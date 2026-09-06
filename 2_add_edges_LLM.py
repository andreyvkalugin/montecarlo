"""
2_add_edges_LLM.py — Поиск LLM-связей (этап 2). Хуки специфичной логики этапа.

После рефакторинга: ТОЛЬКО специфика этапа 2.
Несколько LLM-запросов (consensus) агрегируются в моду/частоту голосов,
после чего результат идёт напрямую в финализацию (строитель/критик удалены).

Вход:  1_data/csg_tasks_wbs.json
Выход: 2_data/csg_edges_final.json (расчётный, компактные рёбра)
       2_data/csg_edges_llm_survey_audit.json (диагностика: ответы LLM + мода, в расчётах не участвует)
"""
import json
import os
import sys

from utils.consensus.base_consensus_step import BaseConsensusStep
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json
from utils.llm.llm_utils import normalize_llm_response
from utils.consensus.pipeline import (
    run_step,
    ensure_acyclic,
    creates_cycle,
    FORBIDDEN_IDS,
)
from utils.general.mermaid_export import export_mermaid_graph, format_edges_for_prompt
from utils.general.paths import paths

# State для шага 2
from typing import TypedDict

class AddEdgesState(TypedDict):
    """Состояние для шага 2 — добавление LLM-связей."""
    input_data: dict
    pipeline_config: dict
    step_label: str
    iterations: int
    save_metadata: dict
    nodes: list
    wbs_edges: list
    candidate_edges: list
    consensus_output: dict
    consensus_raw_rounds: list
    cleaned_edges: list
    selected_edges: list
    edge_limit: int
    final_edges: list

# Единый конфиг пайплайна
PIPELINE_CONFIG = get_pipeline_config()

# ===== НАСТРОЙКИ =====
INPUT_JSON = paths.csg_tasks_wbs_json
OUTPUT_JSON = paths.csg_edges_final_json
OUTPUT_MERMAID = paths.csg_edges_final_mermaid
OUT_DIR = paths.step_dirs[2]


# ===================================================================
#  ЭТАП 2 CONSISTEP — специфика шага 2
# ===================================================================

class Step2ConsensusStep(BaseConsensusStep):
    """Шаг 2: поиск LLM-связей между вершинами графа."""

    STATE_CLASS = AddEdgesState
    STEP_PREFIX = "step2"

    def get_items_key(self) -> str:
        return "edges"

    # ===================================================================
    #  КАНДИДАТЫ
    # ===================================================================

    def build_candidate_prompt(self, state: dict) -> str:
        """Формирует промпт для одного раунда consensus."""
        nodes = state["nodes"]
        wbs_edges = state["wbs_edges"]

        ids_flat = ", ".join(f'"{n["id"]}"' for n in nodes)
        names = "\n".join(f'{n["id"]}: {n["name"]}' for n in nodes)

        try:
            wbs_str = format_edges_for_prompt(wbs_edges)
        except Exception:
            wbs_str = ""

        prompt_file = str(paths.prompt_file(2))
        system_prompt = self._load_prompt(prompt_file)
        if not system_prompt:
            return ""

        # Максимум LLM-связей берём из конфига (не жёсткая константа)
        config = state.get("pipeline_config", {})
        max_edges = config.get("llm_edges", {}).get("max_additional_edges", 50)

        return (system_prompt
                .replace("[WBS_STR]", wbs_str)
                .replace("[IDS_FLAT]", ids_flat)
                .replace("[NAMES]", names)
                .replace("[MAX_EDGES]", str(max_edges)))

    def aggregate_consensus_results(self, raw_results: list) -> dict:
        """Агрегирует голоса за рёбра из раундов (мода — рёбра с максимумом голосов сверху)."""
        from collections import defaultdict

        edge_votes = defaultdict(lambda: {"votes": 0, "reasons": []})

        for round_data in raw_results:
            if not isinstance(round_data, dict):
                continue

            edges = round_data.get("result", [])
            rnd = round_data.get("round", -1)

            # Дедупликация внутри раунда
            seen_edges = {}
            for e in edges:
                if not isinstance(e, dict):
                    continue
                s = str(e.get("source", "")).strip()
                t = str(e.get("target", "")).strip()
                if not s or not t or s == t:
                    continue
                key = f"{s}|{t}"
                typ = e.get("type", "FS")
                reason = e.get("reason", "")

                if key not in seen_edges:
                    seen_edges[key] = {"type": typ, "reasons": []}
                if reason:
                    seen_edges[key]["reasons"].append({
                        "round": rnd, "reason": reason, "type": typ,
                    })

            for key, info in seen_edges.items():
                edge_votes[key]["votes"] += 1
                for r in info.get("reasons", []):
                    edge_votes[key]["reasons"].append(r)

        # Формируем итоговый список
        consensus_output = {
            "llm_model": "",
            "consensus_rounds": len(raw_results),
            "edges": [],
        }
        for key, data in edge_votes.items():
            s, t = key.split("|")
            consensus_output["edges"].append({
                "source": s, "target": t,
                "votes": data["votes"],
                "consensus_total": len(raw_results),
                "reasons": data["reasons"],
            })
        consensus_output["edges"].sort(key=lambda x: x["votes"], reverse=True)

        return consensus_output

    def parse_candidate_response(self, text: str) -> list:
        """Парсит ответ LLM кандидатов."""
        if not text:
            return []
        return normalize_llm_response(text)

    # ===================================================================
    #  CLEANING
    # ===================================================================

    def run_cleaning(self, state: dict) -> dict:
        """Фильтрация кандидатов + лимит."""
        EDGE_LIMIT_PERCENT = float(PIPELINE_CONFIG.get("llm_edges", {}).get("edge_limit_percent", 0.5))
        EDGE_LIMIT_MIN = int(PIPELINE_CONFIG.get("llm_edges", {}).get("edge_limit_min", 10))
        EDGE_LIMIT_MAX = int(PIPELINE_CONFIG.get("llm_edges", {}).get("edge_limit_max", 100))

        def _compute_limit(n: int) -> int:
            total_pairs = n * (n - 1) // 2
            capped = min(EDGE_LIMIT_MAX, round(EDGE_LIMIT_PERCENT * total_pairs))
            return max(EDGE_LIMIT_MIN, capped)

        nodes = state["nodes"]
        wbs_edges = state["wbs_edges"]
        raw_candidates = state.get("candidate_edges", {})

        # candidate_edges может быть dict с полем "edges" или уже список
        if isinstance(raw_candidates, dict):
            candidates = raw_candidates.get("edges", [])
        elif isinstance(raw_candidates, list):
            candidates = raw_candidates
        else:
            candidates = []

        # Преобразуем wbs_edges в нормальный формат (если это строки)
        if wbs_edges and isinstance(wbs_edges[0], str):
            graph_edges = [{"source": src, "target": tgt} for src, tgt in wbs_edges]
        else:
            graph_edges = list(wbs_edges)

        valid_ids = {n["id"] for n in nodes}
        existing_pairs = {(e["source"], e["target"]) for e in graph_edges}
        seen = set()
        cleaned = []

        for e in candidates:
            s = str(e.get("source", "")).strip()
            t = str(e.get("target", "")).strip()
            if s in FORBIDDEN_IDS or t in FORBIDDEN_IDS:
                continue
            if not s or not t or s not in valid_ids or t not in valid_ids:
                continue
            if s == t or (s, t) in existing_pairs or (s, t) in seen:
                continue
            if creates_cycle(graph_edges, s, t):
                continue
            cleaned.append(e)
            seen.add((s, t))
            graph_edges.append({"source": s, "target": t})

        limit = _compute_limit(len(nodes))
        selected = cleaned[:limit]

        print(f"\n[ОТБОР] Чистых кандидатов: {len(cleaned)}, лимит={limit}, отобрано {len(selected)}.")

        # Статистика отбора и ответы LLM пишутся в аудит-файл (main → _save_llm_survey_audit)
        return {
            "cleaned_edges": cleaned,
            "selected_edges": selected,
            "edge_limit": limit,
        }

    # ===================================================================
    #  FINALIZATION
    # ===================================================================

    def get_final_items(self, state: dict) -> list:
        # Итоговые рёбра — отобранные после cleaning (мода голосов + лимит),
        # с гарантией ацикличности (применяется ДО сохранения кэша, чтобы
        # кэш и финальный результат всегда совпадали).
        selected = state.get("selected_edges", [])
        wbs_edges = state.get("wbs_edges", [])
        return ensure_acyclic(selected, wbs_edges)

    def get_cache_context(self, state: dict) -> dict:
        return {
            "nodes": state["nodes"],
            "wbs_edges": state.get("wbs_edges", []),
            "llm_edges": state.get("pipeline_config", {}).get("llm_edges", {}),
        }

    def run_finalization(self, state: dict) -> dict:
        """Финализация. ensure_acyclic выполняется в get_final_items, поэтому
        в кэш и финальный результат попадают одни и те же (ацикличные) рёбра."""
        return super().run_finalization(state)

    # Утилита _load_prompt унаследована от BaseConsensusStep


# ===================================================================
#  MAIN
# ===================================================================

def main():
    """Точка входа."""
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if not os.path.isfile(INPUT_JSON):
        print(f"[ERROR]  Файл не найден: {INPUT_JSON}", file=sys.stderr)
        sys.exit(1)

    print("\n[STEP] ДОБАВЛЕНИЕ LLM-СВЯЗЕЙ (консенсус нескольких запросов → мода)...")
    print(f"[LOAD] {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        wbs_graph = json.load(f)
    nodes = wbs_graph.get("nodes", [])
    wbs_edges = wbs_graph.get("edges", [])
    metadata = wbs_graph.get("metadata", {})

    initial_state = {
        "input_data": {},
        "pipeline_config": PIPELINE_CONFIG,
        "step_label": "step2",
        "nodes": nodes,
        "wbs_edges": list(wbs_edges),
        "candidate_edges": [],
        "consensus_output": {},
        "cleaned_edges": [],
        "selected_edges": [],
        "edge_limit": 0,
        "final_edges": [],
        "iterations": 0,
        "save_metadata": {},
    }

    # Создаём builder и проверяем кэш
    step = Step2ConsensusStep()
    cached = step.check_cache(initial_state)

    # Сырые ответы LLM-опроса и агрегация (мода) доступны только при полном
    # прогоне; при кэш-хите они берутся из ранее сохранённого аудит-файла.
    consensus_raw_rounds = []
    consensus_output = {}
    selection = {}

    if cached is not None:
        final_edges = cached
        # Из кэша недоступны сырые ответы раундов, но голоса и причины
        # сохраняются в самих рёбрах — аудит можно восстановить по ним.
        consensus_output = {"edges": [e for e in cached if isinstance(e, dict)]}
        selection = {"note": "Запуск из кэша: сырые ответы раундов и статистика отбора недоступны."}
    else:
        hooks = step.build_hooks()
        result = run_step(AddEdgesState, hooks, initial_state, label="step2")
        final_edges = result.get("final_edges", [])
        consensus_raw_rounds = result.get("consensus_raw_rounds", []) or []
        consensus_output = result.get("consensus_output", {}) or {}
        candidates = (result.get("candidate_edges") or {}).get("edges", []) or []
        selection = {
            "edge_limit": result.get("edge_limit", 0),
            "total_candidates": len(candidates),
            "clean_candidates": len(result.get("cleaned_edges") or []),
            "selected": [_audit_edge(e) for e in (result.get("selected_edges") or [])],
        }

    # Компактные рёбра для расчётного файла (только source/target/type/generated_by)
    seen = {(e.get("source"), e.get("target")) for e in wbs_edges}
    all_edges = [dict(e) for e in wbs_edges]
    all_edges += [_compact_edge(e) for e in final_edges
                  if (e.get("source"), e.get("target")) not in seen]

    # Диагностический (аудит) файл — ответы раундов, мода и статистика отбора.
    _save_llm_survey_audit(
        consensus_raw_rounds=consensus_raw_rounds,
        consensus_output=consensus_output,
        selection=selection,
        final_edges=final_edges,
        wbs_edges=wbs_edges,
    )

    # Источник рёбер: шаг 1 даёт только WBS, здесь добавляются LLM-связи,
    # поэтому перечисляем фактические уникальные generated_by из итоговых рёбер.
    metadata = {k: v for k, v in metadata.items() if k != "edge_source"}
    edge_sources = sorted({e.get("generated_by") or "WBS" for e in all_edges})

    # Сохраняем final graph (компактный — только для расчётов)
    final_graph = {
        "nodes": nodes,
        "edges": all_edges,
        "metadata": {
            **metadata,
            "total_edges": len(all_edges),
            "wbs_edges": len(wbs_edges),
            "llm_edges": len([e for e in all_edges if e.get("generated_by") == "LLM"]),
            "edge_sources": edge_sources,
            "llm_model": PIPELINE_CONFIG.get("llm", {}).get("model"),
            "version": "2.4-consensus",
        }
    }
    save_json(OUTPUT_JSON, final_graph)
    print(f"[SAVE] {OUTPUT_JSON}")

    # Экспорт Mermaid
    os.makedirs(paths.to_str(OUT_DIR), exist_ok=True)
    export_mermaid_graph(nodes, all_edges, paths.to_str(OUTPUT_MERMAID))

    print(f"\n[RESULTS]")
    print(f"-  Узлов: {len(nodes)}")
    print(f"-  WBS-рёбер: {len(wbs_edges)}")
    print(f"-  Итоговых рёбер: {len(all_edges)}")
    fs_count = sum(1 for e in all_edges if e.get("type") == "FS")
    ss_count = sum(1 for e in all_edges if e.get("type") == "SS")
    print(f"-  Типы: FS={fs_count}, SS={ss_count}")

    # Очистка временных промпт-файлов после завершения шага
    _clear_prompts_temp()


def _edge_type(edge: dict) -> str:
    """
    Тип связи ребра. Агрегация consensus хранит тип только в голосах (reasons),
    поэтому при отсутствии верхнеуровневого поля берём преобладающий тип голосов.
    """
    typ = edge.get("type")
    if typ:
        return typ
    types = [r.get("type") for r in (edge.get("reasons") or []) if r.get("type")]
    if not types:
        return "FS"
    return max(set(types), key=types.count)


def _compact_edge(edge: dict) -> dict:
    """Приводит ребро к компактному формату расчётного файла."""
    return {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": _edge_type(edge),
        "generated_by": edge.get("generated_by") or "WBS",
    }


def _save_llm_survey_audit(consensus_raw_rounds, consensus_output, selection,
                           final_edges, wbs_edges) -> None:
    """
    Сохраняет диагностический (аудит) файл шага 2.

    Содержит полные «сырые» ответы каждого раунда LLM-опроса, их агрегацию
    (моду голосов) и статистику отбора. Используется ТОЛЬКО для аудита и не
    участвует в дальнейших расчётах — расчётный файл это csg_edges_final.json.
    """
    audit_path = paths.to_str(paths.csg_edges_llm_survey_audit_json)
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)

    full_run = bool(consensus_raw_rounds)

    # Прогон из кэша не должен понижать полный аудит ( со сырыми ответами раундов).
    if not full_run and os.path.isfile(audit_path):
        existing = load_json(audit_path) or {}
        if existing.get("source") == "fresh-run":
            print(f"[SKIP]  Полный аудит уже сохранён с прошлого прогона, перезапись не нужна: {audit_path}")
            return

    audit = {
        "description": "Диагностика шага 2 — опрос LLM, агрегация (мода) и отбор. Только для аудита, в расчётах не участвует.",
        "llm_model": PIPELINE_CONFIG.get("llm", {}).get("model"),
        "consensus_rounds": len(consensus_raw_rounds or []),
        "source": "fresh-run" if full_run else "cache",
        "rounds": consensus_raw_rounds or [],
        "mode": _mode_summary(consensus_output),
        "selection": selection or {},
        "final_llm_edges": [_audit_edge(e) for e in (final_edges or [])],
        "wbs_edges": [_compact_edge(e) for e in wbs_edges],
    }
    save_json(audit_path, audit)
    print(f"[SAVE] {audit_path} (аудит, не участвует в расчётах)")


def _audit_edge(edge: dict) -> dict:
    """Представление ребра для аудит-файла: топология + детали голосования."""
    return {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": _edge_type(edge),
        "votes": edge.get("votes", 0),
        "consensus_total": edge.get("consensus_total", 0),
        "reasons": edge.get("reasons", []),
        "generated_by": edge.get("generated_by") or "LLM",
    }


def _mode_summary(consensus_output: dict) -> list:
    """Сводка агрегации голосов: рёбра с частотой голосов (мода) и причинами."""
    return [_audit_edge(e) for e in ((consensus_output or {}).get("edges", []) or [])]


def _clear_prompts_temp() -> None:
    """Удаляет временные файлы промптов из cache/llm_prompts_temp."""
    import shutil

    dir_path = paths.to_str(paths.cache / "llm_prompts_temp")
    if not os.path.isdir(dir_path):
        return

    removed = 0
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        try:
            if os.path.isfile(full):
                os.remove(full)
                removed += 1
            elif os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
        except OSError:
            pass

    if removed:
        print(f"[CLEAN] Удалено временных промпт-файлов: {removed}")


if __name__ == "__main__":
    main()