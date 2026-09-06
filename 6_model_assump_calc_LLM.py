"""
6_model_assump_calc_LLM.py — Расчёт семантических весов связей рисков (этап 6).

Структурно идентичен шагам 2 и 4: наследует BaseConsensusStep и реализует конвейер
candidates → cleaning → finalization. Отличие от шагов 2/4 — ТОЛЬКО процедура
агрегации консенсуса: 2/4 — мода голосов, 6 — среднее семантических весов по раундам.

Единый вызов LLM для скоринга пар (batch_scores) остаётся единственным хуком,
специфичным для шага 6; весь каркас (раунды, кэш, merge, финализация) — из базового класса.

Вход:  risk_graph.json (шаг 5) + risks_processed.json (шаг 3)
Выход: risk_graph_with_weights.json
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from utils.consensus.base_consensus_step import BaseConsensusStep
from utils.consensus.pipeline import run_step
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json
from utils.general.paths import paths
from utils.llm.llm_client import LLMApiClient

# Единый конфиг пайплайна — ЕДИНСТВЕННЫЙ источник всех параметров
PIPELINE_CONFIG = get_pipeline_config()


# ==================== ДАННЫЕ ====================

@dataclass
class RiskNode:
    """Узел графа рисков."""
    risk_id: str
    name: str
    node_id: str
    node_name: str
    description: str = field(default_factory=str)
    cause: str = field(default_factory=str)
    probability_min: float = 0.0
    probability_max: float = 0.0
    delay_min: float = 0.0
    delay_max: float = 0.0
    budget_min: float = 0.0
    budget_max: float = 0.0
    p50: float = 0.0
    expected_delay: float = 0.0
    expected_budget: float = 0.0


@dataclass
class RiskEdge:
    """Ребро графа рисков."""
    source: str
    target: str
    distance: int = -1
    structural_weight: float = 0.0
    semantic_weight: float = 0.0
    semantic_reason: str = ""
    llm_status: str = ""
    # Вычисляется в шаге 7 (final_weight ниже оставлен для совместимости схемы)
    final_weight: float = 0.0


class RiskGraph:
    """Контейнер графа рисков: загрузка КСГ-графа и реестра, расчёт и сохранение."""

    def __init__(self):
        self.risk_nodes: Dict[str, RiskNode] = {}
        self.risk_edges: List[RiskEdge] = []
        self.risk_node_map: Dict[str, str] = {}  # risk_id -> node_id

    def load_risk_graph(self, json_file: str = None, data: dict = None) -> None:
        """Загружает граф рисков из JSON.

        data: граф рисков из состояния LangGraph (in-memory); если None — читается файл.
        """
        if data is None:
            data = load_json(json_file)
        for risk_id, rdata in data['risks'].items():
            self.risk_nodes[risk_id] = RiskNode(
                risk_id=risk_id,
                name=rdata['name'],
                node_id=rdata['node_id'],
                node_name=rdata['node_name'],
            )
            self.risk_node_map[risk_id] = rdata['node_id']

        seen: set = set()
        for risk_id, deps in data['dependencies'].items():
            for succ in deps['successors']:
                key = (risk_id, succ['risk_id'])
                if key in seen:
                    continue
                seen.add(key)
                self.risk_edges.append(RiskEdge(
                    source=risk_id,
                    target=succ['risk_id'],
                    distance=succ['distance'],
                ))
        print(f"[LOAD] {json_file}")
        print(f"-  Узлов: {len(self.risk_nodes)}, связей: {len(self.risk_edges)}")

    def load_risk_register(self, register_file: str = None, data: dict = None) -> None:
        """Загружает текстовые и числовые поля из enriched risks_processed.json.

        data: реестр рисков из состояния LangGraph (in-memory); если None — читается файл.
        """
        if data is None:
            try:
                data = load_json(register_file)
            except FileNotFoundError:
                print(f"[WARN] Реестр рисков не найден ({register_file}), значения 0.0")
                return
            except Exception as e:
                print(f"[WARN] Ошибка загрузки реестра: {e}")
                return

        applied = 0
        for risk in data.get("risks", []):
            node = self.risk_nodes.get(str(risk.get("risk_number", "")))
            if node is None:
                continue
            node.description = risk.get("description", "")
            node.cause = risk.get("cause", "")
            node.probability_min = _to_float(risk.get("probability_min"))
            node.probability_max = _to_float(risk.get("probability_max"))
            node.delay_min = _to_float(risk.get("delay_min"))
            node.delay_max = _to_float(risk.get("delay_max"))
            node.budget_min = _to_float(risk.get("budget_min"))
            node.budget_max = _to_float(risk.get("budget_max"))
            applied += 1

        print(f"[LOAD] {register_file} (полей применено для {applied} рисков)")

    def unique_pairs(self) -> List[Dict]:
        """Уникальные пары рёбер (source_id,target_id) в порядке тонологического появления."""
        pairs: List[Dict] = []
        seen: set = set()
        for edge in self.risk_edges:
            key = f"{edge.source}_{edge.target}"
            if key in seen:
                continue
            seen.add(key)
            src = self.risk_nodes[edge.source]
            tgt = self.risk_nodes[edge.target]
            pairs.append({
                "source_id": edge.source,
                "source_name": src.name,
                "target_id": edge.target,
                "target_name": tgt.name,
                "distance": edge.distance,
                "structural_weight": edge.structural_weight,
            })
        return pairs

    def apply_semantic_weights(self, weights: List[Dict]) -> None:
        """Применяет агрегированные семантические веса к рёбрам."""
        by_key = {(w.get("source_id"), w.get("target_id")): w for w in weights if isinstance(w, dict)}
        for edge in self.risk_edges:
            w = by_key.get((edge.source, edge.target))
            if not w:
                edge.semantic_weight = 0.0
                edge.semantic_reason = ""
                edge.llm_status = "unknown"
                continue
            edge.semantic_weight = max(0.0, min(1.0, _to_float(w.get("semantic_weight"))))
            edge.semantic_reason = w.get("semantic_reason", "") or ""
            edge.llm_status = "ok" if edge.semantic_weight > 0 else "failed"

    def calculate_final_weights(self) -> None:
        """Итоговый вес — min(1.0, semantic * (1 + structural(d)))."""
        for edge in self.risk_edges:
            edge.final_weight = min(1.0, edge.semantic_weight * (1 + edge.structural_weight))

    def save_risk_graph_with_weights(self, output_file: str) -> None:
        """Сохраняет enriched-граф с весами."""
        risks_dict = {}
        for risk_id, r in self.risk_nodes.items():
            risks_dict[risk_id] = {
                "name": r.name,
                "node_id": r.node_id,
                "node_name": r.node_name,
                "probability": {"min": r.probability_min, "max": r.probability_max, "p50": r.p50},
                "impact": {
                    "delay": {"min": r.delay_min, "max": r.delay_max, "expected": r.expected_delay},
                    "budget": {"min": r.budget_min, "max": r.budget_max, "expected": r.expected_budget},
                },
            }
        edges_list = []
        for edge in self.risk_edges:
            edges_list.append({
                "source": edge.source,
                "target": edge.target,
                "distance": edge.distance,
                "weights": {"structural": 0.0, "semantic": round(edge.semantic_weight, 4)},
                "llm_explanation": {
                    "status": edge.llm_status or "unknown",
                    "semantic_reason": edge.semantic_reason or "Нет пояснения",
                },
            })
        data = {
            "risks": risks_dict,
            "edges": edges_list,
            "model_assumptions": {
                "use_llm": PIPELINE_CONFIG.get("use_llm", True),
                "note": "structural_decay используется в шаге 7",
            },
            "summary": {
                "total_risks": len(self.risk_nodes),
                "total_edges": len(self.risk_edges),
                "avg_semantic_weight": round(_avg([e.semantic_weight for e in self.risk_edges]), 4),
                "note": "structural_weight рассчитывается в шаге 7, final — в шаге 7",
            },
        }
        save_json(output_file, data)
        print(f"[SAVE] {output_file}")
        return data


def _to_float(value) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _risk_data_hash(graph: RiskGraph, pairs: List[Dict]) -> str:
    """SHA-256 по description/cause задействованных рисков (для кэша)."""
    import hashlib
    ids: set = set()
    for p in pairs:
        ids.add(p["source_id"])
        ids.add(p["target_id"])
    lines = []
    for pid in sorted(ids):
        rn = graph.risk_nodes[pid]
        lines.append(f"{pid}:{rn.description}|{rn.cause}")
    raw = "\n".join(lines)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ==================== CONSENSUS-ШАГ 6 ====================

class Step6ConsensusStep(BaseConsensusStep):
    """
    Шаг 6: семантические веса рёбер через LLM (consensus-режим, среднее).

    Наследует BaseConsensusStep и отличается от шагов 2/4 только агрегацией:
    - 2/4: мода голосов (aggregate_consensus_results);
    - 6:   среднее семантических весов по раундам (aggregate_consensus_results).
    Единый hook — скоринг-вызов LLM через batch_scores.
    """

    STEP_PREFIX = "step6"

    def __init__(self):
        super().__init__()
        self.llm = LLMApiClient(model=PIPELINE_CONFIG.get("llm", {}).get("model"))

    def get_items_key(self) -> str:
        return "weights"

    # ---------- КАНДИДАТЫ ----------

    def build_candidate_prompt(self, state: dict) -> str:
        """Формирует batch-промпт для скоринга пар (один раунд)."""
        graph: RiskGraph = state["graph"]
        pairs = state["unique_pairs"]
        prompt_header = state["prompt_header"]
        return self._build_batch_prompt(prompt_header, graph, pairs)

    def _build_batch_prompt(self, prompt_header: str, graph: RiskGraph, pairs: List[Dict]) -> str:
        """Полный скоринг-промпт: заголовок + справочник рисков + список пар."""
        seen_ids: set = set()
        for p in pairs:
            seen_ids.add(p['source_id'])
            seen_ids.add(p['target_id'])

        desc_lines = []
        for pid in sorted(seen_ids):
            rn = graph.risk_nodes[pid]
            desc_lines.append(f"{pid}: {rn.name}")
            if rn.description:
                desc_lines.append(f"  описание: {rn.description}")
            if rn.cause:
                desc_lines.append(f"  причина: {rn.cause}")
        desc_block = "\n".join(desc_lines) if desc_lines else ""

        pairs_text = ""
        for idx, pair in enumerate(pairs, 1):
            pairs_text += (
                f"{idx}. {pair['source_id']} | {pair['target_id']}\n"
                f"Название: {pair['source_name']} -> {pair['target_name']}\n"
                f"Расстояние: {pair['distance']}, Структурный вес: {pair['structural_weight']}\n"
            )

        if desc_block:
            return (prompt_header.format(pair_count=len(pairs))
                    + "\n\nСПРАВОЧНИК РИСКОВ (по ID, данные выше):\n" + desc_block + "\n\n" + pairs_text)
        return prompt_header.format(pair_count=len(pairs)) + "\n\n" + pairs_text

    def run_consensus_candidates(self, state: dict, consensus_rounds: int, max_concurrent: int = 4) -> dict:
        """
        Единственный hook-исключение шага 6: скоринг пар через self.llm.batch_scores.

        Шаги 2/4 используют единый run_parallel_consensus (текстовые кандидаты);
        шаг 6 зовёт скоринг-вызов LLM. Каркас round/агрегация/кэш — из базового класса.
        """
        prompt = self.build_candidate_prompt(state)
        ids = [f"{p['source_id']}|{p['target_id']}" for p in state["unique_pairs"]]

        print(f"   LLM запуск {consensus_rounds} последовательных запросов...")

        raw_results = []
        for rnd in range(consensus_rounds):
            try:
                resp = self.llm.batch_scores(prompt, ids)
                pairs_count = len(resp) if resp else 0
                raw_results.append({"round": rnd, "result": resp or {}, "attempt": 1})
                print(f"   LLM запрос {rnd}: {pairs_count}")
            except Exception as e:
                print(f"   LLM запрос {rnd}: ошибка {e}")
                raw_results.append({"round": rnd, "result": {}, "attempt": 1})

        # Агрегация — среднее (как и в шагах 2/4 результат проходит через aggregate_consensus_results)
        aggregated = self.aggregate_consensus_results(raw_results)

        return {
            "candidate_weights": aggregated,
            "consensus_output": aggregated,
            "consensus_raw_rounds": raw_results,
        }

    # ---------- АГРЕГАЦИЯ (СРЕДНЕЕ) ----------

    def aggregate_consensus_results(self, raw_results: list) -> dict:
        """Среднее семантических весов по раундам (отличие от шагов 2/4, где мода)."""
        from collections import defaultdict
        per_pair_scores: Dict[str, List[float]] = defaultdict(list)
        per_pair_reasons: Dict[str, List[str]] = defaultdict(list)

        for rd in raw_results:
            if not isinstance(rd, dict):
                continue
            for key, item in (rd.get("result") or {}).items():
                if not isinstance(item, dict):
                    continue
                per_pair_scores[key].append(_to_float(item.get("score")))
                reason = item.get("reason", "")
                if reason:
                    per_pair_reasons[key].append(reason)

        weights = []
        for key in sorted(per_pair_scores):
            vals = per_pair_scores[key]
            avg = sum(vals) / len(vals) if vals else 0.0
            src, tgt = key.split("|", 1)
            reason = next((r for r in reversed(per_pair_reasons[key]) if r), "")
            weights.append({
                "source_id": src,
                "target_id": tgt,
                "semantic_weight": round(avg, 6),
                "semantic_reason": reason,
                "rounds": len(vals),
                "generated_by": "LLM",
            })
        return {"weights": weights}

    # ---------- CLEANING ----------

    def run_cleaning(self, state: dict) -> dict:
        """Нормализация: вес в [0,1], отсутствующие пары дополняются нулём."""
        cand = state.get("candidate_weights", {})
        items = cand.get("weights", []) if isinstance(cand, dict) else cand if isinstance(cand, list) else []

        result = []
        seen: set = set()
        for w in items:
            if not isinstance(w, dict):
                continue
            pair_key = f"{w.get('source_id')}|{w.get('target_id')}"
            w["semantic_weight"] = max(0.0, min(1.0, _to_float(w.get("semantic_weight"))))
            w["generated_by"] = "LLM"
            result.append(w)
            seen.add(pair_key)

        for p in state["unique_pairs"]:
            pair_key = f"{p['source_id']}|{p['target_id']}"
            if pair_key not in seen:
                result.append({"source_id": p["source_id"], "target_id": p["target_id"],
                               "semantic_weight": 0.0, "semantic_reason": "",
                               "rounds": 0, "generated_by": "LLM"})
        return {"cleaned_weights": result}

    # ---------- FINALIZATION / КЭШ ----------

    def get_cache_context(self, state: dict) -> dict:
        return {
            "pairs": state["unique_pairs"],
            "prompt_header": state.get("prompt_header", ""),
            "risk_data_hash": state.get("risk_data_hash", ""),
        }

    def get_final_items(self, state: dict) -> list:
        return state.get("cleaned_weights", [])

    def run_finalization(self, state: dict) -> dict:
        """Финализация + расчёт финальных весов и применения к графу."""
        result = super().run_finalization(state)
        items_key = self.get_items_key()
        final = result.get(f"final_{items_key}", [])
        graph: RiskGraph = state["graph"]
        graph.apply_semantic_weights(final)
        graph.calculate_final_weights()
        return result


# ==================== MAIN ====================

def main(risk_graph: dict = None, risks: dict = None) -> dict:
    """Точка входа.

    risk_graph: граф рисков из шага 5 (in-memory из состояния LangGraph);
    risks:      реестр рисков из шага 3 (in-memory). Если None — читаются с диска.
    """
    sys.stdout.reconfigure(encoding="utf-8")

    INPUT_FILE = paths.risk_graph_json
    OUTPUT_DIR = paths.step_dirs[6]
    os.makedirs(paths.to_str(OUTPUT_DIR), exist_ok=True)

    graph = RiskGraph()
    graph.load_risk_graph(paths.to_str(INPUT_FILE), data=risk_graph)
    graph.load_risk_register(paths.to_str(paths.risks_processed_json), data=risks)

    # Чтение промпта и построение справочника рисков для кэша
    prompt_header = ""
    try:
        with open(paths.to_str(paths.prompt_file(6)), 'r', encoding='utf-8') as f:
            prompt_header = f.read()
    except FileNotFoundError:
        print(f"[WARN] Файл промпта не найден: {paths.prompt_file(6)}")

    unique_pairs = graph.unique_pairs()

    if not PIPELINE_CONFIG.get("use_llm", True):
        print("[SKIP] use_llm=False — семантические веса = 0.0")
        final_weights = []
        for p in unique_pairs:
            final_weights.append({"source_id": p["source_id"], "target_id": p["target_id"],
                                  "semantic_weight": 0.0, "semantic_reason": "", "generated_by": "LLM"})
        graph.apply_semantic_weights(final_weights)
        graph.calculate_final_weights()
    else:
        initial_state = {
            "pipeline_config": PIPELINE_CONFIG,
            "step_label": "step6",
            "graph": graph,
            "unique_pairs": unique_pairs,
            "prompt_header": prompt_header,
            "risk_data_hash": _risk_data_hash(graph, unique_pairs),
            "iterations": 0,
            "save_metadata": {},
        }
        step = Step6ConsensusStep()
        cached = step.check_cache(initial_state)
        if cached is not None:
            graph.apply_semantic_weights(cached)
            graph.calculate_final_weights()
            # Аудит из кэша
            items_key = step.get_items_key()
            result = {
                items_key: cached,
                f"candidate_{items_key}": {"weights": cached} if isinstance(cached, list) else cached,
                "consensus_raw_rounds": [],
            }
        else:
            hooks = step.build_hooks()
            result = run_step(dict, hooks, initial_state, label="step6")
            graph.apply_semantic_weights(result.get("final_weights", []))
            graph.calculate_final_weights()

    weights_doc = graph.save_risk_graph_with_weights(paths.to_str(paths.risk_graph_with_weights_json))

    # Сохранение аудита консенсуса
    rounds = PIPELINE_CONFIG.get("model_assumptions", {}).get("consensus", {}).get("rounds", 1) or 1
    try:
        items_key = step.get_items_key()

        # Формируем детальный аудит по каждому раунду LLM
        raw_rounds = result.get("consensus_raw_rounds", [])
        rounds_detail = []
        for rnd_data in raw_rounds:
            rnd_num = rnd_data.get("round", 0)
            rnd_result = rnd_data.get("result", {})
            # Переводим dict {pair_id: {score, reason}} в список для читаемости
            scores_detail = []
            if isinstance(rnd_result, dict):
                for pair_id, data in rnd_result.items():
                    if isinstance(data, dict):
                        scores_detail.append({
                            "pair_id": pair_id,
                            "semantic_weight": data.get("score"),
                            "reason": data.get("reason", ""),
                        })
            rounds_detail.append({
                "round": rnd_num + 1,
                "pairs_scored": len(scores_detail),
                "scores": scores_detail,
            })

        # Финальные агрегированные веса
        final_weights = result.get(f"candidate_{items_key}", {})
        if isinstance(final_weights, dict):
            final_weights = final_weights.get("weights", [])

        audit_data = {
            "pipeline_config": {
                "model": PIPELINE_CONFIG.get("llm", {}).get("model", "N/A"),
                "consensus_rounds": rounds,
                "pairs_count": len(unique_pairs),
            },
            "rounds_detail": rounds_detail,
            "final_weights": final_weights,
        }
        save_json(
            os.path.join(paths.to_str(OUTPUT_DIR), "risk_semantic_llm_survey_audit.json"),
            audit_data,
        )
        print(f"[SAVE] risk_semantic_llm_survey_audit.json")
    except Exception as e:
        print(f"[WARN] Не удалось сохранить аудит: {e}", file=sys.stderr)

    print(f"\n[RESULTS]")
    print(f"-  Рисков: {len(graph.risk_nodes)}")
    print(f"-  Рёбер: {len(graph.risk_edges)}")
    print(f"-  Консенсус: {rounds} раундов (усреднение)")

    if graph.risk_edges:
        avg_sem = sum(e.semantic_weight for e in graph.risk_edges) / len(graph.risk_edges)
        print(f"-  Средний семантический вес: {avg_sem:.3f}")
    print(f"-  structural_weight и final_weight рассчитываются в шаге 7")

    # Возвращаем взвешенный граф для состояния LangGraph (JSON уже сохранён)
    return weights_doc


if __name__ == "__main__":
    main()