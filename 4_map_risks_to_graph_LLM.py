"""
4_map_risks_to_graph_LLM.py — Привязка рисков к вершинам графа (этап 4). Хуки.

После рефакторинга: ТОЛЬКО специфика этапа 4.
Несколько LLM-запросов (consensus) агрегируются в моду голосов (по каждому
риску выбирается вершина с максимальным числом голосов), после чего результат
идёт напрямую в финализацию (строитель/критик удалены).

Вход: risks_processed.json, csg_edges_final.json
Выход: risk_mapping_final.json
"""
import json
import os
import re
import sys
from collections import OrderedDict

from utils.consensus.base_consensus_step import BaseConsensusStep
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json
from utils.consensus.pipeline import (
    BaseStepState,
    run_step,
    PipelineLogger,
)
from utils.general.paths import paths

# Единый конфиг пайплайна
PIPELINE_CONFIG = get_pipeline_config()

# ===== НАСТРОЙКИ =====
RISKS_FILE = paths.risks_processed_json
GRAPH_JSON = paths.csg_edges_final_json
OUTPUT_JSON = paths.risk_mapping_json
OUT_DIR = paths.step_dirs[4]


# ===================================================================
#  STATE ЭТАПА 4
# ===================================================================

class Step4State(BaseStepState):
    """Состояние для шага 4 — маппинг рисков к узлам графа."""
    # --- Входные данные ---
    risks: list
    nodes: list
    node_ids: list

    # --- ШАГ 1: Кандидаты (consensus) ---
    candidate_mappings: list
    consensus_output: dict
    consensus_raw_rounds: list

    # --- ШАГ 2: Очистка (мода) ---
    cleaned_mappings: list

    # --- ШАГ 3: Финализация ---
    final_mappings: list

    # --- Специфичные поля ---
    fallback_node: str
    fallback_name: str


# ===================================================================
#  ЭТАП 4 CONSISTEP — специфика шага 4
# ===================================================================

class Step4ConsensusStep(BaseConsensusStep):
    """Шаг 4: привязка рисков к вершинам графа."""

    STATE_CLASS = Step4State
    STEP_PREFIX = "step4"

    def get_items_key(self) -> str:
        return "mappings"

    # ===================================================================
    #  КАНДИДАТЫ
    # ===================================================================

    def build_candidate_prompt(self, state: dict) -> str:
        """Формирует промпт для одного раунда consensus."""
        nodes = state["nodes"]
        risks = state["risks"]

        names = "\n".join(f'{n["id"]}: {n["name"]}' for n in nodes)

        # Листовые (конечные) вершины WBS — та же логика, что и в run_cleaning
        leaf_ids, _ = self._compute_leaf_structure(nodes)
        leaf_ids_str = ", ".join(f'"{i}"' for i in leaf_ids)

        risk_lines = []
        for r in risks:
            cause_text = f"; Причина: {r.get('cause', '')}" if r.get("cause") else ""
            risk_lines.append(
                f'  Номер: {r["risk_number"]} | Название: {r["risk_name"]}{cause_text}'
            )
        risks_block = "\n".join(risk_lines)

        prompt_file = os.path.join(str(paths.prompt), "4_llm_mapping_prompt.txt")
        system_prompt = self._load_prompt(prompt_file)
        if not system_prompt:
            return ""

        return (system_prompt
                .replace("[NAMES]", names)
                .replace("[LEAF_IDS]", leaf_ids_str)
                .replace("[RISKS_BLOCK]", risks_block))

    def _compute_leaf_structure(self, nodes: list):
        """Возвращает листовые вершины WBS и карту parent → children.

        Родителем (декомпозируемым) считается узел, у которого есть потомок
        с id, начинающимся с "<id>.". Листовая вершина — узел без таких потомков.

        Returns:
            (leaf_ids: List[str], parent_to_children: Dict[str, List[str]])
        """
        parent_to_children = {}
        has_child = set()
        for n in nodes:
            nid = str(n["id"]).strip()
            if not nid:
                continue
            prefix = nid + "."
            for n2 in nodes:
                n2id = str(n2["id"]).strip()
                if n2id.startswith(prefix):
                    parent_to_children.setdefault(nid, []).append(n2id)
                    has_child.add(nid)
        leaf_ids = [str(n["id"]) for n in nodes if str(n["id"]).strip() and str(n["id"]) not in has_child]
        # Дедупликация детей (если потомок встречался несколько раз)
        parent_to_children = {k: list(dict.fromkeys(v)) for k, v in parent_to_children.items()}
        return leaf_ids, parent_to_children

    def parse_candidate_response(self, text: str) -> list:
        """Парсит ответ LLM кандидатов."""
        return self._parse_step4_response(text)

    def _parse_step4_response(self, response) -> list:
        """Парсит ответ LLM в список маппингов (несколько стратегий для robustности)."""
        if response is None:
            return []

        if isinstance(response, dict):
            if "mappings" in response:
                return response["mappings"] if isinstance(response["mappings"], list) else []
            if "edges" in response:
                return response["edges"]
            if response.get("risk_number"):
                return [response]
            return []

        if isinstance(response, list):
            return response

        if not isinstance(response, str):
            return []

        text = response.strip()
        if not text:
            return []

        # Стратегия 1: ищем ```json ... ``` блок
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```\s*$', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(text)
            if isinstance(parsed, dict):
                if "mappings" in parsed:
                    return parsed["mappings"]
                elif "edges" in parsed:
                    return parsed["edges"]
                elif parsed.get("risk_number"):
                    return [parsed]
            elif isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Стратегия 2: ищем JSON в тексте (без маркдаун)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = text[start:end + 1]
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    if "mappings" in parsed:
                        return parsed["mappings"]
                    elif parsed.get("risk_number"):
                        return [parsed]
                elif isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Стратегия 3: regex для извлечения объектов маппингов
        mapping_pattern = r'\{\s*"risk_number"\s*:\s*"([^"]+)"[^}]*\}'
        mappings = []
        for match in re.finditer(mapping_pattern, text):
            risk_num = match.group(1)
            block_end = min(match.end() + 200, len(text))
            block = text[match.start():block_end]
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "risk_number" in parsed:
                    mappings.append(parsed)
            except json.JSONDecodeError:
                pass

        return mappings if mappings else []

    def aggregate_consensus_results(self, raw_results: list) -> dict:
        """Агрегирует голоса за маппинги рисков (выбор моды по каждому риску)."""
        # Собираем все предложения по каждому риску
        risk_node_votes = {}
        risk_order = []

        for round_data in raw_results:
            if not isinstance(round_data, dict):
                continue

            # Дедупликация: 1 голос за (risk, node) в одном раунде
            dedup = set()
            for m in round_data.get("result", []):
                risk_num = m.get("risk_number", "")
                risk_name = m.get("risk_name", "")
                if not risk_num:
                    continue
                if risk_num not in risk_node_votes:
                    risk_node_votes[risk_num] = {"risk_name": risk_name, "nodes": {}}
                    risk_order.append(risk_num)

                for node in (m.get("matched_nodes") or []):
                    node = str(node).strip()
                    if not node:
                        continue
                    dedup_key = (risk_num, node)
                    if dedup_key in dedup:
                        continue
                    dedup.add(dedup_key)
                    cell = risk_node_votes[risk_num]["nodes"].setdefault(
                        node, {"votes": 0, "reasons": [], "first_round": None})
                    cell["votes"] += 1
                    if cell["first_round"] is None:
                        cell["first_round"] = round_data.get("round", -1)
                    reason = m.get("reason", "")
                    if reason:
                        cell["reasons"].append({"round": round_data.get("round", -1), "reason": reason})

        # Для каждого риска выбираем моду
        all_mappings = []
        consensus_total = len([r for r in raw_results if isinstance(r, dict)])
        consensus_total = max(consensus_total, 1)

        for risk_num in risk_order:
            info = risk_node_votes[risk_num]
            nodes_map = info["nodes"]
            ordered = sorted(nodes_map.items(),
                             key=lambda kv: (-kv[1]["votes"], kv[1]["first_round"] or 0))
            if not ordered:
                continue

            chosen_node, chosen_data = ordered[0]
            is_mode = len(ordered) > 1 and ordered[0][1]["votes"] > ordered[1][1]["votes"]

            all_mappings.append({
                "risk_number": risk_num,
                "risk_name": info["risk_name"],
                "matched_nodes": [chosen_node],
                "votes": chosen_data["votes"],
                "consensus_total": consensus_total,
                "mode": is_mode,
                "mode_reasons": chosen_data["reasons"],
                "all_votes": {node: cell["votes"] for node, cell in nodes_map.items()},
            })

        return {
            "llm_model": "",
            "consensus_rounds": consensus_total,
            "mappings": all_mappings,
        }

    # ===================================================================
    #  CLEANING
    # ===================================================================

    def run_cleaning(self, state: dict) -> dict:
        """Фильтрация кандидатов + декомпозиция вершин."""
        nodes = state["nodes"]
        raw_candidates = state.get("candidate_mappings", [])

        # candidate_mappings может быть dict с полем "mappings" или уже список
        if isinstance(raw_candidates, dict):
            candidates = raw_candidates.get("mappings", [])
        elif isinstance(raw_candidates, list):
            candidates = raw_candidates
        else:
            candidates = []

        # Строим карту: parent → [children] и листовые вершины (единая логика с промптом)
        leaf_ids, parent_to_children = self._compute_leaf_structure(nodes)
        valid_leaf_ids = set(leaf_ids)

        logger = PipelineLogger("step4")
        seen = set()
        cleaned = []

        for m in candidates:
            risk_num = m.get("risk_number", "")
            matched = m.get("matched_nodes", [])

            if not risk_num or not matched:
                continue

            # Заменяем родительские вершины на их детей
            final_matched = []
            for nid in matched:
                if nid in valid_leaf_ids:
                    final_matched.append(nid)
                elif nid in parent_to_children:
                    final_matched.extend(parent_to_children[nid])

            if not final_matched:
                continue

            # Дедупликация по риску
            key = risk_num
            if key in seen:
                continue
            seen.add(key)

            # Не более 3 вершин
            final_matched = list(dict.fromkeys(final_matched))[:3]
            m["matched_nodes"] = final_matched
            cleaned.append(m)

        logger.cleaning_done(len(cleaned))

        # Save consensus в корень папки шага 4 (for downstream steps and audit)
        try:
            mapping_report = [dict(m, generated_by="LLM") for m in cleaned]
            rounds_report = self.rounds_report(state)

            save_json(os.path.join(paths.to_str(paths.step_dirs[4]), "risk_mapping_llm_survey_audit.json"), {
                "limit": len(cleaned),
                "total_candidates": len(cleaned),
                "clean_candidates": len(cleaned),
                "rounds": rounds_report,
                "llm_edges": mapping_report,
                "mappings": mapping_report,
            })
        except Exception as e:
            print(f"[ERROR] Failed to save consensus (step 4): {e}", file=sys.stderr)

        return {"cleaned_mappings": cleaned}

    # ===================================================================
    #  FINALIZATION
    # ===================================================================

    def get_final_items(self, state: dict) -> list:
        # Итоговые маппинги — очищенные после агрегации по моде
        return state.get("cleaned_mappings", [])

    def get_cache_context(self, state: dict) -> dict:
        return {
            "nodes": state["nodes"],
            "risks": state.get("risks", []),
            "risk_mapping": state.get("pipeline_config", {}).get("risk_mapping", {}),
        }


# ===================================================================
#  СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ===================================================================

def _save_step4_results(mappings, output_json_path, pipeline_config, risks, nodes, state_result) -> None:
    """Сохраняет результаты этапа 4 в файлы."""
    metadata = state_result.get("save_metadata", {})
    consensus_rounds = metadata.get("consensus_rounds", 1)
    llm_model = metadata.get("llm_model", "")

    # Карта ID → название узлов
    id_to_name = {n["id"]: n["name"] for n in nodes}

    clean_mappings = []
    for m in mappings:
        matched = m.get("matched_nodes", [])
        node_name = ""
        if matched:
            node_name = id_to_name.get(matched[0], matched[0])

        clean_mappings.append({
            "risk_number": m.get("risk_number", ""),
            "risk_name": m.get("risk_name", ""),
            "matched_nodes": matched,
            "node_name": node_name,
        })

    result = {
        "total_risks_processed": len(clean_mappings),
        "total_graph_nodes": len(nodes),
        "llm_model": llm_model,
        "consensus_rounds": consensus_rounds,
        "risk_mappings": clean_mappings,
        "version": "2.4-consensus-step4",
    }

    save_json(output_json_path, result)
    return result


# ===================================================================
#  MAIN
# ===================================================================

def main(risks_doc: dict = None, edges: dict = None) -> dict:
    """Точка входа.

    risks_doc: риски из шага 3 (in-memory из состояния LangGraph);
    edges:     граф КСГ из шага 2 (in-memory). Если None — читаются с диска.
    """
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("\n[STEP] ПРИВЯЗКА РИСКОВ К ВЕРШИНАМ ГРАФА (консенсус нескольких запросов → мода)...")

    if risks_doc is None and not os.path.isfile(RISKS_FILE):
        print(f"[ERROR]  Файл рисков не найден: {RISKS_FILE}", file=sys.stderr)
        print("Сначала запустите: python 3_parse_risks.py", file=sys.stderr)
        sys.exit(1)

    if edges is None and not os.path.isfile(GRAPH_JSON):
        print(f"[ERROR]  Файл графа не найден: {GRAPH_JSON}", file=sys.stderr)
        print("Сначала запустите: python main.py", file=sys.stderr)
        sys.exit(1)

    # Чтение рисков (из состояния или с диска)
    risks = _read_risks(RISKS_FILE, data=risks_doc)

    if edges is not None:
        graph = edges
    else:
        print(f"[LOAD] {GRAPH_JSON}")
        with open(GRAPH_JSON, "r", encoding="utf-8") as f:
            graph = json.load(f)
    nodes = graph.get("nodes", [])
    node_ids = [n["id"] for n in nodes]

    if not PIPELINE_CONFIG.get("use_llm", True):
        print("[SKIP] use_llm=False — привязка всех рисков к первой вершине")
        fallback_node = nodes[0]["id"] if nodes else None
        fallback_name = nodes[0]["name"] if nodes else None
        mappings = [
            OrderedDict([
                ("matched_nodes", [fallback_node]),
                ("node_name", fallback_name),
                ("risk_number", r["risk_number"]),
                ("risk_name", r["risk_name"]),
                ("reason", "use_llm=False — manual mapping to first node"),
            ])
            for r in risks
        ]
        result_doc = _save_step4_results(mappings, OUTPUT_JSON, PIPELINE_CONFIG, risks, nodes,
                            {"save_metadata": {}, "iterations": 0})
    else:
        initial_state = {
            "input_data": {},
            "pipeline_config": PIPELINE_CONFIG,
            "step_label": "step4",
            "risks": risks,
            "nodes": nodes,
            "node_ids": node_ids,
            "candidate_mappings": [],
            "consensus_output": {},
            "cleaned_mappings": [],
            "final_mappings": [],
            "fallback_node": nodes[0]["id"] if nodes else None,
            "fallback_name": nodes[0]["name"] if nodes else None,
            "iterations": 0,
            "save_metadata": {},
        }

        # Создаём builder и проверяем кэш
        step = Step4ConsensusStep()
        cached = step.check_cache(initial_state)

        if cached is not None:
            mappings = cached
            result_doc = _save_step4_results(mappings, OUTPUT_JSON, PIPELINE_CONFIG, risks, nodes,
                                {"save_metadata": {}, "iterations": 0})
        else:
            hooks = step.build_hooks()
            result = run_step(Step4State, hooks, initial_state, label="step4")
            mappings = result.get("final_mappings", [])
            result_doc = _save_step4_results(mappings, OUTPUT_JSON, PIPELINE_CONFIG, risks, nodes, result)

    # Вывод
    print(f"\n[SAVE] {OUTPUT_JSON}")
    total_risks = len(risks)
    total_with_nodes = sum(1 for m in mappings if m.get("matched_nodes"))
    total_without_nodes = total_risks - total_with_nodes

    print(f"\n[RESULTS]")
    print(f"-  Рисков обработано: {total_risks}")
    print(f"-  Узлов графа: {len(nodes)}")
    print(f"-  Привязано: {total_with_nodes}")
    print(f"-  Без привязки: {total_without_nodes}")
    llm_model = PIPELINE_CONFIG.get("llm", {}).get("model", "")
    consensus_rounds = (PIPELINE_CONFIG.get("model_assumptions", {})
                        .get("consensus", {}).get("rounds", 1) or 1)
    print(f"-  LLM: {llm_model}")
    print(f"-  Consensus: {consensus_rounds}")

    # Возвращаем расчётный doc привязок для состояния LangGraph (JSON уже сохранён)
    return result_doc


def _read_risks(path: str, data: dict = None) -> list:
    """
    Читает risks_processed.json → [{risk_number, risk_name, cause}, ...].

    Поля берутся из snake_case-записей шага 3; пустые описание/номер отбрасываются,
    как и раньше при чтении CSV. Если передан data (in-memory из состояния
    LangGraph) — файл не читается.
    """
    if data is None:
        if not os.path.isfile(path):
            print(f"[ERROR]  Файл рисков не найден: {path}", file=sys.stderr)
            sys.exit(1)
        data = load_json(path)
    rows = []
    for risk in data.get("risks", []):
        risk_number = str(risk.get("risk_number", "")).strip()
        description = str(risk.get("description", "")).strip()
        cause = str(risk.get("cause", "")).strip()
        if risk_number and description:
            rows.append({
                "risk_number": risk_number,
                "risk_name": description,
                "cause": cause,
            })
    print(f"[LOAD] {path} (рисков: {len(rows)})")
    return rows


if __name__ == "__main__":
    main()