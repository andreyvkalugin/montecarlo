"""
pipeline_graph.py — Оркестратор пайплайна (шаги 1–10) на LangGraph.

Заменяет subprocess-цикл из main.py: вместо запуска каждого скрипта отдельным
процессом строит StateGraph, где каждый шаг — узел графа. Данные между шагами
передаются ЧЕРЕЗ СОСТОЯНИЕ LangGraph (in-memory), а не через чтение файлов с диска.

При этом каждый шаг ПО-ПРЕЖНЕМУ сохраняет свой результат в JSON на диск
(артефакт/персистентность) — меняется только механизм передачи состояния
следующему шагу.

Граница in-memory состояния:
  - все расчётные JSON-документы шагов передаются через состояние;
  - крупные бинарные выборки (.npy шага 8) и iteration_scenarios.json остаются
    дисковыми артефактами: их читает шаг 9 напрямую с диска (файлы только что
    записал шаг 8). Аналогично шаг 8 читает project_duration из csg_tasks_wbs.json
    с диска (записан шагом 1).

Запуск:  python pipeline_graph.py
Standalone-отладка отдельного шага по-прежнему работает: python 5_risk_graph_builder.py
(каждый main() при вызове без аргументов читает входные данные с диска).
"""
import importlib
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from langgraph.graph import StateGraph, START, END

import main as pipeline_main  # переиспользуем setup: ensure_structure/clean_all/check_input_files
from utils.general.logger import install_console_logging
from utils.general.paths import paths


# ===================================================================
#  СОСТОЯНИЕ ПАЙПЛАЙНА
# ===================================================================

class PipelineState(TypedDict, total=False):
    """Состояние, которое LangGraph протягивает от шага к шагу.

    Каждый ключ — расчётный JSON-документ соответствующего шага (in-memory).
    Значения по умолчанию перезаписываются узлами (reducer не нужен).
    """
    wbs: dict            # шаг 1  → csg_tasks_wbs.json
    edges: dict          # шаг 2  → csg_edges_final.json
    risks: dict          # шаг 3  → risks_processed.json
    risk_mapping: dict   # шаг 4  → risk_mapping_final.json
    risk_graph: dict     # шаг 5  → risk_graph.json
    weights: dict        # шаг 6  → risk_graph_with_weights.json
    bayes: dict          # шаг 7  → bayesian_network.json
    simulation: dict     # шаг 8  → simulation_results.json
    summary: dict        # шаг 9  → summary_statistics.json
    report: str          # шаг 10 → llm_report.md (путь)
    timings: List[Tuple[str, float]]


# ===================================================================
#  ИМПОРТ ШАГОВ (имена модулей начинаются с цифры — только через importlib)
# ===================================================================

step1 = importlib.import_module("1_parse_csg")
step2 = importlib.import_module("2_add_edges_LLM")
step3 = importlib.import_module("3_parse_risks")
step4 = importlib.import_module("4_map_risks_to_graph_LLM")
step5 = importlib.import_module("5_risk_graph_builder")
step6 = importlib.import_module("6_model_assump_calc_LLM")
step7 = importlib.import_module("7_bayesian_network_builder")
step8 = importlib.import_module("8_monte_carlo_simulator")
step9 = importlib.import_module("9_analyzer")
step10 = importlib.import_module("10_final_report_LLM")


# ===================================================================
#  УЗЛЫ ГРАФА
#  Каждый узел: читает входы из состояния → зовёт main() шага
#  (тот сохраняет JSON на диск) → возвращает свой output в состояние.
# ===================================================================

def _node_step1(state: PipelineState) -> dict:
    return {"wbs": step1.main()}

def _node_step2(state: PipelineState) -> dict:
    return {"edges": step2.main(wbs=state["wbs"])}

def _node_step3(state: PipelineState) -> dict:
    return {"risks": step3.main()}

def _node_step4(state: PipelineState) -> dict:
    return {"risk_mapping": step4.main(risks_doc=state["risks"], edges=state["edges"])}

def _node_step5(state: PipelineState) -> dict:
    return {"risk_graph": step5.main(edges=state["edges"], risk_mapping=state["risk_mapping"])}

def _node_step6(state: PipelineState) -> dict:
    return {"weights": step6.main(risk_graph=state["risk_graph"], risks=state["risks"])}

def _node_step7(state: PipelineState) -> dict:
    return {"bayes": step7.main(weights=state["weights"])}

def _node_step8(state: PipelineState) -> dict:
    return {"simulation": step8.main(bayes=state["bayes"])}

def _node_step9(state: PipelineState) -> dict:
    return {"summary": step9.main(simulation=state["simulation"], bayes=state["bayes"])}

def _node_step10(state: PipelineState) -> dict:
    return {"report": step10.main(
        summary=state["summary"], bayes=state["bayes"],
        edges=state["edges"], risks=state["risks"],
    )}


# (номер, метка, имя узла, функция-узел)
STEP_NODES: List[Tuple[int, str, str, Callable[[PipelineState], dict]]] = [
    (1, "Парсинг КСГ и WBS-иерархия", "step1", _node_step1),
    (2, "Добавление LLM-связей", "step2", _node_step2),
    (3, "Парсинг карты рисков", "step3", _node_step3),
    (4, "Привязка рисков к вершинам графа", "step4", _node_step4),
    (5, "Построение графа рисков", "step5", _node_step5),
    (6, "Расчёт весов рисков (модельные допущения)", "step6", _node_step6),
    (7, "Построение Байесовской сети", "step7", _node_step7),
    (8, "Монте-Карло симуляция", "step8", _node_step8),
    (9, "Анализ результатов симуляции", "step9", _node_step9),
    (10, "LLM-отчёт", "step10", _node_step10),
]


def _wrap(step_num: int, label: str, fn: Callable[[PipelineState], dict]) -> Callable[[PipelineState], dict]:
    """Оборачивает узел: баннер шага, тайминг и накопление timings в состоянии."""
    def node(state: PipelineState) -> dict:
        print(f"\n{'=' * 60}")
        print(f"Шаг №{step_num}: [{label}] (LangGraph-узел)")
        print(f"{'=' * 60}")
        t0 = time.time()
        updates = fn(state)
        elapsed = time.time() - t0
        print(f"[OK]  Шаг '{label}' выполнен успешно. [{elapsed:.1f}с]")
        updates["timings"] = (state.get("timings") or []) + [(label, elapsed)]
        return updates
    return node


def build_graph():
    """Строит и компилирует StateGraph пайплайна (линейный: 1→2→…→10)."""
    builder = StateGraph(PipelineState)

    for step_num, label, name, fn in STEP_NODES:
        builder.add_node(name, _wrap(step_num, label, fn))

    builder.add_edge(START, STEP_NODES[0][2])
    for (_, _, prev_name, _), (_, _, next_name, _) in zip(STEP_NODES, STEP_NODES[1:]):
        builder.add_edge(prev_name, next_name)
    builder.add_edge(STEP_NODES[-1][2], END)

    return builder.compile()


# ===================================================================
#  ТОЧКА ВХОДА
# ===================================================================

def run_pipeline() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(BASE_DIR)

    _log_file, log_path = install_console_logging()
    print(f"\n[LOG]  Лог вывода терминала: {log_path}")

    print("=" * 60)
    print("[PIPELINE]   КОНВЕЙЕР ОБРАБОТКИ (LangGraph):")
    print("=" * 60)

    # Подготовка структуры и входных файлов (переиспользуем логику main.py)
    print("\n[1] Проверка структуры папок...")
    pipeline_main.ensure_structure()
    pipeline_main.check_data_dir()
    print("    [OK]  Структура папок OK.")

    print("\n[2] Проверка входных файлов...")
    if not pipeline_main.check_input_files():
        sys.exit(1)
    print("    [OK]  Проверка файлов завершена.")

    print("\n[3] Очистка data/data_processed...")
    pipeline_main.clean_all()

    print("\n[4] Запуск графа LangGraph...")
    pipeline_start = time.time()

    graph = build_graph()
    final_state = graph.invoke({})

    total_time = time.time() - pipeline_start

    # Сводка таймингов
    timings = final_state.get("timings", [])
    if timings:
        print("\n" + "=" * 60)
        print("[TIME]   ВРЕМЯ ВЫПОЛНЕНИЯ ШАГОВ")
        print("=" * 60)
        for i, (label, elapsed) in enumerate(sorted(timings, key=lambda x: x[1], reverse=True), 1):
            print(f"{i:<4} {elapsed:>10.1f}с  {label}")
        print("-" * 60)
        print(f"{'ИТОГО':<45} {total_time:>10.1f}с")

    print("\n" + "=" * 60)
    print(f"[DONE]  Конвейер успешно завершён! [Всего: {total_time:.1f}с]")
    print("[OUTPUT]  Результаты находятся в: " + str(paths.processed))
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n[WARN]  Конвейер прерван пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR]  Непредвиденная ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
