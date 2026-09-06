"""
pipeline.py — Универсальный мультиагентный оркестратор.

Фреймворк НЕ содержит специфики отдельных этапов. Каждый этап (2, 4, 6, ...)
передаёт свой набор хуков (StepHooks) и State, а логика этапов живёт в
соответствующих скриптах (2_add_edges_LLM.py, 4_map_risks_to_graph_LLM.py, ...).

Паттерн: [candidates] → [cleaning] → [finalization]

Несколько LLM-запросов (consensus) агрегируются в моду/частоту голосов при
агрегации кандидатов, после чего результат сразу финализируется.

Повторноиспользуемые компоненты:
  BaseStepState            — базовый TypedDict для State
  StepHooks                — набор callable, специфичных для этапа
  create_llm_client()      — фабрика LLMApiClient
  call_llm()               — безопасный вызов LLM
  compute_cache_key()      — SHA256-хеш для кэша
  save_cache()/load_cache()— запись/чтение кэша
  ensure_acyclic()         — удаление рёбер, создающих цикл
  run_step()               — универсальная точка входа

Специфичная для этапов логика НЕ помещается сюда — см. скрипты 2_/4_.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# Путь к корню проекта для sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.general.paths import paths
from utils.llm.cache_llm import CACHE_DIR
from utils.general.config_loader import get_pipeline_config
from utils.llm.llm_client import LLMApiClient

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------
PIPELINE_CONFIG = get_pipeline_config()
CONFIG_PATH = os.path.join(paths.to_str(paths.utils), ".gigacode_config.json")

FORBIDDEN_IDS = {"LLM", "WBS", "WBS_ROOT", "ROOT", "START", "END", "NONE", "SYSTEM"}


# ===================================================================
#  PIPELINE LOGGER — единый вывод для всех этапов
# ===================================================================

class PipelineLogger:
    """Обёртка для единообразного вывода всех этапов пайплайна.

    Методы вызывают print() с префиксом шага (step2, step4, step6, ...).
    """

    def __init__(self, step_label: str = "step"):
        self.step_label = step_label  # например "step4", "step2"

    def _pfx(self) -> str:
        return f"[{self.step_label}]"

    # ——— Candidates (Шаг 1) ———
    def candidates_start(self, consensus_rounds: int, timeout: int) -> None:
        print(f"\n{'='*60}")
        print(f"[CONSENSUS]  {self._pfx()} ШАГ 1 (CANDIDATES): {consensus_rounds} параллельных вызовов LLM (timeout={timeout}с)")
        print(f"{'='*60}")

    def candidates_done(self, total_candidates: int, total_votes: int) -> None:
        print(f"[{self._pfx()}] [OK]  Собрано {total_candidates} кандидатов "
              f"(всего голосов={total_votes})")

    def candidates_error(self, error: str) -> None:
        print(f"[{self._pfx()}] [WARN]  Ошибка consensus: {error}")

    def candidates_round_ok(self, round_idx: int, count: int) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: {count}")

    def candidates_round_error(self, round_idx: int, attempt: int, error: str) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: ошибка #{attempt}: {error}")

    def candidates_round_retry_exhausted(self, round_idx: int) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: исчерпаны retry")

    def candidates_save_error(self, error: str) -> None:
        print(f"   [{self._pfx()}] [WARN]  Не удалось сохранить: {error}")

    # ——— Cleaning (Шаг 2) ———
    def cleaning_done(self, cleaned: int, limit: int = 0, selected: int = 0) -> None:
        if limit and selected:
            print(f"[{self._pfx()}] [FILTER]  ОТБОР: чистых кандидатов={cleaned}, лимит={limit}, отобрано={selected}")
        else:
            print(f"[{self._pfx()}] [FILTER]  ОТБОР: чистых кандидатов={cleaned}")

    def cleaning_save_error(self, error: str) -> None:
        print(f"[{self._pfx()}] [WARN]  Не удалось сохранить: {error}")

    # ——— Finalization ———
    def finalization_done(self, total: int, iterations: int) -> None:
        print(f"\n{'='*60}")
        print(f"[OK]  {self._pfx()} ФИНАЛИЗАЦИЯ: {total} элементов, {iterations} итераций")
        print(f"{'='*60}")

    # ——— Consensus (общий, для параллельного запуска) ———
    def consensus_round_ok(self, round_idx: int, count: int) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: {count}")

    def consensus_round_error(self, round_idx: int, attempt: int, error: str) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: ошибка #{attempt}: {error}")

    def consensus_round_retry(self, round_idx: int) -> None:
        print(f"   [{self._pfx()}] Раунд {round_idx}: исчерпаны retry")

    def consensus_exception(self, error: str) -> None:
        print(f"   [{self._pfx()}] Исключение: {error}")


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def load_prompt(path: str) -> Optional[str]:
    """Загружает текст промпта из файла."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_llm_client(
    model: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 1,
    config_path: Optional[str] = None,
) -> LLMApiClient:
    """Фабрика LLM-клиента."""
    if model is None:
        model = PIPELINE_CONFIG.get("llm", {}).get("model")
    return LLMApiClient(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        config_path=config_path or CONFIG_PATH,
    )


def call_llm(
    llm_client: LLMApiClient,
    prompt: str,
    max_tokens: int = 8192,
    label: str = "LLM",
) -> Optional[str]:
    """Безопасный вызов LLM с обработкой исключений. Возвращает сырую строку или None."""
    try:
        return llm_client.chat(prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"[{label}] [WARN]  Ошибка LLM: {e}")
        return None


def compute_cache_key(*args: Any) -> str:
    """Вычисляет SHA256-хеш для кэширования."""
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def save_cache(key: str, data: Dict[str, Any], prefix: str = "") -> str:
    """Сохраняет результат в кэш и возвращает путь."""
    cache_prefix = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{cache_prefix}{key}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_cache(key: str, prefix: str = "") -> Optional[Dict]:
    """Загружает результат из кэша по ключу."""
    cache_prefix = f"{prefix}_" if prefix else ""
    path = os.path.join(CACHE_DIR, f"{cache_prefix}{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ===================================================================
#  BASE STATE
# ===================================================================

class BaseStepState(TypedDict):
    """Базовое состояние. Конкретные State наследуются и расширяют поля."""
    input_data: dict                        # произвольные входные данные этапа
    pipeline_config: dict                   # полный конфиг пайплайна
    step_label: str                         # метка этапа (например "step4")
    iterations: int                         # число итераций цикла
    save_metadata: dict                     # метаданные для сохранения финального файла


# ===================================================================
#  STEP HOOKS — контракт для специфичной логики этапа
# ===================================================================

@dataclass
class StepHooks:
    """
    Набор callable-хуков, специфичных для этапа.

    Каждый хук принимает state и возвращает dict с обновлением полей state
    (паттерн узла пайплайна). Имена полей, которые хук пишет, задаются на этапе.
    """
    candidates: Callable[[Dict], Dict]
    cleaning: Callable[[Dict], Dict]
    finalization: Callable[[Dict], Dict]
    # Маршрутизаторы не нужны — конвейер линейный (candidates → cleaning → finalization)


# ===================================================================
#  ЦИКЛ-ДЕТЕКТОР / АЦИКЛИЧНОСТЬ
# ===================================================================

def creates_cycle(edges: List[Dict], new_s: str, new_t: str) -> bool:
    """Проверяет, создаст ли добавление edge (new_s → new_t) цикл в DAG."""
    graph: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if isinstance(e, dict) and "source" in e and "target" in e:
            graph[e["source"]].append(e["target"])
    graph[new_s].append(new_t)

    visited, path = set(), set()

    def dfs(node: str) -> bool:
        if node in path:
            return True
        if node in visited:
            return False
        visited.add(node)
        path.add(node)
        for nb in sorted(graph.get(node, [])):
            if dfs(nb):
                return True
        path.remove(node)
        return False

    for node in sorted(graph.keys()):
        if node not in visited:
            if dfs(node):
                return True
    return False


def ensure_acyclic(
    new_edges: Optional[List[Dict]],
    base_edges: Optional[List[Dict]],
) -> List[Dict]:
    """Гарантирует ацикличность: отбрасывает рёбра, создающие цикл."""
    kept = [e for e in (base_edges or []) if e is not None]
    result: List[Dict] = []
    for e in (new_edges or []):
        if e is None:
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None or s == t:
            continue
        if creates_cycle(kept, s, t):
            continue
        kept.append(e)
        result.append(e)
    return result


# ===================================================================
#  ПОСТРОЕНИЕ ГРАФА ИЗ ХУКОВ
# ===================================================================

def run_step(
    state_cls: type,
    hooks: StepHooks,
    initial_state: Dict[str, Any],
    label: str = "step",
) -> Dict[str, Any]:
    """
    Выполняет мультиагентный консенсус-пайплайн.

    Порядок вызова хуков:
      candidates → cleaning → finalization

    Несколько LLM-запросов (consensus) уже агрегированы в моду/частоту голосов
    на этапе candidates.
    """
    print(f"\n[PIPELINE] Запуск консенсус-пайплайна ({label})...")
    state = dict(initial_state)

    # ── Линейный проход (merge: обновляем только возвращённые поля) ──
    state = {**state, **hooks.candidates(state)}
    state = {**state, **hooks.cleaning(state)}
    state = {**state, **hooks.finalization(state)}

    return state