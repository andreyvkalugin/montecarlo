"""
consensus_builder.py — Общие утилиты для кэширования consensus-шагов (2, 4, 6, ...).

Содержит только используемые функции кэширования консенсус-результатов.
Парсеры ответов LLM живут в utils.llm.llm_utils, агрегация голосов — в методах
шагов (2_add_edges_LLM.py, 4_map_risks_to_graph_LLM.py).
"""
import hashlib
import json
import os
import time
from typing import Optional

from utils.general.json_io import save_json
from utils.llm.cache_llm import CACHE_DIR

# Версия логики агрегации/конвейера. Включается в хеш кэша, чтобы после изменения
# формата пайплайна устаревшие кэши не переиспользовались.
PIPELINE_VER = "2.4-consensus"


def _cache_key(data_context: dict, rounds: int) -> str:
    """Вычисляет ключ кэша по контексту задачи и числу раундов."""
    hash_input = json.dumps({
        **data_context,
        "rounds": rounds,
        "feedback": "",
        "pipeline_ver": PIPELINE_VER,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]


def _prefix_suffix(prefix: str) -> str:
    """Извлекает суффикс из префикса кэша ('step2_consensus' -> '2')."""
    return prefix.replace("step", "").replace("_consensus", "")


def _cache_path(prefix: str, key: str) -> str:
    """Возвращает полный путь к файлу кэша консенсуса."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{prefix}_{key}.json")


# ===================================================================
#  КЭШИРОВАНИЕ
# ===================================================================

def save_consensus_cache(
    prefix: str,
    validated_data: list,
    data_context: dict,
    iterations: int,
    consensus_rounds: int,
    pipeline_config: dict,
) -> str:
    """
    Универсальное сохранение кэша consensus-результатов.

    Args:
        prefix: префикс имени файла (например "step2_consensus")
        validated_data: финальные данные (edges/mappings)
        data_context: контекст задачи (nodes, risks/wbs_edges)
        iterations: число итераций диалога
        consensus_rounds: число раундов консенсуса
        pipeline_config: полный конфиг пайплайна

    Returns:
        Путь к сохранённому файлу
    """
    key = _cache_key(data_context, consensus_rounds)
    cache_path = _cache_path(prefix, key)

    clean = [d for d in validated_data if d is not None and isinstance(d, dict)]
    config_model = pipeline_config.get("llm", {}).get("model", "")
    suffix = _prefix_suffix(prefix)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "consensus_output": {
                "iterations": iterations,
                f"final_{suffix}_count": len(clean),
                f"final_{suffix}": clean,
            },
            f"validated_{suffix}": clean,
            "iterations": iterations,
            "consensus_rounds": consensus_rounds,
            "model": config_model,
            "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    return cache_path


def load_consensus_cache(
    prefix: str,
    data_context: dict,
    pipeline_config: dict,
) -> Optional[list]:
    """
    Универсальная загрузка кэша consensus-результатов.

    Args:
        prefix: префикс имени файла
        data_context: контекст задачи для вычисления хеша
        pipeline_config: полный конфиг пайплайна

    Returns:
        Кэшированные данные или None
    """
    config_rounds = (pipeline_config.get("model_assumptions", {})
                     .get("consensus", {}).get("rounds", 1) or 1)
    key = _cache_key(data_context, config_rounds)
    path = _cache_path(prefix, key)

    if not os.path.exists(path):
        print(f"\n[CACHE MISS] {prefix}_{key}.json")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # Универсальное извлечение ключа (edges/mappings)
        suffix = _prefix_suffix(prefix)
        cached_data = cache_data.get(f"validated_{suffix}")
        if not cached_data:
            return None

        cached_data = [d for d in cached_data if isinstance(d, dict)]
        if not cached_data:
            return None

        config_model = pipeline_config.get("llm", {}).get("model", "")

        if (cache_data.get("model") != config_model or
                cache_data.get("consensus_rounds") != config_rounds):
            print(f"\n[CACHE MISS] Конфиг изменился")
            return None

        print(f"\n[CACHE HIT] {prefix}_{key}.json "
              f"(iterations={cache_data.get('iterations')}, "
              f"items={len(cached_data)})")
        return cached_data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"\n[CACHE MISS] Ошибка чтения: {e}")
        return None