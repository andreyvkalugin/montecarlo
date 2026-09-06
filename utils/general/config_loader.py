"""
config_loader.py — Централизованный загрузчик PIPELINE_CONFIG.

Используется всеми шагами пайплайна для получения единого источника параметров
без прямого импорта main.py.
"""
import importlib
import sys
from typing import Dict, Optional


# Единый глобальный кэш конфига
_config_cache: Optional[Dict] = None


def get_pipeline_config(cache: Optional[Dict] = None) -> Dict:
    """
    Загружает PIPELINE_CONFIG из main.py.
    Использует кэш для избежания повторных импортов.

    Args:
        cache: Внешний кэш (для тестов или передачи из main.py).

    Returns:
        Словарь PIPELINE_CONFIG или пустой словарь при ошибке импорта.
    """
    global _config_cache

    if cache is not None:
        return cache

    if _config_cache is not None:
        return _config_cache

    try:
        import main as main_module
        _config_cache = main_module.PIPELINE_CONFIG
        return _config_cache
    except (ImportError, AttributeError) as e:
        print(f"[WARN] Не удалось загрузить PIPELINE_CONFIG: {e}", file=sys.stderr)
        return {}
