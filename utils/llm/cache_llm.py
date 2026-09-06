"""
cache_llm.py - Модуль кэширования LLM-запросов.

Все результаты LLM-запросов кэшируются по SHA256-хешу от:
  - входных данных
  - промпта
  - полного конфига PIPELINE_CONFIG

При повторном запуске с теми же данными LLM не вызывается,
результат берётся из кэша.

Использование:
    from utils.llm.cache_llm import check_and_cache

    result = check_and_cache(
        input_content=input_data,
        prompt_content=prompt_text,
        pipeline_config=PIPELINE_CONFIG,
        llm_fn=my_llm_call,
        llm_args=(prompt,),
        llm_kwargs={},
    )
"""

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Optional


# ===== ПУТИ =====
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ===== УТИЛИТЫ ХЕШИРОВАНИЯ =====

def _compute_hash(
    input_content: Any,
    prompt_content: str,
    pipeline_config: dict,
) -> str:
    """
    Вычисляет SHA256-хеш от трёх компонентов:
      1. Входные данные (сериализуется в JSON)
      2. Текст промпта (строка)
      3. Полный конфиг PIPELINE_CONFIG (JSON, сортированные ключи)

    Результат — первые 16 hex-символов хеша (достаточно для уникальности
    и для использования как имя файла).
    """
    try:
        input_str = json.dumps(input_content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        input_str = str(input_content)

    config_str = json.dumps(pipeline_config, ensure_ascii=False, sort_keys=True)

    raw = input_str + "\n" + prompt_content + "\n" + config_str

    full_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return full_hash[:16]


def _cache_path(cache_key: str, prefix: str = "") -> str:
    """Полный путь к файлу кэша.
    
    Args:
        cache_key: SHA256-хеш (16 символов).
        prefix: Префикс шага пайплайна (например 'step2', 'step4').
    """
    key = f"{prefix}_{cache_key}" if prefix else cache_key
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cached_result(cache_key: str, prefix: str = "") -> Optional[dict]:
    """
    Загружает кэшированный результат по ключу.
    """
    path = _cache_path(cache_key, prefix)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        # Если кэш помечен как failed — удаляем и возвращаем None (повторный вызов LLM)
        if cached.get("is_failure", False):
            os.remove(path)
            print(f"[CACHE REMOVED] Failed cache for {prefix}_{cache_key} — will retry LLM")
            return None

        # Старые кэши (до добавления is_failure) могут иметь llm_response: null,
        # но без флага is_failure. Это артефакты ошибок LLM (WinError 206 и т.д.).
        # Применяем проверку ТОЛЬКО к файлам check_and_cache (имеют cache_key).
        if "cache_key" in cached and cached.get("llm_response") is None:
            os.remove(path)
            print(f"[CACHE REMOVED] Cache with null response for {prefix}_{cache_key} — will retry LLM")
            return None

        return cached
    except (json.JSONDecodeError, IOError):
        return None


def _save_cached_result(
    cache_key: str,
    llm_response: Any,
    input_hash: str,
    prompt_content: str,
    pipeline_config: dict,
    is_failure: bool = False,
    prefix: str = "",
) -> None:
    """Сохраняет результат LLM в кэш."""
    path = _cache_path(cache_key, prefix)

    data = {
        "cache_key": f"{prefix}_{cache_key}" if prefix else cache_key,
        "input_hash": input_hash,
        "model": pipeline_config.get("llm", {}).get("model", "unknown"),
        "cached_at": datetime.now().isoformat(timespec="seconds"),
        "llm_response": llm_response,
        "is_failure": is_failure,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===== ОСНОВНАЯ ФУНКЦИЯ =====

def check_and_cache(
    input_content: Any,
    prompt_content: str,
    pipeline_config: dict,
    llm_fn: Callable,
    llm_args: tuple = (),
    llm_kwargs: Optional[dict] = None,
    llm_client: Any = None,
    cache_prefix: str = "",
) -> Any:
    """
    Проверяет кэш и при необходимости вызывает LLM.

    Логика:
      1. Вычисляется хеш от входных данных + промпта + конфига.
      2. Проверяется наличие файла кэша.
         - Если файл найден → результат берётся из кэша (LLM не вызывается).
         - Если файл не найден → вызывается LLM, результат сохраняется в кэш.

    Args:
        input_content: Входные данные (сериализуемый в JSON объект или строка).
        prompt_content: Текст промпта (строка).
        pipeline_config: Полный конфиг PIPELINE_CONFIG из main.py.
        llm_fn: Функция, вызывающая LLM (например, llm.chat).
        llm_args: Позиционные аргументы для LLM-функции.
        llm_kwargs: Именованные аргументы для LLM-функции.
        cache_prefix: Префикс шага пайплайна для имени файла кэша (например 'step2').

    Returns:
        Результат вызова LLM (из кэша или свежий).
    """
    if llm_kwargs is None:
        llm_kwargs = {}

    # --- Шаг 1: Вычисляем хеш ---
    cache_key = _compute_hash(input_content, prompt_content, pipeline_config)

    # --- Шаг 2: Проверяем кэш ---
    cached = _load_cached_result(cache_key, cache_prefix)

    if cached is not None:
        # --- КЭШ НАЙДЕН ---
        label = f"{cache_prefix}_{cache_key}" if cache_prefix else cache_key
        print(f"\n[CACHE HIT] Key {label}")
        print(f"   model: {cached.get('model', 'unknown')}")
        print(f"   cached: {cached.get('cached_at', 'unknown')}")
        print(f"   LLM не вызывается — используется кэшированный результат")
        return cached["llm_response"]

    # --- КЭША НЕТ ---
    label = f"{cache_prefix}_{cache_key}" if cache_prefix else cache_key
    print(f"\n[CACHE MISS] Key {label}")
    print(f"   model: {pipeline_config.get('llm', {}).get('model', 'unknown')}")
    print(f"   Вызов LLM...")

    # --- Шаг 3: Вызываем LLM ---
    start = time.time()
    llm_result = llm_fn(*llm_args, **llm_kwargs)
    elapsed = time.time() - start

    # --- Конвертируем ответ из CP1251 → UTF-8 (LLM может вернуть CP1251) ---
    if isinstance(llm_result, str):
        # LLM возвращает строку, которая была неверно декодирована из UTF-8 байтов как CP1251.
        # Такие строки содержат "битые" символы вроде Ê, î, ö, å, ð, æ вместо русских букв.
        # Обратим это: encode('cp1251') → байты, decode('utf-8') → корректная строка.
        try:
            # Проверяем, содержит ли строка типичные артефакты CP1251 декодирования
            has_broken = any(c in llm_result for c in 'ÂÃÊÎÖÐÆàáâãäåæçèéêìîïðóöýûü')
            if has_broken:
                corrected = llm_result.encode('cp1251').decode('utf-8')
                # Проверяем, что исправленная строка содержит валидные русские буквы
                has_valid_russian = any(c in corrected for c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя')
                if has_valid_russian:
                    llm_result = corrected
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass  # Если не получилось, оставляем как есть

    # --- Статистика токенов ---
    if llm_client and hasattr(llm_client, "print_token_stats"):
        llm_client.print_token_stats()

    # --- Шаг 4: Сохраняем в кэш ---
    input_hash = hashlib.sha256(
        (json.dumps(input_content, ensure_ascii=False, sort_keys=True)
         if not isinstance(input_content, str)
         else input_content).encode("utf-8")
    ).hexdigest()[:16]

    # Не кэшируем None — это признак ошибки LLM, нужно retry
    is_failure = llm_result is None
    save_as_failure = is_failure

    _save_cached_result(
        cache_key,
        llm_result,
        input_hash,
        prompt_content,
        pipeline_config,
        is_failure=save_as_failure,
        prefix=cache_prefix,
    )

    if save_as_failure:
        print(f"   [WARN] LLM вернул None — результат НЕ закэширован (ошибка запроса)")
    else:
        label = f"{cache_prefix}_{cache_key}" if cache_prefix else cache_key
        print(f"   Результат сохранён в кэш: {_cache_path(cache_key, cache_prefix)}")
    print(f"   Время LLM: {elapsed:.1f}с")

    return llm_result
