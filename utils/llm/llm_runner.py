"""
llm_runner.py — Единый инструмент для всех LLM-вызовов в пайплайне.

Все шаги (2, 4, 6, ...) используют ТОЛЬКО этот модуль для вызовов LLM.
Специфика шагов (промпт, парсинг ответа) передаётся как callback-функции.

Функции:
  run_llm(prompt, config) → str | None
  run_parallel_consensus(prompts, config, response_parser) → list
  create_runner_config(model, timeout, max_retries, token_file, config_path) → dict
"""
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Утилиты из utils (независимо от шагов)
# ---------------------------------------------------------------------------
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.llm.llm_client import _gigachat_chat
from gigachat.exceptions import AuthenticationError

PIPELINE_CONFIG = get_pipeline_config()
CONFIG_PATH = os.path.join(paths.to_str(paths.utils), ".gigacode_config.json")


def _backoff(attempt: int, retry_wait_base: int, cap: int = 32) -> float:
    """Экспоненциальная пауза между ретраями (бэкофф) с ограничением сверху."""
    return min(retry_wait_base * (2 ** (attempt - 1)), cap)


# ===================================================================
#  КОНФИГУРАЦИЯ
# ===================================================================

def create_runner_config(
    model: Optional[str] = None,
    timeout: int = 180,
    max_retries: int = 10,
    retry_wait_base: int = 2,
    token_file: Optional[str] = None,
    config_path: Optional[str] = None,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Создаёт единый конфиг для всех LLM-вызовов шага.
    
    Args:
        model: имя модели (если None — из PIPELINE_CONFIG)
        timeout: таймаут CLI-вызова в секундах
        max_retries: максимум попыток на один вызов
        retry_wait_base: базовая пауза между ретраями (бэкофф)
        token_file: путь к файлу токена (если None — из конфигфайла)
        config_path: путь к файлу конфигурации
        max_tokens: максимум токенов в ответе
    
    Returns:
        Словарь конфига, который передаётся в run_llm / run_parallel_consensus
    """
    if model is None:
        model = PIPELINE_CONFIG.get("llm", {}).get("model", "vllm/DeepSeek-V4-Flash-0731-262k")
    return {
        "model": model,
        "timeout": timeout,
        "max_retries": max_retries,
        "retry_wait_base": retry_wait_base,
        "token_file": token_file,
        "config_path": config_path or CONFIG_PATH,
        "max_tokens": max_tokens,
    }


# ===================================================================
#  ЕДИНЫЙ ВЫЗОВ LLM
# ===================================================================

def run_llm(prompt: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Отправляет промпт к LLM и возвращает текстовый ответ.
    
    Это ЕДИНЫЙ путь вызова LLM для всех шагов. Обрабатывает:
    - вызов GigaChat через библиотечный клиент
    - аутентификацию и токены (через переменные окружения GIGACHAT_*)
    - таймауты и ретраи с бэкоффом
    - обработку JSON-ответа

    Args:
        prompt: текст промпта
        config: результат create_runner_config

    Returns:
        Текстовый ответ LLM или None при ошибке
    """
    model = config.get("model") or PIPELINE_CONFIG.get("llm", {}).get("model", "GigaChat-Pro")
    timeout = config.get("timeout", 180)
    max_retries = config.get("max_retries", 10)
    retry_wait_base = config.get("retry_wait_base", 2)
    max_tokens = config.get("max_tokens", 4096)
    temperature = config.get("temperature")

    for attempt in range(1, max_retries + 1):
        try:
            text, _tokens = _gigachat_chat(
                prompt,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = text.strip()

            if text:
                # Извлекаем JSON если нужно
                text_clean = re.sub(r"^```json\s*", "", text)
                text_clean = re.sub(r"\s*```\s*$", "", text_clean)
                return text_clean

            # Пустой ответ (код 0, но нет текста) — ретраим
            error_msg = "пустой ответ GigaChat"
            if attempt < max_retries:
                backoff = _backoff(attempt, retry_wait_base)
                print(f"[LLM][RETRY] Попытка {attempt}/{max_retries}: {error_msg}...", flush=True)
                print(f"[LLM][WAIT] Повтор через {backoff:.1f}s...", flush=True)
                time.sleep(backoff)
            else:
                print(f"[LLM][ERROR] GigaChat ошибка (исчерпаны повторы): {error_msg}", flush=True)

        except AuthenticationError as e:
            # Аутентификация — не повторяем, сразу возвращаем None
            print(f"[LLM][AUTH] Ошибка аутентификации: {str(e)[:100]}", flush=True)
            return None

        except Exception as e:
            if attempt < max_retries:
                backoff = _backoff(attempt, retry_wait_base)
                print(f"[LLM][RETRY] Исключение #{attempt}/{max_retries}: {e}")
                time.sleep(backoff)
            else:
                print(f"[LLM][ERROR] Исключение (исчерпаны повторы): {e}")
            continue

    return None


# ===================================================================
#  КОНСЕНСУС (единый, последовательный для всех шагов)
# ===================================================================

def run_parallel_consensus(
    prompts: List[str],
    config: Dict[str, Any],
    max_concurrent: int = 4,
    max_retries_per_round: int = 3,
    response_parser: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Запускает N ПОСЛЕДОВАТЕЛЬНЫХ вызовов LLM и собирает результаты.

    Все вызовы выполняются один за другим через единый run_llm, что
    обеспечивает единую логику обработки ошибок, аутентификации и ретраев.
    `max_concurrent` оставлен для совместимости сигнатуры, но не используется
    (выполнение строго последовательное).

    Args:
        prompts: список промптов для последовательного запуска
        config: результат create_runner_config
        max_concurrent: не используется (оставлен для совместимости)
        max_retries_per_round: максимум повторных попыток на один вызов
        response_parser: callable(response) -> parsed_value (если None — возвращает сырой текст)

    Returns:
        Список результатов: [{round, result, attempt}]
    """
    print(f"   LLM запуск {len(prompts)} последовательных запросов...", flush=True)

    parsed = []
    for round_idx, prompt in enumerate(prompts):
        for attempt in range(1, max_retries_per_round + 1):
            try:
                response = run_llm(prompt, config)
                if response:
                    if response_parser:
                        parsed_response = response_parser(response)
                        count = len(parsed_response) if isinstance(parsed_response, list) else "OK"
                        print(f"   LLM запрос {round_idx}: {count}", flush=True)
                        parsed.append({"round": round_idx, "result": parsed_response, "attempt": attempt})
                    else:
                        print(f"   LLM запрос {round_idx}: OK", flush=True)
                        parsed.append({"round": round_idx, "result": response, "attempt": attempt})
                    break
            except Exception as e:
                print(f"   LLM запрос {round_idx}: ошибка #{attempt}: {e}", flush=True)

            if attempt < max_retries_per_round:
                time.sleep(5 * attempt)

        # Если ни одна попытка не дала ответа — фиксируем пустой результат
        if not any(r.get("round") == round_idx for r in parsed):
            print(f"   LLM запрос {round_idx}: исчерпаны retry", flush=True)
            parsed.append({"round": round_idx, "result": [], "attempt": max_retries_per_round})

    return parsed
