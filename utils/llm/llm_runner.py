"""
llm_runner.py — Единый инструмент для всех LLM-вызовов в пайплайне.

Все шаги (2, 4, 6, ...) используют ТОЛЬКО этот модуль для вызовов LLM.
Специфика шагов (промпт, парсинг ответа) передаётся как callback-функции.

Функции:
  run_llm(prompt, config) → str | None
  run_parallel_consensus(prompts, config, response_parser) → list
  create_runner_config(model, timeout, max_retries, token_file, config_path) → dict
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Утилиты из utils (независимо от шагов)
# ---------------------------------------------------------------------------
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.llm.llm_client import _gigacode_run

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
    - запуск CLI GigaCode
    - аутентификацию и токены
    - таймауты и ретраи с бэкоффом
    - обработку JSON-ответа
    
    Args:
        prompt: текст промпта
        config: результат create_runner_config
    
    Returns:
        Текстовый ответ LLM или None при ошибке
    """
    model = config.get("model", "vllm/DeepSeek-V4-Flash-0731-262k")
    timeout = config.get("timeout", 180)
    max_retries = config.get("max_retries", 10)
    retry_wait_base = config.get("retry_wait_base", 2)
    token_file = config.get("token_file")
    config_path = config.get("config_path", CONFIG_PATH)
    max_tokens = config.get("max_tokens", 4096)

    # Создаём каталог для временных файлов, если не существует
    prompts_dir = os.path.join(paths.to_str(paths.cache), "llm_prompts_temp")
    os.makedirs(prompts_dir, exist_ok=True)
    
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8',
        dir=prompts_dir,
    ).name
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(prompt)

        project_root = paths.to_str(paths.root)

        for attempt in range(1, max_retries + 1):
            try:
                # Единый надёжный запуск CLI через stdin (node-рантайм)
                args = [
                    "--approval-mode", "auto-edit",
                    "--channel", "CI",
                    "--bare",
                    "--chat-recording", "false",
                ]
                result = _gigacode_run(prompt, args, timeout, project_root)

                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip()
                    # Извлекаем JSON если нужно
                    text_clean = re.sub(r"^```json\s*", "", text)
                    text_clean = re.sub(r"\s*```\s*$", "", text_clean)
                    # Выводим статус раунда в консенсусе (через caller),
                    # здесь только silent success — caller сам выведет.
                    return text_clean

                # Обрабатываем ошибки
                error_msg = result.stderr[:200] if result.stderr else f"Код возврата: {result.returncode}"
                if not error_msg and not result.stdout.strip():
                    error_msg = "stdout пуст (код 0, но нет ответа)"
                if any(k in error_msg.lower() for k in ["unauthorized", "auth", "token", "bearer"]):
                    print(f"[LLM][AUTH] Ошибка аутентификации: {error_msg[:100]}", flush=True)
                    # Аутентификация — не повторяем, сразу возвращаем None
                    return None

                if attempt < max_retries:
                    backoff = _backoff(attempt, retry_wait_base)
                    print(f"[LLM][RETRY] Попытка {attempt}/{max_retries}: {error_msg[:80]}...", flush=True)
                    print(f"[LLM][WAIT] Повтор через {backoff:.1f}s...", flush=True)
                    time.sleep(backoff)
                else:
                    print(f"[LLM][ERROR] CLI ошибка (исчерпаны повторы): {error_msg[:100]}", flush=True)

            except subprocess.TimeoutExpired:
                if attempt < max_retries:
                    backoff = _backoff(attempt, retry_wait_base)
                    print(f"[LLM][TIMEOUT] Попытка {attempt}/{max_retries}")
                    print(f"[LLM][WAIT] Повтор через {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    print(f"[LLM][TIMEOUT] Исчерпаны повторы {max_retries}")
                continue

            except Exception as e:
                if attempt < max_retries:
                    backoff = _backoff(attempt, retry_wait_base)
                    print(f"[LLM][RETRY] Исключение #{attempt}/{max_retries}: {e}")
                    time.sleep(backoff)
                else:
                    print(f"[LLM][ERROR] Исключение (исчерпаны повторы): {e}")
                continue

        return None

    finally:
        # Удаляем временный файл
        try:
            os.unlink(tmp)
        except Exception:
            pass


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
