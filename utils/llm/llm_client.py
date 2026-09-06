"""
llm_client.py - Единый клиент для запросов к LLM через GigaCode CLI.

Использует GigaCode CLI для автоматической аутентификации.
Совместим с новыми версиями CLI (>= 26.8.x).
"""

import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union


# Стандартные пути установки GigaCode CLI
GIGACODE_EXE_PATHS = [
    r"C:\Program Files\.gigacode\bin\gigacode.exe",
    r"C:\Program Files\GigaCode\bin\gigacode.exe",
]

GIGACODE_CMD_PATHS = [
    r"C:\Program Files\.gigacode\bin\gigacode.cmd",
    r"C:\Program Files\GigaCode\bin\gigacode.cmd",
]


@lru_cache()
def _find_gigacode():
    """Находит путь к gigacode CLI (результат кэшируется).

    Возвращает (path, wrapper):
      - path: путь к gigacode (может быть .cmd или .exe)
      - wrapper: None для .exe, "cmd.exe" для .cmd файлов
    """
    # Сначала проверяем .exe в стандартных местах
    for path in GIGACODE_EXE_PATHS:
        if os.path.isfile(path):
            return path, None

    # Проверяем .cmd в стандартных местах
    for path in GIGACODE_CMD_PATHS:
        if os.path.isfile(path):
            return path, "cmd.exe"

    # Fallback: через PATH
    try:
        result = subprocess.run(
            ["where", "gigacode"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().split('\n')[0].strip()
            if os.path.isfile(first_line):
                if first_line.lower().endswith('.cmd'):
                    return first_line, "cmd.exe"
                return first_line, None
    except Exception:
        pass

    return "gigacode", "cmd.exe"


def _gigacode_run(prompt: str, args, timeout: int, cwd: str) -> subprocess.CompletedProcess:
    """Запускает GigaCode CLI, передавая промпт через stdin надёжным способом.

    Вложенный `cmd.exe /c type file | gigacode` не доставляет stdin до CLI
    (CLI отвечает "No input provided via stdin"). Поэтому для .cmd-оболочки
    запускаем node-рантайм напрямую и передаём промпт через `input=`, который
    не имеет лимита длины командной строки Windows.
    """
    path, wrapper = _find_gigacode()
    if wrapper == "cmd.exe":
        root = os.path.dirname(os.path.dirname(path))
        node = os.path.join(root, "runtime", "node", "node.exe")
        cli = os.path.join(root, "lib", "cli-entry.js")
        identity = ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.match(r'\s*set\s+"GIGACODE_PACKAGE_IDENTITY=(.*?)"\s*', line)
                    if m:
                        identity = m.group(1)
                        break
        except Exception:
            pass
        base = [node, "--max-old-space-size=6144", cli]
        env = os.environ.copy()
        if identity:
            env["GIGACODE_PACKAGE_IDENTITY"] = identity
        env["GIGACODE_PACKAGE_LAUNCHER"] = path
        cmd = base + list(args)
    else:
        cmd = [path] + list(args)
        env = os.environ.copy()
    return subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=cwd,
        env=env,
        shell=False,
    )


class LLMApiClient:
    """Клиент для запросов к LLM через GigaCode CLI."""

    @staticmethod
    @lru_cache()
    def _supports_json_output() -> bool:
        """Проверяет, поддерживает ли CLI --output-format json (результат кэшируется)."""
        try:
            out = subprocess.getoutput("gigacode --help 2>&1")
            return "--output-format" in out
        except Exception:
            return False

    def __init__(
        self,
        url: str = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        retry_wait: int = None,
        token_file: str = None,
        config_path: str = None,
    ):
        self.model = model or "vllm/DeepSeek-V4-Flash-0731-262k"
        self.timeout = timeout or 180
        self.max_retries = max_retries or 10
        self._last_tokens = None
        self._supports_json = self._supports_json_output()

        # Находим gigacode (результат кэшируется на уровне модуля)
        self._gigacode_path, self._cli_wrapper = _find_gigacode()

    def _send_prompt_via_file(self, prompt: str) -> Optional[str]:
        """Отправляет промпт в GigaCode CLI через stdin с ретраями и бэкоффом."""

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for attempt in range(self.max_retries):
            try:
                # gigacode CLI читает промпт из stdin (передаётся через input в _gigacode_run).
                # --chat-recording false — не показывать каждый вызов как отдельный диалог в Desktop
                args = [
                    "--max-wall-time", "60s",
                    "--approval-mode", "auto-edit",
                    "--channel", "CI",
                    "--bare",
                    "--chat-recording", "false",
                ]
                if self._supports_json:
                    args.extend(["--output-format", "json"])

                result = _gigacode_run(prompt, args, self.timeout, project_root)

                if result.returncode == 0 and result.stdout.strip():
                    if self._supports_json:
                        try:
                            stream = json.loads(result.stdout)
                            last = stream[-1] if isinstance(stream, list) and len(stream) > 0 else {}
                            if last.get("type") == "result":
                                text_parts = []
                                for msg in stream:
                                    if msg.get("type") == "assistant" and isinstance(msg.get("message", {}).get("content"), list):
                                        for block in msg["message"]["content"]:
                                            if block.get("type") == "text":
                                                text_parts.append(block.get("text", ""))
                                text = "\n".join(text_parts).strip()
                                usage = last.get("usage") or {}
                                if not isinstance(usage, dict):
                                    usage = {}
                                self._last_tokens = {
                                    "prompt_tokens": usage.get("input_tokens", 0) or 0,
                                    "output_tokens": usage.get("output_tokens", 0) or 0,
                                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0) or 0,
                                    "total_tokens": usage.get("total_tokens", 0) or 0,
                                }
                                return text if text else result.stdout.strip()
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
                    return result.stdout.strip()
                else:
                    error_msg = result.stderr[:200] if result.stderr else f"Код возврата: {result.returncode}"
                    if "unauthorized" in error_msg.lower() or "auth" in error_msg.lower() or "token" in error_msg.lower():
                        print(f"[ERROR] Ошибка аутентификации: {error_msg}")
                        return None
                    print(f"[WARN] CLI возвратил ошибку (attempt {attempt+1}/{self.max_retries}): {error_msg[:100]}")
                    if attempt < self.max_retries - 1:
                        backoff = min(2 ** attempt, 32)
                        print(f"[WAIT] Повтор через {backoff:.1f}s...")
                        time.sleep(backoff)
                    continue

            except subprocess.TimeoutExpired:
                backoff = min(2 ** attempt, 32)
                if attempt < self.max_retries - 1:
                    print(f"[WAIT] Таймаут CLI, повтор #{attempt+1}/{self.max_retries} через {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    print(f"[WAIT] Таймаут CLI, последний повтор ({attempt+1}/{self.max_retries})")
                    return None

            except Exception as e:
                backoff = min(2 ** attempt, 32)
                if attempt < self.max_retries - 1:
                    print(f"[WAIT] Ошибка: {e}, повтор #{attempt+1}/{self.max_retries} через {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    print(f"[ERROR] Ошибка CLI: {e}, последний повтор")
                    return None

        return None

    def _chat_with_cli(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Отправляет запрос через GigaCode CLI."""
        return self._send_prompt_via_file(prompt)

    def print_token_stats(self) -> None:
        """Выводит статистику по токенам последнего запроса."""
        if not self._last_tokens:
            return
        t = self._last_tokens
        if t["prompt_tokens"] == 0 and t["output_tokens"] == 0:
            return
        cached_pct = (t["cache_read_tokens"] / t["prompt_tokens"] * 100) if t["prompt_tokens"] > 0 else 0
        print(f"Токены: запрос {t['prompt_tokens']:,} | ответ {t['output_tokens']:,} | "
              f"кэш {t['cache_read_tokens']:,} ({cached_pct:.0f}%) | "
              f"итого {t['total_tokens']:,}")

    def chat(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Отправляет промпт и возвращает текстовый ответ."""
        return self._chat_with_cli(prompt, max_tokens=max_tokens)

    def chat_json(self, prompt: str, max_tokens: int = 4096) -> Optional[Union[dict, list]]:
        """Отправляет промпт, ожидает JSON-ответ."""
        text = self._chat_with_cli(prompt, max_tokens=max_tokens)
        if text is None:
            return None

        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'(\[\{.*?\}\]|\{.*?\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        def extract_balanced_json(text):
            for i, ch in enumerate(text):
                if ch == '{':
                    depth = 0
                    start = i
                    for j in range(i, len(text)):
                        if text[j] == '{':
                            depth += 1
                        elif text[j] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(text[start:j + 1])
                                except json.JSONDecodeError:
                                    return None
                elif ch == '[':
                    depth = 0
                    start = i
                    for j in range(i, len(text)):
                        if text[j] == '[':
                            depth += 1
                        elif text[j] == ']':
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(text[start:j + 1])
                                except json.JSONDecodeError:
                                    return None
            return None

        result = extract_balanced_json(text)
        if result:
            return result

        print(f"[WARN] JSON не распарсился (LLM вернул не-JSON): {text[:200]}...")
        if "English" in text or "английском" in text.lower() or "English JSON" in text:
            print("[WARN] LLM ответил на английском — попробуем извлечь JSON из конца текста.")
            result = extract_balanced_json(text)
            if result:
                return result
        return None

    def batch_scores(
        self,
        prompt: str,
        ids: List[str],
        max_tokens: int = 4096,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Batch-запрос для оценки всех пар."""
        items = self.chat_json(prompt, max_tokens=max_tokens)
        if items is None:
            return None

        if not isinstance(items, list):
            print(f"[ERROR] Ожидается список, получено: {type(items)}")
            return None

        ids_set = set(ids)
        result = {}
        skipped = 0
        for item in items:
            if not isinstance(item, dict) or "pair" not in item or "score" not in item:
                continue
            pair_key = item["pair"]
            if pair_key in ids_set:
                score = max(0.0, min(1.0, float(item["score"])))
                reason = item.get("reason", "") or f"LLM не дала обоснование для {pair_key}"
                result[pair_key] = {"score": score, "reason": reason}
            else:
                skipped += 1

        if skipped > 0:
            print(f"[WARN] Пропущено {skipped} пар (не совпали с запрошенными ID)")

        for ident in ids:
            if ident not in result:
                result[ident] = {"score": 0.0, "reason": "LLM не вернула эту пару (неполный ответ)"}

        return result
