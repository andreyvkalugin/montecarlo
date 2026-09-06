"""
llm_client.py - Единый клиент для запросов к LLM через GigaChat.

Использует официальную библиотеку gigachat (класс GigaChat) вместо
внешнего CLI. Параметры подключения берутся из переменных окружения
GIGACHAT_* (их читает Settings самой библиотеки):

  GIGACHAT_CREDENTIALS       — ключ авторизации для OAuth (обмен на access_token);
  GIGACHAT_ACCESS_TOKEN      — готовый access_token (JWE), альтернатива credentials;
  GIGACHAT_BASE_URL          — адрес API (по умолчанию gigachat.devices.sberbank.ru);
  GIGACHAT_SCOPE             — версия API (GIGACHAT_API_PERS / _CORP / _B2B);
  GIGACHAT_VERIFY_SSL_CERTS  — "true"/"false", проверять ли TLS (по умолчанию false).
"""

import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from gigachat import GigaChat
from gigachat.exceptions import AuthenticationError


def _verify_ssl_default() -> bool:
    """Читает GIGACHAT_VERIFY_SSL_CERTS (по умолчанию False для корпоративных стендов)."""
    val = os.environ.get("GIGACHAT_VERIFY_SSL_CERTS")
    if val is None:
        return False
    return val.strip().lower() not in ("0", "false", "no", "off")


@lru_cache()
def _build_gigachat_client(model: Optional[str], timeout: float, verify_ssl_certs: bool) -> GigaChat:
    """Создаёт (и кэширует) синхронный клиент GigaChat.

    credentials / access_token / base_url / scope подхватываются из
    переменных окружения GIGACHAT_* самим Settings библиотеки, поэтому
    здесь их указывать не нужно.
    """
    LLM_CRED = "Njk5OGM4NjktZTYxYi00ZWZlLWE5NDgtZjk5NGQxZDg2MDIzOjIzMzE2NjQ5LWExMDAtNGQ3NS04YmYyLTlhODdiNzk4YWVjOQ=="
    cmodel = "GigaChat-Pro" 

    return GigaChat(
        credentials=LLM_CRED,
        model=cmodel,
        timeout=float(timeout),
        verify_ssl_certs=verify_ssl_certs,
    )


def _gigachat_chat(
    prompt: str,
    model: Optional[str] = None,
    timeout: float = 180,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    verify_ssl_certs: Optional[bool] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Один синхронный вызов GigaChat.

    Возвращает кортеж (text, tokens), где tokens — статистика по токенам.
    Исключения библиотеки (AuthenticationError, ResponseError, httpx.*)
    пробрасываются наверх: логика ретраев/бэкоффа остаётся на стороне
    вызывающего кода.

    response_format: если задан (например, {"type": "json_schema", "schema": ...}),
    просит модель вернуть структурированный JSON — это резко снижает долю
    синтаксически битого JSON в ответах.
    """
    if verify_ssl_certs is None:
        verify_ssl_certs = _verify_ssl_default()

    client = _build_gigachat_client(model, float(timeout), verify_ssl_certs)

    payload: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format

    completion = client.chat(payload)

    text = completion.choices[0].message.content if completion.choices else ""
    usage = completion.usage
    tokens = {
        "prompt_tokens": usage.prompt_tokens or 0,
        "output_tokens": usage.completion_tokens or 0,
        "cache_read_tokens": usage.precached_prompt_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }
    return text or "", tokens


# ---------------------------------------------------------------------------
# Разбор JSON-ответов LLM (устойчивый к синтаксическим огрехам модели)
# ---------------------------------------------------------------------------

# response_format для батч-скоринга пар: просим модель вернуть массив
# объектов {pair, score, reason}. GigaChat при этом возвращает валидный JSON.
SCORING_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair": {"type": "string"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["pair", "score", "reason"],
                },
            }
        },
        "required": ["pairs"],
    },
}


def _strip_json_fences(text: str) -> str:
    """Убирает markdown-обёртку ```json ... ``` вокруг JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    return text


def _extract_balanced_json(text: str) -> Optional[Union[dict, list]]:
    """Ищет первый сбалансированный JSON-объект/массив в тексте."""
    openers = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch not in openers:
            continue
        closer = openers[ch]
        depth = 0
        for j in range(i, len(text)):
            if text[j] == ch:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break  # переходим к следующему открывающему символу
    return None


def _parse_json_lenient(text: str) -> Optional[Union[dict, list]]:
    """Пытается распарсить JSON из ответа LLM разными способами.

    Порядок: прямой json.loads → поиск сбалансированного фрагмента.
    Возвращает dict/list или None, если ничего не удалось.
    """
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return _extract_balanced_json(text)


def _extract_scored_pairs(text: str) -> List[Dict[str, Any]]:
    """Извлекает записи скоринга {pair, score, reason} даже из битого JSON.

    Модель иногда ломает разделители между объектами (например,
    `"reason":"...."","pair":...` вместо `"reason":"..."},{"pair":...`),
    из-за чего строгий парсер возвращает пустой результат. Здесь мы режем
    текст по маркерам `"pair":` и вытаскиваем поля по отдельности: id пары и
    числовой score восстанавливаются надёжно, reason — по возможности.
    """
    results: List[Dict[str, Any]] = []
    for chunk in re.split(r'(?="pair"\s*:)', text):
        m_pair = re.search(r'"pair"\s*:\s*"([^"]+)"', chunk)
        m_score = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', chunk)
        if not m_pair or not m_score:
            continue
        m_reason = re.search(r'"reason"\s*:\s*"(.*?)"\s*[,}\]]', chunk, re.DOTALL)
        results.append({
            "pair": m_pair.group(1),
            "score": float(m_score.group(1)),
            "reason": m_reason.group(1) if m_reason else "",
        })
    return results


def _coerce_scored_list(parsed: Any, raw_text: str) -> List[Dict[str, Any]]:
    """Приводит ответ скоринга к списку записей.

    Учитывает, что response_format может вернуть как массив, так и объект
    вида {"pairs": [...]}. Если распознать не удалось — устойчивый разбор
    по regex из сырого текста.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return _extract_scored_pairs(raw_text)


class LLMApiClient:
    """Клиент для запросов к LLM через библиотеку GigaChat."""

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
        self.model = model or "GigaChat-Pro"
        self.timeout = timeout or 180
        self.max_retries = max_retries or 10
        self._last_tokens = None

    def _send_prompt_via_file(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Отправляет промпт в GigaChat и возвращает текст ответа.

        Имя метода сохранено для совместимости с вызывающим кодом; промпт
        больше не пишется во временный файл — запрос идёт напрямую через
        библиотечный клиент GigaChat с ретраями и экспоненциальным бэкоффом.
        """
        for attempt in range(self.max_retries):
            try:
                text, tokens = _gigachat_chat(
                    prompt,
                    model=self.model,
                    timeout=self.timeout,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self._last_tokens = tokens
                text = text.strip()
                return text if text else None

            except AuthenticationError as e:
                print(f"[ERROR] Ошибка аутентификации GigaChat: {e}")
                return None

            except Exception as e:
                backoff = min(2 ** attempt, 32)
                if attempt < self.max_retries - 1:
                    print(f"[WAIT] Ошибка GigaChat: {e}, повтор #{attempt+1}/{self.max_retries} через {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    print(f"[ERROR] Ошибка GigaChat: {e}, последний повтор ({attempt+1}/{self.max_retries})")
                    return None

        return None

    def _chat_with_cli(
        self,
        prompt: str,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Отправляет запрос через библиотеку GigaChat."""
        return self._send_prompt_via_file(prompt, max_tokens=max_tokens, response_format=response_format)

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

    def chat_json(
        self,
        prompt: str,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union[dict, list]]:
        """Отправляет промпт, ожидает JSON-ответ."""
        text = self._chat_with_cli(prompt, max_tokens=max_tokens, response_format=response_format)
        if text is None:
            return None

        result = _parse_json_lenient(text)
        if result is not None:
            return result

        print(f"[WARN] JSON не распарсился (LLM вернул не-JSON): {_strip_json_fences(text)[:200]}...")
        return None

    def batch_scores(
        self,
        prompt: str,
        ids: List[str],
        max_tokens: int = 4096,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Batch-запрос для оценки всех пар.

        Просит модель вернуть структурированный JSON (response_format), а при
        синтаксически битом ответе восстанавливает записи скоринга по regex,
        чтобы не терять весь раунд из-за одной сломанной запятой/кавычки.
        """
        text = self._chat_with_cli(
            prompt, max_tokens=max_tokens, response_format=SCORING_RESPONSE_FORMAT
        )
        if text is None:
            return None

        items = _coerce_scored_list(_parse_json_lenient(text), text)
        if not items:
            print(f"[WARN] Скоринг: не удалось извлечь пары из ответа: {_strip_json_fences(text)[:200]}...")
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
