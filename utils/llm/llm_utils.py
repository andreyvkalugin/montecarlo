"""
llm_utils.py — Общие утилиты для работы с LLM.

Содержит функции для:
- Парсинга JSON-ответов от LLM
- Нормализации структуры рёбер
"""
import json
import re
from typing import Any, Dict, List, Optional


def normalize_llm_response(response: Any) -> List[Dict]:
    """
    Универсальный парсер ответа LLM для рёбер графа.

    Поддерживает форматы:
    - {"edges": [...]}
    - [...]-список рёбер
    - JSON-строка с markdown-обёрткой

    Возвращает нормализованный список рёбер с полями:
    {source, target, type, reason}
    """
    if response is None:
        return []
    if isinstance(response, dict):
        raw = response.get("edges", [])
    elif isinstance(response, list):
        raw = response
    elif isinstance(response, str):
        raw = extract_edges_from_json(response)
        if not raw:
            raw = extract_edges_from_text(response).get("edges", [])
    else:
        raw = []

    edges = []
    for e in raw:
        if isinstance(e, dict):
            s = str(e.get("source", "")).strip()
            t = str(e.get("target", "")).strip()
            entry = {
                "source": s,
                "target": t,
                "type": str(e.get("type", "FS")).strip() or "FS",
            }
            if e.get("reason"):
                entry["reason"] = (e.get("reason") or "").strip()
            edges.append(entry)

    return edges


def extract_edges_from_json(text: str) -> List[Dict]:
    """Извлекает рёбра из JSON-ответа LLM."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    edges = parsed.get("edges", [])
    if isinstance(edges, list):
        return edges
    return [edges] if isinstance(edges, dict) else []


def extract_edges_from_text(text: str) -> Dict:
    """Извлекает рёбра из текстового ответа LLM (regex-парсинг)."""
    result = {"edges": []}
    pattern = r'"source"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"'
    for match in re.finditer(pattern, text):
        s, t = match.group(1), match.group(2)
        start = match.start()
        before = text.rfind("{", 0, start)
        end = text.find("}", start)
        block = text[before:end + 1] if before != -1 and end != -1 else text[start:match.end()]
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', block)
        type_match = re.search(r'"type"\s*:\s*"([^"]*)"', block)
        reason = reason_match.group(1) if reason_match else ""
        edge_type = type_match.group(1) if type_match else "FS"
        result["edges"].append({"source": s, "target": t, "type": edge_type, "reason": reason})
    return result
