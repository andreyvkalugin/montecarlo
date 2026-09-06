"""
json_io.py — Централизованные функции загрузки/сохранения JSON.

Заменяют дублирующиеся load_json/save_json в скриптах шагов пайплайна.
Используют os.makedirs для автосоздания директорий.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def load_json(path: Union[str, Path], encoding: str = "utf-8") -> Dict:
    """
    Загружает JSON-файл по пути.

    Args:
        path: Путь к файлу.
        encoding: Кодировка файла.

    Returns:
        Словарь/список из JSON.
    """
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(path: Union[str, Path], data: Dict, encoding: str = "utf-8") -> None:
    """
    Сохраняет JSON-данные по пути.
    Автоматически создаёт директории, если их нет.

    Args:
        path: Путь сохранения.
        data: Данные для сохранения.
        encoding: Кодировка файла.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
