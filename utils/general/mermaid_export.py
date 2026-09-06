"""
mermaid_export.py — Экспорт графов в формат Mermaid.

Объединяет функции форматирования и экспорта Mermaid-графов,
используемые в 2_add_edges_LLM.py и 5_risk_graph_builder.py.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


def escape_mermaid_text(text: str) -> str:
    """
    Экранирует спецсимволы для Mermaid.

    Заменяет:
    - Переносы строк → <br/>
    - Кавычки, фигурные скобки, круглые скобки → экранированные
    - Квадратные скобки, вертикальную черту → экранированные
    """
    if not text:
        return ""
    nbs = text.replace("\\n", "<br/>").replace("\n", "<br/>").replace("\r", "")
    nbs = nbs.replace('"', '\\"').replace("{", "\\{").replace("}", "\\}")
    nbs = nbs.replace("(", "&#40;").replace(")", "&#41;")
    return (nbs.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|"))


def _csg_node_line(node: Dict) -> str:
    """Возвращает Mermaid-строку узла КСГ вида '{id}["id: name"]'."""
    nid = str(node["id"]).replace(".", "_")
    name = escape_mermaid_text(str(node.get("name", "")))
    return f'    {nid}["{node["id"]}: {name}"]'


def _write_mermaid(output_path: Union[str, Path], content: str) -> None:
    """Записывает Mermaid-содержимое в файл, создавая директории при необходимости."""
    os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def format_edges_for_prompt(edges: List[Dict], max_edges: int = 30) -> str:
    """
    Форматирует список рёбер для отображения в промпте.

    Args:
        edges: Список рёбер {"source": ..., "target": ...}.
        max_edges: Максимум строк в выводе.

    Returns:
        Отформатированная строка с рёбрами.
    """
    if not edges:
        return "  (нет связей)"
    lines = [f"  {e['source']} → {e['target']}" for e in edges[:max_edges]]
    if len(edges) > max_edges:
        lines.append(f"  ... и ещё {len(edges) - max_edges} связей")
    return "\n".join(lines)


def export_mermaid_graph(
    nodes: List[Dict],
    edges: List[Dict],
    output_path: Union[str, Path],
    graph_type: str = "TD",
    edge_label_field: Optional[str] = None,
) -> None:
    """
    Экспортирует граф в Mermaid-формат.

    Args:
        nodes: Список узлов [{"id": ..., "name": ...}].
        edges: Список рёбер [{"source": ..., "target": ..., "type": ..., "reason": ...}].
        output_path: Путь сохранения .mermaid файла.
        graph_type: Направление графа ("TD", "LR", "BT", "RL").
        edge_label_field: Поле для метки ребра (если None, используется "generated_by" + "reason").
    """
    lines = [f"graph {graph_type}"]

    # Узлы
    for node in nodes:
        lines.append(_csg_node_line(node))

    # Рёбра
    for edge in edges:
        s = str(edge.get("source", "")).replace(".", "_")
        t = str(edge.get("target", "")).replace(".", "_")

        if edge_label_field and edge_label_field in edge:
            label = str(edge[edge_label_field])
        else:
            gen = str(edge.get("generated_by", "WBS"))
            reason = str(edge.get("reason", ""))
            label = f"{gen}\\n{reason}" if reason else gen

        escaped_label = escape_mermaid_text(label)
        lines.append(f'    {s} -->|{escaped_label}| {t}')

    _write_mermaid(output_path, "\n".join(lines))
