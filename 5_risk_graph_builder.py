"""
5_risk_graph_builder.py - Модуль 5
Построение графа рисков и таблицы попарных расстояний.

Вход:  2_data/csg_edges_final.json, 4_data/risk_mapping_final.json
Выход: 5_data/risk_graph.json, 5_data/pairwise_distances.json,
       5_data/risk_graph.mermaid
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools
from utils.general.paths import paths
from utils.general.json_io import load_json, save_json


@dataclass
class Risk:
    risk_id: str
    risk_name: str
    node_id: str
    node_name: str


@dataclass
class RiskPair:
    risk_a_id: str
    risk_a_name: str
    risk_b_id: str
    risk_b_name: str
    distance: int           # -1 если пути нет
    is_predecessor: bool    # True если risk_a -> risk_b


class RiskGraphBuilder:
    def __init__(self):
        self.work_nodes = {}        # вершины КСГ (для расчета расстояний)
        self.work_edges = []        # ребра КСГ (для расчета расстояний)
        self.risks = []             # список рисков
        self.risk_map = {}          # {risk_id: Risk}
        self.risk_node_map = defaultdict(list)  # {node_id: [Risk, ...]}
        self.dist_matrix = {}       # расстояния между вершинами КСГ

    def load_dag(self, dag_file: str) -> None:
        """Загружает граф КСГ для расчета топологических расстояний"""
        data = load_json(dag_file)

        for node in data['nodes']:
            self.work_nodes[node['id']] = node['name']

        for edge in data['edges']:
            self.work_edges.append((edge['source'], edge['target']))

        print(f"[LOAD] {dag_file}")

    def load_risks(self, mapping_file: str) -> None:
        """Загружает риски и их привязку к вершинам КСГ"""
        data = load_json(mapping_file)

        self.risks_dict = {}  # {risk_id: {"name": ..., "node_id": ..., "node_name": ...}}

        for m in data['risk_mappings']:
            risk = Risk(
                risk_id=m['risk_number'],
                risk_name=m['risk_name'],
                node_id=m['matched_nodes'][0],
                node_name=m['node_name']
            )
            self.risks.append(risk)
            self.risk_map[risk.risk_id] = risk
            self.risk_node_map[risk.node_id].append(risk)
            self.risks_dict[risk.risk_id] = {
                "name": risk.risk_name,
                "node_id": risk.node_id,
                "node_name": risk.node_name
            }

        unique_nodes = len(set(r.node_id for r in self.risks))
        print(f"[LOAD] {mapping_file}")

    def _floyd_warshall(self) -> Dict[Tuple[str, str], int]:
        """Вычисляет расстояния между всеми парами вершин КСГ"""
        nodes = list(self.work_nodes.keys())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}

        INF = 999
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for src, dst in self.work_edges:
            dist[idx[src]][idx[dst]] = 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        result = {}
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j and dist[i][j] < INF:
                    result[(a, b)] = dist[i][j]

        # Вычисляем диаметр графа: максимальное расстояние между любыми двумя вершинами
        self.diameter = max(result.values()) if result else 1

        return result

    def _get_risk_distance(self, risk_a: Risk, risk_b: Risk) -> Tuple[int, bool]:
        """
        Определяет расстояние и направление между двумя рисками
        Возвращает: (distance, is_predecessor)
        где is_predecessor = True если risk_a -> risk_b
        """
        # Если риски на одной вершине — расстояние 0, нет предшественника
        if risk_a.node_id == risk_b.node_id:
            return 0, False

        d_forward = self.dist_matrix.get((risk_a.node_id, risk_b.node_id), -1)
        d_backward = self.dist_matrix.get((risk_b.node_id, risk_a.node_id), -1)

        if d_forward > 0:
            return d_forward, True
        elif d_backward > 0:
            return d_backward, False
        else:
            return -1, False

    def _get_all_pairs(self) -> List[RiskPair]:
        """Возвращает все пары рисков с расстояниями (единый источник построения пар)."""
        pairs = []
        for a, b in itertools.combinations(self.risks, 2):
            distance, is_predecessor = self._get_risk_distance(a, b)
            pairs.append(RiskPair(
                risk_a_id=a.risk_id,
                risk_a_name=a.risk_name,
                risk_b_id=b.risk_id,
                risk_b_name=b.risk_name,
                distance=distance,
                is_predecessor=is_predecessor
            ))
        return pairs

    def build(self) -> List[RiskPair]:
        """
        Строит граф рисков:
        - вершины: риски
        - ребра: зависимости между рисками
        - расстояние: топологическое расстояние между вершинами КСГ
        """
        # Вычисляем расстояния между вершинами КСГ
        self.dist_matrix = self._floyd_warshall()

        # Перебираем все пары рисков через общий метод
        pairs = self._get_all_pairs()

        # Сортируем по расстоянию (сначала связанные)
        pairs.sort(key=lambda x: (x.distance == -1, x.distance))

        return pairs

    def save_json(self, pairs: List[RiskPair], output_file: str) -> None:
        """
        Сохраняет граф рисков в компактном словарном формате.
        Каждый риск содержит:
        - predecessors: риски, которые влияют на него
        - successors: риски, на которые он влияет
        - unrelated: риски без топологической связи
        """
        # 1. Словарь рисков
        risks_dict = {
            r.risk_id: {
                "name": r.risk_name,
                "node_id": r.node_id,
                "node_name": r.node_name
            }
            for r in self.risks
        }

        # 2. Инициализация структуры зависимостей
        deps = {
            r.risk_id: {
                "predecessors": [],   # кто влияет на этот риск
                "successors": [],     # на кого влияет этот риск
                "unrelated": []       # нет пути между вершинами
            }
            for r in self.risks
        }

        # 3. Заполнение связей
        for p in pairs:
            if p.distance == -1:
                # Несвязанные риски
                deps[p.risk_a_id]["unrelated"].append({
                    "risk_id": p.risk_b_id,
                    "distance": -1
                })
                deps[p.risk_b_id]["unrelated"].append({
                    "risk_id": p.risk_a_id,
                    "distance": -1
                })
            elif p.is_predecessor:
                # A -> B (A предшествует B)
                deps[p.risk_a_id]["successors"].append({
                    "risk_id": p.risk_b_id,
                    "distance": p.distance
                })
                deps[p.risk_b_id]["predecessors"].append({
                    "risk_id": p.risk_a_id,
                    "distance": p.distance
                })
            else:
                # B -> A (B предшествует A)
                deps[p.risk_b_id]["successors"].append({
                    "risk_id": p.risk_a_id,
                    "distance": p.distance
                })
                deps[p.risk_a_id]["predecessors"].append({
                    "risk_id": p.risk_b_id,
                    "distance": p.distance
                })

        # 4. Сортируем связи по расстоянию
        for risk_id in deps:
            deps[risk_id]["predecessors"].sort(key=lambda x: x["distance"])
            deps[risk_id]["successors"].sort(key=lambda x: x["distance"])

        # 5. Сборка итогового JSON
        data = {
            "risks": risks_dict,
            "dependencies": deps,
            "summary": {
                "total_risks": len(self.risks),
                "total_pairs": len(pairs),
                "nodes_with_risks": len({r.node_id for r in self.risks}),
                "total_work_nodes": len(self.work_nodes),
                "total_work_edges": len(self.work_edges),
                "diameter": self.diameter
            }
        }

        save_json(output_file, data)

        print(f"[SAVE] {output_file}")

    def save_pairs_json(self, pairs: List[RiskPair], output_file: str) -> None:
        """
        Сохраняет попарные расстояния между рисками в JSON.

        Плоский список пар (аналог прежнего CSV, по строке на пару) + metadata
        со сводкой. source всегда предшествует target; для несвязанных пар
        relation_type = "unrelated".
        """
        rows = []
        for p in pairs:
            if p.distance == -1:
                rel, s_id, s_name, t_id, t_name = ('unrelated', p.risk_a_id, p.risk_a_name,
                                                   p.risk_b_id, p.risk_b_name)
            elif p.is_predecessor:
                rel, s_id, s_name, t_id, t_name = ('predecessor', p.risk_a_id, p.risk_a_name,
                                                   p.risk_b_id, p.risk_b_name)
            else:
                rel, s_id, s_name, t_id, t_name = ('successor', p.risk_b_id, p.risk_b_name,
                                                   p.risk_a_id, p.risk_a_name)

            rows.append({
                "source_risk_id": s_id,
                "source_risk_name": s_name,
                "source_node_id": self.risk_map[s_id].node_id,
                "target_risk_id": t_id,
                "target_risk_name": t_name,
                "target_node_id": self.risk_map[t_id].node_id,
                "distance": p.distance,
                "relation_type": rel,
            })

        by_relation: Dict[str, int] = {}
        for row in rows:
            by_relation[row["relation_type"]] = by_relation.get(row["relation_type"], 0) + 1

        data = {
            "metadata": {
                "description": "Попарные топологические расстояния между рисками на графе КСГ.",
                "total_risks": len(self.risks),
                "total_pairs": len(rows),
                "connected": sum(v for k, v in by_relation.items() if k != "unrelated"),
                "unrelated": by_relation.get("unrelated", 0),
                "by_relation_type": by_relation,
                "nodes_with_risks": len({r.node_id for r in self.risks}),
                "total_work_nodes": len(self.work_nodes),
                "total_work_edges": len(self.work_edges),
                "diameter": self.diameter,
                "distance_semantics": "-1 — пути нет; 0 — одна вершина КСГ; N>0 — топологическое расстояние",
                "version": "5.0-json",
            },
            "pairs": rows,
        }

        save_json(output_file, data)

        print(f"[SAVE] {output_file}")

    def to_mermaid_risk_graph(self, output_file: str) -> None:
        """
        Визуализация ГРАФА РИСКОВ (не КСГ)
        ПОЛНЫЙ текст рисков без обрезания
        """
        lines = ["flowchart TD"]

        # Создаем узлы для рисков с ПОЛНЫМ названием
        for risk in self.risks:
            # Экранируем спецсимволы для Mermaid
            clean_name = (risk.risk_name
                        .replace('"', '&quot;')
                        .replace("'", "&#39;")
                        .replace("[", "&#91;")
                        .replace("]", "&#93;")
                        .replace("(", "&#40;")
                        .replace(")", "&#41;"))
            label = f"{risk.risk_id}<br>{clean_name}"
            lines.append(f'    {risk.risk_id}["{label}"]')

        lines.append("")

        # Добавляем ребра между рисками
        edges = set()
        for p in self._get_all_pairs():
            if p.distance >= 0 and p.is_predecessor:
                edges.add((p.risk_a_id, p.risk_b_id, p.distance))
            elif p.distance >= 0 and not p.is_predecessor:
                edges.add((p.risk_b_id, p.risk_a_id, p.distance))

        # Сортируем по расстоянию, затем по source, затем по target
        edges = sorted(edges, key=lambda x: (x[2], x[0], x[1]))

        for src, dst, dist in edges:
            if dist == 0:
                style = "==>"          # одна вершина — двойная стрелка
            elif dist <= 2:
                style = "-->"          # расстояние 1-2 — обычная
            elif dist <= 4:
                style = "---"          # расстояние 3-4 — штриховая (без стрелки, только связь)
            else:
                style = "-.->"         # расстояние 5+ — пунктирная
            lines.append(f'    {src} {style}|dist {dist}| {dst}')

        lines.append("")
        lines.append("    %% Легенда:")
        lines.append("    %% ==>  расстояние 0 (одна вершина)")
        lines.append("    %% -->  расстояние 1-2 (прямое влияние)")
        lines.append("    %% ---  расстояние 3-4 (опосредованное)")
        lines.append("    %% -.-> расстояние 5+ (слабое влияние)")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"[SAVE] {output_file}")


def main():
    """Точка входа"""
    # UTF-8 вывод для Windows
    sys.stdout.reconfigure(encoding="utf-8")

    OUT = paths.step_dirs[5]
    os.makedirs(paths.to_str(OUT), exist_ok=True)

    builder = RiskGraphBuilder()

    # Загрузка данных
    dag_path = paths.csg_edges_final_json
    mapping_path = paths.risk_mapping_json

    builder.load_dag(paths.to_str(dag_path))
    builder.load_risks(paths.to_str(mapping_path))

    # Построение графа рисков
    pairs = builder.build()

    # Сохранение результатов
    builder.save_json(pairs, paths.to_str(paths.risk_graph_json))
    builder.save_pairs_json(pairs, paths.to_str(paths.pairwise_distances_json))

    # Визуализация
    builder.to_mermaid_risk_graph(paths.to_str(paths.risk_graph_mermaid))

    print(f"\n[RESULTS]")
    print(f"-  Рисков: {len(builder.risks)}")
    print(f"-  Пар рисков: {len(pairs)}")
    connected = sum(1 for p in pairs if p.distance > 0)
    unrelated = sum(1 for p in pairs if p.distance == -1)
    print(f"-  Связанных: {connected}")
    print(f"-  Несвязанных: {unrelated}")


if __name__ == "__main__":
    main()