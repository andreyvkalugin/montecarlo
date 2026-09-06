"""
7_bayesian_network_builder.py - Модуль 7
Построение Байесовской сети из графа рисков с весами.

На вход:
  - risk_graph_with_weights.json (риски + вероятности + веса связей)

На выход:
  - bayesian_network.json (структура сети + CPT)
  - bayesian_network.mermaid (Mermaid-диаграмма DAG BN)
"""

import os
import sys
from typing import Dict, List
from dataclasses import dataclass
import itertools
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json

# Единый конфиг пайплайна — ЕДИНСТВЕННЫЙ источник всех параметров
PIPELINE_CONFIG = get_pipeline_config()

# Извлекаем конфигурацию bayesian_network — ТОЛЬКО из PIPELINE_CONFIG
BN = PIPELINE_CONFIG.get("bayesian_network", {})

# Извлекаем конфигурацию model_assumptions (порог связи используется единым
# для шагов 8 и 9) — ТОЛЬКО из PIPELINE_CONFIG
MA = PIPELINE_CONFIG.get("model_assumptions", {})


# ==================== КЛАССЫ ДАННЫХ ====================

@dataclass
class RiskData:
    """Данные по риску из графа"""
    risk_id: str
    name: str
    node_id: str
    node_name: str
    p50: float
    expected_delay: float
    expected_budget: float
    prob_min: float
    prob_max: float
    delay_min: float
    delay_max: float
    budget_min: float
    budget_max: float


@dataclass
class ParentInfo:
    """Информация о родителе в Байесовской сети"""
    risk_id: str
    name: str
    weight: float  # final_weight


@dataclass
class CptRow:
    """Одна строка CPT"""
    parents_states: Dict[str, int]  # risk_id -> 0/1
    probability: float


class BayesianNetworkBuilder:
    def __init__(self, config: Dict = None):
        # Загружаем единственную конфигурацию из BN (PIPELINE_CONFIG)
        self.config = BN.copy()
        if config:
            self._deep_update(self.config, config)

        # Извлекаем параметры CPT из BN (PIPELINE_CONFIG)
        self.cpt_config = self.config.get('cpt', {})
        # alpha/beta удалены — CPT использует final_weight напрямую
        # Порог связи — единый из model_assumptions (используется шагами 8 и 9)
        self.threshold = MA.get('threshold')

        # Проверяем что все обязательные параметры загружены
        if self.threshold is None:
            print("[ERROR]  ERROR: Missing threshold in PIPELINE_CONFIG['model_assumptions']", file=sys.stderr)
            sys.exit(1)

        self.risks: Dict[str, RiskData] = {}
        self.risk_edges: List[Dict] = []
        self.risk_names: Dict[str, str] = {}
        
        # Результаты
        self.parents: Dict[str, List[ParentInfo]] = {}  # risk_id -> [ParentInfo]
        self.cpt_tables: Dict[str, List[CptRow]] = {}   # risk_id -> [CptRow]
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    # -------------------- 1. ЗАГРУЗКА ДАННЫХ --------------------
    
    def load_risk_graph_with_weights(self, json_file: str = None, data: dict = None) -> None:
        """Загрузка обогащенного графа рисков.

        data: взвешенный граф из состояния LangGraph (in-memory); если None — читается файл.
        """
        if data is None:
            data = load_json(json_file)

        # Загружаем риски
        for risk_id, risk_data in data['risks'].items():
            self.risks[risk_id] = RiskData(
                risk_id=risk_id,
                name=risk_data['name'],
                node_id=risk_data['node_id'],
                node_name=risk_data['node_name'],
                p50=risk_data['probability']['p50'],
                prob_min=risk_data['probability']['min'],
                prob_max=risk_data['probability']['max'],
                expected_delay=risk_data['impact']['delay']['expected'],
                delay_min=risk_data['impact']['delay']['min'],
                delay_max=risk_data['impact']['delay']['max'],
                expected_budget=risk_data['impact']['budget']['expected'],
                budget_min=risk_data['impact']['budget']['min'],
                budget_max=risk_data['impact']['budget']['max']
            )
            self.risk_names[risk_id] = risk_data['name']
        
        # Загружаем связи
        self.risk_edges = data.get('edges', [])

        print(f"Загружено: {len(self.risks)} рисков, {len(self.risk_edges)} связей")

    # -------------------- 1.1 РАСЧЕТ p50 --------------------

    def _calculate_risk_probabilities(self) -> None:
        """Расчет p50, expected_delay, expected_budget из реестра."""
        print(f"[STEP] Расчет базовых характеристик рисков (p50)...")
        for risk in self.risks.values():
            risk.p50 = (risk.prob_min + risk.prob_max) / 2
            risk.expected_delay = (risk.delay_min + risk.delay_max) / 2
            risk.expected_budget = (risk.budget_min + risk.budget_max) / 2

    # -------------------- 2. РАСЧЕТ СТРУКТУРНЫХ ВЕСОВ --------------------

    def calculate_structural_weights(self) -> None:
        """
        Расчет structural_weight из PIPELINE_CONFIG['model_assumptions']['structural_decay'].
        Формула: structural(d) = a + (b - a) * (1 - d / D)
        """
        print("\n[STEP] Расчет структурных весов (continuous)...")

        MA = PIPELINE_CONFIG.get("model_assumptions", {})
        decay = MA.get("structural_decay", {})
        diameter = decay.get("diameter", 1)

        if diameter <= 0:
            try:
                graph_data = load_json(paths.to_str(paths.risk_graph_json))
                summary = graph_data.get("summary", {})
                d_from_graph = summary.get("diameter", 0)
                if d_from_graph and d_from_graph > 0:
                    diameter = d_from_graph
            except Exception:
                diameter = max(1, max(e.get("distance", 1) for e in self.risk_edges))

        a = decay.get("a", -0.2)
        b = decay.get("b", 0.7)
        w0 = decay.get("w0", 1.0)

        for edge in self.risk_edges:
            d = edge.get("distance", 1)
            if diameter <= 0:
                diameter = 1
            if d == 0:
                edge["weights"]["structural"] = max(0.0, min(1.0, w0))
            else:
                structural = a + (b - a) * (1 - d / diameter)
                edge["weights"]["structural"] = max(0.0, min(1.0, structural))

        print(f"-  diameter={diameter}, a={a}, b={b}, w0={w0}")

    # -------------------- 3. РАСЧЕТ ФИНАЛЬНЫХ ВЕСОВ --------------------

    def calculate_final_weights(self) -> None:
        """
        Расчет final_weight из structural + semantic (перед построением CPT).
        Формула: final = min(1.0, semantic × (1 + structural))
        """
        print("\n[STEP] Расчет финальных весов из structural + semantic...")

        calculated = 0
        for edge in self.risk_edges:
            structural = edge['weights']['structural']
            semantic = edge['weights']['semantic']
            edge['weights']['final'] = min(1.0, semantic * (1 + structural))
            calculated += 1

        print(f"-  Рассчитано финальных весов: {calculated}")

        avg_final = sum(e['weights']['final'] for e in self.risk_edges) / len(self.risk_edges) if self.risk_edges else 0
        print(f"-  Средний final_weight: {avg_final:.4f}")

    # -------------------- 3. ОПРЕДЕЛЕНИЕ РОДИТЕЛЕЙ --------------------

    def build_parents(self) -> None:
        """
        Для каждого риска определяем родители — риски, которые на него влияют.
        Родители — это предшественники с final_weight > threshold.
        """
        print("\n[STEP] Построение структуры Байесовской сети...")
        
        # Строим карту: для каждого риска собираем всех, кто на него влияет
        parents_map: Dict[str, List[ParentInfo]] = {}
        
        for edge in self.risk_edges:
            source = edge['source']
            target = edge['target']
            final_weight = edge['weights']['final']
            
            if final_weight > self.threshold:
                if target not in parents_map:
                    parents_map[target] = []
                parents_map[target].append(ParentInfo(
                    risk_id=source,
                    name=self.risk_names.get(source, source),
                    weight=final_weight
                ))
        
        # Сортируем родителей по весу (убывание)
        for target in parents_map:
            parents_map[target].sort(key=lambda x: x.weight, reverse=True)
        
        self.parents = parents_map
        
        # Статистика
        total_children = len(parents_map)
        total_parents = sum(len(p) for p in parents_map.values())
        avg_parents = total_parents / total_children if total_children > 0 else 0
        
        print(f"- Узлов с родителями: {total_children}")
        print(f"- Всего связей: {total_parents}")
        print(f"- Среднее число родителей: {avg_parents:.2f}")
        
        # Список рисков без родителей
        no_parents = [r for r in self.risks.keys() if r not in parents_map or not parents_map[r]]
        if no_parents:
            print(f"- Рисков без родителей: {len(no_parents)}")
    
    # -------------------- 3. РАСЧЕТ CPT --------------------

    def calculate_cpt(self) -> None:
        """
        Расчет таблиц условных вероятностей (CPT) для каждого риска.

        Формула:
        P(risk=1 | parents) = base_p * ∏ adjustment_i
        где adjustment_i = (1 + final_weight) если parent state=1
                     или 1.0 если parent state=0 (alpha/beta удалены)
        """
        print("\n[STEP] Расчет CPT (таблиц условных вероятностей)...")

        for risk_id in sorted(self.risks.keys()):  # Deterministic order
            risk = self.risks[risk_id]
            # Родители для этого риска
            risk_parents = self.parents.get(risk_id, [])
            base_p = risk.p50

            if not risk_parents:
                # Нет родителей — независимый риск (только базовая вероятность)
                self.cpt_tables[risk_id] = [
                    CptRow(parents_states={}, probability=base_p)
                ]
                continue

            # Все возможные комбинации состояний родителей
            n = len(risk_parents)
            combinations = list(itertools.product([0, 1], repeat=n))

            cpt_rows = []
            for combo in combinations:
                parents_states = {
                    risk_parents[i].risk_id: combo[i]
                    for i in range(n)
                }

                # Расчет вероятности
                p = base_p
                for i, parent in enumerate(risk_parents):
                    state = combo[i]
                    weight = parent.weight

                    if state == 1:
                        # Родитель наступил → усиливаем на weight (final_weight)
                        adjustment = 1 + weight
                    else:
                        # Родитель не наступил → множитель 1.0 (alpha/beta удалены)
                        adjustment = 1.0

                    p = p * adjustment

                # Ограничиваем [0, 1]
                p = max(0.0, min(1.0, p))

                cpt_rows.append(CptRow(
                    parents_states=parents_states,
                    probability=p
                ))

            self.cpt_tables[risk_id] = cpt_rows
            
        
        # Статистика
        total_rows = sum(len(rows) for rows in self.cpt_tables.values())
        avg_rows = total_rows / len(self.cpt_tables) if self.cpt_tables else 0
        print(f"Всего CPT-строк: {total_rows}")
        print(f"Среднее строк на риск: {avg_rows:.2f}")
    
    # -------------------- 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ --------------------
    
    def save_bayesian_network(self, output_file: str) -> None:
        """Сохранение структуры Байесовской сети с CPT"""
        data = {
            "risks": {
                risk_id: {
                    "name": risk.name,
                    "node_id": risk.node_id,
                    "node_name": risk.node_name,
                    "p50": risk.p50,
                    "prob_min": risk.prob_min,
                    "prob_max": risk.prob_max,
                    "expected_delay": risk.expected_delay,
                    "expected_budget": risk.expected_budget,
                    "delay_min": risk.delay_min,
                    "delay_max": risk.delay_max,
                    "budget_min": risk.budget_min,
                    "budget_max": risk.budget_max,
                    "cpt": [
                        {
                            "states": row.parents_states,
                            "probability": round(row.probability, 4)
                        }
                        for row in self.cpt_tables.get(risk_id, [])
                    ]
                }
                for risk_id in sorted(self.risks.keys())  # Deterministic order
                for risk in [self.risks[risk_id]]
            },
            "parents": {
                risk_id: [
                    {
                        "risk_id": p.risk_id,
                        "name": p.name,
                        "weight": p.weight
                    }
                    for p in parents
                ]
                for risk_id in sorted(self.parents.keys())  # Deterministic order
                for parents in [self.parents[risk_id]]
            },
            "summary": {
                "total_risks": len(self.risks),
                "total_parent_relationships": sum(len(p) for p in self.parents.values()),
                "risks_with_parents": len([p for p in self.parents.values() if p]),
                "risks_without_parents": len([r for r in self.risks.keys() if r not in self.parents or not self.parents[r]]),
                "cpt_parameters": {
                    "threshold": self.threshold,
                    "note": "alpha/beta удалены — CPT использует final_weight напрямую"
                }
            }
        }

        save_json(output_file, data)
        print(f"[SAVE] Сохранена структура сети: {output_file}")
        return data

    def print_summary(self) -> None:
        """Вывод краткой сводки по Байесовской сети"""
        print("\n[RESULTS]")

        # Риски с родителями
        with_parents = [r for r in self.risks.keys() if r in self.parents and self.parents[r]]
        without_parents = [r for r in self.risks.keys() if r not in self.parents or not self.parents[r]]

        print(f"-  Всего рисков: {len(self.risks)}")
        print(f" - с родителями: {len(with_parents)}")
        print(f" - без родителей (независимые): {len(without_parents)}")

        # Параметры CPT
        print(f"\n-  Параметры CPT:")
        print(f" - threshold (порог): {self.threshold}")

    # -------------------- 5. MERMAIDI-ДИАГРАММА --------------------

    def _escape_md(self, text: str) -> str:
        """Экранирование спецсимволов для Mermaid-меток."""
        return text.replace("\n", "\\n").replace('"', '\\"')

    def _format_cpt_lines(self, risk_id: str, parent_ids: List[str]) -> str:
        """
        Форматирует строки CPT для отображения внутри вершины Mermaid.
        Если комбинаций > 6, показывает первые 3, троеточие и последние 3.
        """
        rows = []
        n = len(parent_ids)

        for combo in itertools.product([0, 1], repeat=n):
            combo_str = "".join(str(combo[i]) for i in range(n))
            states = {parent_ids[i]: combo[i] for i in range(n)}
            prob = 0.0
            for row in self.cpt_tables[risk_id]:
                if row.parents_states == states:
                    prob = row.probability
                    break
            rows.append(f"{combo_str}→{prob:.3f}")

        if len(rows) > 6:
            return "\n".join(rows[:3] + ["..."] + rows[-3:])
        return "\n".join(rows)


# ==================== ТОЧКА ВХОДА ====================

def main(weights: dict = None) -> dict:
    """Точка входа.

    weights: взвешенный граф рисков из шага 6 (in-memory из состояния LangGraph).
    Если None — читается с диска (standalone-режим).
    """
    sys.stdout.reconfigure(encoding="utf-8")

    # Пути
    INPUT_FILE = paths.risk_graph_with_weights_json
    OUTPUT_DIR = paths.step_dirs[7]
    os.makedirs(paths.to_str(OUTPUT_DIR), exist_ok=True)

    # Конфиг (переопределяем только если нужно)
    custom_config = {}  # Используем DEFAULT_CONFIG по умолчанию

    print("\n[STEP] ПОСТРОЕНИЕ БАЙЕСОВСКОЙ СЕТИ")

    # Строим сеть
    builder = BayesianNetworkBuilder(custom_config)
    builder.load_risk_graph_with_weights(paths.to_str(INPUT_FILE), data=weights)
    builder._calculate_risk_probabilities()
    builder.calculate_structural_weights()  # ← structural из decay
    builder.calculate_final_weights()       # ← final = semantic * (1 + structural)
    builder.build_parents()
    builder.calculate_cpt()

    # Сохраняем
    bayes_doc = builder.save_bayesian_network(paths.to_str(paths.bayesian_network_json))

    # Статистика
    builder.print_summary()

    # Возвращаем сеть для состояния LangGraph (JSON уже сохранён)
    return bayes_doc

if __name__ == "__main__":
    main()
