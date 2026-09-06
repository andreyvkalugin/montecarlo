"""
8_monte_carlo_simulator.py - Модуль 8
Монте-Карло симуляция на основе Байесовской сети.

ВЕРОЯТНОСТНАЯ ГЕНЕРАЦИЯ СЦЕНАРИЕВ (логика ModelRisk).
Вероятность сценария оценивается через CPT.
Ущерб добавляется ПОЛНОСТЬЮ (без взвешивания), если риск наступил.

На вход:
  - paths.bayesian_network_json (структура сети + CPT)
  - paths.csg_tasks_wbs.json (для получения длительности проекта)

На выход:
  - paths.simulation_results_json (распределения + статистика)
  - paths.final_delay_samples_npy (абсолютные даты завершения)
  - paths.final_budget_samples_npy (абсолютный бюджет)
  - paths.iteration_scenarios_json (сценарии с вероятностями)
"""

import json
import os
import sys
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json

# Единый конфиг пайплайна — ЕДИНСТВЕННЫЙ источник параметров
PIPELINE_CONFIG = get_pipeline_config()

# Извлекаем конфигурацию Monte Carlo — ТОЛЬКО из PIPELINE_CONFIG
MC = PIPELINE_CONFIG.get("monte_carlo", {})

# Базовые срок и бюджет проекта — из раздела "project"
PROJECT = PIPELINE_CONFIG.get("project", {})

# Конфигурация копулы
COUPULA_CONFIG = PIPELINE_CONFIG.get("copula", {})
COUPULA_CORRELATION = COUPULA_CONFIG.get("correlation", 0.0)


# ==================== КЛАССЫ ДАННЫХ ====================

@dataclass
class RiskData:
    """Данные по риску"""
    risk_id: str
    name: str
    node_id: str
    node_name: str
    p50: float                      # базовая вероятность
    prob_min: float
    prob_max: float
    delay_min: float                # минимальная задержка (дни)
    delay_max: float                # максимальная задержка (дни)
    expected_delay: float           # ожидаемая задержка
    budget_min: float               # минимальный перерасход (рубли)
    budget_max: float               # максимальный перерасход (рубли)
    expected_budget: float          # ожидаемый перерасход
    parents: List[Dict]             # список родителей с весами
    cpt: List[Dict]                 # таблица условных вероятностей


@dataclass
class SimulationResult:
    """Результат одного прогона"""
    iteration: int
    states: Dict[str, int]                  # полный вектор состояний
    activated_risks: List[str]              # ID рисков, которые наступили (state=1)
    total_delay: float                      # итоговая задержка в днях (прирост)
    delay_by_risk: Dict[str, float]         # вклад в задержку по каждому риску (дни)
    budget_risk_impact: float               # влияние рисков из реестра (прирост, суммарно)
    budget_copula_impact: float             # влияние за счет копулы (время↔бюджет, суммарно)
    budget_risk_impact_by_risk: Dict[str, float]   # прямой эффект риска из реестра (по каждому риску)
    budget_copula_impact_by_risk: Dict[str, float] # копула-эффект для задержки риска (по каждому риску)
    total_budget_impact: float              # итоговый перерасход (budget_risk + copula)
    final_duration: float                   # итоговый срок (база + прирост)
    final_budget: float                     # итоговый бюджет (база + прирост)
    scenario_probability: float             # вероятность этого сценария по CPT
    risk_probabilities: Dict[str, float]    # вероятность каждого риска в этом сценарии


# ==================== ОСНОВНОЙ КЛАСС ====================

class MonteCarloSimulator:
    def __init__(self, config: Dict = None):
        # Прямое извлечение параметров без _deep_update
        self.iterations = MC.get('iterations')
        self.distribution_type = MC.get('distribution', {}).get('type', 'normal')
        self.sigma_div = MC.get('distribution', {}).get('normal_sigma_div', 6)
        self.random_seed = MC.get('random_seed', 42)
        
        # Целевая дата завершения проекта — из раздела "project"
        self.complete_date_str = PROJECT.get('complete_date', '2026-02-05')
        self.base_budget = PROJECT.get('base_budget')

        # Загрузка длительности проекта из результатов шага 1
        self.project_duration_days = 0
        try:
            if os.path.exists(str(paths.csg_tasks_wbs_json)):
                csg_data = load_json(str(paths.csg_tasks_wbs_json))
                meta = csg_data.get('metadata', {})
                self.project_duration_days = meta.get('project_duration_days', 0)
            print(f"Загружена длительность проекта: {self.project_duration_days} дней")
        except Exception as e:
            print(f"[WARN]  Не удалось загрузить duration из csg_tasks_wbs.json: {e}")

        # Парсим целевую дату и вычисляем базовый срок в днях от текущей даты
        self.complete_date = datetime.strptime(self.complete_date_str, "%Y-%m-%d")
        self.project_start_date = datetime.now()  # Начало от текущей даты
        self.base_duration = (self.complete_date - self.project_start_date).days

        # Проверяем обязательные параметры
        if self.iterations is None or self.base_duration is None or self.base_budget is None:
            print("[ERROR]  ERROR: Missing required config in PIPELINE_CONFIG['monte_carlo'] "
                  "('.iterations') or PIPELINE_CONFIG['project'] "
                  "('.complete_date', '.base_budget')", file=sys.stderr)
            sys.exit(1)

        self.risks: Dict[str, RiskData] = {}
        self.risk_order: List[str] = []    # топологический порядок
        self.results: List[SimulationResult] = []

    # -------------------- 1. ЗАГРУЗКА ДАННЫХ --------------------

    def load_bayesian_network(self, network_file: str = None, data: dict = None) -> None:
        """Загрузка Байесовской сети с CPT из единого файла.

        data: сеть из состояния LangGraph (in-memory); если None — читается файл.
        """
        if data is None:
            data = load_json(network_file)
        parents_map = data.get('parents', {})

        for risk_id, risk_data in data['risks'].items():
            self.risks[risk_id] = RiskData(
                risk_id=risk_id,
                name=risk_data['name'],
                node_id=risk_data['node_id'],
                node_name=risk_data['node_name'],
                p50=risk_data['p50'],
                prob_min=risk_data.get('prob_min', 0.0),
                prob_max=risk_data.get('prob_max', 0.0),
                delay_min=risk_data.get('delay_min', 0.0),
                delay_max=risk_data.get('delay_max', 0.0),
                expected_delay=risk_data.get('expected_delay', 0.0),
                budget_min=risk_data.get('budget_min', 0.0),
                budget_max=risk_data.get('budget_max', 0.0),
                expected_budget=risk_data.get('expected_budget', 0.0),
                parents=parents_map.get(risk_id, []),
                cpt=risk_data.get('cpt', []),
            )

        print(f"Загружено {len(self.risks)} рисков")
        self._build_topological_order()

        cpt_count = sum(1 for r in self.risks.values() if r.cpt)
        print(f"[LOAD] Загружены CPT для {cpt_count} рисков")

    def _build_topological_order(self) -> None:
        """
        Построение топологического порядка для розыгрыша рисков.
        Сначала разыгрываются риски без родителей, затем их дети.
        """
        # Строим граф зависимостей
        in_degree = {risk_id: len(risk.parents) for risk_id, risk in self.risks.items()}
        children = defaultdict(list)

        for risk_id, risk in self.risks.items():
            for parent in risk.parents:
                children[parent['risk_id']].append(risk_id)

        # Топологическая сортировка (алгоритм Кана)
        queue = [risk_id for risk_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            risk_id = queue.pop(0)
            order.append(risk_id)
            for child_id in children[risk_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        self.risk_order = order
        print(f"Топологический порядок: {len(order)} рисков")

    # -------------------- 2. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ --------------------

    def _random_bernoulli(self, probability: float) -> bool:
        """Бернулли (монетка с заданной вероятностью)"""
        return random.random() < probability

    def _random_triangular(self, min_val: float, max_val: float, mode: float) -> float:
        """Треугольное распределение"""
        if min_val == max_val:
            return min_val
        return random.triangular(min_val, max_val, mode)

    def _random_normal(self, min_val: float, max_val: float, expected: float) -> float:
        """Нормальное распределение: mean=(min+max)/2, sigma=(max-min)/N"""
        if min_val == max_val:
            return min_val
        mean = (min_val + max_val) / 2
        sigma = (max_val - min_val) / self.sigma_div
        # Генерируем и обрезаем по границам, чтобы не выйти за пределы
        return max(min_val, min(max_val, random.gauss(mean, sigma)))

    # -------------------- 3. РАЗЫГРЫШ РИСКОВ В ОДНОМ ПРОГОНЕ --------------------

    def _run_single_iteration(self, iteration: int) -> SimulationResult:
        """
        Один прогон Монте-Карло (логика ModelRisk).

        Шаг 1: Определяем вероятность риска из CPT (с учётом родителей)
        Шаг 2: Разыгрываем состояние С ЭТОЙ ВЕРОЯТНОСТЬЮ (_random_bernoulli)
        Шаг 3: state × delta (если state=1 → полный ущерб, если state=0 → 0)
        """
        # ========== ШАГ 1: ОПРЕДЕЛЯЕМ ВЕРОЯТНОСТИ ==========
        states: Dict[str, int] = {}
        activated_risks: List[str] = []
        risk_probabilities: Dict[str, float] = {}
        scenario_probability = 1.0

        for risk_id in self.risk_order:
            risk = self.risks[risk_id]

            # Определяем вероятность (условную, с учётом родителей)
            if risk.parents:
                parent_states = {
                    parent['risk_id']: states.get(parent['risk_id'], 0)
                    for parent in risk.parents
                }
                prob = self._get_cpt_probability(risk.cpt, parent_states)
            else:
                prob = risk.p50

            risk_probabilities[risk_id] = prob

            # ========== ШАГ 2: РАЗЫГРЫВАЕМ СОСТОЯНИЕ С ЭТОЙ ВЕРОЯТНОСТЬЮ ==========
            if self._random_bernoulli(prob):
                states[risk_id] = 1
                activated_risks.append(risk_id)
            else:
                states[risk_id] = 0

            # Для оценки вероятности всего сценария (справочно)
            if states[risk_id] == 1:
                scenario_probability *= prob
            else:
                scenario_probability *= (1 - prob)

        # ========== ШАГ 3: ДОБАВЛЯЕМ УЩЕРБ ==========
        total_delay = 0.0
        budget_risk_impact = 0.0

        # Индивидуальные эффекты по каждому риску (для корректного Tornado)
        delay_by_risk: Dict[str, float] = {}
        budget_risk_by_risk: Dict[str, float] = {}

        for risk_id in self.risk_order:
            risk = self.risks[risk_id]
            state = states[risk_id]

            delay_delta = state * (self._random_normal(risk.delay_min, risk.delay_max, risk.expected_delay)
                                  if self.distribution_type == "normal"
                                  else self._random_triangular(risk.delay_min, risk.delay_max, risk.expected_delay))
            budget_delta = state * (self._random_normal(risk.budget_min, risk.budget_max, risk.expected_budget)
                                   if self.distribution_type == "normal"
                                   else self._random_triangular(risk.budget_min, risk.budget_max, risk.expected_budget))
            total_delay += delay_delta
            budget_risk_impact += budget_delta
            delay_by_risk[risk_id] = delay_delta
            budget_risk_by_risk[risk_id] = budget_delta

        # Расчет влияния копулы (связь времени и бюджета)
        cost_per_day = self.base_budget / self.project_duration_days if self.project_duration_days > 0 else 0.0
        budget_copula_impact = total_delay * cost_per_day * COUPULA_CORRELATION
        budget_copula_by_risk = {}
        if budget_copula_impact > 0:
            for risk_id, delay_delta in delay_by_risk.items():
                if risk_id in activated_risks and delay_delta > 0:
                    budget_copula_by_risk[risk_id] = budget_copula_impact * (delay_delta / total_delay)
            
        total_budget_impact = budget_risk_impact + budget_copula_impact

        return SimulationResult(
            iteration=iteration,
            states=states,
            activated_risks=activated_risks,
            total_delay=total_delay,
            delay_by_risk=delay_by_risk,
            budget_risk_impact=budget_risk_impact,
            budget_copula_impact=budget_copula_impact,
            budget_risk_impact_by_risk=budget_risk_by_risk,
            budget_copula_impact_by_risk=budget_copula_by_risk,
            total_budget_impact=total_budget_impact,
            final_duration=self.complete_date + timedelta(days=total_delay),
            final_budget=self.base_budget + total_budget_impact,
            scenario_probability=scenario_probability,
            risk_probabilities=risk_probabilities
        )

    def _get_cpt_probability(self, cpt: List[Dict], parent_states: Dict[str, int]) -> float:
        """Поиск вероятности в CPT по состояниям родителей"""
        for row in cpt:
            states = row.get('states', {})
            match = True
            for parent_id, state in parent_states.items():
                if states.get(parent_id) != state:
                    match = False
                    break
            if match:
                return row.get('probability', 0.0)

        # Если не найдено — возвращаем 0.0 (сценарий невозможен)
        return 0.0

    # -------------------- 4. ЗАПУСК СИМУЛЯЦИИ --------------------

    def run_simulation(self) -> None:
        """Запуск полной симуляции"""
        # Принудительная настройка line buffering ДО ЛЮБОГО вывода (Windows fix)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=1)

        print(f"\n[STEP] Запуск Монте-Карло симуляции...")
        print(f"- Итераций: {self.iterations}")
        print(f"- Базовый срок: {self.base_duration} дней")
        print(f"- Целевая дата завершения: {self.complete_date.strftime('%d.%m.%Y')}")
        print(f"- Базовый бюджет: {self.base_budget:,.0f} руб")
        print(f"- Распределение дельт: {self.distribution_type} (sigma_div={self.sigma_div})\n")

        random.seed(self.random_seed)

        self.results = []
        self._sum_delay = 0.0
        self._sum_budget_impact = 0.0
        self._sum_budget_risk = 0.0
        self._sum_budget_copula = 0.0
        self._sum_final_budget = 0.0
        self._sum_scenario_prob = 0.0

        # Прогресс-бар (обновляется каждые 10000 итераций)
        for i in range(self.iterations):
            result = self._run_single_iteration(i + 1)
            self.results.append(result)

            # Аккумуляторы (O(1) вместо O(N) в конце)
            self._sum_delay += result.total_delay
            self._sum_budget_impact += result.total_budget_impact
            self._sum_budget_risk += result.budget_risk_impact
            self._sum_budget_copula += result.budget_copula_impact
            self._sum_final_budget += result.final_budget
            self._sum_scenario_prob += result.scenario_probability

            # Выводим прогресс каждые 10000 итераций
            if (i + 1) % 10000 == 0 or (i + 1) == self.iterations:
                pct = int((i + 1) / self.iterations * 100)
                print(f"Прогресс: {pct}% ({i + 1}/{self.iterations})")

        n = len(self.results)
        print(f"Симуляция завершена!")
        print(f"[RESULTS]")
        print(f"- средняя задержка (прирост): {self._sum_delay / n:.1f} дней")
        avg_finish_date = self.complete_date + timedelta(days=self._sum_delay / n)
        print(f"- средняя дата завершения: {avg_finish_date.strftime('%d.%m.%Y')}")
        print(f"- средний перерасход (прирост): {self._sum_budget_impact / n:,.0f} руб")
        print(f"- средний итоговый бюджет: {self._sum_final_budget / n:,.0f} руб")
        print(f"- средняя вероятность сценария: {self._sum_scenario_prob / n:.6f}")

    # -------------------- 5. СТАТИСТИКА --------------------

    def calculate_statistics(self, bootstrap_n: int = 500, bootstrap_ci: float = 0.95) -> Dict:
        """Расчёт статистики (дельты в днях и рублях) + bootstrap-доверительные интервалы."""
        n = len(self.results)
        if n == 0:
            return {}

        # Общие выборки (без предсортировки — np.percentile сортирует сам)
        delay_samples = [r.total_delay for r in self.results]
        budget_samples = [r.total_budget_impact for r in self.results]
        budget_risk_samples = [r.budget_risk_impact for r in self.results]
        budget_copula_samples = [r.budget_copula_impact for r in self.results]
        final_durations = [r.final_duration for r in self.results]

        print(f"\n[STEP] Вычисление статистики + bootstrap-CI (B={bootstrap_n})...")

        def _percentile(data: list, p: float) -> float:
            """Общий расчёт персентиля через интерполяцию."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f if k == f else f + 1
            if f == c:
                return float(data[f])
            return float(data[f]) * (c - k) + float(data[c]) * (k - f)

        def _percentile_dates(data: list, p: float) -> datetime:
            """Процентиль для дат."""
            if not data:
                return self.complete_date
            k = (len(data) - 1) * p
            f = int(k)
            c = f if k == f else f + 1
            if f == c:
                return data[f]
            delta_days = (data[c] - data[f]).days * (k - f)
            return data[f] + timedelta(days=delta_days)

        # Точечные оценки
        delay_p50 = _percentile(delay_samples, 0.50)
        delay_p80 = _percentile(delay_samples, 0.80)
        delay_p90 = _percentile(delay_samples, 0.90)
        delay_p95 = _percentile(delay_samples, 0.95)
        delay_mean = sum(delay_samples) / n

        budget_p50 = _percentile(budget_samples, 0.50)
        budget_p80 = _percentile(budget_samples, 0.80)
        budget_p90 = _percentile(budget_samples, 0.90)
        budget_p95 = _percentile(budget_samples, 0.95)
        budget_mean = sum(budget_samples) / n

        budget_risk_p50 = _percentile(budget_risk_samples, 0.50)
        budget_risk_p95 = _percentile(budget_risk_samples, 0.95)
        budget_risk_mean = sum(budget_risk_samples) / n

        budget_copula_p50 = _percentile(budget_copula_samples, 0.50)
        budget_copula_p95 = _percentile(budget_copula_samples, 0.95)
        budget_copula_mean = sum(budget_copula_samples) / n

        # ========== BOOTSTRAP: доверительные интервалы для персцентилей ==========
        percentiles_to_compute = [0.50, 0.80, 0.90, 0.95]
        alpha = 1.0 - bootstrap_ci  # 0.05 для 95% CI
        ci_lo_pct = alpha / 2 * 100       # 2.5
        ci_hi_pct = (1.0 - alpha / 2) * 100  # 97.5

        def _bootstrap_ci(data: np.ndarray, pct_list: List[float],
                          n_boot: int, ci_lo: float, ci_hi: float) -> Dict:
            """
            Векторизованный bootstrap-CI через percentiles (numpy).
            Генерирует все n_boot выборок сразу и находит квантили через argpartition.
            """
            N = len(data)
            if N < 10:
                sorted_data = np.sort(data)
                result = {}
                for p in pct_list:
                    val = _percentile(list(sorted_data), p)
                    key = f"p{int(p * 100)}"
                    result[key] = {"point": val, "ci_lo": None, "ci_hi": None}
                return result

            rng = np.random.RandomState(123)
            boot_keys = [f"p{int(p * 100)}" for p in pct_list]
            sorted_pct = np.sort(pct_list)

            # Векторизованный bootstrap:
            # 1. Генерируем (n_boot, N) индексы для ресемплинга
            indices = rng.randint(0, N, size=(n_boot, N))
            samples = data[indices]  # (n_boot, N)

            # 2. Для каждой bootstrap-выборки находим нужные квантили
            #    Используем numpy.percentile (векторизовано по axis=1)
            result = {}
            for i, p in enumerate(pct_list):
                key = boot_keys[i]
                # percentiles на axis=1 → массив длины n_boot
                boot_percentiles = np.percentile(samples, p * 100, axis=1)
                point = float(np.median(boot_percentiles))
                ci_lo_val = float(np.percentile(boot_percentiles, ci_lo))
                ci_hi_val = float(np.percentile(boot_percentiles, ci_hi))
                result[key] = {
                    "point": point,
                    "ci_lo": ci_lo_val,
                    "ci_hi": ci_hi_val,
                }
            return result

        # === Вычисление bootstrap-CI с прогрессом ===
        def _run_bootstrap(data: np.ndarray, label: str, percentiles: list, decimals: int = 1) -> Dict:
            result = _bootstrap_ci(data, percentiles, bootstrap_n, ci_lo_pct, ci_hi_pct)
            for key in result:
                p50 = result[key]
                fmt = f"{{:.{decimals}f}}" if decimals > 0 else f"{{:,}}"
                print(f"   [OK] {label}: P50={fmt.format(p50['point'])} "
                      f"(95% CI: [{fmt.format(p50['ci_lo'])} .. {fmt.format(p50['ci_hi'])}])")
            return result

        data_delay = np.array(delay_samples)
        data_budget = np.array(budget_samples)

        print(f"   Bootstrap-выборки для задержки (B={bootstrap_n})...")
        delay_ci = _run_bootstrap(data_delay, "Задержка", percentiles_to_compute, 1)

        print(f"   Bootstrap-выборки для бюджета (B={bootstrap_n})...")
        budget_ci = _run_bootstrap(data_budget, "Бюджет", percentiles_to_compute, 0)

        print(f"   Bootstrap-выборки для бюджета-риски (B={bootstrap_n})...")
        budget_risk_ci = _run_bootstrap(np.array(budget_risk_samples), "Бюджет-риски", [0.50, 0.95], 0)

        print(f"   Bootstrap-выборки для бюджета-копула (B={bootstrap_n})...")
        budget_copula_ci = _run_bootstrap(np.array(budget_copula_samples), "Бюджет-копула", [0.50, 0.95], 0)

        print(f"   [OK] Статистика вычислена.")

        # Конвертируем даты в ISO строки для JSON
        date_p50 = _percentile_dates(final_durations, 0.50)
        date_p80 = _percentile_dates(final_durations, 0.80)
        date_p90 = _percentile_dates(final_durations, 0.90)
        date_p95 = _percentile_dates(final_durations, 0.95)
        date_min = min(final_durations) if final_durations else self.complete_date
        date_max = max(final_durations) if final_durations else self.complete_date

        stats = {
            "delay": {
                "p50": delay_ci["p50"]["point"],
                "p50_ci": {"lo": delay_ci["p50"]["ci_lo"], "hi": delay_ci["p50"]["ci_hi"]},
                "p80": delay_ci["p80"]["point"],
                "p80_ci": {"lo": delay_ci["p80"]["ci_lo"], "hi": delay_ci["p80"]["ci_hi"]},
                "p90": delay_ci["p90"]["point"],
                "p90_ci": {"lo": delay_ci["p90"]["ci_lo"], "hi": delay_ci["p90"]["ci_hi"]},
                "p95": delay_ci["p95"]["point"],
                "p95_ci": {"lo": delay_ci["p95"]["ci_lo"], "hi": delay_ci["p95"]["ci_hi"]},
                "mean": delay_mean,
                "min": delay_samples[0] if delay_samples else 0,
                "max": delay_samples[-1] if delay_samples else 0,
                # Абсолютные даты для гистограммы (ISO строки)
                "date_p50": date_p50.isoformat(),
                "date_p80": date_p80.isoformat(),
                "date_p90": date_p90.isoformat(),
                "date_p95": date_p95.isoformat(),
                "date_mean": date_p50.isoformat(),
                "date_min": date_min.isoformat(),
                "date_max": date_max.isoformat(),
            },
            "budget": {
                "p50": budget_ci["p50"]["point"],
                "p50_ci": {"lo": budget_ci["p50"]["ci_lo"], "hi": budget_ci["p50"]["ci_hi"]},
                "p80": budget_ci["p80"]["point"],
                "p80_ci": {"lo": budget_ci["p80"]["ci_lo"], "hi": budget_ci["p80"]["ci_hi"]},
                "p90": budget_ci["p90"]["point"],
                "p90_ci": {"lo": budget_ci["p90"]["ci_lo"], "hi": budget_ci["p90"]["ci_hi"]},
                "p95": budget_ci["p95"]["point"],
                "p95_ci": {"lo": budget_ci["p95"]["ci_lo"], "hi": budget_ci["p95"]["ci_hi"]},
                "mean": budget_mean,
                "min": budget_samples[0] if budget_samples else 0,
                "max": budget_samples[-1] if budget_samples else 0
            },
            "budget_risk": {
                "p50": budget_risk_ci["p50"]["point"],
                "p50_ci": {"lo": budget_risk_ci["p50"]["ci_lo"], "hi": budget_risk_ci["p50"]["ci_hi"]},
                "p95": budget_risk_ci["p95"]["point"],
                "p95_ci": {"lo": budget_risk_ci["p95"]["ci_lo"], "hi": budget_risk_ci["p95"]["ci_hi"]},
                "mean": budget_risk_mean,
                "min": budget_risk_samples[0] if budget_risk_samples else 0,
                "max": budget_risk_samples[-1] if budget_risk_samples else 0
            },
            "budget_copula": {
                "p50": budget_copula_ci["p50"]["point"],
                "p50_ci": {"lo": budget_copula_ci["p50"]["ci_lo"], "hi": budget_copula_ci["p50"]["ci_hi"]},
                "p95": budget_copula_ci["p95"]["point"],
                "p95_ci": {"lo": budget_copula_ci["p95"]["ci_lo"], "hi": budget_copula_ci["p95"]["ci_hi"]},
                "mean": budget_copula_mean,
                "min": budget_copula_samples[0] if budget_copula_samples else 0,
                "max": budget_copula_samples[-1] if budget_copula_samples else 0
            },
            "scenario_probabilities": {
                "mean": self._sum_scenario_prob / n,
                "min": min(r.scenario_probability for r in self.results),
                "max": max(r.scenario_probability for r in self.results)
            },
            "bootstrap": {
                "n": bootstrap_n,
                "ci_level": bootstrap_ci,
                "ci_percentile_lo": ci_lo_pct,
                "ci_percentile_hi": ci_hi_pct,
            },
            "summary": {
                "avg_activated_risks": sum(len(r.activated_risks) for r in self.results) / n,
                "max_activated_risks": max(len(r.activated_risks) for r in self.results),
                "min_activated_risks": min(len(r.activated_risks) for r in self.results),
                "total_iterations": n
            },
            "sum_delay": self._sum_delay,
            "sum_budget_impact": self._sum_budget_impact,
            "sum_budget_risk": self._sum_budget_risk,
            "sum_budget_copula": self._sum_budget_copula,
            "sum_final_budget": self._sum_final_budget,
            "sum_scenario_prob": self._sum_scenario_prob,
        }

        return stats

    # -------------------- 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ --------------------

    def _ensure_stats_cached(self) -> Dict:
        """Кэширует результат calculate_statistics — вычисляется только один раз."""
        if not hasattr(self, '_stats_cache'):
            self._stats_cache = self.calculate_statistics()
        return self._stats_cache

    def _serialize_result(self, r: SimulationResult) -> dict:
        """Сериализация результата одного прогона."""
        return {
            "iteration": r.iteration,
            "states": r.states,
            "activated_risks": r.activated_risks,
            "total_delay": round(r.total_delay, 2),
            "delay_impact_by_risk": {k: round(v, 2) for k, v in r.delay_by_risk.items()},
            "budget_risk_impact": round(r.budget_risk_impact, 2),
            "budget_copula_impact": round(r.budget_copula_impact, 2),
            "budget_risk_impact_by_risk": {k: round(v, 2) for k, v in r.budget_risk_impact_by_risk.items()},
            "budget_copula_impact_by_risk": {k: round(v, 2) for k, v in r.budget_copula_impact_by_risk.items()},
            "total_budget_impact": round(r.total_budget_impact, 2),
            "final_duration": r.final_duration.isoformat(),
            "final_budget": round(r.final_budget, 2),
            "scenario_probability": r.scenario_probability,
            "risk_probabilities": {k: round(v, 6) for k, v in r.risk_probabilities.items()},
        }

    def save_results(self, output_file: str) -> None:
        """Сохранение результатов симуляции"""
        stats = self._ensure_stats_cached()

        # Сохраняем АБСОЛЮТНЫЕ значения (даты и бюджет) — для визуализации
        final_durations_arr = np.array([r.final_duration.isoformat() for r in self.results])
        final_budgets_arr = np.array([r.final_budget for r in self.results])

        # Подготовка данных для сохранения
        data = {
            "config": {
                "iterations": self.iterations,
                "base_duration": self.base_duration,
                "complete_date": self.complete_date_str,
                "base_budget": self.base_budget,
                "random_seed": self.random_seed
            },
            "statistics": stats,
            "summary": {
                "total_activated_risks": sum(len(r.activated_risks) for r in self.results),
                "avg_activated_risks": sum(len(r.activated_risks) for r in self.results) / len(self.results),
                "max_activated_risks": max(len(r.activated_risks) for r in self.results)
            }
        }

        save_json(output_file, data)

        # Сохраняем АБСОЛЮТНЫЕ значения (даты и бюджет) — для визуализации
        final_delay_npy = paths.final_delay_samples_npy
        final_budget_npy = paths.final_budget_samples_npy
        np.save(paths.to_str(final_delay_npy), final_durations_arr)
        np.save(paths.to_str(final_budget_npy), final_budgets_arr)
        print(f"[SAVE] {final_delay_npy.name} (даты завершения, {len(final_durations_arr)} значений)")
        print(f"[SAVE] {final_budget_npy.name} (абсолютный бюджет, {len(final_budgets_arr)} значений)")

        # Сохраняем детали каждой итерации (сценарии с вероятностями)
        scenarios = [self._serialize_result(r) for r in self.results]
        scenarios_file = paths.iteration_scenarios_json
        save_json(paths.to_str(scenarios_file), scenarios)
        print(f"[SAVE] {scenarios_file.name} ({len(scenarios)} сценариев)")

        print(f"[SAVE] Результаты сохранены: {output_file}")
        return data


# ==================== ТОЧКА ВХОДА ====================

def main(bayes: dict = None) -> dict:
    """Запуск Монте-Карло симуляции.

    bayes: Байесовская сеть из шага 7 (in-memory из состояния LangGraph).
    Если None — читается с диска (standalone-режим).
    """
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    STEP_DIR_7 = paths.to_str(paths.step_dirs[7])
    STEP_DIR_8 = paths.to_str(paths.step_dirs[8])
    BN_INPUT = paths.to_str(paths.bayesian_network_json)
    SIM_OUTPUT = os.path.join(STEP_DIR_8, "simulation_results.json")

    # Проверяем входные файлы (только если сеть не передана из состояния)
    if bayes is None and not os.path.isfile(BN_INPUT):
        print(f"\n[ERROR]  ERROR: Входной файл не найден: {BN_INPUT}", file=sys.stderr)
        print("   Сначала выполните шаг 7: 7_bayesian_network_builder.py", file=sys.stderr)
        sys.exit(1)

    # Создаём симулятор
    sim = MonteCarloSimulator()
    sim.load_bayesian_network(BN_INPUT, data=bayes)
    sim.run_simulation()
    sim_doc = sim.save_results(SIM_OUTPUT)

    print(f"\n[OK]  Шаг 'Монте-Карло симуляция' выполнен успешно.")

    # Возвращаем результаты симуляции для состояния LangGraph (JSON уже сохранён)
    return sim_doc


if __name__ == "__main__":
    main()
