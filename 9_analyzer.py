"""
9_analyzer.py — Модуль 9
Визуализация и анализ результатов Monte Carlo симуляции.

Включает:
  - Дашборд: гистограмма + KDE с линиями квантилей
  - Tornado-графики с полными названиями и ID рисков
  - Тепловая карта плотности сценариев с контурами
  - Анализ чувствительности: тепловая карта корреляций + Radar Chart
  - Экспорт сводных статистик в JSON для LLM

Результаты сохраняются в data/data_processed/9_data/
"""
import os
import re
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from utils.general.paths import paths
from utils.general.json_io import load_json, save_json


# ==================== НАСТРОЙКИ ГРАФИКОВ ====================
HIST_CONFIG = {
    "bins": 40, "alpha": 0.7, "linewidth": 0.5,
    "color": "#2196F3", "kde_color": "#E91E63", "kde_linewidth": 3, "kde_npoints": 200,
    "base_color": "black", "base_linewidth": 2.5,
    "p50_color": "green", "p80_color": "orange", "p90_color": "purple", "p95_color": "brown",
    "p_linewidth": 2,
}

TORNADO_CONFIG = {
    "bar_color": "#FF5722", "edge": "white", "height": 0.7,
}

HEATMAP_CONFIG = {
    "cmap": "Blues_r", "contour_color": "red",
}

CORRELATION_CONFIG = {
    "cmap": "coolwarm", "text_fontsize": 18, "text_fontweight": "bold",
}

RADAR_CONFIG = {
    "area_color": "#2196F3", "area_alpha": 0.6,
    "line_color": "#E91E63", "marker_color": "red",
}

PERCENTILES = [("P50", 0.50, "p50_color"), ("P80", 0.80, "p80_color"),
               ("P90", 0.90, "p90_color"), ("P95", 0.95, "p95_color")]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def _load_scenarios(iteration_scenarios_path):
    """Загрузка итерационных сценариев из симуляции."""
    if not os.path.isfile(iteration_scenarios_path):
        print(f"   [WARN]  Файл сценариев не найден: {iteration_scenarios_path}")
        return None
    scenarios_data = load_json(iteration_scenarios_path)
    return scenarios_data if isinstance(scenarios_data, list) else scenarios_data.get('scenarios', [])


def _npx(d, rid, key, default=0.0):
    """Безопасное чтение вложенного словаря вкладов по риску."""
    return (d.get(key) or {}).get(rid, default)


def _clean_risk_name(name: str) -> str:
    """Убирает '; Причина: ...' из имени риска."""
    if not name:
        return name
    return re.sub(r';\s*Причина\s*:\s*.*$', '', name, flags=re.IGNORECASE).strip()


def _risk_display_name(name: str, rid: str, width: int = 40) -> str:
    """Форматирует имя риска для подписи: 'RISK-xx: <название>'."""
    wrapped = textwrap.fill(_clean_risk_name(name), width=width)
    return f"{rid}: {wrapped}"


def _load_bootstrap_ci(results_file: str) -> Dict:
    """Загружает bootstrap-CI из simulation_results.json."""
    try:
        data = load_json(results_file)
        stats = data.get('statistics', {})
        return {
            'bootstrap': stats.get('bootstrap', {}),
            'delay_ci': {k: stats.get('delay', {}).get(f'{k}_ci') for k in ['p50', 'p80', 'p90', 'p95']},
            'budget_ci': {k: stats.get('budget', {}).get(f'{k}_ci') for k in ['p50', 'p80', 'p90', 'p95']},
        }
    except (FileNotFoundError, ValueError, KeyError):
        return {'bootstrap': {}, 'delay_ci': {}, 'budget_ci': {}}


def _draw_stacked_tornado(ax, labels, seg1_values, seg2_values, total_values,
                          seg1_label=None, seg2_label=None, xlabel="",
                          total_suffix="", show_inner_labels=True):
    """Отрисовка stacked горизонтального графика на заданной оси."""
    n = len(labels)
    seg1 = [max(0.0, v) for v in seg1_values]
    seg2 = [max(0.0, v) for v in seg2_values]
    total = [a + b for a, b in zip(seg1, seg2)] if total_values is None else total_values

    ax.barh(range(n), seg1, color='#3498db', edgecolor='white', height=0.7, label=seg1_label)
    ax.barh(range(n), seg2, left=seg1, color='#e67e22', edgecolor='white', height=0.7, label=seg2_label)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    if seg2_label:
        ax.legend(loc='lower right', fontsize=11)

    max_total = max(total) if total else 1
    ax.set_xlim(0, max_total * 1.25)

    for i in range(n):
        if show_inner_labels:
            if seg1[i] > 0.01 and max_total:
                ax.text(seg1[i] * 0.5, i, f'+{seg1[i]:.1f}',
                        va='center', ha='center', fontsize=14, color='white', fontweight='bold')
            if seg2[i] > 0.01 and max_total:
                ax.text(seg1[i] + seg2[i] * 0.5, i, f'+{seg2[i]:.1f}',
                        va='center', ha='center', fontsize=14, color='white', fontweight='bold')
        if total[i] > 0.01:
            suffix = f' {total_suffix}' if total_suffix else ''
            ax.text(total[i] + max_total * 0.025, i, f'+{total[i]:.1f}{suffix}',
                    va='center', ha='left', fontsize=14, color='#333', fontweight='bold')


def _compute_risk_impacts(scenarios, bayesian_network):
    """
    Расчёт фактического влияния каждого риска по данным симуляции.
    Возвращает dict risk_id -> {delay, budget_direct, budget_copula, name}.
    """
    risks = bayesian_network.get("risks", {})
    n_total = len(scenarios)
    if n_total == 0:
        return {}

    acc = {rid: {"sum_delay": 0.0, "sum_bd": 0.0, "sum_bc": 0.0,
                 "name": risks.get(rid, {}).get("name", rid)}
           for rid in risks}

    for s in scenarios:
        for rid in acc:
            acc[rid]["sum_delay"] += _npx(s, rid, "delay_impact_by_risk")
            acc[rid]["sum_bd"] += _npx(s, rid, "budget_risk_impact_by_risk")
            acc[rid]["sum_bc"] += _npx(s, rid, "budget_copula_impact_by_risk")

    return {rid: {**a, "budget_total": (a["sum_bd"] + a["sum_bc"]) / n_total,
                  "delay": a["sum_delay"] / n_total,
                  "budget_direct": a["sum_bd"] / n_total,
                  "budget_copula": a["sum_bc"] / n_total}
            for rid, a in acc.items()}


# ==================== КЛАСС АНАЛИЗАТОРА ====================

class Analyzer:
    """Анализатор результатов Monte Carlo симуляции."""

    def __init__(self, bayesian_network: Dict, base_duration: float,
                 base_budget: float, complete_date: Optional[datetime] = None,
                 scenarios: List = None):
        self.network = bayesian_network
        self.base_duration = base_duration
        self.base_budget = base_budget
        self.complete_date = complete_date
        self._scenarios = scenarios
        self._percentiles = {}
        self._risk_impacts = {}  # кэш вычислений risk impacts

    def _get_scenarios(self, iteration_scenarios_path: str) -> Optional[List]:
        """Загружает сценарии из кэша или с диска."""
        if self._scenarios is not None:
            return self._scenarios
        return _load_scenarios(iteration_scenarios_path)

    def _get_percentiles(self, data, probs):
        """Кэширует вычисление percentiles."""
        key = id(data)
        if key not in self._percentiles:
            self._percentiles[key] = np.percentile(data, probs)
        return self._percentiles[key]

    # ==================== 1. ДАШБОРД ====================

    def _load_simulation_inputs(self, final_delay_path: str, final_budget_path: str):
        """Загружает данные симуляции. Возвращает (durations_days, budgets, start_date)."""
        if not os.path.isfile(final_delay_path):
            print(f"[ERROR] {final_delay_path}")
            return None, None, None
        date_strings = np.load(final_delay_path, allow_pickle=True)
        durations = np.array([datetime.fromisoformat(str(d)) for d in date_strings])
        print(f"[LOAD] {final_delay_path}")

        if not os.path.isfile(final_budget_path):
            print(f"[ERROR] {final_budget_path}")
            return None, None, None
        budgets = np.load(final_budget_path)
        print(f"[LOAD] {final_budget_path}")

        start_date = self.complete_date - timedelta(days=self.base_duration)
        durations_days = np.array([(d - start_date).days for d in durations])
        return durations_days, budgets, start_date

    def _draw_histogram_kde(self, ax, values, value_label, digits):
        """Рисует гистограмму + KDE с линиями базы и квантилей."""
        ax.hist(values, bins=HIST_CONFIG["bins"], color=HIST_CONFIG["color"],
                edgecolor="white", alpha=HIST_CONFIG["alpha"], density=True,
                linewidth=HIST_CONFIG["linewidth"])

        degenerate = values.std() < 1e-6
        if not degenerate and len(values) > 1:
            kde = gaussian_kde(values)
            x_kde = np.linspace(max(0, values.min()), values.max(),
                                HIST_CONFIG["kde_npoints"])
            ax.plot(x_kde, kde(x_kde), color=HIST_CONFIG["kde_color"],
                    linewidth=HIST_CONFIG["kde_linewidth"], label="KDE")
        elif len(values) > 0:
            ax.axvline(values[0], color=HIST_CONFIG["kde_color"], linestyle="--",
                       linewidth=HIST_CONFIG["kde_linewidth"],
                       label=f"Константа: {values[0]:.{digits}f} {value_label}")

        ax.axvline(0, color=HIST_CONFIG["base_color"], linestyle="-",
                   linewidth=HIST_CONFIG["base_linewidth"], label=f"База: 0 {value_label}")

        for name, prob, color_key in PERCENTILES:
            val = np.percentile(values, prob * 100)
            color = HIST_CONFIG[color_key]
            ax.axvline(val, color=color, linestyle="--", linewidth=HIST_CONFIG["p_linewidth"],
                       label=f"{name}: +{val:.{digits}f} {value_label}")

        ax.set_xlim(0, values.max() * 1.05)

    def plot_dashboard(self, final_delay_path: str, final_budget_path: str) -> Optional[Dict]:
        """Дашборд: гистограмма + KDE для сроков и бюджета."""
        print("\n[SUMMARY]  Построение дашборда...")

        durations_days, budgets, start_date = self._load_simulation_inputs(
            final_delay_path, final_budget_path)
        if durations_days is None or budgets is None:
            return None

        n = len(durations_days)
        delay_all = durations_days - self.base_duration
        budget_increase = budgets - self.base_budget

        pcts = [50, 80, 90, 95]
        delay_pcts = dict(zip([f"p50_d","p80_d","p90_d","p95_d"],
                              np.percentile(delay_all, pcts)))
        budget_pcts = dict(zip([f"p50_b","p80_b","p90_b","p95_b"],
                               np.percentile(budget_increase, pcts)))

        # ==================== ФИГУРА 1: СРОК ====================
        fig1 = plt.figure(figsize=(14, 6.5))
        fig1.suptitle(f"Монте-Карло симуляция ({n:,} итераций): влияние на срок реализации проекта",
                      fontsize=18, fontweight="bold")
        gs1 = GridSpec(1, 1, figure=fig1, left=0.08, right=0.92, bottom=0.10, top=0.85)

        delay_positive = delay_all[delay_all > 0]
        if len(delay_positive) == 0:
            delay_positive = delay_all
        ax1 = fig1.add_subplot(gs1[0, :])
        self._draw_histogram_kde(ax1, delay_positive, "дн", 0)
        ax1.set_xlabel("Прирост срока (дн)", fontsize=13)
        ax1.set_ylabel("Плотность вероятности", fontsize=13)
        ax1.set_title("Гистограмма распределения прироста срока", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, alpha=0.3)
        fig1.savefig(str(paths.dashboard_delay_png), dpi=150, bbox_inches="tight")
        print("[SAVE] 01_dashboard_delay.png")
        plt.close(fig1)

        # ==================== ФИГУРА 2: БЮДЖЕТ ====================
        fig2 = plt.figure(figsize=(14, 6.5))
        fig2.suptitle(f"Монте-Карло симуляция ({n:,} итераций): влияние на бюджет проекта",
                      fontsize=18, fontweight="bold")
        gs2 = GridSpec(1, 1, figure=fig2, left=0.08, right=0.92, bottom=0.10, top=0.85)

        budget_increase_millions = budget_increase / 1_000_000
        budget_positive = budget_increase_millions[budget_increase_millions > 0]
        if len(budget_positive) == 0:
            budget_positive = budget_increase_millions
        ax2 = fig2.add_subplot(gs2[0, :])
        self._draw_histogram_kde(ax2, budget_positive, "млн руб", 1)
        ax2.set_xlabel("Прирост бюджета (млн руб)", fontsize=13)
        ax2.set_ylabel("Плотность вероятности", fontsize=13)
        ax2.set_title("Гистограмма распределения прироста бюджета", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, alpha=0.3)
        fig2.savefig(str(paths.dashboard_budget_png), dpi=150, bbox_inches="tight")
        print("[SAVE] 01_dashboard_budget.png")
        plt.close(fig2)

        return {
            "durations": durations_days, "budgets": budgets, "n": n,
            "start_date": start_date, "complete_date": self.complete_date,
            **delay_pcts, **budget_pcts,
            "mean_d": float(delay_all.mean()), "mean_b": float(budget_increase.mean()),
            "std_d": float(delay_all.std()), "std_b": float(budget_increase.std()),
            "min_d": float(delay_all.min()), "max_d": float(delay_all.max()),
            "min_b": float(budget_increase.min()), "max_b": float(budget_increase.max()),
        }

    def _get_risk_impacts(self, scenarios):
        """Кэширует вычисление risk impacts — вызывается один раз, используется для delay и budget."""
        key = id(scenarios)
        if key not in self._risk_impacts:
            self._risk_impacts[key] = _compute_risk_impacts(scenarios, self.network)
        return self._risk_impacts[key]

    # ==================== 2. ТОРНАДО-ГРАФИКИ ====================

    def plot_stacked_delay_tornado(self, iteration_scenarios_path: str) -> list:
        """Stacked Tornado по сроку."""
        print("\n[SUMMARY]  Построение tornado-графика (задержки)...")
        scenarios = self._get_scenarios(iteration_scenarios_path)
        if not scenarios:
            return []

        impacts = self._get_risk_impacts(scenarios)
        items = [(rid, imp["name"], imp["delay"]) for rid, imp in impacts.items() if imp["delay"] > 0]
        items.sort(key=lambda x: (x[2], x[0]), reverse=True)

        names = [_risk_display_name(name, rid) for rid, name, _ in items]
        values = [i[2] for i in items]

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.suptitle("Влияние рисков на срок (данные симуляции)", fontsize=18, fontweight="bold", y=0.95)
        _draw_stacked_tornado(ax, names, values, [0.0]*len(values), values,
                              xlabel="Средний прирост срока (дни)", total_suffix="дн",
                              show_inner_labels=False)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(paths.tornado_delay_png), dpi=150, bbox_inches="tight")
        print(f"[SAVE] {paths.tornado_delay_png}")
        plt.close(fig)
        return items

    def plot_stacked_budget_tornado(self, iteration_scenarios_path: str) -> list:
        """Stacked Tornado по бюджету (прямой + копула эффект)."""
        print("\n[SUMMARY]  Построение tornado-графика (бюджет)...")
        scenarios = self._get_scenarios(iteration_scenarios_path)
        if not scenarios:
            return []

        impacts = self._get_risk_impacts(scenarios)
        items = [(rid, imp["name"], imp["budget_direct"], imp["budget_copula"], imp["budget_total"])
                 for rid, imp in impacts.items() if imp["budget_total"] > 0]
        items.sort(key=lambda x: (x[4], x[0]), reverse=True)

        names = [_risk_display_name(name, rid) for rid, name, *_ in items]
        direct_m = [i[2] / 1_000_000 for i in items]
        copula_m = [i[3] / 1_000_000 for i in items]
        total_m = [i[4] / 1_000_000 for i in items]

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.suptitle("Влияние рисков на бюджет (данные симуляции)", fontsize=18, fontweight="bold", y=0.95)
        _draw_stacked_tornado(ax, names, direct_m, copula_m, total_m,
                              seg1_label="Прямой эффект", seg2_label="Косвенный эффект",
                              xlabel="Прирост бюджета (млн руб)", total_suffix="млн руб")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(paths.tornado_budget_png), dpi=150, bbox_inches="tight")
        print(f"[SAVE] {paths.tornado_budget_png}")
        plt.close(fig)

        print(f"   ТОП-5 рисков по общему бюджетному влиянию:")
        for i, (rid, name, d, c, t) in enumerate(items[:5]):
            print(f"      {i+1}. {rid}: Прямой={d/1e6:.1f} + Копула={c/1e6:.1f} = {t/1e6:.1f} млн руб")

        return [(rid, name, t) for rid, name, _bd, _bc, t in items]

    # ==================== 3. ЭКСПОРТ СТАТИСТИКИ ДЛЯ LLM ====================

    def export_summary_statistics(self, stats: Dict, risk_delay: list, risk_budget: list) -> Dict:
        """Экспорт всех сводных статистик в JSON для LLM."""
        print("\n[SUMMARY]  Экспорт сводных статистик...")

        bootstrap_info = _load_bootstrap_ci(str(paths.simulation_results_json))
        bootstrap_meta = bootstrap_info.get('bootstrap', {})
        n_boot = bootstrap_meta.get('n', '')
        ci_level = bootstrap_meta.get('ci_level', 0)
        ci_pct_label = f"{ci_level * 100:.0f}%" if ci_level else "95%"

        base_budget = stats.get("base_budget", 0)
        delay_ci = bootstrap_info.get('delay_ci', {})
        budget_ci = bootstrap_info.get('budget_ci', {})

        def _adjust_budget_ci(ci_dict: Optional[Dict]) -> Optional[Dict]:
            """CI для дельты → абсолютные значения: добавляем base_budget."""
            if not ci_dict or ci_dict.get('lo') is None or ci_dict.get('hi') is None:
                return None
            return {"lo": ci_dict["lo"] + base_budget, "hi": ci_dict["hi"] + base_budget}

        summary = {
            "project": {"base_duration_days": stats["base_duration"],
                        "base_budget_rub": stats["base_budget"],
                        "iterations": stats["n"]},
            "bootstrap": {"iterations": n_boot, "ci_level": ci_pct_label},
            "delay": {
                "p50_days": stats["p50_d"], "p50_days_ci": delay_ci.get('p50'),
                "p80_days": stats["p80_d"], "p80_days_ci": delay_ci.get('p80'),
                "p90_days": stats["p90_d"], "p90_days_ci": delay_ci.get('p90'),
                "p95_days": stats["p95_d"], "p95_days_ci": delay_ci.get('p95'),
                "mean_days": stats["mean_d"], "std_days": stats["std_d"],
                "min_days": stats["min_d"], "max_days": stats["max_d"]
            },
            "budget": {
                "p50_rub": stats["p50_b"], "p50_rub_ci": _adjust_budget_ci(budget_ci.get('p50')),
                "p80_rub": stats["p80_b"], "p80_rub_ci": _adjust_budget_ci(budget_ci.get('p80')),
                "p90_rub": stats["p90_b"], "p90_rub_ci": _adjust_budget_ci(budget_ci.get('p90')),
                "p95_rub": stats["p95_b"], "p95_rub_ci": _adjust_budget_ci(budget_ci.get('p95')),
                "mean_rub": stats["mean_b"], "std_rub": stats["std_b"],
                "min_rub": stats["min_b"], "max_rub": stats["max_b"]
            },
            "top_risks_delay": [
                {"risk_id": rid, "name": name, "expected_impact_days": round(val, 2)}
                for rid, name, val in risk_delay[:5] if val > 0.5
            ],
            "top_risks_budget": [
                {"risk_id": rid, "name": name, "expected_impact_rub": round(val, 2)}
                for rid, name, val in risk_budget[:5] if val > 1e6
            ],
            "all_risks": self._all_risks_export(),
            "generated_at": datetime.now().isoformat()
        }

        json_file = str(paths.summary_statistics_json)
        save_json(json_file, summary)
        print(f"[SAVE] {json_file}")
        return summary

    def _all_risks_export(self) -> list:
        """Список всех рисков (детерминированный порядок)."""
        result = []
        for rid in sorted(self.network.get("risks", {}).keys()):
            rdata = self.network["risks"][rid]
            result.append({
                "risk_id": rid, "name": rdata.get("name", ""),
                "p50": rdata.get("p50", 0),
                "expected_delay": rdata.get("expected_delay", 0),
                "expected_budget": rdata.get("expected_budget", 0),
                "node_id": rdata.get("node_id", ""),
                "node_name": rdata.get("node_name", "")
            })
        return result

    # ==================== 4. ТЕПЛОВАЯ КАРТА ПЛОТНОСТИ ====================

    def plot_risk_heatmaps(self, stats: Dict):
        """Тепловая карта плотности сценариев."""
        print("\n[SUMMARY]  Построение тепловой карты плотности сценариев...")

        durations = stats.get("durations")
        budgets = stats.get("budgets")
        if durations is None or budgets is None:
            print("   [WARN]  Нет данных для тепловой карты")
            return

        delay_days = durations - self.base_duration
        budget_increase = budgets / 1_000_000 - self.base_budget / 1_000_000

        positive_mask = (delay_days >= 0) & (budget_increase >= 0)
        dd_pos = delay_days[positive_mask]
        bi_pos = budget_increase[positive_mask]
        if len(dd_pos) == 0:
            dd_pos, bi_pos = delay_days, budget_increase

        fig, ax = plt.subplots(figsize=(14, 11))
        h, _, _, im = ax.hist2d(dd_pos, bi_pos, bins=60, cmap=plt.cm.Blues_r, density=True)
        if np.max(h) > 0:
            levels = [np.percentile(h[h > 0], p) for p in (50, 75, 90)]
            ax.contour(h.T, levels=levels, colors='red', alpha=0.6, linewidths=1.5,
                       extent=[dd_pos.min(), dd_pos.max(), bi_pos.min(), bi_pos.max()])

        self._draw_quantile_lines(ax, dd_pos, bi_pos)
        self._draw_intersections(ax, dd_pos, bi_pos)

        ax.set_xlabel('Прирост срока (дни)', fontsize=14)
        ax.set_ylabel('Прирост бюджета (млн руб)', fontsize=14)
        ax.set_title('Тепловая карта плотности сценариев\n(Прирост срока vs Прирост бюджета)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.15)
        ax.set_xlim(0, dd_pos.max() * 1.05)
        ax.set_ylim(0, bi_pos.max() * 1.05)

        plt.colorbar(im, ax=ax).set_label('Плотность вероятности', fontsize=12)
        fig.tight_layout()
        fig.savefig(str(paths.heatmap_png), dpi=150, bbox_inches="tight")
        print("[SAVE] 04_heatmap_density.png")
        plt.close(fig)

    @staticmethod
    def _draw_quantile_lines(ax, delay_days_pos, budget_increase_pos):
        """Рисует линии квантилей на тепловой карте."""
        for name, color in [('P50', 'cyan'), ('P80', 'magenta'), ('P90', 'yellow')]:
            p = int(name[1:]) / 100
            p_d = np.percentile(delay_days_pos, p * 100)
            p_b = np.percentile(budget_increase_pos, p * 100)
            ax.axvline(p_d, color=color, linestyle='--', linewidth=2,
                       label=f'{name} срока: +{p_d:.0f} дн')
            ax.axhline(p_b, color=color, linestyle='--', linewidth=2,
                       label=f'{name} бюджета: +{p_b:.1f} млн руб')
        ax.axvline(0, color='red', linestyle='-', linewidth=2.5, label='База: 0')
        ax.axhline(0, color='red', linestyle='-', linewidth=2.5)

    @staticmethod
    def _draw_intersections(ax, delay_days_pos, budget_increase_pos):
        """Рисует точки пересечения персентилей."""
        pcts = [(np.percentile(delay_days_pos, p), np.percentile(budget_increase_pos, p))
                for p in (50, 80, 90)]
        points = [(0, 0)] + pcts
        labels = ['База', 'P50', 'P80', 'P90']

        for (x, y), label in zip(points, labels):
            ax.plot(x, y, 'o', color='white', markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5, alpha=0.9)
            ax.text(x, y + 0.3, label, color='white', fontsize=9, fontweight='bold',
                    ha='center', va='bottom', alpha=0.9)

        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='-|>', color='red', lw=5,
                                        mutation_scale=40))

    # ==================== 5. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ====================

    def _compute_correlations(self, scenarios):
        """Вычисляет корреляции Пирсона между рисками и с задержкой/бюджетом."""
        risk_ids = sorted(self.network.get("risks", {}).keys())
        if not risk_ids:
            return [], None, [], [], []

        n = len(scenarios)
        risk_to_idx = {rid: i for i, rid in enumerate(risk_ids)}
        X = np.zeros((n, len(risk_ids)))

        for i, s in enumerate(scenarios):
            states = s.get("states", {})
            for rid in risk_ids:
                X[i, risk_to_idx[rid]] = states.get(rid, 0)

        y_delay = np.array([s.get("total_delay", 0) for s in scenarios])
        y_budget = np.array([s.get("total_budget_impact", 0) for s in scenarios])

        corr_delay = np.zeros(len(risk_ids))
        corr_budget = np.zeros(len(risk_ids))
        activation_prob = np.zeros(len(risk_ids))

        for i in range(len(risk_ids)):
            activation_prob[i] = X[:, i].mean()
            if np.std(X[:, i]) > 0:
                corr_delay[i] = np.corrcoef(X[:, i], y_delay)[0, 1]
                corr_budget[i] = np.corrcoef(X[:, i], y_budget)[0, 1]

        corr_matrix = np.corrcoef(X.T) if len(risk_ids) > 1 else np.ones((1, 1))
        return risk_ids, corr_matrix, corr_delay, corr_budget, activation_prob

    def _plot_correlation_heatmap(self, fig, ax, corr_matrix, risk_ids):
        """Тепловая карта корреляций рисков."""
        fig.suptitle('Тепловая карта корреляций рисков', fontsize=18, fontweight='bold', y=1.02)
        im = ax.imshow(corr_matrix, cmap=CORRELATION_CONFIG["cmap"], vmin=-1, vmax=1, aspect='auto')

        labels = [rid.replace("RISK-", "R-") for rid in risk_ids]
        ax.set_xticks(range(len(risk_ids)))
        ax.set_yticks(range(len(risk_ids)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=12, fontweight='bold')

        for i in range(len(risk_ids)):
            for j in range(len(risk_ids)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                        color='#2d2d2d', fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8).set_label('Корреляция', fontsize=14)

    def _plot_radar(self, risk_ids, corr_values, title, output_path):
        """Radar Chart корреляций (ТОП-10 рисков по модулю)."""
        sorted_idx = np.argsort(np.abs(corr_values))[::-1][:10]
        n_items = len(sorted_idx)
        if n_items == 0:
            return

        angles = np.linspace(0, 2 * np.pi, n_items, endpoint=False).tolist()
        max_corr = max(np.abs(corr_values)) if max(np.abs(corr_values)) > 0 else 1
        values = [abs(corr_values[i]) / max_corr for i in sorted_idx]

        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]
        labels = [risk_ids[i].replace("RISK-", "R-") for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(projection='polar'))
        ax.plot(angles_plot, values_plot, 'o-', linewidth=3, color='#FF5722', markersize=12)
        ax.fill(angles_plot, values_plot, alpha=0.3, color='#FF5722')
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', pad=15)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3)

        for i, (angle, val) in enumerate(zip(angles, values)):
            if val > 0.01:
                va = 'top' if (np.pi / 2 < angle < 3 * np.pi / 2) else 'center'
                ax.text(angle, val + 0.08, f'{val:.2f}', ha='center', va=va,
                        fontsize=11, color='#FF5722', fontweight='bold')

        plt.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"[SAVE] {output_path.name}")
        plt.close(fig)

    def plot_sensitivity_analysis(self, stats: Dict, iteration_scenarios_path: str) -> Optional[Dict]:
        """Анализ чувствительности (корреляции Пирсона)."""
        print("\n[SUMMARY]  Построение анализа чувствительности...")

        scenarios = self._get_scenarios(iteration_scenarios_path)
        if not scenarios:
            print("   [WARN]  Нет данных для анализа")
            return None

        risk_ids, corr_matrix, corr_delay, corr_budget, activation_prob = self._compute_correlations(scenarios)
        if not risk_ids:
            return None

        # 1. Тепловая карта корреляций
        fig1, ax1 = plt.subplots(figsize=(14, 11))
        self._plot_correlation_heatmap(fig1, ax1, corr_matrix, risk_ids)
        plt.tight_layout()
        fig1.savefig(str(paths.correlation_png), dpi=150, bbox_inches="tight")
        print("[SAVE] 05_correlation_matrix.png")
        plt.close(fig1)

        # 2-3. Radar charts
        self._plot_radar(risk_ids, corr_delay, 'Radar Chart: корреляции (Срок)', paths.radar_delay_png)
        self._plot_radar(risk_ids, corr_budget, 'Radar Chart: корреляции (Бюджет)', paths.radar_budget_png)

        # 4. JSON
        self._export_correlations_json(risk_ids, activation_prob, corr_delay, corr_budget)

        return {"risk_ids": risk_ids, "activation_prob": activation_prob,
                "corr_delay": corr_delay, "corr_budget": corr_budget,
                "corr_matrix": corr_matrix}

    def _export_correlations_json(self, risk_ids, activation_prob, corr_delay, corr_budget):
        """Сохраняет сводную таблицу корреляций в JSON."""
        sorted_idx = np.argsort(np.abs(corr_delay))[::-1]
        correlations = [{
            "risk_id": risk_ids[idx],
            "activation_probability": round(float(activation_prob[idx]), 4),
            "correlation_with_delay": round(float(corr_delay[idx]), 4),
            "correlation_with_budget": round(float(corr_budget[idx]), 4)
        } for idx in sorted_idx]
        save_json(str(paths.sensitivity_correlations_json), correlations)
        print(f"[SAVE] sensitivity_correlations.json")


# ==================== ТОЧКА ВХОДА ====================

def main(simulation: dict = None, bayes: dict = None) -> dict:
    """Точка входа.

    simulation: результаты симуляции из шага 8 (in-memory из состояния LangGraph);
    bayes:      Байесовская сеть из шага 7 (in-memory). Если None — читаются с диска.

    Примечание: массивы выборок (.npy) и iteration_scenarios.json остаются
    дисковыми артефактами (их только что записал шаг 8) и читаются с диска.
    """
    import sys
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("\n[SUMMARY]  ANALYZER: Monte Carlo Analysis")

    sim_data = simulation if simulation is not None else load_json(str(paths.simulation_results_json))
    base_duration = sim_data["config"].get("base_duration")
    complete_date_str = sim_data["config"].get("complete_date", "2026-02-05")
    if base_duration is None:
        base_duration = (datetime.strptime(complete_date_str, "%Y-%m-%d") - datetime.now()).days
    base_budget = sim_data["config"].get("base_budget", 2000000000)
    complete_date = datetime.strptime(complete_date_str, "%Y-%m-%d")

    bayesian_network = bayes if bayes is not None else load_json(str(paths.bayesian_network_json))

    # Загружаем сценарии ОДИН РАЗ
    scenarios = _load_scenarios(str(paths.iteration_scenarios_json))

    analyzer = Analyzer(bayesian_network, base_duration, base_budget, complete_date, scenarios)

    stats = analyzer.plot_dashboard(str(paths.final_delay_samples_npy), str(paths.final_budget_samples_npy))

    if stats:
        # base_duration/base_budget добавляем в stats для export_summary_statistics
        stats["base_duration"] = base_duration
        stats["base_budget"] = base_budget

        risk_delay = analyzer.plot_stacked_delay_tornado(str(paths.iteration_scenarios_json))
        risk_budget = analyzer.plot_stacked_budget_tornado(str(paths.iteration_scenarios_json))

        summary_doc = analyzer.export_summary_statistics(stats, risk_delay, risk_budget)
        analyzer.plot_risk_heatmaps(stats)
        analyzer.plot_sensitivity_analysis(stats, str(paths.iteration_scenarios_json))

        # Возвращаем сводную статистику для состояния LangGraph (JSON уже сохранён)
        return summary_doc

    # stats пуст — вернуть None (сводка не сформирована)
    return None


if __name__ == "__main__":
    main()
