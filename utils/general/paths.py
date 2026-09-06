"""
ProjectPaths - Централизованное управление путями проекта.

Все пути к файлам и директориям хранятся в одном месте,
что упрощает поддержку и изменение структуры проекта.
"""

from pathlib import Path
from typing import Dict, Optional


class ProjectPaths:
    """Централизованный менеджер путей для всего пайплайна."""

    def __init__(self, project_root: Optional[str] = None):
        """
        Инициализация путей.

        Args:
            project_root: Корневая директория проекта.
                         Если None, используется dirname текущего файла.
        """
        self.root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.data = self.root / "data"
        self.raw_data = self.data / "raw_data"
        # Подпапки входных данных: КСГ (графики) и реестр рисков
        self.raw_cmp = self.raw_data / "CMP"
        self.raw_risks = self.raw_data / "risks"
        self.processed = self.data / "data_processed"
        self.prompt = self.root / "prompt"
        self.utils = self.root / "utils"
        self.cache = self.root / "cache"
        self.logs = self.root / "logs"

        # Папки для каждого шага
        self.step_dirs: Dict[int, Path] = {
            i: self.processed / f"{i}_data"
            for i in range(1, 11)
        }

    # ========== ПУТИ ВХОДНЫХ ДАННЫХ ==========

    @property
    def risks_csv(self) -> Path:
        return self.raw_risks / "risks.csv"

    # ========== ПУТИ ВЫХОДНЫХ ДАННЫХ ПО ШАГАМ ==========

    # Шаг 1: Парсинг КСГ
    @property
    def csg_tasks_wbs_json(self) -> Path:
        return self.step_dirs[1] / "csg_tasks_wbs.json"

    @property
    def csg_tasks_wbs_mermaid(self) -> Path:
        return self.step_dirs[1] / "csg_tasks_wbs.mermaid"

    # Шаг 2: LLM-связи
    @property
    def csg_edges_final_json(self) -> Path:
        """Компактный расчётный граф КСГ (узлы + рёбра WBS/LLM)."""
        return self.step_dirs[2] / "csg_edges_final.json"

    @property
    def csg_edges_final_mermaid(self) -> Path:
        return self.step_dirs[2] / "csg_edges_final.mermaid"

    @property
    def csg_edges_llm_survey_audit_json(self) -> Path:
        """Диагностический файл шага 2 (только для аудита).

        Содержит полные «сырые» ответы всех раундов LLM-опроса, их агрегацию (моду)
        и статистику отбора. Не участвует в дальнейших расчётах — расчётный файл
        это csg_edges_final.json.
        """
        return self.step_dirs[2] / "csg_edges_llm_survey_audit.json"

    # Шаг 3: Парсинг рисков
    @property
    def risks_processed_json(self) -> Path:
        """Обогащённый реестр рисков: metadata + risks (поля в snake_case)."""
        return self.step_dirs[3] / "risks_processed.json"

    # Шаг 4: Привязка рисков
    @property
    def risk_mapping_json(self) -> Path:
        return self.step_dirs[4] / "risk_mapping_final.json"

    @property
    def risk_mapping_raw_response_json(self) -> Path:
        return self.step_dirs[4] / "risk_mapping_raw_response.json"

    # Шаг 5: Граф рисков
    @property
    def risk_graph_json(self) -> Path:
        return self.step_dirs[5] / "risk_graph.json"

    @property
    def pairwise_distances_json(self) -> Path:
        """Попарные расстояния между рисками: metadata + pairs (диагностический артефакт)."""
        return self.step_dirs[5] / "pairwise_distances.json"

    @property
    def risk_graph_mermaid(self) -> Path:
        return self.step_dirs[5] / "risk_graph.mermaid"

    # Шаг 6: Модельные допущения
    @property
    def risk_graph_with_weights_json(self) -> Path:
        return self.step_dirs[6] / "risk_graph_with_weights.json"

    @property
    def semantic_cache_json(self) -> Path:
        return self.step_dirs[6] / "semantic_cache.json"

    # Шаг 7: Байесовская сеть
    @property
    def bayesian_network_json(self) -> Path:
        """Структура сети + CPT (единый файл)."""
        return self.step_dirs[7] / "bayesian_network.json"

    # Шаг 8: Монте-Карло
    @property
    def simulation_results_json(self) -> Path:
        return self.step_dirs[8] / "simulation_results.json"

    @property
    def final_delay_samples_npy(self) -> Path:
        return self.step_dirs[8] / "final_delay_samples.npy"

    @property
    def final_budget_samples_npy(self) -> Path:
        return self.step_dirs[8] / "final_budget_samples.npy"

    @property
    def iteration_scenarios_json(self) -> Path:
        return self.step_dirs[8] / "iteration_scenarios.json"

    # Шаг 9: Анализ
    @property
    def dashboard_delay_png(self) -> Path:
        return self.step_dirs[9] / "01_dashboard_delay.png"

    @property
    def dashboard_budget_png(self) -> Path:
        return self.step_dirs[9] / "01_dashboard_budget.png"

    @property
    def tornado_delay_png(self) -> Path:
        return self.step_dirs[9] / "02_tornado_delay.png"

    @property
    def tornado_budget_png(self) -> Path:
        return self.step_dirs[9] / "02_tornado_budget.png"

    @property
    def heatmap_png(self) -> Path:
        return self.step_dirs[9] / "04_heatmap_density.png"

    @property
    def correlation_png(self) -> Path:
        return self.step_dirs[9] / "05_correlation_matrix.png"

    @property
    def radar_delay_png(self) -> Path:
        return self.step_dirs[9] / "05_radar_chart_delay.png"

    @property
    def radar_budget_png(self) -> Path:
        return self.step_dirs[9] / "05_radar_chart_budget.png"

    @property
    def summary_statistics_json(self) -> Path:
        return self.step_dirs[9] / "summary_statistics.json"

    @property
    def sensitivity_correlations_json(self) -> Path:
        return self.step_dirs[9] / "sensitivity_correlations.json"

    # ========== ПРОМПТЫ ==========

    def prompt_file(self, step: int, suffix: Optional[str] = None) -> Path:
        """Получить путь к файлу промпта для шага.

        Args:
            step: Номер шага пайплайна.
            suffix: Дополнительный суффикс для имени файла (например "consensus_reason").
                   Если указан, suffix подставляется перед "weights_prompt": step_06_semantic_{suffix}.txt
                   Т.е. для шага 6: "06_semantic_{suffix}.txt"
        """
        names = {
            2: "2_llm_edges_prompt.txt",
            4: "4_llm.txt",
            6: "6_semantic_weights_prompt.txt",
            10: "10_llm_report_prompt.txt",
        }
        name = names.get(step)
        if name is None:
            name = f"{int(step):02d}_prompt.txt"
        if suffix:
            stem = Path(name).stem
            suffix_ext = Path(name).suffix
            # Для шага 6 (semantic) — suffix подставляется вместо "weights"
            if stem == "06_semantic_weights_prompt":
                name = f"{step:02d}_semantic_{suffix}{suffix_ext}"
            else:
                # Для других шагов — добавляем suffix в конец
                name = f"{stem}_{suffix}{suffix_ext}"
        return self.prompt / name

    # ========== УТИЛИТЫ ==========

    def to_str(self, path: Path) -> str:
        """Конвертировать Path в строку (для совместимости с os.path.join)."""
        return str(path)

    def __repr__(self) -> str:
        return f"ProjectPaths(root={self.root})"


# Глобальный экземпляр для удобства
paths = ProjectPaths()
