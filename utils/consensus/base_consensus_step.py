"""
base_consensus_step.py — Базовый класс для consensus-шагов.

Позволяет реализовать весь цикл (candidates → cleaning → finalization)
с минимальным дублированием кода.

Каждый шаг (2, 4, ...) наследуется от BaseConsensusStep и реализует ТОЛЬКО специфику:
- Конкретный State-класс
- Специфичные хуки (cleaning и т.д.)
- Формирование промптов

Всё остальное (параллельный consensus, кэш, merge, parsing) — из базового класса.

Пример использования:
    class AddEdgesState(TypedDict):
        nodes: list
        candidate_edges: list
        ...

    class EdgeConsensusStep(BaseConsensusStep):
        STATE_CLASS = AddEdgesState
        STEP_PREFIX = "step2"

        def build_candidate_prompt(self, state):
            return f"nodes: {state['nodes']}"

        def parse_candidate_response(self, text):
            return normalize_llm_response(text)

        def run_cleaning(self, state):
            # специфичная логика очистки
            return state
"""
from typing import Optional

from utils.consensus.consensus_builder import (
    load_consensus_cache,
    save_consensus_cache,
)
from utils.consensus.pipeline import load_prompt
from utils.llm.llm_utils import normalize_llm_response
from utils.llm.llm_runner import create_runner_config, run_parallel_consensus
from utils.general.paths import paths


class BaseConsensusStep:
    """
    Базовый класс для всех consensus-шагов.

    Атрибуты для переопределения в подклассах:
        STATE_CLASS: класс State (TypedDict)
        STEP_PREFIX: префикс для кэша и логов ("step2", "step4", ...)
        PROMPT_DIR: директория с промптами (по умолчанию paths.prompt)

    Методы для переопределения:
        build_candidate_prompt(state): формирование промпта для кандидатов
        parse_candidate_response(text): парсинг ответа LLM кандидатов
        run_cleaning(state): очистка кандидатов
        get_final_items(state): итоговые элементы для финализации (по умолчанию cleaned_<items>)
        run_finalization(state): финализация
    """

    STATE_CLASS = None
    STEP_PREFIX = "step"
    PROMPT_DIR = None

    def __init__(self):
        self.prompt_dir = self.PROMPT_DIR or paths.prompt

    # ===================================================================
    #  КАНДИДАТЫ (CONSENSUS)
    # ===================================================================

    def run_consensus_candidates(
        self,
        state: dict,
        consensus_rounds: int,
        max_concurrent: int = 4,
    ) -> dict:
        """
        Запускает ПОСЛЕДОВАТЕЛЬНЫЙ consensus-опрос N LLM.

        Returns:
            Обновлённый state с candidate_* и consensus_output
        """
        # Получаем промпты из подкласса
        prompt_builder = self.build_candidate_prompt(state)
        prompts = [prompt_builder for _ in range(consensus_rounds)]

        # Парсер из подкласса
        response_parser = self.parse_candidate_response

        # Конфиг LLM
        runner_config = create_runner_config(
            model=self._llm_model(state),
            timeout=180,
            max_retries=3,
            max_tokens=24000,
        )

        # Запускаем последовательный consensus
        raw_results = run_parallel_consensus(
            prompts=prompts,
            config=runner_config,
            max_concurrent=max_concurrent,
            max_retries_per_round=3,
            response_parser=response_parser,
        )

        # Агрегируем голоса (мода/частота)
        aggregated = self.aggregate_consensus_results(raw_results)

        return {
            "candidate_{}".format(self.get_items_key()): aggregated,
            "consensus_output": aggregated,
            "consensus_raw_rounds": raw_results,
        }

    def build_candidate_prompt(self, state: dict) -> str:
        """Формирует промпт для одного раунда consensus. Переопределяется в подклассе."""
        raise NotImplementedError("Подкласс должен реализовать build_candidate_prompt()")

    def parse_candidate_response(self, text: str) -> list:
        """Парсит ответ LLM. Переопределяется в подклассе или используется normalize_llm_response."""
        return normalize_llm_response(text)

    def aggregate_consensus_results(self, raw_results: list) -> dict:
        """Агрегирует результаты раундов (мода голосов). Переопределяется в подклассе."""
        raise NotImplementedError("Подкласс должен реализовать aggregate_consensus_results()")

    def get_items_key(self) -> str:
        """Возвращает ключ для хранения результатов (edges, mappings, ...)."""
        raise NotImplementedError("Подкласс должен реализовать get_items_key()")

    def _llm_model(self, state: dict) -> str:
        """Возвращает имя LLM-модели из конфига пайплайна."""
        return state.get("pipeline_config", {}).get("llm", {}).get("model", "")

    def _load_prompt(self, path: str) -> str:
        """Загружает промпт из файла (пустая строка, если файла нет)."""
        return load_prompt(path) or ""

    def rounds_report(self, state: dict, field: str = "result") -> list:
        """
        Собирает отчёт по раундам consensus из state.

        Args:
            state: Состояние шага с полем consensus_raw_rounds.
            field: Поле результата в каждом раунде (по умолчанию "result").

        Returns:
            Список словарей {round, edges, attempt} для сохранения отчёта.
        """
        return [
            {"round": r.get("round"), "edges": r.get(field, []) or [],
             "attempt": r.get("attempt", 0)}
            for r in (state.get("consensus_raw_rounds", []) or [])
            if isinstance(r, dict)
        ]

    # ===================================================================
    #  CLEANING
    # ===================================================================

    def run_cleaning(self, state: dict) -> dict:
        """Очистка кандидатов. Переопределяется в подклассе."""
        raise NotImplementedError("Подкласс должен реализовать run_cleaning()")

    # ===================================================================
    #  FINALIZATION
    # ===================================================================

    def get_final_items(self, state: dict) -> list:
        """
        Возвращает итоговые элементы для финализации.

        По умолчанию — cleaned_<items_key> (уже агрегированные по моде).
        Подклассы могут переопределить (например, step2 → selected_edges).
        """
        return state.get(f"cleaned_{self.get_items_key()}", [])

    def run_finalization(self, state: dict) -> dict:
        """
        Финализация: mark generated_by, сохранение в кэш.

        Итоговые элементы берутся из get_final_items(state) — результат
        агрегации N запросов (мода) после cleaning.

        Диагностические артефакты (сырые ответы, мода) пишутся самими шагами —
        здесь сохраняется только кэш и возвращаются итоги в state.
        """
        items_key = self.get_items_key()
        final = self.get_final_items(state)
        consensus_rounds = self._consensus_rounds(state)

        # Помечаем generated_by
        final = [dict(i, generated_by="LLM") for i in final if isinstance(i, dict)]

        # Сохраняем кэш
        data_context = self.get_cache_context(state)
        save_consensus_cache(
            prefix=f"{self.STEP_PREFIX}_consensus",
            validated_data=final,
            data_context=data_context,
            iterations=0,
            consensus_rounds=consensus_rounds,
            pipeline_config=state.get("pipeline_config", {}),
        )

        # Метаданные
        metadata = {
            f"total_{items_key}": len(final),
            "llm_model": self._llm_model(state),
            "consensus_rounds": consensus_rounds,
            "version": f"{self.STEP_PREFIX}-consensus",
        }

        return {
            f"final_{items_key}": final,
            "save_metadata": metadata,
            "consensus_output": state.get("consensus_output", {}),
        }

    def get_cache_context(self, state: dict) -> dict:
        """Возвращает контекст для кэша. Переопределяется в подклассе."""
        raise NotImplementedError("Подкласс должен реализовать get_cache_context()")

    # ===================================================================
    #  CACHE HELPERS
    # ===================================================================

    def check_cache(self, state: dict) -> Optional[list]:
        """Проверяет кэш перед запуском."""
        items_key = self.get_items_key()
        data_context = self.get_cache_context(state)

        return load_consensus_cache(
            prefix=f"{self.STEP_PREFIX}_consensus",
            data_context=data_context,
            pipeline_config=state.get("pipeline_config", {}),
        )

    # ===================================================================
    #  HOOKS BUILDER
    # ===================================================================

    def build_hooks(self):
        """
        Возвращает StepHooks с привязкой к методам класса.

        Реализует упрощённый конвейер: candidates → cleaning → finalization.
        """
        from utils.consensus.pipeline import StepHooks

        return StepHooks(
            candidates=lambda s: self.run_consensus_candidates_sync(s),
            cleaning=self.run_cleaning,
            finalization=self.run_finalization,
        )

    def run_consensus_candidates_sync(self, state: dict) -> dict:
        """Запускает последовательный consensus-опрос N LLM (синхронно)."""
        return self.run_consensus_candidates(state, self._consensus_rounds(state))

    def _consensus_rounds(self, state: dict) -> int:
        """Возвращает число consensus-раундов из конфига (не менее 1)."""
        return state.get("pipeline_config", {}).get(
            "model_assumptions", {}
        ).get("consensus", {}).get("rounds", 1) or 1