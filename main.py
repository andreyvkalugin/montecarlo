"""
main.py — Оркестратор пайплайна (шаги 1–10).

Порядок вызова: python main.py
"""
import os
import shutil
import subprocess
import sys
import time

# Относительно корня проекта (где лежит main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Импорт из пакета utils (по конвенции проекта)
sys.path.insert(0, BASE_DIR)
from utils.general.logger import install_console_logging
from utils.general.paths import paths

DATA_DIR = paths.data
RAW_DIR = paths.raw_data
DATA_PROCESSED_DIR = paths.processed

# Папки для каждого шага
STEP_DIRS = paths.step_dirs

# ===== ЕДИНЫЙ КОНФИГ ПАЙПЛАЙНА =====
PIPELINE_CONFIG = {
    "use_llm": True,  # Единый флаг: False = все скрипты пропускают LLM-обращения

    "project": {
        "name": "Строительный проект",
        "type": "строительный проект",
        "complete_date": "2028-04-01",  # ISO формат YYYY-MM-DD
        "base_budget": 9_154_000_000,  # базовый бюджет в рублях
    },

    "wbs": {
        "max_depth": 2,  # Максимальная глубина WBS-иерархии (1, 1.1, 1.2, ...)
    },

    "llm": {
        "model": "GigaChat-3.1-Ultra-128k",
    },

    "llm_edges": {
        "max_additional_edges": 100,  # Максимальное количество дополнительных рёбер
        "edge_limit_percent": 0.5,  # Процент от total_pairs (0.0 — 1.0)
        "edge_limit_min": 10,       # Минимальный лимит связей
        "edge_limit_max": 100,      # Максимальный лимит связей
    },

    "risk_mapping": {
        "batch_size": 20,  # Сколько рисков отправлять за один запрос к LLM
    },

    "model_assumptions": {
        "structural_decay": {
            "mode": "continuous",
            "a": 0.0,   # вес связи на максимальном расстоянии (d = D)
            "b": 0.7,   # вес связи на минимальном расстоянии (d = 1)
            "w0": 1.0,  # вес для рисков на одной вершине
            "diameter": 0,  # 0 = автоматически из шага 5
        },
        "threshold": 0.0,  # минимальный final_weight для связи
        "llm": {
            "temperature": 0.7,
        },
        "consensus": {
            "rounds": 3  # N = 1 (single-shot), N > 1 = усреднение по раундам
        }
    },

    "bayesian_network": {
        "cpt": {}  # CPT использует final_weight напрямую
    },

    "monte_carlo": {
        "iterations": 200_000,
        "random_seed": 42,
        "distribution": {
            "type": "normal",  # normal или triangular
            "normal_sigma_div": 6,  # sigma = (max-min) / N. N=6 → ±3σ ≈ 99.7%
        },
    },

    "copula": {
        "correlation": 0.5  # Коэффициент корреляции между срывом сроков и ростом бюджета
    }
}
# =========================

# Шаги конвейера: (описание, имя скрипта, обязателен ли)
STEPS = [
    ("Парсинг КСГ и WBS-иерархия", "1_parse_csg.py", True),
    ("Добавление LLM-связей", "2_add_edges_LLM.py", True),
    ("Парсинг карты рисков", "3_parse_risks.py", True),
    ("Привязка рисков к вершинам графа", "4_map_risks_to_graph_LLM.py", True),
    ("Построение графа рисков", "5_risk_graph_builder.py", True),
    ("Расчёт весов рисков (модельные допущения)", "6_model_assump_calc_LLM.py", True),
    ("Построение Байесовской сети", "7_bayesian_network_builder.py", True),
    ("Монте-Карло симуляция", "8_monte_carlo_simulator.py", True),
    ("Анализ результатов симуляции", "9_analyzer.py", True),
    ("LLM-отчёт", "10_final_report_LLM.py", True),
]


def ensure_structure():
    """Создаёт папки raw_data и data_processed с подпапками, если их нет."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    for step_dir in STEP_DIRS.values():
        os.makedirs(step_dir, exist_ok=True)


def check_data_dir():
    """Проверяет, что в data/ нет посторонних файлов (не в папках)."""
    allowed_root_files = {".gitkeep", "README.md"}

    for entry in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, entry)

        if entry.startswith(".") or entry in allowed_root_files:
            continue

        if os.path.isfile(full):
            print(f"[!] В data/ найден лишний файл: {entry}", file=sys.stderr)
            print("    В data/ должны находиться только папки raw_data и data_processed.", file=sys.stderr)
            sys.exit(1)


def clean_all():
    """Полностью очищает data/data_processed со всеми подпапками."""
    if os.path.isdir(DATA_PROCESSED_DIR):
        print("    Удаление старых файлов...")
        shutil.rmtree(DATA_PROCESSED_DIR)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    for step_dir in STEP_DIRS.values():
        os.makedirs(step_dir, exist_ok=True)
    print("    data/data_processed очищена и создана заново.")


def check_input_files():
    """Проверяет наличие файла рисков (риски.csv или risks.xlsx)."""
    risks_csv = str(paths.risks_csv)
    risks_xlsx = str(paths.raw_risks / "risks.xlsx")
    has_risks = os.path.isfile(risks_csv) or os.path.isfile(risks_xlsx)

    if not has_risks:
        print("\n[WARN] ПРЕДУПРЕЖДЕНИЕ: Отсутствуют файлы рисков (risks.csv / risks.xlsx):", file=sys.stderr)
        print(f"   - risks.csv не найден: {risks_csv}", file=sys.stderr)
        print(f"   - risks.xlsx не найден: {risks_xlsx}", file=sys.stderr)
        return False

    return True


def run_step(step_num, step_info):
    """Запускает один шаг конвейера."""
    label, script, required = step_info
    script_path = os.path.join(BASE_DIR, script)

    if not os.path.isfile(script_path):
        if required:
            print(f"   [ERROR]  ОШИБКА: Скрипт не найден: {script}", file=sys.stderr)
            return False, 0.0
        else:
            print(f"   [WARN]  ПРЕДУПРЕЖДЕНИЕ: Скрипт не найден: {script} (пропускаем)", file=sys.stderr)
            return True, 0.0

    print(f"\n{'=' * 60}")
    print(f"Шаг №{step_num}: [{label}] Запуск: python {script}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        env = os.environ.copy()
        env["MAX_WBS_DEPTH"] = str(PIPELINE_CONFIG["wbs"]["max_depth"])

        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()

        proc.wait()
        elapsed = time.time() - start_time
        if proc.returncode != 0:
            print(f"\n[ERROR]  Шаг '{label}' завершился с ошибкой (код {proc.returncode}) [{elapsed:.1f}с]", file=sys.stderr)
            return False, elapsed

        print(f"[OK]  Шаг '{label}' выполнен успешно. [{elapsed:.1f}с]")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR]  Критическая ошибка при запуске '{label}': {e} [{elapsed:.1f}с]", file=sys.stderr)
        return False, elapsed


def print_summary(successful_steps, failed_steps, skipped_steps):
    """Выводит сводку о выполнении конвейера."""
    print("\n" + "=" * 60)
    print("[SUMMARY]  СВОДКА ВЫПОЛНЕНИЯ")
    print("=" * 60)

    total = len(STEPS)
    success_count = len(successful_steps)
    failed_count = len(failed_steps)
    skipped_count = len(skipped_steps)

    print(f"\nВсего шагов: {total}")
    print(f"[OK]  Успешно: {success_count}")
    print(f"[ERROR]  Ошибок: {failed_count}")
    print(f"[SKIP]  Пропущено: {skipped_count}")

    if successful_steps:
        print("\n[OK]  Успешно выполнены:")
        for step in successful_steps:
            print(f"   - {step}")

    if failed_steps:
        print("\n[ERROR]  Ошибки:")
        for step in failed_steps:
            print(f"   - {step}")

    if skipped_steps:
        print("\n[SKIP]  Пропущены:")
        for step in skipped_steps:
            print(f"   - {step}")


def print_timing_summary(step_timings, total_time):
    """Выводит таблицу с временем выполнения каждого шага и итогом."""
    print("\n" + "=" * 60)
    print("[TIME]   ВРЕМЯ ВЫПОЛНЕНИЯ ШАГОВ")
    print("=" * 60)
    print(f"\n{'№':<4} {'Шаг':<45} {'Время':>10}")
    print("-" * 60)

    sorted_timings = sorted(step_timings, key=lambda x: x[1], reverse=True)
    for i, (label, elapsed) in enumerate(sorted_timings, 1):
        print(f"{i:<4} {elapsed:>10.1f}с  {label}")

    print("-" * 60)
    print(f"{'ИТОГО':<45} {total_time:>10.1f}с")

    if sorted_timings:
        slowest_label, slowest_time = sorted_timings[0]
        print(f"\n[SLOW]  Самый медленный шаг: {slowest_label} ({slowest_time:.1f}с, {slowest_time/total_time*100:.0f}% от общего)")

    print("\nРаспределение времени:")
    for label, elapsed in sorted_timings:
        pct = elapsed / total_time * 100 if total_time > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<40} {pct:5.1f}% {bar}")


def run_pipeline():
    """Основная функция запуска конвейера."""
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(BASE_DIR)

    _log_file, log_path = install_console_logging()
    print(f"\n[LOG]  Лог вывода терминала: {log_path}")

    def msg(*a, **kw):
        kw.setdefault("flush", True)
        print(*a, **kw)

    msg("=" * 60)
    msg("[PIPELINE]   КОНВЕЙЕР ОБРАБОТКИ:")
    msg("=" * 60)

    mcfg = PIPELINE_CONFIG["monte_carlo"]
    msg("\n[CONFIG]  Ключевые настройки конфига:")
    msg(f"   -  WBS: максимальная глубина = {PIPELINE_CONFIG['wbs']['max_depth']}")
    msg(f"   -  LLM: модель = {PIPELINE_CONFIG['llm']['model']}, "
        f"макс. LLM-связей = {PIPELINE_CONFIG['llm_edges']['max_additional_edges']}")
    msg(f"   -  Привязка рисков: batch_size = {PIPELINE_CONFIG['risk_mapping']['batch_size']}")
    msg(f"   -  Допущения: порог связи (Шаг 6) = {PIPELINE_CONFIG['model_assumptions']['threshold']}")
    msg(f"   -  Байес. сеть: использует final_weight напрямую")
    msg(f"   -  Monte Carlo: итераций = {mcfg['iterations']}, "
        f"seed = {mcfg['random_seed']}, дата = {PIPELINE_CONFIG['project']['complete_date']}, "
        f"бюджет = {PIPELINE_CONFIG['project']['base_budget']:,.0f} руб")

    msg("\n[1] Проверка структуры папок...")
    ensure_structure()
    check_data_dir()
    msg("    [OK]  Структура папок OK.")

    msg("\n[2] Проверка входных файлов...")
    if not check_input_files():
        sys.exit(1)
    msg("    [OK]  Проверка файлов завершена.")

    msg("\n[3] Очистка data/data_processed...")
    clean_all()

    msg("\n[4] Запуск конвейера...")
    pipeline_start = time.time()

    successful_steps = []
    failed_steps = []
    skipped_steps = []
    step_timings = []

    for step_idx, step_info in enumerate(STEPS, start=1):
        label, script, required = step_info

        if not required:
            if "рисков" in label.lower() or "risks" in label.lower():
                has_file = (os.path.isfile(str(paths.risks_csv)) or
                            os.path.isfile(str(paths.raw_risks / "risks.xlsx")))
                if not has_file:
                    print(f"\n[SKIP]  Шаг '{label}' пропущен (файлы рисков не найдены)")
                    skipped_steps.append(label)
                    continue
        else:
            if "привязка рисков" in label.lower():
                has_file = (os.path.isfile(str(paths.risks_csv)) or
                            os.path.isfile(str(paths.raw_risks / "risks.xlsx")))
                if not has_file:
                    print(f"\n[WARN]  Шаг '{label}' пропущен (файлы рисков не найдены)")
                    skipped_steps.append(label)
                    continue

        success, elapsed = run_step(step_idx, step_info)

        if success:
            successful_steps.append(label)
            step_timings.append((label, elapsed))
        else:
            failed_steps.append(label)
            step_timings.append((label, elapsed))
            if required:
                msg(f"\n[ERROR]  КОНВЕЙЕР ОСТАНОВЛЕН из-за ошибки в шаге '{label}'", file=sys.stderr)
                break
            else:
                msg(f"\n[WARN]  Шаг '{label}' пропущен из-за ошибки", file=sys.stderr)
                skipped_steps.append(label)

    total_pipeline_time = time.time() - pipeline_start

    print_summary(successful_steps, failed_steps, skipped_steps)

    if step_timings:
        print_timing_summary(step_timings, total_pipeline_time)

    if failed_steps:
        msg("\n[ERROR]  Конвейер завершился с ошибками.", file=sys.stderr)
        sys.exit(1)
    elif not successful_steps:
        msg("\n[WARN]  Конвейер не выполнил ни одного шага.", file=sys.stderr)
        sys.exit(1)
    else:
        bn_output = os.path.join(STEP_DIRS[7], "bayesian_network.json")
        sim_output = os.path.join(STEP_DIRS[8], "simulation_results.json")
        analysis_output = os.path.join(STEP_DIRS[9], "01_dashboard_delay.png")

        if os.path.isfile(analysis_output):
            msg("\n[OK]  Анализ результатов завершён:")
            msg(f"   - Байесовская сеть: {bn_output}")
            msg(f"   - Симуляция MC: {sim_output}")
            msg(f"   - Графики анализа: {analysis_output}")
        else:
            msg("\n[WARN]  Итоговые файлы не найдены. Проверьте выполнение шагов 8-10.", file=sys.stderr)

    msg("\n" + "=" * 60)
    msg(f"[DONE]  Конвейер успешно завершён! [Всего: {total_pipeline_time:.1f}с]")
    msg("[OUTPUT]  Результаты находятся в: " + str(paths.processed))
    msg("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n[WARN]  Конвейер прерван пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR]  Непредвиденная ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
