"""
10_final_report_LLM.py — Модуль 10
Формирование расширенного итогового заключения по анализу.

Что делает:
  - Собирает основную информацию о проекте: сводные метрики (шаг 9),
    Байесовскую сеть (шаг 7), КСГ/граф задач (шаг 2), карту рисков (шаг 3).
  - Строит «обложку» отчёта (фактические таблицы и метаданные) в markdown.
  - Читает промпт prompt/10_llm_report_prompt.txt и отправляет LLM запрос на составление
    аналитической части заключения (анализ КСГ, рисков, значений).
  - Собирает итоговый расширенный markdown-документ и сохраняет в data/data_processed/10_data/.

Результаты сохраняются в data/data_processed/10_data/:
  - llm_report.md       — итоговый расширенный отчёт (обложка + аналитика LLM).
  - report_request.json — полный запрос (контекст + промпт) и полученное заключение.
"""
import io
import json
import os
import sys
from datetime import datetime
from typing import Optional
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import save_json

# Входные данные
SUMMARY_FILE = paths.summary_statistics_json
NETWORK_FILE = paths.bayesian_network_json
EDGES_FILE = paths.csg_edges_final_json
RISKS_FILE = paths.risks_processed_json

# Промпт и конфиг
PROMPT_FILE = paths.prompt_file(10)
CONFIG_PATH = os.path.join(paths.to_str(paths.utils), ".gigacode_config.json")

# Результат
OUTPUT_DIR = paths.step_dirs[10]
os.makedirs(paths.to_str(OUTPUT_DIR), exist_ok=True)

# Единый LLM-клиент (настройки из utils/llm_client.py); модель — из конфига main.py
sys.path.insert(0, paths.to_str(paths.root))
from utils.llm.llm_client import LLMApiClient
from utils.llm.cache_llm import check_and_cache

PIPELINE_CONFIG = get_pipeline_config()

# Название проекта — из конфига (универсально для разных запусков)
PROJECT_NAME = PIPELINE_CONFIG.get("project", {}).get("name", "Строительный проект")


# ===================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====================

def load_json(path):
    """Читает JSON-файл, возвращает None при отсутствии или ошибке."""
    if not os.path.isfile(path):
        print(f"   [WARN] Файл не найден: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"   [WARN] Не удалось прочитать {path}: {e}")
        return None


def load_prompt(path):
    """Читает текст промпта."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fmt_rub(value):
    """Форматирует сумму в рублях в читаемый вид (млрд/млн/руб)."""
    value = value or 0
    v = abs(value)
    if v >= 1e9:
        return f"{value/1e9:,.2f} млрд руб"
    if v >= 1e6:
        return f"{value/1e6:,.1f} млн руб"
    return f"{value:,.0f} руб"


def fmt_days(value):
    """Форматирует дни в читаемый вид (годы/месяцы/дни)."""
    value = value or 0
    if abs(value) >= 365:
        return f"{value:,.0f} дн (~{value/365:.1f} года)"
    return f"{value:,.0f} дн"


def fmt_ci_days(value, ci_dict):
    """Форматирует значение дней с CI."""
    if not ci_dict or ci_dict.get("lo") is None or ci_dict.get("hi") is None:
        return f"{value:,.0f} дн"
    return f"{value:,.0f} дн (CI: [{ci_dict['lo']:,.0f} .. {ci_dict['hi']:,.0f}])"


def fmt_rub_ci(value, ci_dict):
    """Форматирует сумму в рублях (fmt_rub) с CI: '1.0 млн руб (CI: [0 .. 0])'."""
    lo = (ci_dict or {}).get("lo", 0)
    hi = (ci_dict or {}).get("hi", 0)
    return f"{fmt_rub(value)} (CI: [{lo:,.0f} .. {hi:,.0f}])"


def load_risks(path, data=None):
    """
    Читает карту рисков (risks_processed.json) и возвращает список словарей
    с ключевыми фактами: номер, описание, причина, вероятность, влияние, статус.

    data: реестр рисков из состояния LangGraph (in-memory); если None — читается файл.
    """
    if data is None:
        if not os.path.isfile(path):
            print(f"   [WARN] Файл рисков не найден: {path}")
            return []
        try:
            data = load_json(path)
        except Exception as e:
            print(f"   [WARN] Не удалось прочитать риски {path}: {e}")
            return []

    risks = []
    for risk in data.get("risks", []):
        risks.append({
            "id": risk.get("risk_number", ""),
            "name": risk.get("description", ""),
            "cause": risk.get("cause", ""),
            "status": risk.get("status", ""),
            "probability": risk.get("probability_text", ""),
            "impact_schedule": risk.get("delay_text", ""),
            "impact_budget": risk.get("budget_text", ""),
            "total_weight": risk.get("total_weight", ""),
            "level": risk.get("level", ""),
        })
    return risks


def format_risks_for_prompt(risks):
    """Форматирует карту рисков (реестр шага 3) для передачи LLM."""
    if not risks:
        return "  (нет данных по рискам)"
    lines = []
    for r in risks:
        lines.append(
            f"- {r['id']}: {r['name']} | статус: {r['status']} | "
            f"вероятность: {r['probability']} | влияние на срок: {r['impact_schedule']} | "
            f"влияние на бюджет: {r['impact_budget']} | уровень: {r['level']}"
        )
        if r.get("cause"):
            lines.append(f"    причина: {r['cause']}")
    return "\n".join(lines)


def format_risk_table(risks):
    """
    Превращает список рисков (из summary_statistics.json) в компактный
    текстовый блок для передачи LLM, отсортированный по ожидаемому влиянию.
    """
    if not risks:
        return "  (нет данных по рискам)"

    # Сортируем по убыванию: сначала по бюджету, потом по задержке, потом по ID
    s = sorted(
        risks,
        key=lambda r: (r.get("expected_budget", 0) or 0, r.get("expected_delay", 0) or 0, r.get("risk_id", "")),
        reverse=True,
    )

    lines = []
    for r in s:
        delay = r.get("expected_delay", 0) or 0
        budget = r.get("expected_budget", 0) or 0
        lines.append(
            f"- {r['risk_id']}: {r.get('name', '')} | "
            f"P50={r.get('p50', 0):.3f} | "
            f"задержка={fmt_days(delay)} | "
            f"бюджет={fmt_rub(budget)} | "
            f"вершина {r.get('node_id', '')} ({r.get('node_name', '')})"
        )
    return "\n".join(lines)


def format_top_risks(top_items, key_label):
    """Форматирует топ-списки рисков (по задержке/бюджету)."""
    if not top_items:
        return "  (нет данных)"
    lines = []
    for item in top_items:
        val = item.get(key_label, 0) or 0
        if key_label == "expected_impact_days":
            val_txt = fmt_days(val)
        else:
            val_txt = fmt_rub(val)
        lines.append(f"- {item['risk_id']}: {item.get('name', '')} — {val_txt}")
    return "\n".join(lines)


# ===================== СБОР ИНФОРМАЦИИ =====================

def collect_project_info(summary, network, tasks, risk_register):
    """
    Собирает весь контекст в текстовом виде для передачи LLM:
    метрики, КСГ, карта рисков, Байесовская сеть, риски симуляции.
    """
    sections = []

    proj = summary.get("project", {})
    delay = summary.get("delay", {})
    budget = summary.get("budget", {})

    base_dur = proj.get("base_duration_days", 0)
    base_bud = proj.get("base_budget_rub", 0)

    # ВАЖНО: все значения в delay.*_days — это ЗАДЕРЖКА относительно базового срока,
    # а не абсолютная длительность. Для получения абсолютного срока нужно прибавить base_duration_days.
    # Например: P50_abs = p50_days + base_duration_days

    # --- 1. Параметры и метрики ---
    bootstrap = summary.get("bootstrap", {})
    n_bootstrap = bootstrap.get("iterations", "")
    ci_level = bootstrap.get("ci_level", "95%")

    lines = []
    lines.append("**Параметры проекта:**")
    lines.append(f"- Название: {PROJECT_NAME}")
    lines.append(f"- Базовый срок: {base_dur:,.0f} дней")
    lines.append(f"- Базовый бюджет: {fmt_rub(base_bud)}")
    lines.append(f"- Число прогонов Monte Carlo: {proj.get('iterations', 0):,}")
    lines.append(f"- Bootstrap-итераций: {n_bootstrap} ({ci_level})")
    lines.append("")
    lines.append("**Распределение СРОКА (дни) с доверительными интервалами:**")
    for p in ("P50", "P80", "P90", "P95"):
        key = f"{p.lower()}_days"
        ci_key = f"{key}_ci"
        lines.append(f"- {p} = {fmt_ci_days(delay.get(key, 0), delay.get(ci_key))}")
    lines.append(f"- Среднее = {delay.get('mean_days', 0):,.0f} (±{delay.get('std_days', 0):,.0f})")
    lines.append(f"- Мин..Макс = {delay.get('min_days', 0):,.0f}..{delay.get('max_days', 0):,.0f}")
    lines.append("")
    lines.append("[WARN]  ВАЖНО: Значения выше (p50_days, p80_days, p90_days, p95_days, mean_days, min_days, max_days)")
    lines.append("  — это ЗАДЕРЖКА относительно базового срока, а не абсолютная длительность проекта.")
    lines.append(f"  Для получения абсолютного срока к этим значениям нужно прибавить базовый срок: {base_dur:,.0f} дн.")
    lines.append(f"  Например: P50_абс = {delay.get('p50_days', 0):,.0f} + {base_dur:,.0f} = {delay.get('p50_days', 0) + base_dur:,.0f} дн.")
    lines.append("")
    lines.append("**Распределение БЮДЖЕТА (руб) с доверительными интервалами:**")
    for p in ("P50", "P80", "P90", "P95"):
        key = f"{p.lower()}_rub"
        ci_key = f"{key}_ci"
        lines.append(f"- {p} = {fmt_rub_ci(budget.get(key, 0), budget.get(ci_key))}")
    lines.append(f"- Среднее = {fmt_rub(budget.get('mean_rub', 0))} (±{fmt_rub(budget.get('std_rub', 0))})")
    lines.append(f"- Мин..Макс = {fmt_rub(budget.get('min_rub', 0))}..{fmt_rub(budget.get('max_rub', 0))}")
    sections.append("### 1. Параметры проекта и метрики симуляции\n" + "\n".join(lines))

    # --- 2. Карта рисков (реестр шага 3) ---
    sections.append(
        "### 2. Карта рисков (полный реестр)\n"
        + format_risks_for_prompt(risk_register)
    )

    # --- 3. Риски симуляции (итоги Monte Carlo) ---
    top_delay = summary.get("top_risks_delay", [])
    top_budget = summary.get("top_risks_budget", [])
    all_risks = summary.get("all_risks", [])
    sections.append(
        "### 3. Итоговые риски симуляции (по влиянию)\n"
        "**Топ-5 по влиянию на СРОК (дни):**\n"
        + format_top_risks(top_delay, "expected_impact_days")
        + "\n\n**Топ-5 по влиянию на БЮДЖЕТ (руб):**\n"
        + format_top_risks(top_budget, "expected_impact_rub")
        + "\n\n**Полный список с ожидаемым воздействием:**\n"
        + format_risk_table(all_risks)
    )

    return "\n\n".join(sections)


def build_llm_request(prompt, project_info):
    """Собирает итоговый промпт для LLM: инструкция + данные проекта."""
    return (
        "ИНСТРУКЦИЯ (роль и требования):\n"
        "--------------------\n"
        f"{prompt}\n\n"
        "--------------------\n"
        "ДАННЫЕ О ПРОЕКТЕ (используй ТОЛЬКО эти значения):\n"
        "--------------------\n"
        f"{project_info}\n"
    )


# ===================== ФОРМИРОВАНИЕ MD-ОТЧЁТА =====================

def build_report_cover(summary, tasks, risk_register, network):
    """
    Формирует факт-обложку отчёта (шапка, таблицы, параметры) в markdown.
    Этот блок всегда присутствует в итоговом документе независимо от LLM.
    """
    proj = summary.get("project", {})
    delay = summary.get("delay", {})
    budget = summary.get("budget", {})
    base_dur = proj.get("base_duration_days", 0)
    base_bud = proj.get("base_budget_rub", 0)

    lines = []
    lines.append(f"# Итоговое заключение по проекту: {PROJECT_NAME}")
    lines.append("")
    lines.append(f"> Сформировано автоматически: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 0. Фактографическая сводка (исходные данные анализа)")
    lines.append("")
    lines.append("| Параметр | Значение |")
    lines.append("|---|---|")
    lines.append(f"| Базовый срок | **{base_dur:,.0f} дн** |")
    lines.append(f"| Базовый бюджет | **{fmt_rub(base_bud)}** |")
    lines.append(f"| Прогонов Monte Carlo | {proj.get('iterations', 0):,} |")
    lines.append("")
    lines.append("### Распределение срока (дни)")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for p in ("P50", "P80", "P90", "P95"):
        key = f"{p.lower()}_days"
        ci = delay.get(f"{key}_ci")
        ci_txt = f" (CI: [{ci.get('lo', 0):,.0f} .. {ci.get('hi', 0):,.0f}])" if ci else ""
        lines.append(f"| {p} | {delay.get(key, 0):,.0f} дн{ci_txt} |")
    lines.append(f"| Среднее ± ст.откл. | {delay.get('mean_days', 0):,.0f} ± {delay.get('std_days', 0):,.0f} |")
    lines.append(f"| Мин..Макс | {delay.get('min_days', 0):,.0f}..{delay.get('max_days', 0):,.0f} |")
    lines.append("")
    lines.append("### Распределение бюджета")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    for p in ("P50", "P80", "P90", "P95"):
        key = f"{p.lower()}_rub"
        ci_key = f"{key}_ci"
        lines.append(f"| {p} | {fmt_rub_ci(budget.get(key, 0), budget.get(ci_key))} |")
    lines.append(f"| Среднее ± ст.откл. | {fmt_rub(budget.get('mean_rub', 0))} ± {fmt_rub(budget.get('std_rub', 0))} |")
    lines.append(f"| Мин..Макс | {fmt_rub(budget.get('min_rub', 0))}..{fmt_rub(budget.get('max_rub', 0))} |")
    lines.append("")

    # --- Сводка рисков симуляции ---
    all_risks = summary.get("all_risks", [])
    if all_risks:
        lines.append("### Риски в модели (ожидаемое влияние)")
        lines.append("| Риск | Название | P50 | Задержка | Бюджет | Вершина |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(all_risks, key=lambda x: (x.get("expected_budget", 0) or 0, x.get("risk_id", "")), reverse=True):
            lines.append(
                f"| {r['risk_id']} | {r.get('name','')[:60]} | {r.get('p50',0):.3f} | "
                f"{fmt_days(r.get('expected_delay',0))} | {fmt_rub(r.get('expected_budget',0))} | "
                f"{r.get('node_id','')} |"
            )
        lines.append("")

    # --- Описание КСГ ---
    nodes = (tasks or {}).get("nodes", [])
    edges = (tasks or {}).get("edges", [])
    if nodes:
        lines.append("### Календарно-сетевой график")
        lines.append("**Рабочие пакеты (вершины):**")
        lines.append("| ID | Название |")
        lines.append("|---|---|")
        for n in nodes:
            lines.append(f"| {n['id']} | {n['name']} |")
        lines.append("")
        lines.append("**Связи (зависимости):**")
        lines.append("| От | К | Тип |")
        lines.append("|---|---|---|")
        for e in edges:
            lines.append(f"| {e['source']} | {e['target']} | {e.get('type','')} |")
        lines.append("")

    # --- Краткая карта рисков (реестр шага 3) ---
    if risk_register:
        lines.append("### Краткая карта рисков")
        lines.append("| Риск | Описание | Статус | Уровень |")
        lines.append("|---|---|---|---|")
        for r in risk_register[:15]:
            lines.append(
                f"| {r['id']} | {r['name'][:60]} | {r['status']} | {r['level']} |"
            )
        if len(risk_register) > 15:
            lines.append(f"| | … и ещё {len(risk_register)-15} рисков | | |")
        lines.append("")

    # --- Байесовская сеть ---
    if network:
        sm = network.get("summary", {})
        risks_n = network.get("risks", {})
        parents = network.get("parents", {})
        lines.append("### Байесовская сеть")
        lines.append(f"- Рисков в сети: {len(risks_n)}")
        lines.append(f"- Связей «риск→риск»: {sm.get('total_parent_relationships', 0)}")
        # Катализаторы
        pf = {}
        for rl in parents.values():
            for p in rl:
                pf[p["risk_id"]] = pf.get(p["risk_id"], 0) + 1
        if pf:
            top_cat = sorted(pf.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("- Ключевые «катализаторы» (влияют на другие риски):")
            for rid, cnt in top_cat:
                lines.append(f"    - {rid} — {cnt} связей")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Аналитическое заключение")
    lines.append("")
    return "\n".join(lines)


def assemble_report(cover_md, llm_analysis):
    """Склеивает факт-обложку и аналитическую часть отчёта."""
    analysis = (llm_analysis or "").strip()
    if not analysis:
        analysis = "*(Аналитическое заключение не получено. Фактовая сводка приведена выше.)*"
    # Если LLM вернула собственные заголовки уровня ##, начинаем под шапкой.
    return cover_md.rstrip() + "\n" + analysis + "\n"


def save_report(report_md, request, report_json, llm_analysis):
    """Сохраняет итоговый md-отчёт, запрос и отдельно заключение LLM."""
    report_file = os.path.join(OUTPUT_DIR, "llm_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"   [SAVE] {report_file}")

    analysis_file = os.path.join(OUTPUT_DIR, "llm_analysis.md")
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write(llm_analysis or "")
    print(f"   [SAVE] {analysis_file}")

    request_out = {
        "meta": report_json.get("meta", {}),
        "prompt": request,
        "llm_analysis": llm_analysis,
    }
    request_file = os.path.join(OUTPUT_DIR, "report_request.json")
    save_json(request_file, request_out)
    print(f"   [SAVE] {request_file}")


def main(summary: dict = None, bayes: dict = None, edges: dict = None, risks: dict = None) -> str:
    """Точка входа.

    summary: сводная статистика из шага 9 (in-memory из состояния LangGraph);
    bayes:   Байесовская сеть из шага 7; edges: граф КСГ из шага 2;
    risks:   реестр рисков из шага 3. Любой None читается с диска.
    """
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("\n[LLM]  LLM REPORT: Формирование расширенного заключения")

    # 1. Загружаем входные данные (из состояния или с диска)
    summary = summary if summary is not None else load_json(SUMMARY_FILE)
    if not summary:
        print("\n[ERROR]  Ошибка: не найдено сводных статистик (шаг 9). "
              f"Ожидается файл: {SUMMARY_FILE}")
        sys.exit(1)

    network = bayes if bayes is not None else load_json(NETWORK_FILE)
    tasks = edges if edges is not None else load_json(EDGES_FILE)
    risk_register = load_risks(RISKS_FILE, data=risks)

    # 2. Читаем промпт
    prompt = load_prompt(PROMPT_FILE)
    if not prompt:
        print(f"\n[ERROR]  Ошибка: не найден промпт: {PROMPT_FILE}")
        sys.exit(1)

    # 3. Собираем контекст и строим факт-обложку
    print("\n[КОЛЛЕКЦИЯ] Сбор данных о проекте (метрики, КСГ, риски)...")
    project_info = collect_project_info(summary, network, tasks, risk_register)
    report_cover = build_report_cover(summary, tasks, risk_register, network)
    print("   [OK]  Данные собраны, обложка отчёта построена.")

    # 4. Отправка LLM (если включена)
    request = build_llm_request(prompt, project_info)
    llm_analysis: Optional[str] = None

    if PIPELINE_CONFIG.get("use_llm", True):
        print("\n[LLM] Отправка запроса к нейросети...")
        llm = LLMApiClient(config_path=CONFIG_PATH,
                           model=PIPELINE_CONFIG.get("llm", {}).get("model"))

        # --- Кэшируем вызов LLM ---
        def llm_call(r):
            return llm.chat(r, max_tokens=8000)

        llm_analysis = check_and_cache(
            input_content=project_info,
            prompt_content=prompt,
            pipeline_config=PIPELINE_CONFIG,
            llm_fn=llm_call,
            llm_args=(request,),
            llm_client=llm,
            cache_prefix="step10",
        )
    else:
        print("[SKIP] use_llm=False — LLM-аналитика пропущена, отчёт только по фактам.")

    meta = {
        "model": PIPELINE_CONFIG.get("llm", {}).get("model") if PIPELINE_CONFIG.get("use_llm", True) else "N/A",
        "generated_at": datetime.now().isoformat(),
        "sources": {
            "summary": os.path.basename(SUMMARY_FILE),
            "network": os.path.basename(NETWORK_FILE),
            "edges": os.path.basename(EDGES_FILE),
            "risks": os.path.basename(RISKS_FILE),
        },
    }

    # 5. Склеиваем полный отчёт и сохраняем
    print("\n[СОХРАНЕНИЕ] Сборка и сохранение отчёта...")
    report_md = assemble_report(report_cover, llm_analysis)
    save_report(report_md, request, {"meta": meta}, llm_analysis or "")

    # 6. Превью
    print("\n[OK]  Отчёт собран и сохранён.")
    print(f"   Полный отчёт: {os.path.join(OUTPUT_DIR, 'llm_report.md')}")
    print(f"   Запрос: {os.path.join(OUTPUT_DIR, 'report_request.json')}")
    print("\n--- ПРЕВЬЮ ФАКТ-СВОДКИ ---")
    print(report_cover[:1200])
    if llm_analysis:
        print("\n--- ПРЕВЬЮ АНАЛИТИКИ LLM ---")
        print(llm_analysis[:1200])
    print("\n")

    # Возвращаем путь к отчёту для состояния LangGraph (Markdown уже сохранён)
    return os.path.join(OUTPUT_DIR, "llm_report.md")


if __name__ == "__main__":
    main()
