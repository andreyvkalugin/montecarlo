"""
3_parse_risks.py — Парсинг и обогащение реестра рисков (этап 3).

Вход:  raw_data/risks/risks.csv | risks.xlsx
Выход: 3_data/risks_processed.json — {metadata, risks}, поля реестра приведены
       к snake_case, добавлены числовые probability_*/delay_*/budget_*.
"""
import csv
import math
import os
import re
import sys

from utils.general.json_io import save_json
from utils.general.paths import paths

# ===== НАСТРОЙКИ =====
# Поддержка обоих форматов: CSV и Excel (.xlsx)
def _resolve_risks_file():
    csv_path = paths.risks_csv
    xlsx_path = paths.raw_risks / "risks.xlsx"
    if os.path.isfile(csv_path):
        return csv_path
    if os.path.isfile(xlsx_path):
        return xlsx_path
    return csv_path  # вернёт CSV-путь, чтобы получить внятное сообщение об ошибке


SRC_FILE = _resolve_risks_file()
OUT_FILE = paths.risks_processed_json

# Колонки реестра (рус.) → поля JSON (snake_case). Колонки, отсутствующие здесь,
# сохраняются под исходными именами и попадают в metadata.unmapped_columns.
COLUMN_MAP = {
    "Название проекта": "project_name",
    "Номер риска": "risk_number",
    "Описание риска": "description",
    "Причина возникновения": "cause",
    "Статус риска": "status",
    "Вероятность реализации": "probability_text",
    "Вес вероятности": "probability_weight",
    "Влияние на срок": "delay_text",
    "Вес влияния на срок": "delay_weight",
    "Влияние на бюджет": "budget_text",
    "Вес влияния на бюджет": "budget_weight",
    "Средний вес влияния": "impact_weight_avg",
    "Суммарный вес риска": "total_weight",
    "Уровень риска": "level",
    "Комментарий к риску": "comment",
    "Описание КИР": "kir_description",
    "Контрольная дата КИР": "kir_control_date",
    "Предупреждающая дата КИР": "kir_warning_date",
    "Статус КИР": "kir_status",
    "Описание МПР": "mpr_description",
    "Плановая дата МПР": "mpr_planned_date",
    "Фактическая дата МПР": "mpr_actual_date",
    "Статус МПР": "mpr_status",
    "Описание события": "event_description",
    "Дата события": "event_date",
    "Ущерб от события": "event_damage",
    "Статус события": "event_status",
}

# Числовые поля, которые добавляет этот шаг — пишутся числами, а не строками
NUMERIC_FIELDS = ("probability_min", "probability_max",
                  "delay_min", "delay_max",
                  "budget_min", "budget_max")
# =====================


def _is_missing(value) -> bool:
    """Пустое значение: None, NaN/NaT из pandas или заглушка 'nan'/'NaT'."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in ("nan", "nat", "none")


def _to_text(value) -> str:
    """
    Текстовое поле реестра: пропуски → "", иначе строка AS-IS.

    Значения намеренно не обрезаются: реестр хранится 1-в-1 с источником,
    чтобы хэши consensus-кэшей последующих шагов не менялись без причины.
    """
    if _is_missing(value):
        return ""
    return str(value)


def _to_number(value, default: float = 0.0) -> float:
    """Числовое поле для JSON (не NaN, чтобы файл оставался валидным JSON)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else round(number, 6)


def _to_record(row: dict) -> dict:
    """
    Одна запись risk_processed.json: русские колонки → snake_case,
    числовые поля шага 3 → числа, текстовые → строки.
    """
    record = {}
    for column, value in row.items():
        if column in NUMERIC_FIELDS:
            record[column] = _to_number(value)
        else:
            record[COLUMN_MAP.get(column, column)] = _to_text(value)
    return record


def parse_probability(text: str) -> tuple[float, float]:
    """
    Парсит вероятность из форматов:
      "10-25%", "Низкая (до 25%)", "Высокая (от 50% до 75%)"
    Возвращает (min, max) в диапазоне 0.0–1.0.
    """
    text = text.strip()
    numbers = re.findall(r"(\d+)%", text)

    # "от X до Y%" → X% – Y% (два числа)
    if len(numbers) >= 2:
        low = float(numbers[0]) / 100.0
        high = float(numbers[1]) / 100.0
        return round(low, 2), round(high, 2)

    # "до X%" → 0.0 – X% (одно число + "до")
    if "до" in text.lower() and len(numbers) == 1:
        high = float(numbers[0]) / 100.0
        return 0.0, round(high, 2)

    return 0.0, 0.0


def parse_delay(text: str) -> tuple[float, float]:
    """
    Парсит влияние на срок из форматов:
      "Без влияния", "Увеличение от 1 до 3 месяцев", "Увеличение до 1 месяца"
    Возвращает (min, max) в днях.
    """
    text = text.strip().lower()

    if "без влияния" in text or "не влияет" in text:
        return 0.0, 0.0

    numbers = re.findall(r"(\d+)", text)

    # "от X до Y месяцев" → X*30 – Y*30 дней
    if len(numbers) >= 2:
        low_days = float(numbers[0]) * 30
        high_days = float(numbers[1]) * 30
        return low_days, high_days

    # "до X месяцев" → 0 – X*30 дней
    if "до" in text and len(numbers) == 1:
        high_days = float(numbers[0]) * 30
        return 0.0, high_days

    return 0.0, 0.0


def parse_budget(text: str) -> tuple[float, float]:
    """
    Парсит влияние на бюджет из форматов:
      "Без влияния", "до 50 млн", "Критичное. 500 млн и более", "Серьезное. 50 - 250 млн"
    Возвращает (min, max) в базовых единицах (полные числа).
    """
    text = text.strip().lower()

    if "без влияния" in text or "не влияет" in text:
        return 0.0, 0.0

    suffixes = {
        "млрд": 1_000_000_000,
        "млн": 1_000_000,
        "тыс": 1_000,
    }

    # Определяем множитель
    multiplier = 1
    for suffix, mult in suffixes.items():
        if suffix in text:
            multiplier = mult
            break

    # Извлекаем все числа
    numbers = re.findall(r"(\d+)", text)

    # "X - Y млн" → два числа (от X до Y)
    if len(numbers) >= 2:
        low = float(numbers[0]) * multiplier
        high = float(numbers[1]) * multiplier
        return low, high

    # "до X млн" → 0 – X*multiplier
    if "до" in text and len(numbers) == 1:
        high_val = float(numbers[0]) * multiplier
        return 0.0, high_val

    # Остальные случаи — одно число
    if len(numbers) == 1:
        val = float(numbers[0]) * multiplier
        return val, val

    return 0.0, 0.0


def enrich_risk_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Добавляет в каждую строку числовые поля:
      probability_min, probability_max
      delay_min, delay_max
      budget_min, budget_max
    Возвращает (обогащённые_строки, список_новых_полей).
    """
    NEW_COLS = list(NUMERIC_FIELDS)

    enriched: list[dict] = []
    for row in rows:
        new_row: dict = dict(row)

        new_row["probability_min"], new_row["probability_max"] = parse_probability(
            _to_text(row.get("Вероятность реализации")) or "0-0")

        new_row["delay_min"], new_row["delay_max"] = parse_delay(
            _to_text(row.get("Влияние на срок")) or "0-0")

        new_row["budget_min"], new_row["budget_max"] = parse_budget(
            _to_text(row.get("Влияние на бюджет")) or "0-0")

        enriched.append(new_row)

    return enriched, NEW_COLS


def detect_encoding(path):
    """Определяет кодировку по первым байтам (UTF-8 vs Windows-1251)."""
    with open(path, "rb") as f:
        raw = f.read(4000)
    try:
        raw.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1251"


def read_risks(path):
    """Читает risks.csv / risks.xlsx (с определением кодировки и разделителя) в список словарей."""
    path_lower = str(path).lower()

    # --- Excel ---
    if path_lower.endswith((".xlsx", ".xls")):
        print(f"[LOAD] Excel: {path}")
        import pandas as pd
        df = pd.read_excel(path)
        # Приводим все ключи к str — pandas может сделать int/float из заголовков
        df.columns = [str(c).strip() for c in df.columns]
        return df.where(pd.notna(df), None).to_dict(orient="records")

    # --- CSV ---
    encoding = detect_encoding(path)

    # Определяем разделитель по первой строке
    with open(path, "r", encoding=encoding) as f:
        first_line = f.readline()
    delimiter = ";" if ";" in first_line else ","

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(paths.to_str(paths.step_dirs[3]), exist_ok=True)

    if not os.path.isfile(SRC_FILE):
        print(f"Файл не найден: {SRC_FILE}", file=sys.stderr)
        sys.exit(1)

    rows = read_risks(SRC_FILE)

    print("\n[STEP] Загрузка реестра рисков...")
    print(f"[LOAD] {SRC_FILE}")

    # Убираем дубликаты по номеру риска, оставляем первую строку
    # (номер сравнивается без обрамляющих пробелов, хранится — как в источнике)
    seen_numbers = set()
    unique_rows = []
    duplicate_count = 0
    for row in rows:
        risk_number = _to_text(row.get("Номер риска")).strip()
        if risk_number not in seen_numbers:
            seen_numbers.add(risk_number)
            unique_rows.append(row)
        else:
            duplicate_count += 1
    if duplicate_count:
        print(f"[SKIP] Дубликатов по номеру риска: {duplicate_count}")

    # ===== Добавляем числовые поля =====
    print(f"\n[STEP] Обработка числовых полей...")
    enriched_rows, new_fields = enrich_risk_rows(unique_rows)

    # Исключаем архивные и черновые риски
    ARCHIVE_STATUSES = set() #"Архив"
    excluded_count = sum(1 for r in enriched_rows
                         if _to_text(r.get("Статус риска")).strip() in ARCHIVE_STATUSES)
    if excluded_count:
        print(f"[SKIP] Исключено (Архив/Черновик): {excluded_count}")
        enriched_rows = [r for r in enriched_rows
                         if _to_text(r.get("Статус риска")).strip() not in ARCHIVE_STATUSES]

    # Колонки реестра → snake_case, пропуски → "", числовые поля → числа
    risks = [_to_record(row) for row in enriched_rows]
    fields = list(risks[0].keys()) if risks else []

    source_columns = list(rows[0].keys()) if rows else []
    unmapped_columns = [c for c in source_columns
                        if c not in COLUMN_MAP and c not in NUMERIC_FIELDS]
    if unmapped_columns:
        print(f"[WARN]  Колонки без snake_case-маппинга (сохранены как есть): {unmapped_columns}")

    output = {
        "metadata": {
            "source_file": os.path.basename(str(SRC_FILE)),
            "source_format": os.path.splitext(str(SRC_FILE))[1].lstrip(".").lower(),
            "total_risks": len(risks),
            "duplicates_removed": duplicate_count,
            "excluded_by_status": excluded_count,
            "fields": fields,
            "unmapped_columns": unmapped_columns,
            "added_numeric_fields": list(new_fields),
            "version": "3.0-json",
        },
        "risks": risks,
    }

    save_json(paths.to_str(OUT_FILE), output)
    print(f"[SAVE] {OUT_FILE}")

    print(f"\n[RESULTS]")
    print(f"-  Рисков: {len(risks)}")
    print(f"-  Полей: {len(fields)}")

    # Возвращаем результат для передачи в состояние LangGraph (JSON уже сохранён)
    return output


if __name__ == "__main__":
    main()
