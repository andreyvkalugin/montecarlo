"""
1_parse_csg.py — Парсинг КСГ и построение WBS-иерархии.

Вход: data/raw_data/CMP/*.xlsx или *.csv
Выход: 1_data/csg_tasks_wbs.json, 1_data/csg_tasks_wbs.mermaid
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from datetime import date, datetime

import pandas as pd

from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import save_json
from utils.general.mermaid_export import escape_mermaid_text

# Единый конфиг пайплайна
PIPELINE_CONFIG = get_pipeline_config()

# ===== НАСТРОЙКИ ПРОЕКТА =====
INPUT_DIR = paths.raw_cmp
OUT_WBS_JSON = paths.csg_tasks_wbs_json
MAX_WBS_DEPTH = PIPELINE_CONFIG.get("wbs", {}).get("max_depth", 2)
# Требуемые колонки (совпадают с русскими заголовками в файле)
REQUIRED_COLUMNS = {"Код WBS", "Название"}
# =====================


# ==============================
# Поиск входного файла
# ==============================

def find_input_file(directory: Path) -> Path | None:
    """Находит первый файл .xlsx или .csv в директории."""
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        for fpath in sorted(directory.glob(ext)):
            if fpath.is_file():
                return fpath
    return None


# ==============================
# Основная логика: Извлечение задач
# ==============================

def _has_header_row(df_test: pd.DataFrame, row_idx: int) -> bool:
    """
    Проверяет, является ли строка с индексом row_idx строкой заголовков.
    Критерий: содержит хотя бы одну из требуемых колонок.
    """
    if row_idx >= len(df_test):
        return False
    row_vals = [str(df_test.iloc[row_idx, i]).strip() for i in range(df_test.shape[1])]
    return any(col in row_vals for col in REQUIRED_COLUMNS)


def parse_excel(path: Path) -> pd.DataFrame:
    """
    Читает Excel-файл (xlsx/xls) через openpyxl.

    Если в файле два заголовка (первая строка — вспомогательный,
    вторая строка — основной с «Код WBS», «Название» и т.д.),
    то используется вторая строка как header.
    """
    # Читаем без заголовка для анализа структуры
    df_test = pd.read_excel(path, header=None)

    if df_test.shape[0] >= 2 and _has_header_row(df_test, 1):
        # Вторая строка — основной заголовок
        df = pd.read_excel(path, header=1)
    else:
        # Одна строка заголовка (первая)
        df = pd.read_excel(path, header=0)

    # Нормализуем заголовки: убираем пробелы
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_csv(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    Читает CSV-файл.

    Если в файле два заголовка, используется вторая строка как header.
    """
    # Считываем все строки
    with open(path, "r", encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) >= 2:
        # Проверяем вторую строку
        row1_cols = [c.strip() for c in lines[1].split(",")]
        if any(col in row1_cols for col in REQUIRED_COLUMNS):
            # Вторая строка — заголовок
            df = pd.read_csv(path, header=1, encoding=encoding)
        else:
            df = pd.read_csv(path, header=0, encoding=encoding)
    else:
        df = pd.read_csv(path, header=0, encoding=encoding)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_date(val) -> str | None:
    """Конвертирует дату из Excel/CSV в строку YYYY-MM-DD."""
    # pd.isna() корректно обрабатывает None, NaT, NaN
    if pd.isna(val):
        return None
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")
    # Если строка — пробуем распарсить, если нет — возвращаем как есть
    s = str(val).strip()
    if not s:
        return None
    try:
        parsed = datetime.strptime(s, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        pass
    try:
        parsed = datetime.strptime(s, "%d.%m.%Y")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        pass
    return s


def extract_records(df: pd.DataFrame) -> list[dict]:
    """Извлекает записи из DataFrame в нужный формат."""
    records = []
    for _, row in df.iterrows():
        code = row.get("Код WBS")
        if code is None:
            continue
        code_str = str(code).strip()
        # Пропускаем строки без числового кода WBS
        if not code_str or not code_str[0].isdigit():
            continue

        record: dict = {
            "Код WBS": code_str,
            "Название": str(row.get("Название", "")).replace("\n", " ").strip(),
            "Начало": normalize_date(row.get("Начало")),
            "Окончание": normalize_date(row.get("Окончание")),
        }
        records.append(record)

    return records


# ==============================
# Основная логика: WBS-иерархия
# ==============================

def code_key(t: dict) -> str:
    """Извлекает ID задачи."""
    return str(t.get("Код WBS", t.get("id", ""))).strip()


def get_depth(code: str) -> int:
    """Возвращает глубину WBS-кода (количество уровней)."""
    return len([p for p in code.split(".") if p])


def filter_by_depth(nodes: list, max_depth: int) -> list:
    """
    Фильтрует узлы, оставляя только те, чья глубина <= max_depth.
    """
    return [n for n in nodes if get_depth(n["id"]) <= max_depth]


def _num_key(seg: str) -> int:
    """Числовой ключ для сортировки сегментов WBS."""
    nums = re.findall(r"\d+", seg)
    return int(nums[0]) if nums else 1e9


def build_wbs_edges(nodes: list) -> list:
    """
    Строит полный граф связей на основе WBS-иерархии.

    Принципы:
    1. Корневые узлы идут последовательно: 0 → 1 → 2 → ...
       НО: если у корневого узла есть дети, связь идёт через последнего потомка
    2. Родитель → первый дочерний элемент: 5 → 5.1
    3. Дочерние элементы идут последовательно: 5.1 → 5.2 → 5.3 → ...
    4. Последний дочерний → следующий корневой узел: 5.5 → 6

    Возвращает:
        list: Список рёбер [{"source": "id", "target": "id", "type": "FS", "generated_by": "WBS"}]
    """
    existing = set()
    edges = []

    def _add(src: str, tgt: str) -> None:
        """Добавляет ребро, если оно уникально и оба узла существуют."""
        if src == tgt or (src, tgt) in existing:
            return

        src_exists = any(n["id"] == src for n in nodes)
        tgt_exists = any(n["id"] == tgt for n in nodes)

        if not src_exists or not tgt_exists:
            return

        edges.append({
            "source": src,
            "target": tgt,
            "type": "FS",
            "generated_by": "WBS"
        })
        existing.add((src, tgt))

    # Индексируем узлы по уровням
    by_level: dict[int, list] = defaultdict(list)
    for n in nodes:
        parts = n["id"].split(".")
        depth = len(parts)
        by_level[depth].append((n["id"], parts))

    # Сортируем узлы на каждом уровне
    for depth in by_level:
        by_level[depth].sort(key=lambda x: [_num_key(s) for s in x[1]])

    # ---- 1. КОРНЕВОЙ УРОВЕНЬ (глубина 1) ----
    roots = by_level.get(1, [])

    # Определяем, у каких корней есть дети (глубина 2)
    parents_with_children = set()
    for nid, parts in by_level.get(2, []):
        parent = ".".join(parts[:-1])
        parents_with_children.add(parent)

    # Связи между корневыми узлами
    for i in range(len(roots) - 1):
        cur = roots[i][0]
        nxt = roots[i + 1][0]

        if cur not in parents_with_children:
            _add(cur, nxt)

    # ---- 2. УРОВЕНЬ 2 (глубина 2) ----
    level2 = by_level.get(2, [])

    # Группируем детей по родителям
    children_by_parent: dict[str, list] = defaultdict(list)
    for nid, parts in level2:
        parent = ".".join(parts[:-1])
        children_by_parent[parent].append((nid, parts))

    # Для каждого родителя
    for parent, children in children_by_parent.items():
        children.sort(key=lambda x: [_num_key(s) for s in x[1]])

        # 2.1. Родитель → первый ребёнок
        _add(parent, children[0][0])

        # 2.2. Цепочка детей: 5.1 → 5.2 → 5.3 → ...
        for i in range(len(children) - 1):
            _add(children[i][0], children[i + 1][0])

        # 2.3. Последний ребёнок → следующий корневой узел
        last_child = children[-1][0]

        parent_idx = None
        for i, (root_id, _) in enumerate(roots):
            if root_id == parent:
                parent_idx = i
                break

        if parent_idx is not None and parent_idx + 1 < len(roots):
            next_root = roots[parent_idx + 1][0]
            _add(last_child, next_root)

    return edges


# ==============================
# Главная функция
# ==============================

def main() -> None:
    # Настройка кодировки для Windows
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    sys.stdout.reconfigure(encoding="utf-8")

    print("\n[STEP] ПАРСИНГ КСГ ИЗ EXCEL/CSV...")

    # Создаём директории
    for d in [paths.step_dirs[1]]:
        os.makedirs(str(d), exist_ok=True)

    # Ищем входной файл
    input_file = find_input_file(INPUT_DIR)
    if input_file is None:
        print(f"[ERROR]  Файлы .xlsx/.csv не найдены в: {INPUT_DIR}", file=sys.stderr)
        print(f"Положите файл КСГ (.xlsx/.csv) в: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    ext = input_file.suffix.lower()
    print(f"[INPUT] {input_file.name} (формат: {ext})")

    # Парсим файл
    if ext in (".xlsx", ".xls"):
        print(f"[LOAD] {input_file}")
        df = parse_excel(input_file)
    elif ext == ".csv":
        print(f"[LOAD] {input_file}")
        df = parse_csv(input_file)
    else:
        print(f"[WARN]  Неизвестный формат: {ext}", file=sys.stderr)
        sys.exit(1)

    # === ОТЛАДКА: выводим колонки и первые строки ===
    print(f"[DEBUG] Колонки файла: {list(df.columns)}")
    print(f"[DEBUG] Форма DataFrame: {df.shape}")
    print(f"[DEBUG] Первые 3 строки:")
    for i, row in df.head(3).iterrows():
        print(f"   {i}: {dict(row)}")

    # Проверяем наличие обязательных колонок
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"[WARN]  Колонка '{col}' не найдена в файле. Доступные колонки: {list(df.columns)}",
                  file=sys.stderr)
            sys.exit(1)

    # === Отбор задач: исключаем этапы, которые уже завершились ===
    # Фильтруем DataFrame по дате окончания
    if 'Окончание' in df.columns:
        # Конвертируем даты в datetime для сравнения
        df['Окончание_dt'] = pd.to_datetime(df['Окончание'], errors='coerce')

        # Текущая дата (без времени)
        today = pd.Timestamp.now().normalize()

        # Фильтруем: оставляем задачи, у которых end >= today
        df = df[df['Окончание_dt'] >= today]

        # Удаляем временную колонку
        df.drop(columns=['Окончание_dt'], inplace=True)

        # Рассчитываем длительность проекта на основе оставшихся задач
        project_duration_days = 0
        if len(df) > 0 and 'Начало' in df.columns:
            # Парсим даты начала и окончания
            start_dates = pd.to_datetime(df['Начало'], errors='coerce')
            end_dates = pd.to_datetime(df['Окончание'], errors='coerce')

            # Исключаем строки без корректных дат
            valid_mask = end_dates.notna() & start_dates.notna()
            if valid_mask.any():
                min_start = start_dates[valid_mask].min()
                max_end = end_dates[valid_mask].max()
                project_duration_days = (max_end - min_start).days
            else:
                project_duration_days = 0
        else:
            project_duration_days = 0

    # Извлекаем записи (уже с фильтром)
    records = extract_records(df)

    if not records:
        print("[WARN]  Не удалось извлечь задачи из файла.", file=sys.stderr)
        sys.exit(1)

    print(f"  Извлечено задач: {len(records)}")
    print(f"  Длительность проекта: {project_duration_days} дней")

    # Проверяем, есть ли даты
    with_dates = sum(1 for r in records if r.get("Начало") or r.get("Окончание"))
    if with_dates > 0:
        print(f"  Задач с датами: {with_dates}")

    # ==============================
    print("\n[STEP] ПОСТРОЕНИЕ WBS-ИЕРАРХИИ...")

    # Готовим узлы из извлечённых записей (в памяти, без промежуточного файла)
    all_nodes = [
        {
            "id": code_key(t),
            "name": t.get("Название", t.get("name", "")),
            "start": t.get("Начало"),
            "end": t.get("Окончание"),
        }
        for t in records
    ]

    # Фильтрация по глубине
    nodes = filter_by_depth(all_nodes, MAX_WBS_DEPTH)
    filtered_out = len(all_nodes) - len(nodes)

    # Построение WBS-связей
    print("\n[STEP] Построение WBS-связей...")
    wbs_edges = build_wbs_edges(nodes)

    # Формирование графа
    graph = {
        "nodes": nodes,
        "edges": wbs_edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(wbs_edges),
            "edge_source": "WBS",
            "max_depth": MAX_WBS_DEPTH,
            "filtered_out": filtered_out,
            "input_file": input_file.name,
            "version": "2.0",
            "project_duration_days": project_duration_days
        }
    }

    # Сохранение
    save_json(OUT_WBS_JSON, graph)
    print(f"[SAVE] {OUT_WBS_JSON}")

    # ==============================
    # Экспорт Mermaid (WBS только)
    # ==============================
    print("\n[STEP] Экспорт WBS-графа в Mermaid...")

    MERMAID_OUT = paths.csg_tasks_wbs_mermaid
    mermaid_lines = ["graph TD"]

    # Узлы
    for node in nodes:
        nid = str(node["id"]).replace(".", "_")
        name = escape_mermaid_text(str(node.get("name", "")))
        mermaid_lines.append(f'    {nid}["{node["id"]}: {name}"]')

    # Рёбра (WBS)
    for edge in wbs_edges:
        s = str(edge.get("source", "")).replace(".", "_")
        t = str(edge.get("target", "")).replace(".", "_")
        mermaid_lines.append(f'    {s} --> {t}')

    mermaid_content = "\n".join(mermaid_lines)
    os.makedirs(os.path.dirname(str(MERMAID_OUT)) or ".", exist_ok=True)
    with open(MERMAID_OUT, "w", encoding="utf-8") as f:
        f.write(mermaid_content)
    print(f"[SAVE] {MERMAID_OUT}")

    print(f"\n[RESULTS]")
    print(f"-  Узлов: {len(nodes)}")
    print(f"-  Рёбер: {len(wbs_edges)}")
    print(f"-  Макс. глубина WBS: {MAX_WBS_DEPTH}")
    if filtered_out > 0:
        print(f"-  Исключено (глубже {MAX_WBS_DEPTH}): {filtered_out}")


if __name__ == "__main__":
    main()
