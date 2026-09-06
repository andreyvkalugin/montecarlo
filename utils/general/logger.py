"""
logger.py — Логирование вывода консоли в файл logs/log.txt.

Назначение:
  Перехватывать все сообщения, выводимые в терминал (включая вывод
  подпроцессов шагов пайплайна), и дублировать их в единый лог-файл.

Основные компоненты:
  - LOG_DIR / LOG_FILE — путь к каталогу и файлу логов.
  - Tee — поток-дублёр: пишет одновременно в консоль и в файл лога.
  - open_log_file() — создаёт каталог логов и открывает файл на запись.
"""

import os

# Корень проекта: utils/ -> каталог проекта
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")


class Tee:
    """Поток-дублёр: дублирует запись в реальный поток и в файл лога."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        # Консоль
        try:
            self.stream.write(data)
            self.stream.flush()
        except Exception:
            pass
        # Лог-файл
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass

    def reconfigure(self, **kwargs):
        # Проксируем reconfigure на реальный поток (encoding/errors/line_buffering/…)
        rc = getattr(self.stream, "reconfigure", None)
        if rc is not None:
            return rc(**kwargs)
        return None

    def __getattr__(self, name):
        # Прочие атрибуты (encoding, buffer, isatty, fileno, ...) берём у
        # реального потока — шаги пайплайна теперь исполняются в одном процессе
        # и обращаются к sys.stdout.encoding / .buffer напрямую.
        return getattr(self.stream, name)


def open_log_file(mode="w"):
    """Создаёт каталог logs/ и открывает файл logs/log.txt.

    Args:
        mode: режим открытия файла (по умолчанию 'w' — перезапись).

    Returns:
        (file_handle, абсолютный_путь_до_файла)
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(LOG_FILE, mode, encoding="utf-8")
    return log_file, LOG_FILE


def install_console_logging():
    """Заменяет sys.stdout/sys.stderr на Tee, дублирующий вывод в лог.

    Открывает logs/log.txt (перезапись) и подменяет стандартные потоки,
    чтобы все print() и сообщения об ошибках попадали в лог-файл.

    Returns:
        (file_handle, абсолютный_путь_до_файла)
    """
    import sys

    log_file, log_path = open_log_file("w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file, log_path