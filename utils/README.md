# Utils — Общие утилиты проекта

## Структура

```
utils/
├── llm/              # Все скрипты для работы с LLM
│   ├── __init__.py
│   ├── llm_runner.py          # Единый вызов LLM через CLI
│   ├── llm_client.py          # Клиент для API LLM
│   ├── llm_utils.py           # Утилиты для работы с ответами LLM
│   ├── async_llm.py           # Асинхронные вызовы LLM
│   ├── cache_llm.py           # Кэширование LLM-ответов
│   └── count_tokens.py        # Подсчёт токенов
├── consensus/       # Все, что относится к шагам 2, 4, 6, 10 (consensus-оркестрация)
│   ├── __init__.py
│   ├── base_consensus_step.py   # Базовый класс для consensus-шагов
│   ├── consensus_builder.py     # Общие утилиты для consensus
│   └── pipeline.py               # Кастомный оркестратор пайплайна (функции/StepHooks)
└── general/        # Все остальные утилиты
    ├── __init__.py
    ├── paths.py               # Управление путями проекта
    ├── config_loader.py       # Загрузка конфигурации
    ├── json_io.py             # Чтение/запись JSON
    ├── logger.py              # Логирование
    └── mermaid_export.py      # Экспорт в Mermaid
```

## Использование

### LLM модули
```python
from utils.llm.llm_runner import run_llm, run_parallel_consensus
from utils.llm.llm_client import LLMApiClient
from utils.llm.llm_utils import normalize_llm_response
from utils.llm.cache_llm import CACHE_DIR
```

### Consensus модули
```python
from utils.consensus.pipeline import run_step, StepHooks, ensure_acyclic
from utils.consensus.base_consensus_step import BaseConsensusStep
from utils.consensus.consensus_builder import load_consensus_cache, save_consensus_cache
```

### General модули
```python
from utils.general.paths import paths
from utils.general.config_loader import get_pipeline_config
from utils.general.json_io import load_json, save_json
from utils.general.mermaid_export import export_mermaid_graph
```

## Миграция

Если у вас были старые импорты вида:
- `from utils.paths import ...` → `from utils.general.paths import ...`
- `from utils.llm_runner import ...` → `from utils.llm.llm_runner import ...`
- `from utils.langgraph_edges import ...` → `from utils.consensus.pipeline import ...`
- `from utils.langgraph.* import ...` → `from utils.consensus.* import ...`

Все импорты в проекте уже обновлены.
