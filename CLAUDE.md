# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 10-step probabilistic pipeline that turns a construction schedule (КСГ) + a risk register into a Monte Carlo forecast of project delay and budget overrun, plus an LLM-written analytical report. Steps mix deterministic computation (WBS parsing, Floyd–Warshall distances, Bayesian network, MC simulation) with LLM "candidate agents" that infer edges/weights via multi-round consensus. See `README.md` for the domain math (structural decay, CPT construction, copula, bootstrap CI).

The primary language of the codebase, prompts, logs, and comments is **Russian**. Match that when editing.

## Commands

```powershell
# Setup (venv already present at .\venv)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Run the whole pipeline (LangGraph orchestrator; cleans data_processed/ then runs steps 1–10)
.\venv\Scripts\python.exe pipeline_graph.py

# Legacy subprocess orchestrator (still works; runs each step as its own process)
.\venv\Scripts\python.exe main.py

# Run a single step standalone — MUST be run from the project root, and the
# step's input files (produced by earlier steps) must already exist in data/data_processed/
.\venv\Scripts\python.exe 8_monte_carlo_simulator.py
```

There is **no test suite, linter, or build step** configured. "Verifying" a change means running the relevant step (or `main.py`) and reading its console output / the JSON it writes under `data/data_processed/<N>_data/`.

## Architecture

**Two orchestrators exist.** The primary one is `pipeline_graph.py` (LangGraph): it builds a linear `StateGraph` (step1→…→step10) and passes each step's output **through the graph state in-memory**. The legacy `main.py` runs each numbered script as a separate `python` subprocess, communicating only via disk files.

**Step contract (used by both orchestrators).** Each `N_*.py` exposes `main(<upstream docs>=None) -> output_doc`: it takes its inputs as in-memory dicts (from LangGraph state), **still saves its result JSON to `data/data_processed/<N>_data/`** (persistence is unconditional), and returns that dict. When called with no args (standalone / `main.py` subprocess), it falls back to loading its inputs from disk — the file-reading loaders (`load_dag`, `load_bayesian_network`, `_read_risks`, …) all take an optional `data=` param for exactly this. `PipelineState` keys in `pipeline_graph.py` map 1:1 to step outputs (`wbs, edges, risks, risk_mapping, risk_graph, weights, bayes, simulation, summary, report`).

**In-memory boundary:** JSON documents flow through state; bulk/binary artifacts do not. Step 8's `.npy` sample arrays and `iteration_scenarios.json`, and step 1's duration field read by step 8, remain disk-only (written by an earlier node, read directly from disk by the consumer).

**Single source of config truth.** All tunable parameters live in one dict, `PIPELINE_CONFIG` in `main.py`. Every module reads it via `utils.general.config_loader.get_pipeline_config()`, which imports `main` and caches the dict — do **not** duplicate config values elsewhere. The master switch `use_llm: False` makes every LLM step fall back to a deterministic baseline (independent risks, zero semantic weights).

**Paths are centralized.** `utils.general.paths.paths` (a `ProjectPaths` singleton) is the only place that knows the directory layout and per-step output filenames. Use it (`paths.step_dirs[6]`, `paths.risk_graph_json`, etc.) instead of hardcoding paths.

**LLM steps (2, 4, 6) share a consensus framework** under `utils/consensus/`:
- `pipeline.py` — provider-agnostic orchestrator: `run_step()` runs the linear `candidates → cleaning → finalization` hook chain, plus reusable helpers (`compute_cache_key`, `save_cache`/`load_cache`, `ensure_acyclic`/`creates_cycle` for keeping the edge graph a DAG).
- `base_consensus_step.py` — `BaseConsensusStep`, subclassed by each step. A step overrides only its specifics (prompt building, response parsing, cleaning, aggregation). The framework runs N LLM rounds and aggregates: **steps 2 & 4 vote by mode**, **step 6 averages** semantic weights.
- Each step's `.py` file (e.g. `6_model_assump_calc_LLM.py`) contains the step-specific subclass and its data classes.

**LLM access layer** is `utils/llm/`:
- `llm_runner.py` — `run_llm()` / `run_parallel_consensus()`, the single call path used by the consensus framework (text prompts, retries + exponential backoff).
- `llm_client.py` — `LLMApiClient` (`chat`, `chat_json`, `batch_scores`) used by step 6's scoring. Both modules call GigaChat through the **`gigachat` library** (`GigaChat` sync client), not a CLI. `batch_scores` requests structured JSON via `response_format` and falls back to a tolerant regex extractor (`_extract_scored_pairs`) because the model intermittently emits malformed JSON.
- Auth/connection come from `GIGACHAT_*` env vars read by the library's `Settings` (`GIGACHAT_CREDENTIALS`, `GIGACHAT_ACCESS_TOKEN`, `GIGACHAT_BASE_URL`, `GIGACHAT_SCOPE`, `GIGACHAT_VERIFY_SSL_CERTS`).

**LLM caching.** Results of steps 2/4/6/10 are cached in `cache/llm_cache/step<N>_*.json`, keyed by a SHA256 of input data + prompt text + relevant config. Re-running with unchanged inputs skips the LLM entirely (see `[CACHE MISS]` / cache hits in logs). Changing a prompt file in `prompt/` or the input data invalidates the key automatically.

## Conventions & gotchas

- **Every step script** does `sys.path.insert(0, <project root>)` and `sys.stdout.reconfigure(encoding="utf-8")` at startup (Windows console is cp1251) — keep this when adding a step. Console logs use `[TAG]` ASCII prefixes (`[OK]`, `[WARN]`, `[STEP]`, `[SAVE]`) instead of emoji.
- **`utils/` is a package with subfolders** `general/`, `llm/`, `consensus/` — imports are `from utils.general.paths import paths`, etc. (The layout in `README.md`'s tree is out of date; trust the actual folders.)
- **Model config mismatch (known issue):** `PIPELINE_CONFIG["llm"]["model"]` is `GigaChat-3.1-Ultra-128k`, which is **not** a model this account exposes (valid ids include `GigaChat`, `GigaChat-Pro`, `GigaChat-Max`, `GigaChat-2*`). `llm_client._build_gigachat_client` currently hardcodes `GigaChat-Pro` and an inline base64 credential — treat both as tech debt to move into config/env, and prefer a valid model id if you touch this.
- **LangGraph** now drives top-level orchestration (`pipeline_graph.py`). The **intra-step LLM consensus** framework in `utils/consensus/` is still the hand-rolled one (not LangGraph). `langchain`/`langchain-*` remain unused.
- **Step modules are imported via `importlib.import_module("5_risk_graph_builder")`** in `pipeline_graph.py` — their filenames start with digits, so a plain `import` is a syntax error.
- **`utils/general/logger.py`'s `Tee`** wraps `sys.stdout` to also write `logs/log.txt`. Because steps now run in-process (not subprocesses), they call `sys.stdout.encoding`/`.buffer`/`.reconfigure(...)` directly, so `Tee` proxies those to the real stream — keep that when touching the logger.
- Output files are typed by role: **расчётный** (feeds later steps), **аудит** (diagnostics, ignored downstream), **визуализация** (`.png`). Only touch downstream logic when changing a "расчётный" file's schema.
