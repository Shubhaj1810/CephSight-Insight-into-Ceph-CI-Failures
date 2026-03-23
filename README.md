# Teuthology AI Log Analyzer

AI-powered tool for analyzing failed Ceph/Teuthology CI jobs. Fetches logs, clusters failures by error signature, analyzes root causes with a local Ollama LLM, and produces an interactive HTML report.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                            │
└──────────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      ▼                       ▼                       ▼
┌─────────────┐      ┌──────────────┐        ┌──────────────────┐
│  fetcher    │      │  log_parser  │        │     cluster      │
│             │      │              │        │                  │
│ • Pulpito   │─────▶│ • Tracebacks │───────▶│ • Fingerprint    │
│ • qa-proxy  │      │ • Errors     │        │ • Fuzzy merge    │
│ • Parallel  │      │ • Summary    │        │ • Run health     │
│ • Thread-   │      │ • Condense   │        │ • Representative │
│   safe DL   │      └──────────────┘        └────────┬─────────┘
└─────────────┘               │                       │
                              ▼                       ▼
                     ┌──────────────────┐    ┌──────────────────┐
                     │    analyzer      │    │ report_generator  │
                     │                  │    │                  │
                     │ • Ollama LLM     │───▶│ • HTML report    │
                     │ • Two-pass       │    │ • Search/filter  │
                     │   analysis       │    │ • JSON/CSV export│
                     │ • Pattern match  │    │ • Light/dark mode│
                     │ • Cache + TTL    │    └──────────────────┘
                     │ • Heuristics     │
                     └──────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with a model pulled
ollama pull llama3:8b
ollama serve

# Analyze a Pulpito run
python3 main.py "https://pulpito.ceph.com/your-run-trial/" -o output

# Use a different model
python3 main.py URL --model mistral -o output

# Use a larger context window for more log visibility
python3 main.py URL --model llama3.1:70b --num-ctx 32768 -o output

# Skip fetching, reuse previously downloaded logs
python3 main.py URL --skip-fetch --logs-dir output/logs -o output
```

## Features

- **Multi-source log fetching** — Fetches `teuthology.log` (with `ansible.log` fallback) from qa-proxy with parallel, thread-safe downloads
- **Smart log parsing** — Extracts tracebacks, error blocks, and summary sections; condenses to ~24K chars for maximum LLM signal
- **Failure clustering** — Groups jobs by error fingerprint with fuzzy similarity merging (Jaccard); one LLM call per cluster
- **Run health classification** — Mass failure / partial / isolated based on actual pass/fail ratio from the run page
- **Two-pass analysis** — Low-confidence results automatically get a deeper second analysis pass
- **Concrete pattern validation** — Post-LLM regex-based cross-checks correct misclassifications using known Ceph error signatures
- **Deep Ceph domain knowledge** — System prompt includes OSD crashes, MON elections, BlueStore errors, RGW failures, and more
- **Model-aware caching** — Cache keys include model name + prompt version; 7-day TTL; executive summary caching
- **Interactive HTML report** — Search, filter by severity/type, sort, JSON/CSV export, light/dark mode toggle
- **Memory efficient** — Lazy raw log loading from disk after initial parse

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llama3:8b` | Ollama model name |
| `--ollama-url` | `localhost:11434` | Ollama API URL |
| `--timeout` | `60` | HTTP timeout for log fetching (seconds) |
| `--llm-timeout` | `300` | LLM request timeout (seconds) |
| `--num-ctx` | `0` (auto) | Override model context window size |
| `--skip-fetch` | — | Load logs from disk instead of fetching |
| `--jobs` | all | Comma-separated job IDs to analyze |
| `--no-cache` | — | Disable analysis caching |
| `--clear-cache` | — | Wipe cache before running |
| `--no-parallel` | — | Disable parallel fetching and analysis |
| `--verbose` | — | Verbose output |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Docker

```bash
docker build -t teuth-analyzer .
docker run --rm -v $(pwd)/output:/app/output teuth-analyzer \
    "https://pulpito.ceph.com/your-run/" -o output \
    --ollama-url http://host.docker.internal:11434
```

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, pipeline orchestration |
| `fetcher.py` | Log fetching from Pulpito/qa-proxy |
| `log_parser.py` | Log preprocessing and error extraction |
| `cluster.py` | Failure clustering and run health |
| `analyzer.py` | Ollama LLM analysis with two-pass accuracy |
| `report_generator.py` | Interactive HTML report generation |
| `tests/` | Unit tests for parser, cluster, analyzer |
