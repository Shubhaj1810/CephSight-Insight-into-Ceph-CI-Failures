# CephSight — AI-Powered Teuthology CI Failure Analyzer

Automated tool that analyzes failed Ceph Teuthology CI jobs using a local LLM and generates interactive reports with root cause, severity, failure type, and fix suggestions.

## Model

- **LLM**: `qwen3.5` (9B) via [Ollama](https://ollama.com/) (self-hosted, free, no data leaves infra)
- **Why**: Fits in ~6.6 GB VRAM, strong JSON output, good instruction following, fast inference (~15s/job)
- **Alternatives**: Pass `--model llama3.1:70b` or `--model qwen3.5:27b` if you have a larger GPU

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CephSight                          │
├──────────────────────┬──────────────────────────────────┤
│     Frontend         │           Backend                │
│  (frontend/)         │         (backend/)               │
│                      │                                  │
│  index.html          │  app.py ─── Flask REST API       │
│  - URL input form    │  fetcher.py ── Log fetching      │
│  - Live progress     │  log_parser.py ── Error extract  │
│  - Interactive       │  cluster.py ── Failure grouping  │
│  - Filter / Sort     │  analyzer.py ── LLM analysis     │
│  - JSON/CSV export   │  main.py ── CLI entry point      │
│                      │  report_generator.py ── HTML CLI  │
└──────────────────────┴──────────────────────────────────┘
```

## Project Structure

```
teutology_analysis/
├── backend/
│   ├── app.py                # Flask API server (web mode)
│   ├── main.py               # CLI entry point (command-line mode)
│   ├── fetcher.py            # Log fetching from Pulpito / qa-proxy
│   ├── log_parser.py         # Extract tracebacks, errors, condense logs
│   ├── cluster.py            # Group similar failures, run health
│   ├── analyzer.py           # Ollama LLM analysis + heuristic checks + cache
│   ├── report_generator.py   # Self-contained HTML report (CLI mode)
│   └── requirements.txt      # Python dependencies
├── frontend/
│   └── index.html            # Web UI (single-page app)
└── README.md
```

## How to Run

### Option A: Web UI (Frontend + Backend)

#### 1. Copy files to the IBM Cloud VM

```bash
scp -i ~/Downloads/ibmcloud_generated.rsa -r \
    backend/ frontend/ \
    root@128.168.142.78:/root/CephSight/
```

#### 2. SSH into the VM

```bash
ssh -i ~/Downloads/ibmcloud_generated.rsa root@128.168.142.78
```

#### 3. Install dependencies and pull model

```bash
cd /root/CephSight/backend
pip install -r requirements.txt
ollama pull qwen3.5
```

#### 4. Start the backend server

```bash
cd /root/CephSight/backend
python3 app.py
```

Server starts on `http://0.0.0.0:5000`. Open in browser and paste a Pulpito URL to analyze.

#### 5. Access from local machine

From your local terminal, set up an SSH tunnel:

```bash
ssh -i ~/Downloads/ibmcloud_generated.rsa -L 5000:localhost:5000 root@128.168.142.78
```

Then open `http://localhost:5000` in your browser.

---

### Option B: CLI Mode (command-line only)

#### 1. Copy files to the VM

```bash
scp -i ~/Downloads/ibmcloud_generated.rsa -r \
    backend/ \
    root@128.168.142.78:/root/CephSight/
```

#### 2. SSH into the VM

```bash
ssh -i ~/Downloads/ibmcloud_generated.rsa root@128.168.142.78
```

#### 3. Run analysis

```bash
cd /root/CephSight/backend
pip install -r requirements.txt
python3 main.py "https://pulpito.ceph.com/<your-run-url>/" -o output
```

#### 4. Copy report back locally

```bash
exit
scp -i ~/Downloads/ibmcloud_generated.rsa \
    root@128.168.142.78:/root/CephSight/backend/output/report.html \
    ~/cephsight_report.html
```

#### 5. Open the report

```bash
xdg-open ~/cephsight_report.html
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen3.5` | Ollama model name |
| `--ollama-url` | `localhost:11434` | Ollama API URL |
| `--timeout` | `60` | Log fetch timeout (seconds) |
| `--llm-timeout` | `300` | LLM request timeout (seconds) |
| `--num-ctx` | `0` (auto) | Override model context window size |
| `--skip-fetch` | — | Load logs from disk instead of fetching |
| `--jobs` | all | Comma-separated job IDs to analyze |
| `--no-cache` | — | Disable analysis caching |
| `--clear-cache` | — | Wipe cache before running |
| `--no-parallel` | — | Disable parallel fetching/analysis |
| `--verbose` | — | Verbose output |

## API Endpoints (Web Mode)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Submit `{"url": "...", "model": "qwen3.5"}`, returns `task_id` |
| `GET` | `/api/status/<task_id>` | Check analysis progress |
| `GET` | `/api/results/<task_id>` | Fetch full results (when complete) |
| `GET` | `/api/health` | Backend health check |
