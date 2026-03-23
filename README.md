# CephSight — AI-Powered Teuthology CI Failure Analyzer

Automated tool that analyzes failed Ceph Teuthology CI jobs using a local LLM and generates an interactive HTML report with root cause, severity, failure type, and fix suggestions.

## Model

- **LLM**: `llama3:8b` via [Ollama](https://ollama.com/) (self-hosted, free, no data leaves infra)
- **Why**: Fits in ~5 GB VRAM, fast inference (~15s/job), and the tool's heuristic layer compensates for the small model's limitations
- **Alternatives**: Pass `--model llama3.1:70b` or `--model mistral` if you have a larger GPU

## Pipeline

```
Pulpito URL → Fetch Logs → Parse & Extract → Cluster Failures → LLM Analysis → HTML Report
(fetcher.py)   (log_parser.py)  (cluster.py)    (analyzer.py)   (report_generator.py)
```

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, pipeline orchestration |
| `fetcher.py` | Log fetching from Pulpito/qa-proxy (parallel) |
| `log_parser.py` | Extract tracebacks, errors, summary; condense to ≤12K chars |
| `cluster.py` | Group similar failures by error fingerprint + run health |
| `analyzer.py` | Ollama LLM analysis + 18 regex heuristic cross-checks + cache |
| `report_generator.py` | Self-contained interactive HTML report |

## How to Run (IBM Cloud VM)

The analysis runs on an IBM Cloud VM where Ollama + GPU are available.

### 1. Copy files to the VM

```bash
scp -i ~/Downloads/ibmcloud_generated.rsa \
    *.py requirements.txt \
    root@128.168.142.78:/root/AI_teutology/
```

### 2. SSH into the VM

```bash
ssh -i ~/Downloads/ibmcloud_generated.rsa root@128.168.142.78
```

### 3. Run the analysis

```bash
cd /root/AI_teutology
pip install -r requirements.txt
python3 main.py "https://pulpito.ceph.com/<your-run-url>/" -o output
```

### 4. Exit the VM

```bash
exit
```

### 5. Copy the report back locally

```bash
scp -i ~/Downloads/ibmcloud_generated.rsa \
    root@128.168.142.78:/root/AI_teutology/output/report.html \
    ~/AI_teutology_report.html
```

### 6. Open the report

```bash
xdg-open ~/AI_teutology_report.html
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llama3:8b` | Ollama model name |
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
