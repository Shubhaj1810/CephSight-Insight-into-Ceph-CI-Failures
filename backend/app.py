#!/usr/bin/env python3
"""
CephSight Backend — Flask API server.

Exposes REST endpoints for the frontend to submit Pulpito URLs,
track analysis progress, and retrieve results.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from fetcher import (
    fetch_failed_logs, save_logs_to_disk, validate_url,
    is_suite_listing_url, extract_run_urls_from_suite_page, make_session,
)
from log_parser import ParsedLog, parse_log
from analyzer import AnalysisResult, OllamaAnalyzer
from cluster import cluster_parsed_logs, classify_run_health

log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = os.environ.get("CEPHSIGHT_OUTPUT", "output")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("CEPHSIGHT_MODEL", "cephsight")

tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


def _new_task(run_url: str, model: str) -> str:
    task_id = uuid.uuid4().hex[:12]
    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "run_url": run_url,
            "model": model,
            "status": "queued",
            "stage": "",
            "progress": "",
            "error": "",
            "results": None,
            "created_at": time.time(),
        }
    return task_id


def _update_task(task_id: str, **kwargs):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)


def _run_analysis(task_id: str, run_url: str, model: str):
    try:
        _update_task(task_id, status="running", stage="Fetching logs")

        url_ok, url_msg = validate_url(run_url)
        if not url_ok:
            _update_task(task_id, status="failed", error=f"Invalid URL: {url_msg}")
            return

        fetch_result = fetch_failed_logs(
            run_url=run_url, timeout=60, verbose=False, parallel=True,
        )
        job_logs = fetch_result.job_logs

        if not job_logs:
            _update_task(task_id, status="failed", error="No failed jobs found on the run page.")
            return

        task_output = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_output, exist_ok=True)
        logs_dir = os.path.join(task_output, "logs")
        save_logs_to_disk(job_logs, logs_dir)

        ok_logs = [jl for jl in job_logs if jl.status == "ok" and jl.log_text]
        failed_fetches = [jl for jl in job_logs if jl.status != "ok" or not jl.log_text]

        _update_task(
            task_id, stage="Parsing logs",
            progress=f"{len(ok_logs)} logs fetched, {len(failed_fetches)} failed",
        )

        parsed_logs: List[ParsedLog] = []
        for jl in ok_logs:
            pl = parse_log(jl.job_id, jl.log_text)
            parsed_logs.append(pl)
            jl.log_text = None

        fallback_analyses: List[dict] = []
        failed_fetches_by_id = {jl.job_id: jl for jl in failed_fetches}
        for jl in failed_fetches:
            fallback_raw = (
                f"=== ARTIFACT FETCH FAILURE ===\njob_id: {jl.job_id}\n"
                f"status: {jl.status}\nfetch_error: {jl.error_message or 'artifact unavailable'}\n"
            )
            pl = parse_log(jl.job_id, fallback_raw)
            parsed_logs.append(pl)
            fallback_analyses.append({
                "job_id": jl.job_id,
                "root_cause": f"Log artifact unavailable from qa-proxy ({jl.log_name})",
                "severity": "medium",
                "error_category": "artifact availability issue",
                "failure_type": "infra",
                "confidence": 1.0,
                "explanation": "Logs could not be fetched. Classified as infrastructure issue.",
                "fix_suggestions": ["Verify qa-proxy artifact retention."],
                "recommended_action": "Recover missing artifact, then rerun analysis.",
                "affected_components": ["qa-proxy", "artifact storage"],
            })

        _update_task(task_id, stage="Clustering failures")

        effective_total = max(fetch_result.total_jobs_on_page, len(job_logs))
        run_health = classify_run_health(
            total_jobs=effective_total,
            failed_jobs=fetch_result.failed_jobs_on_page,
        )

        llm_parsed = [
            pl for pl in parsed_logs if pl.job_id not in failed_fetches_by_id
        ]
        clusters, job_to_cluster = cluster_parsed_logs(llm_parsed)

        _update_task(
            task_id, stage="AI Analysis",
            progress=f"{len(clusters)} clusters from {len(llm_parsed)} jobs",
        )

        cache_dir = os.path.join(OUTPUT_DIR, ".analysis_cache")
        llm = OllamaAnalyzer(
            model=model, base_url=OLLAMA_URL,
            timeout=300, cache_dir=cache_dir,
        )

        ready, ready_msg = llm.check_ready()
        if not ready:
            _update_task(task_id, status="failed", error=ready_msg)
            return

        cluster_results: Dict[str, dict] = {}
        all_analyses: List[dict] = list(fallback_analyses)

        for ci, cluster in enumerate(clusters, start=1):
            _update_task(
                task_id,
                progress=f"Analyzing cluster {ci}/{len(clusters)} ({cluster.size} jobs)",
            )
            rep = cluster.representative_parsed
            if rep is None:
                continue

            result = llm.analyze_cluster(
                representative_job_id=cluster.representative_job_id,
                condensed_text=rep.condensed_text,
                cluster_size=cluster.size,
                signature=cluster.signature,
                job_ids=cluster.job_ids,
                run_health_hint=run_health.hint_for_llm,
            )

            cr_dict = {
                "cluster_id": cluster.cluster_id,
                "signature": cluster.signature,
                "job_ids": cluster.job_ids,
                "size": cluster.size,
                "representative_job_id": cluster.representative_job_id,
                "root_cause": result.root_cause,
                "severity": result.severity,
                "error_category": result.error_category,
                "failure_type": result.failure_type,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "fix_suggestions": result.fix_suggestions,
                "recommended_action": result.recommended_action,
                "affected_components": result.affected_components,
                "cached": result.cached,
            }
            cluster_results[cluster.cluster_id] = cr_dict

            for jid in cluster.job_ids:
                all_analyses.append({
                    "job_id": jid,
                    "root_cause": result.root_cause,
                    "severity": result.severity,
                    "error_category": result.error_category,
                    "failure_type": result.failure_type,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "fix_suggestions": list(result.fix_suggestions),
                    "recommended_action": result.recommended_action,
                    "affected_components": list(result.affected_components),
                    "cached": result.cached,
                })

        _update_task(task_id, stage="Generating summary")

        exec_summary = ""
        summary_results = [r for r in cluster_results.values()]
        if summary_results:
            rh_context = (
                f"Run health: {run_health.classification.upper()} — "
                f"{run_health.failed_jobs}/{run_health.total_jobs} jobs failed "
                f"({run_health.pct}).\n{run_health.hint_for_llm}"
            )
            summary_analysis_results = []
            for cr in cluster_results.values():
                summary_analysis_results.append(AnalysisResult(
                    job_id=cr["representative_job_id"],
                    root_cause=cr["root_cause"],
                    severity=cr["severity"],
                    error_category=cr["error_category"],
                    failure_type=cr["failure_type"],
                    confidence=cr["confidence"],
                    recommended_action=cr["recommended_action"],
                ))
            exec_summary = llm.generate_executive_summary(
                results=summary_analysis_results,
                run_health_context=rh_context,
                cluster_count=len(clusters),
            )

        results = {
            "run_url": run_url,
            "model": model,
            "total_jobs": effective_total,
            "failed_jobs": fetch_result.failed_jobs_on_page,
            "fetched_ok": len(ok_logs),
            "fetched_failed": len(failed_fetches),
            "run_health": {
                "classification": run_health.classification,
                "failure_ratio": run_health.failure_ratio,
                "pct": run_health.pct,
                "total_jobs": run_health.total_jobs,
                "failed_jobs": run_health.failed_jobs,
                "hint": run_health.hint_for_llm,
            },
            "clusters": list(cluster_results.values()),
            "analyses": all_analyses,
            "executive_summary": exec_summary,
        }

        _update_task(task_id, status="complete", stage="Done", results=results)

    except Exception as exc:
        log.exception("Analysis failed for task %s", task_id)
        _update_task(task_id, status="failed", error=str(exc))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    run_url = data.get("url", "").strip()
    model = data.get("model", MODEL).strip()

    if not run_url:
        return jsonify({"error": "Missing 'url' field"}), 400

    ok, msg = validate_url(run_url)
    if not ok:
        return jsonify({"error": f"Invalid URL: {msg}"}), 400

    task_id = _new_task(run_url, model)
    thread = threading.Thread(target=_run_analysis, args=(task_id, run_url, model), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/status/<task_id>")
def status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "task_id": task["task_id"],
        "status": task["status"],
        "stage": task["stage"],
        "progress": task["progress"],
        "error": task["error"],
    })


@app.route("/api/results/<task_id>")
def results(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task["status"] != "complete":
        return jsonify({
            "task_id": task_id,
            "status": task["status"],
            "error": task.get("error", ""),
        })
    return jsonify(task["results"])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL, "ollama_url": OLLAMA_URL})


@app.route("/")
def serve_frontend():
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    return send_from_directory(frontend_dir, "index.html")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"CephSight Backend starting...")
    print(f"  Model  : {MODEL}")
    print(f"  Ollama : {OLLAMA_URL}")
    print(f"  Output : {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False)
