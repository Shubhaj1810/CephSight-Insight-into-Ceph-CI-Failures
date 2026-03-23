#!/usr/bin/env python3
"""
Teuthology AI Log Analyzer — CLI entry point.

Usage:
    python3 main.py <URL1> [URL2 URL3 ...] [options]

Supports multiple Pulpito run URLs — generates a separate report for each.

Full pipeline:  Fetch logs → Parse / extract errors → Analyze with Ollama → HTML report
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

from fetcher import (
    fetch_failed_logs, load_logs_from_disk, save_logs_to_disk,
    is_suite_listing_url, extract_run_urls_from_suite_page, make_session,
    validate_url, FetchResult, JobLog,
)
from log_parser import ParsedLog, parse_log
from analyzer import AnalysisResult, OllamaAnalyzer
from cluster import cluster_parsed_logs, classify_run_health, FailureCluster, RunHealth
from report_generator import generate_html_report

log = logging.getLogger(__name__)

# Maximum parallel LLM analysis threads.
MAX_ANALYSIS_WORKERS = 4

# Flag set by SIGINT handler for graceful shutdown
_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    if _interrupted:
        print("\n[!] Forced exit.")
        sys.exit(1)
    _interrupted = True
    print("\n[!] Interrupt received — finishing current job then stopping...")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Teuthology AI Log Analyzer — fetch, analyze, and report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single URL
  python3 main.py "https://pulpito.ceph.com/run-name-trial/" -o output

  # Multiple URLs — one report per URL
  python3 main.py "https://pulpito.ceph.com/run1-trial/" "https://pulpito.ceph.com/run2-trial/" -o output

  # Skip fetching, reuse downloaded logs
  python3 main.py URL --skip-fetch --logs-dir output/run-name/logs -o output

  # Use a different model (cache is model-aware — old results won't interfere)
  python3 main.py URL --model mistral -o output

  # Wipe cache and re-analyze everything with a new model
  python3 main.py URL --model llama3.1:70b --clear-cache -o output

  # Override the context window size (e.g. for models with large context)
  python3 main.py URL --model llama3.1:70b --num-ctx 32768 -o output

  # Suite page (multi-user) — combined into one report
  python3 main.py "https://pulpito.ceph.com/?suite=orch:cephadm" --combined -o output
""",
    )
    ap.add_argument(
        "run_urls",
        nargs="+",
        help="One or more Pulpito run URLs.",
    )
    ap.add_argument(
        "-o", "--output",
        default="output",
        help="Output base directory (default: output/).",
    )
    ap.add_argument(
        "--model",
        default="llama3:8b",
        help="Ollama model name (default: llama3:8b).",
    )
    ap.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout for log fetching in seconds (default: 60).",
    )
    ap.add_argument(
        "--llm-timeout",
        type=int,
        default=300,
        help="Timeout for each LLM request in seconds (default: 300).",
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=0,
        help=(
            "Override the model's context window size (num_ctx). "
            "0 = use model default. Larger values let the model see more "
            "log text but need more VRAM (e.g. 8192, 16384, 32768)."
        ),
    )
    ap.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching; load logs from --logs-dir instead (single URL only).",
    )
    ap.add_argument(
        "--logs-dir",
        default=None,
        help="Directory with previously-fetched .log files (used with --skip-fetch).",
    )
    ap.add_argument(
        "--jobs",
        default=None,
        help="Comma-separated list of job IDs to analyze (default: all failed).",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable analysis caching (re-analyze everything).",
    )
    ap.add_argument(
        "--clear-cache",
        action="store_true",
        help=(
            "Delete all cached analysis results before running. "
            "Useful when switching models or prompts to force re-analysis."
        ),
    )
    ap.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel log fetching and analysis.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Combine all runs (from a suite/multi-user URL) into a single "
            "unified report instead of one report per run."
        ),
    )
    return ap.parse_args()


def _stage_banner(name: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  STAGE: {name}")
    print("=" * width)
    print()


def _slug_from_url(run_url: str) -> str:
    """Extract a short directory name from a Pulpito URL."""
    path = urlparse(run_url).path.strip("/")
    slug = path.split("/")[-1] if "/" in path else path
    slug = re.sub(r"[^a-zA-Z0-9_\-.]", "_", slug)
    if len(slug) > 80:
        slug = slug[:80]
    return slug or "run"


def _user_from_url(run_url: str) -> str:
    """Extract the user name from a Pulpito run URL slug."""
    slug = _slug_from_url(run_url)
    m = re.match(r"^([a-zA-Z][\w.-]*?)-\d{4}-\d{2}-\d{2}_", slug)
    return m.group(1) if m else ""


@dataclass
class RunResult:
    """Intermediate data produced by processing a single run."""
    report_path: Optional[str] = None
    analyses: List[AnalysisResult] = field(default_factory=list)
    parsed_logs: List[ParsedLog] = field(default_factory=list)
    clusters: List[FailureCluster] = field(default_factory=list)
    cluster_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    run_health: Optional[RunHealth] = None
    executive_summary: str = ""


def _process_single_run(
    run_url: str,
    args: argparse.Namespace,
    run_output_dir: str,
    cache_dir: str,
    skip_report: bool = False,
) -> Optional[RunResult]:
    """
    Process a single Pulpito run URL through the full pipeline.
    Returns a RunResult with analyses, parsed logs, and report path.
    """
    global _interrupted

    os.makedirs(run_output_dir, exist_ok=True)
    logs_dir = os.path.join(run_output_dir, "logs")
    report_path = os.path.join(run_output_dir, "report.html")

    job_ids: Optional[List[str]] = None
    if args.jobs:
        job_ids = [j.strip() for j in args.jobs.split(",") if j.strip()]

    # ------------------------------------------------------------------
    # STAGE 1: Fetch logs
    # ------------------------------------------------------------------
    _stage_banner("FETCH LOGS")

    total_jobs_on_page = 0
    failed_jobs_on_page = 0

    if args.skip_fetch:
        source_dir = args.logs_dir or logs_dir
        print(f"[skip-fetch] Loading logs from: {source_dir}")
        disk_logs = load_logs_from_disk(source_dir)
        if job_ids:
            job_ids_set = set(job_ids)
            disk_logs = [jl for jl in disk_logs if jl.job_id in job_ids_set]
        if not disk_logs:
            print("ERROR: No .log files found. Check --logs-dir path.")
            return None
        job_logs = disk_logs
        total_jobs_on_page = len(job_logs)
        failed_jobs_on_page = len(job_logs)
        print(f"[skip-fetch] Loaded {len(job_logs)} log file(s).\n")
    else:
        fetch_result = fetch_failed_logs(
            run_url=run_url,
            timeout=args.timeout,
            verbose=args.verbose,
            job_ids=job_ids,
            parallel=not args.no_parallel,
        )
        job_logs = fetch_result.job_logs
        total_jobs_on_page = fetch_result.total_jobs_on_page
        failed_jobs_on_page = fetch_result.failed_jobs_on_page

        if not job_logs:
            print("ERROR: No failed jobs found or fetching failed completely.")
            return None

        idx = save_logs_to_disk(job_logs, logs_dir)
        print(f"[fetch] Logs saved to {logs_dir}/  (index: {idx})")

    if _interrupted:
        print("[!] Interrupted during fetch — skipping remaining stages.")
        return None

    ok_logs = [jl for jl in job_logs if jl.status == "ok" and jl.log_text]
    failed_fetches = [jl for jl in job_logs if jl.status != "ok" or not jl.log_text]
    failed_fetches_by_id = {jl.job_id: jl for jl in failed_fetches}
    print(f"\n[fetch] {len(ok_logs)} of {len(job_logs)} logs fetched successfully.")
    if total_jobs_on_page > len(job_logs):
        print(f"[fetch] Total jobs on page: {total_jobs_on_page} "
              f"({len(job_logs)} failed, {total_jobs_on_page - len(job_logs)} passed)")
    if failed_fetches:
        failed_ids = ", ".join(jl.job_id for jl in failed_fetches)
        print(f"[fetch] WARNING: {len(failed_fetches)} job(s) could not be fetched: {failed_ids}")
    if not ok_logs:
        print("ERROR: All log fetches failed. Cannot continue.")
        return None

    # ------------------------------------------------------------------
    # STAGE 2: Parse / preprocess logs
    # ------------------------------------------------------------------
    _stage_banner("PARSE & EXTRACT")

    parsed_logs: List[ParsedLog] = []
    for jl in ok_logs:
        print(
            f"  Parsing job {jl.job_id} ({len(jl.log_text):,} chars) "
            f"[{jl.log_name}]..."
        )
        pl = parse_log(jl.job_id, jl.log_text)
        parsed_logs.append(pl)
        print(
            f"    -> {len(pl.tracebacks)} traceback(s), "
            f"{len(pl.error_blocks)} error block(s), "
            f"condensed to {len(pl.condensed_text):,} chars"
        )

    # Set raw_log_path and release raw log from memory to reduce footprint
    for pl in parsed_logs:
        log_file = os.path.join(logs_dir, f"{pl.job_id}.log")
        if os.path.exists(log_file):
            pl.raw_log_path = log_file
            pl.release_raw_log()

    # Also release log_text from JobLog objects to free memory
    for jl in ok_logs:
        jl.log_text = None

    for jl in failed_fetches:
        fallback_raw = (
            "=== ARTIFACT FETCH FAILURE ===\n"
            f"job_id: {jl.job_id}\n"
            f"status: {jl.status}\n"
            f"primary_url: {jl.source_url}\n"
            f"attempted_log: {jl.log_name}\n"
            f"fetch_error: {jl.error_message or 'artifact unavailable'}\n"
            "note: teuthology and ansible logs could not be downloaded.\n"
            "classification_hint: infrastructure/artifact availability issue.\n"
        )
        pl = parse_log(jl.job_id, fallback_raw)
        parsed_logs.append(pl)
        print(
            f"  Parsing job {jl.job_id} (fallback metadata)... "
            f"condensed to {len(pl.condensed_text):,} chars"
        )

    print(f"\n[parse] Parsed {len(parsed_logs)} log(s).")

    if _interrupted:
        print("[!] Interrupted during parse — skipping analysis.")
        return None

    # ------------------------------------------------------------------
    # STAGE 2.5: RUN HEALTH & FAILURE CLUSTERING
    # ------------------------------------------------------------------
    _stage_banner("CLUSTER & TRIAGE")

    llm_parsed: List[ParsedLog] = []
    fallback_analyses: List[AnalysisResult] = []
    for pl in parsed_logs:
        fallback_job = failed_fetches_by_id.get(pl.job_id)
        if fallback_job is not None:
            result = AnalysisResult(
                job_id=pl.job_id,
                root_cause=(
                    "Log artifact unavailable from qa-proxy "
                    f"({fallback_job.log_name})"
                ),
                severity="medium",
                error_category="artifact availability issue",
                failure_type="infra",
                confidence=1.0,
                explanation=(
                    "Teuthology/ansible logs could not be fetched for this job. "
                    "This is classified as infrastructure/artifact retention issue "
                    "until the original logs are restored."
                ),
                fix_suggestions=[
                    "Verify qa-proxy artifact retention for this run/job.",
                    "Retry artifact fetch later or regenerate archived logs.",
                ],
                recommended_action=(
                    "Recover the missing artifact, then rerun deep analysis."
                ),
                affected_components=["qa-proxy", "artifact storage"],
                raw_llm_response=fallback_job.error_message,
                success=True,
            )
            fallback_analyses.append(result)
            print(
                f"  [fallback] job {pl.job_id}: artifact unavailable"
            )
        else:
            llm_parsed.append(pl)

    # --- Run-health classification (now with real total from the page) ---
    effective_total = max(total_jobs_on_page, len(job_logs))
    run_health = classify_run_health(
        total_jobs=effective_total,
        failed_jobs=failed_jobs_on_page,
    )
    print(f"  Run health : {run_health.classification} ({run_health.pct} failure ratio)")
    print(f"  Hint       : {run_health.hint_for_llm[:120]}...")
    print()

    clusters, job_to_cluster = cluster_parsed_logs(llm_parsed)
    print(f"  Clusters   : {len(clusters)} distinct failure pattern(s)")
    print(f"  Jobs       : {len(llm_parsed)} (to analyze)")
    print()

    for ci, cluster in enumerate(clusters, start=1):
        sig_short = cluster.signature[:80] + ("..." if len(cluster.signature) > 80 else "")
        print(
            f"    Cluster {ci}: {cluster.size:3d} job(s) | "
            f"rep={cluster.representative_job_id} | {sig_short}"
        )

    llm_calls_saved = len(llm_parsed) - len(clusters)
    if llm_calls_saved > 0:
        print(
            f"\n  [cluster] {llm_calls_saved} LLM call(s) saved by clustering "
            f"({len(llm_parsed)} jobs → {len(clusters)} cluster analyses)"
        )

    # ------------------------------------------------------------------
    # STAGE 3: AI analysis (cluster-aware, parallel)
    # ------------------------------------------------------------------
    _stage_banner("AI ANALYSIS")

    print(f"  Backend : Ollama")
    print(f"  Model   : {args.model}")
    print(f"  Endpoint: {args.ollama_url}")
    print(f"  Timeout : {args.llm_timeout}s per job")
    if args.num_ctx > 0:
        print(f"  Context : {args.num_ctx} tokens (num_ctx override)")
    print(f"  Cache   : {'disabled' if args.no_cache else cache_dir}")
    print(f"  Clusters: {len(clusters)} to analyze (covering {len(llm_parsed)} jobs)")
    print()

    llm = OllamaAnalyzer(
        model=args.model,
        base_url=args.ollama_url,
        timeout=args.llm_timeout,
        num_ctx=args.num_ctx,
        cache_dir=cache_dir,
        cache_enabled=not args.no_cache,
    )

    if args.clear_cache:
        removed = llm.clear_cache()
        print(f"[cache] Cleared {removed} cached analysis file(s).\n")

    ready, ready_msg = llm.check_ready()
    if not ready:
        print(f"ERROR: {ready_msg}")
        print(
            "Start Ollama with `ollama serve` "
            f"and ensure model is pulled: `ollama pull {args.model}`"
        )
        return None

    cluster_results: Dict[str, AnalysisResult] = {}
    analyses: List[AnalysisResult] = list(fallback_analyses)
    cached_count = 0
    cluster_total = len(clusters)

    # Parallel cluster analysis when not disabled
    if not args.no_parallel and cluster_total > 1:
        workers = min(MAX_ANALYSIS_WORKERS, cluster_total)
        print(f"  [parallel] Analyzing {cluster_total} clusters with {workers} workers\n")

        def _analyze_one(ci_cluster):
            ci, cluster = ci_cluster
            rep = cluster.representative_parsed
            assert rep is not None
            t0 = time.time()
            result = llm.analyze_cluster(
                representative_job_id=cluster.representative_job_id,
                condensed_text=rep.condensed_text,
                cluster_size=cluster.size,
                signature=cluster.signature,
                job_ids=cluster.job_ids,
                run_health_hint=run_health.hint_for_llm,
            )
            elapsed = time.time() - t0
            return ci, cluster, result, elapsed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_analyze_one, (ci, cluster)): ci
                for ci, cluster in enumerate(clusters, start=1)
            }
            for fut in as_completed(futures):
                if _interrupted:
                    break
                ci, cluster, result, elapsed = fut.result()
                cluster_results[cluster.cluster_id] = result
                if result.cached:
                    cached_count += 1
                    print(
                        f"  [{ci}/{cluster_total}] Cluster {cluster.cluster_id[:8]} "
                        f"({cluster.size} job(s)) CACHED | {result.severity.upper()} | "
                        f"{result.failure_type} | {result.confidence:.0%}"
                    )
                elif result.success:
                    print(
                        f"  [{ci}/{cluster_total}] Cluster {cluster.cluster_id[:8]} "
                        f"({cluster.size} job(s)) {result.severity.upper()} | "
                        f"{result.failure_type} | {result.confidence:.0%} | "
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(
                        f"  [{ci}/{cluster_total}] Cluster {cluster.cluster_id[:8]} "
                        f"FAILED: {result.error_message} | {elapsed:.1f}s"
                    )
    else:
        for ci, cluster in enumerate(clusters, start=1):
            if _interrupted:
                print("[!] Interrupted — stopping analysis.")
                break

            rep = cluster.representative_parsed
            assert rep is not None

            print(
                f"  [{ci}/{cluster_total}] Cluster {cluster.cluster_id[:8]} "
                f"({cluster.size} job(s), rep={cluster.representative_job_id})...",
                end=" ", flush=True,
            )

            t0 = time.time()
            result = llm.analyze_cluster(
                representative_job_id=cluster.representative_job_id,
                condensed_text=rep.condensed_text,
                cluster_size=cluster.size,
                signature=cluster.signature,
                job_ids=cluster.job_ids,
                run_health_hint=run_health.hint_for_llm,
            )
            elapsed = time.time() - t0

            cluster_results[cluster.cluster_id] = result

            if result.cached:
                cached_count += 1
                print(
                    f"CACHED | {result.severity.upper()} | "
                    f"{result.failure_type} | {result.confidence:.0%}"
                )
            elif result.success:
                print(
                    f"{result.severity.upper()} | "
                    f"{result.failure_type} | {result.confidence:.0%} | "
                    f"{elapsed:.1f}s"
                )
            else:
                print(f"FAILED: {result.error_message} | {elapsed:.1f}s")

    # Fan-out: create an AnalysisResult for every job in every cluster
    for cluster in clusters:
        base_result = cluster_results.get(cluster.cluster_id)
        if base_result is None:
            continue

        for jid in cluster.job_ids:
            job_result = AnalysisResult(
                job_id=jid,
                root_cause=base_result.root_cause,
                severity=base_result.severity,
                error_category=base_result.error_category,
                failure_type=base_result.failure_type,
                confidence=base_result.confidence,
                explanation=base_result.explanation,
                fix_suggestions=list(base_result.fix_suggestions),
                recommended_action=base_result.recommended_action,
                affected_components=list(base_result.affected_components),
                raw_llm_response=base_result.raw_llm_response,
                success=base_result.success,
                error_message=base_result.error_message,
                cached=base_result.cached,
            )
            analyses.append(job_result)

    fallback_count = len(fallback_analyses)
    print(
        f"\n[analysis] {len(analyses)} total job results "
        f"({len(clusters)} cluster analyses, {cached_count} cached, "
        f"{fallback_count} fallback)."
    )
    if llm_calls_saved > 0:
        print(
            f"[analysis] Clustering saved {llm_calls_saved} LLM call(s)."
        )

    # Executive summary
    successful = [a for a in analyses if a.success]
    artifact_unavail = [
        a for a in analyses
        if "artifact availability" in (a.error_category or "").lower()
    ]

    exec_summary = ""
    summary_results = [
        r for r in cluster_results.values() if r and r.success
    ]
    if summary_results and not _interrupted:
        print("\n  Generating executive summary...", flush=True)
        t0 = time.time()

        rh_context = (
            f"Run health: {run_health.classification.upper()} — "
            f"{run_health.failed_jobs}/{run_health.total_jobs} jobs failed "
            f"({run_health.pct}).\n{run_health.hint_for_llm}"
        )

        exec_summary = llm.generate_executive_summary(
            results=[r for r in cluster_results.values() if r and r.success],
            artifact_unavailable_count=len(artifact_unavail),
            run_health_context=rh_context,
            cluster_count=len(clusters),
        )
        print(f"    done in {time.time() - t0:.1f}s")

    result = RunResult(
        analyses=analyses,
        parsed_logs=parsed_logs,
        clusters=clusters,
        cluster_results=cluster_results,
        run_health=run_health,
        executive_summary=exec_summary,
    )

    if skip_report:
        return result

    # ------------------------------------------------------------------
    # STAGE 4: Generate HTML report
    # ------------------------------------------------------------------
    _stage_banner("GENERATE REPORT")

    rpath = generate_html_report(
        run_url=run_url,
        analyses=analyses,
        parsed_logs=parsed_logs,
        executive_summary=exec_summary,
        model_name=args.model,
        output_path=report_path,
        clusters=clusters,
        cluster_results=cluster_results,
        run_health=run_health,
    )

    print(f"  Report written to: {rpath}")
    result.report_path = rpath
    return result


def _process_combined_runs(
    run_urls: List[str],
    url_to_user: Dict[str, str],
    args: argparse.Namespace,
    cache_dir: str,
    suite_source_url: str,
) -> Optional[str]:
    """
    Process multiple runs and merge all results into one combined report.
    Each job is tagged with its run's user name and run slug.
    """
    global _interrupted

    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "report.html")

    all_analyses: List[AnalysisResult] = []
    all_parsed_logs: List[ParsedLog] = []
    all_clusters: List[FailureCluster] = []
    all_cluster_results: Dict[str, AnalysisResult] = {}
    processed_run_urls: List[str] = []

    for idx, run_url in enumerate(run_urls, start=1):
        if _interrupted:
            print("[!] Interrupted — skipping remaining runs.")
            break

        user = url_to_user.get(run_url, _user_from_url(run_url))
        run_slug = _slug_from_url(run_url)

        print(f"\n{'#' * 60}")
        print(f"  RUN {idx} of {len(run_urls)} [user: {user or 'unknown'}]")
        print(f"  {run_url}")
        print(f"{'#' * 60}")

        run_output_dir = os.path.join(args.output, run_slug)
        run_result = _process_single_run(
            run_url, args, run_output_dir, cache_dir, skip_report=True,
        )
        if not run_result:
            continue

        processed_run_urls.append(run_url)

        for a in run_result.analyses:
            a.run_user = user
            a.run_name = run_slug
            all_analyses.append(a)

        all_parsed_logs.extend(run_result.parsed_logs)
        all_clusters.extend(run_result.clusters)
        all_cluster_results.update(run_result.cluster_results)

    if not all_analyses:
        print("ERROR: No analyses to combine.")
        return None

    _stage_banner("GENERATE COMBINED REPORT")
    print(f"  Combining {len(all_analyses)} job analyses from {len(processed_run_urls)} runs")

    all_users = sorted({a.run_user for a in all_analyses if a.run_user})
    print(f"  Users: {', '.join(all_users) if all_users else 'unknown'}")

    rpath = generate_html_report(
        run_url=suite_source_url,
        analyses=all_analyses,
        parsed_logs=all_parsed_logs,
        executive_summary="",
        model_name=args.model,
        output_path=report_path,
        clusters=all_clusters,
        cluster_results=all_cluster_results,
        run_health=None,
        all_run_urls=processed_run_urls,
    )

    print(f"  Combined report written to: {rpath}")
    return rpath


def main() -> int:
    signal.signal(signal.SIGINT, _sigint_handler)

    args = _parse_args()

    # Setup structured logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output, exist_ok=True)

    cache_dir = os.path.join(args.output, ".analysis_cache")

    # Validate all URLs upfront
    for url in args.run_urls:
        ok, msg = validate_url(url)
        if not ok:
            print(f"ERROR: Invalid URL '{url}': {msg}")
            return 1

    # Expand suite listing URLs into individual run URLs
    expanded_urls: List[str] = []
    url_to_user: Dict[str, str] = {}
    suite_source_url: str = ""
    session = make_session(retries=3)

    for url in args.run_urls:
        if is_suite_listing_url(url):
            suite_source_url = suite_source_url or url
            print(f"\n{'#' * 60}")
            print(f"  SUITE PAGE DETECTED")
            print(f"  {url}")
            print(f"{'#' * 60}\n")
            run_entries = extract_run_urls_from_suite_page(
                url, session=session, timeout=args.timeout
            )
            for run_url, fail_count, user in run_entries:
                expanded_urls.append(run_url)
                if user:
                    url_to_user[run_url] = user
        else:
            expanded_urls.append(url)
            user = _user_from_url(url)
            if user:
                url_to_user[url] = user

    total_urls = len(expanded_urls)
    if total_urls == 0:
        print("ERROR: No run URLs found to process.")
        return 1

    is_combined = args.combined and total_urls > 1

    print(f"\n[main] Total runs to process: {total_urls}")
    if is_combined:
        print(f"[main] Mode: COMBINED (all runs merged into one report)")
    print(f"[main] Backend: Ollama")
    print(f"[main] Model: {args.model}")
    if args.num_ctx > 0:
        print(f"[main] Context window override: {args.num_ctx} tokens")
    if args.clear_cache:
        print("[main] Cache will be cleared before analysis")
    print()

    pipeline_t0 = time.time()
    reports: List[str] = []

    if is_combined:
        rpath = _process_combined_runs(
            expanded_urls, url_to_user, args, cache_dir,
            suite_source_url or expanded_urls[0],
        )
        if rpath:
            reports.append(rpath)
    else:
        for idx, run_url in enumerate(expanded_urls, start=1):
            if _interrupted:
                print("[!] Interrupted — skipping remaining runs.")
                break

            if total_urls > 1:
                print(f"\n{'#' * 60}")
                print(f"  RUN {idx} of {total_urls}")
                print(f"  {run_url}")
                print(f"{'#' * 60}")

            if total_urls > 1:
                slug = _slug_from_url(run_url)
                run_output_dir = os.path.join(args.output, slug)
            else:
                run_output_dir = args.output

            run_result = _process_single_run(
                run_url, args, run_output_dir, cache_dir,
            )
            if run_result and run_result.report_path:
                reports.append(run_result.report_path)

    pipeline_elapsed = time.time() - pipeline_t0

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE  ({pipeline_elapsed:.1f}s total)")
    print(f"  {len(reports)} of {total_urls if not is_combined else 1} reports generated:")
    for rp in reports:
        print(f"    - {rp}")
    if _interrupted:
        skipped = total_urls - len(reports)
        if skipped > 0:
            print(f"  ({skipped} run(s) skipped due to interrupt)")
    print(f"{'=' * 60}\n")

    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
