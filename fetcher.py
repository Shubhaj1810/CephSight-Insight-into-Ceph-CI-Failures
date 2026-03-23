#!/usr/bin/env python3
"""
Teuthology log fetcher — importable module.

Fetches teuthology.log files for failed jobs from a Pulpito run URL
via qa-proxy.ceph.com.  Can be used as a library (fetch_failed_logs())
or standalone (python3 fetcher.py <URL>).

Features:
- Discovers dead / error / hung jobs in addition to "fail".
- Downloads logs in parallel via ``concurrent.futures.ThreadPoolExecutor``.
- Thread-safe: each worker uses its own session.
- Returns total job count from the run page for accurate run-health.
- Basic URL validation and rate-limiting.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urljoin, parse_qs

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UA = "log-fetcher/2.0 (+https://pulpito.ceph.com)"

QA_PROXY_BASE_RE = re.compile(
    r"https?://qa-proxy\.ceph\.com/teuthology/([^/]+)/"
)
RUN_NAME_RE = re.compile(r"/([^/]+)/?$")  # last path segment

# Statuses that indicate a job worth analyzing
FAILED_STATUSES_RE = re.compile(r"\b(fail|dead|error|hung)\b", re.IGNORECASE)

MAX_DOWNLOAD_WORKERS = 8

# Minimum delay between requests to the same host (rate limiting)
_RATE_LIMIT_DELAY = 0.05  # 50 ms

# Thread-local storage for per-thread sessions
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class JobLog:
    """Result of fetching a single job's artifact log text."""
    job_id: str
    status: str                     # "ok" | "fetch_failed"
    log_text: Optional[str] = None  # full log content (None on failure)
    source_url: str = ""
    log_name: str = "teuthology.log"
    elapsed_s: float = 0.0
    error_message: str = ""


@dataclass
class FetchResult:
    """Aggregate result of fetching logs for a run."""
    job_logs: List[JobLog]
    total_jobs_on_page: int   # all jobs (pass + fail) discovered on run page
    failed_jobs_on_page: int  # jobs matching failed statuses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def make_session(retries: int = 3) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_cfg = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    session.mount("http://", HTTPAdapter(max_retries=retry_cfg))
    return session


def _get_thread_session(retries: int = 3) -> requests.Session:
    """Get or create a thread-local session (thread-safe)."""
    if not hasattr(_thread_local, "session") or _thread_local.session is None:
        _thread_local.session = make_session(retries)
    return _thread_local.session


def validate_url(url: str) -> Tuple[bool, str]:
    """Basic validation that the URL looks like a real HTTP(S) URL.
    Returns (ok, message)."""
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"Could not parse URL: {exc}"

    if parsed.scheme not in ("http", "https"):
        return False, f"URL scheme must be http or https, got: {parsed.scheme!r}"
    if not parsed.netloc:
        return False, "URL has no host/domain"
    return True, "ok"


def _fetch_url(url: str, session: requests.Session, timeout: int) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_run_name_from_url(run_url: str) -> str:
    """Pulpito run URL ends with .../<RUN_NAME>/ (or without trailing /)."""
    path = urlparse(run_url).path.rstrip("/")
    m = RUN_NAME_RE.search(path)
    if not m:
        raise ValueError(f"Could not parse run name from URL: {run_url}")
    return m.group(1)


def extract_qaproxy_run_name_from_html(run_html: str) -> Optional[str]:
    m = QA_PROXY_BASE_RE.search(run_html)
    return m.group(1) if m else None


def count_all_jobs(run_html: str) -> int:
    """Count ALL job rows on a Pulpito run page (pass + fail).

    Scans every table row for links whose text is a numeric job ID
    (4+ digits). Returns the count of unique job IDs found.
    """
    soup = BeautifulSoup(run_html, "html.parser")
    all_ids: set[str] = set()
    for tr in soup.find_all("tr"):
        for a in tr.find_all("a", href=True):
            txt = (a.text or "").strip()
            if txt.isdigit() and len(txt) >= 4:
                all_ids.add(txt)
                break
    return len(all_ids)


def find_failed_job_ids(run_html: str) -> List[str]:
    """Find job IDs for rows whose status contains fail, dead, error, or hung."""
    soup = BeautifulSoup(run_html, "html.parser")
    failed_ids: set[str] = set()

    for tr in soup.find_all("tr"):
        row_text = tr.get_text(" ", strip=True).lower()
        if not FAILED_STATUSES_RE.search(row_text):
            continue
        for a in tr.find_all("a", href=True):
            txt = (a.text or "").strip()
            if txt.isdigit() and len(txt) >= 4:
                failed_ids.add(txt)

    return sorted(failed_ids)


def is_suite_listing_url(url: str) -> bool:
    """
    Detect if a URL is a Pulpito suite/branch listing page (not a single run).
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if any(k in qs for k in ("suite", "branch", "machine_type", "status", "sha1")):
        return True

    path = parsed.path.strip("/")
    if not path:
        return True

    if re.match(r"^[a-zA-Z\-]+$", path) and len(path) < 30:
        return True

    return False


def _user_from_slug(slug: str) -> str:
    """Extract the user name from a Pulpito run slug like 'adking-2026-02-26_19:29:28-...'."""
    m = re.match(r"^([a-zA-Z][\w.-]*?)-\d{4}-\d{2}-\d{2}_", slug)
    return m.group(1) if m else ""


def extract_run_urls_from_suite_page(
    suite_url: str,
    session: requests.Session,
    timeout: int = 60,
    verbose: bool = True,
) -> List[Tuple[str, int, str]]:
    """
    Scrape a Pulpito suite/branch listing page and extract individual run URLs.

    Returns a list of (run_url, fail_count, user_name) tuples,
    sorted by fail count descending.
    Only returns runs that have at least 1 failure.
    """
    if verbose:
        print(f"[fetcher] Detected suite listing page: {suite_url}")
        print(f"[fetcher] Scraping for individual run URLs...")

    html = _fetch_url(suite_url, session=session, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")

    base = f"{urlparse(suite_url).scheme}://{urlparse(suite_url).netloc}"
    runs: List[Tuple[str, int, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/\w+-\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}-", href):
            full_url = urljoin(base, href)
            parts = urlparse(full_url).path.strip("/").split("/")
            if parts:
                run_slug = parts[0]
                run_url = f"{base}/{run_slug}/"
                if run_url not in seen:
                    seen.add(run_url)

    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue

        row_url = None
        row_slug = ""
        for a in tr.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/\w+-\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}-", href):
                parts = urlparse(urljoin(base, href)).path.strip("/").split("/")
                if parts:
                    row_slug = parts[0]
                    row_url = f"{base}/{row_slug}/"
                    break

        if not row_url:
            continue

        row_text = tr.get_text(" ", strip=True)
        fail_match = re.search(r"(\d+)\s+fail", row_text, re.IGNORECASE)
        fail_count = int(fail_match.group(1)) if fail_match else 0
        user = _user_from_slug(row_slug)

        if fail_count > 0 and row_url not in {r[0] for r in runs}:
            runs.append((row_url, fail_count, user))

    if not runs:
        runs = [(url, 0, _user_from_slug(urlparse(url).path.strip("/").split("/")[0]))
                for url in seen]

    runs.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"[fetcher] Found {len(runs)} run(s) with failures:")
        for url, fc, user in runs:
            slug = urlparse(url).path.strip("/").split("/")[0]
            print(f"    {fc:3d} failures | {user:20s} | {slug}")
        print()

    return runs


def qaproxy_teuthology_log_url(qaproxy_run_name: str, job_id: str) -> str:
    return (
        f"https://qa-proxy.ceph.com/teuthology/"
        f"{qaproxy_run_name}/{job_id}/teuthology.log"
    )


def _candidate_log_urls(primary_url: str) -> List[Tuple[str, str]]:
    """
    Candidate artifact URLs for a failed job in priority order.
    Start with teuthology.log, then fallback to ansible.log.
    """
    candidates: List[Tuple[str, str]] = [(primary_url, "teuthology.log")]
    if primary_url.endswith("/teuthology.log"):
        base = primary_url[: -len("teuthology.log")]
        candidates.append((f"{base}ansible.log", "ansible.log"))
    return candidates


def _download_one(
    job_id: str,
    log_url: str,
    session: Optional[requests.Session],
    timeout: int,
    verbose: bool,
    retries: int = 3,
) -> JobLog:
    """Download a single job's log. Uses thread-local session when session=None."""
    if session is None:
        session = _get_thread_session(retries)

    t0 = time.time()
    errors: List[str] = []
    for candidate_url, log_name in _candidate_log_urls(log_url):
        if verbose:
            print(f"    GET {candidate_url}")
        try:
            txt = _fetch_url(candidate_url, session=session, timeout=timeout)
            elapsed = time.time() - t0
            return JobLog(
                job_id=job_id,
                status="ok",
                log_text=txt,
                source_url=candidate_url,
                log_name=log_name,
                elapsed_s=elapsed,
            )
        except Exception as e:
            errors.append(f"{log_name}: {type(e).__name__}: {e}")
            if verbose:
                print(f"    failed: {type(e).__name__}: {e}")
        # Rate limit between attempts
        time.sleep(_RATE_LIMIT_DELAY)

    elapsed = time.time() - t0
    return JobLog(
        job_id=job_id,
        status="fetch_failed",
        log_text=None,
        source_url=log_url,
        log_name="teuthology.log",
        elapsed_s=elapsed,
        error_message=" | ".join(errors),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_failed_logs(
    run_url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
    force_run_name: Optional[str] = None,
    verbose: bool = True,
    job_ids: Optional[List[str]] = None,
    parallel: bool = True,
) -> FetchResult:
    """
    Fetch teuthology.log for every failed job in a Pulpito run.

    Returns a FetchResult containing the job logs plus total/failed counts
    scraped from the run page (for accurate run-health classification).
    """
    url_ok, url_msg = validate_url(run_url)
    if not url_ok:
        print(f"[fetcher] ERROR: Invalid URL: {url_msg}")
        return FetchResult(job_logs=[], total_jobs_on_page=0,
                           failed_jobs_on_page=0)

    session = make_session(retries)
    total_jobs_on_page = 0
    failed_jobs_on_page = 0

    if job_ids is None:
        if verbose:
            print(f"[fetcher] Fetching run page: {run_url}")
        run_html = _fetch_url(run_url, session=session, timeout=timeout)

        total_jobs_on_page = count_all_jobs(run_html)
        job_ids = find_failed_job_ids(run_html)
        failed_jobs_on_page = len(job_ids)

        if not job_ids:
            print("[fetcher] WARNING: No failed job IDs found on the run page.")
            return FetchResult(
                job_logs=[],
                total_jobs_on_page=total_jobs_on_page,
                failed_jobs_on_page=0,
            )

        qaproxy_run_name = (
            force_run_name
            or extract_qaproxy_run_name_from_html(run_html)
            or extract_run_name_from_url(run_url)
        )
    else:
        failed_jobs_on_page = len(job_ids)
        total_jobs_on_page = len(job_ids)  # best we can do without the page
        run_html = ""
        qaproxy_run_name = (
            force_run_name or extract_run_name_from_url(run_url)
        )

    total = len(job_ids)
    if verbose:
        print(f"[fetcher] Found {total} failed job(s) "
              f"(total on page: {total_jobs_on_page})")
        print(f"[fetcher] qa-proxy run name: {qaproxy_run_name}\n")

    download_tasks = [
        (jid, qaproxy_teuthology_log_url(qaproxy_run_name, jid))
        for jid in job_ids
    ]

    results: List[JobLog] = []

    if parallel and total > 1:
        workers = min(MAX_DOWNLOAD_WORKERS, total)
        if verbose:
            print(f"[fetcher] Downloading {total} logs in parallel "
                  f"({workers} workers)...\n")

        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for jid, log_url in download_tasks:
                # session=None → thread-local session (thread-safe)
                fut = pool.submit(
                    _download_one, jid, log_url, None, timeout, False,
                    retries,
                )
                futures[fut] = jid

            for i, fut in enumerate(as_completed(futures), start=1):
                jid = futures[fut]
                try:
                    job_log = fut.result()
                except Exception as exc:
                    job_log = JobLog(
                        job_id=jid, status="fetch_failed",
                        error_message=f"ThreadPool error: {exc}",
                    )
                results.append(job_log)
                if verbose:
                    if job_log.status == "ok" and job_log.log_text is not None:
                        nb = len(job_log.log_text.encode("utf-8",
                                                         errors="replace"))
                        print(
                            f"  [{i}/{total}] job {jid}: "
                            f"{nb:,} bytes from {job_log.log_name} "
                            f"({job_log.elapsed_s:.1f}s)"
                        )
                    else:
                        print(
                            f"  [{i}/{total}] job {jid}: "
                            f"FAILED ({job_log.elapsed_s:.1f}s)"
                        )

        id_order = {jid: idx for idx, (jid, _) in enumerate(download_tasks)}
        results.sort(key=lambda jl: id_order.get(jl.job_id, 0))
    else:
        for i, (jid, log_url) in enumerate(download_tasks, start=1):
            if verbose:
                banner = f"[{i}/{total}] job {jid}"
                print("=" * len(banner))
                print(banner)
                print("=" * len(banner))

            job_log = _download_one(jid, log_url, session, timeout, verbose)

            if verbose:
                if job_log.status == "ok" and job_log.log_text is not None:
                    nb = len(job_log.log_text.encode("utf-8",
                                                     errors="replace"))
                    print(
                        f"  -> saved {nb} bytes from {job_log.log_name} "
                        f"in {job_log.elapsed_s:.1f}s\n"
                    )
                else:
                    print(f"  -> FAILED after {job_log.elapsed_s:.1f}s\n")

            results.append(job_log)

    return FetchResult(
        job_logs=results,
        total_jobs_on_page=total_jobs_on_page,
        failed_jobs_on_page=failed_jobs_on_page,
    )


def save_logs_to_disk(
    logs: List[JobLog],
    outdir: str = "logs",
) -> str:
    """
    Write fetched logs to *outdir* and produce an index.csv.
    Returns the path to index.csv.
    """
    ensure_dir(outdir)
    index_path = os.path.join(outdir, "index.csv")

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "job_id", "status", "log_path", "fetched_from",
            "log_name", "bytes", "elapsed_s", "error_message",
        ])

        for jl in logs:
            if jl.status == "ok" and jl.log_text is not None:
                out_path = os.path.join(outdir, f"{jl.job_id}.log")
                with open(out_path, "w", encoding="utf-8",
                          errors="replace") as lf:
                    lf.write(jl.log_text)
                nb = len(jl.log_text.encode("utf-8", errors="replace"))
                w.writerow([
                    jl.job_id, "ok", out_path, jl.source_url, jl.log_name,
                    nb, f"{jl.elapsed_s:.3f}", jl.error_message,
                ])
            else:
                w.writerow([
                    jl.job_id, "fetch_failed", "", jl.source_url, jl.log_name,
                    0, f"{jl.elapsed_s:.3f}", jl.error_message,
                ])

    return index_path


def load_logs_from_disk(logdir: str) -> List[JobLog]:
    """
    Load previously-fetched .log files from a directory.
    Returns a list of JobLog objects with status='ok'.
    """
    results: List[JobLog] = []
    for fname in sorted(os.listdir(logdir)):
        if not fname.endswith(".log"):
            continue
        job_id = fname.removesuffix(".log")
        fpath = os.path.join(logdir, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        results.append(JobLog(
            job_id=job_id,
            status="ok",
            log_text=text,
            source_url=f"file://{os.path.abspath(fpath)}",
            log_name="file.log",
        ))
    return results


# ---------------------------------------------------------------------------
# Standalone CLI (backward-compatible)
# ---------------------------------------------------------------------------
def _cli_main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch teuthology.log for failed jobs in a Pulpito run"
    )
    ap.add_argument("run_url",
                     help="Pulpito run URL (ends with ...-trial/)")
    ap.add_argument("-o", "--outdir", default="logs",
                     help="Output directory (default: logs)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force-run-name", default=None)
    ap.add_argument("--no-parallel", action="store_true",
                     help="Disable parallel downloads")
    args = ap.parse_args()

    result = fetch_failed_logs(
        run_url=args.run_url,
        timeout=args.timeout,
        retries=args.retries,
        force_run_name=args.force_run_name,
        verbose=not args.quiet,
        parallel=not args.no_parallel,
    )

    if not result.job_logs:
        return 2

    idx = save_logs_to_disk(result.job_logs, args.outdir)
    print(f"Done.\nLogs: {args.outdir}/\nIndex: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
