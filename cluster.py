#!/usr/bin/env python3
"""
Failure clustering for mass-failure runs.

When 64 out of 70 jobs fail with the same error, analysing each one
individually is wasteful and produces a report that is impossible to
read.  This module solves that by:

1. **Fingerprinting** each parsed log based on its key error signals
   (final exception line, traceback tail, first fatal message).
2. **Grouping** jobs with identical or near-identical fingerprints
   into *clusters* — includes fuzzy similarity merging for near-misses.
3. **Run-health classification** — using the failure ratio to decide
   whether the run is a mass-failure (likely infra), partial failure
   (likely regression), or isolated failure (likely test bug / flake).

The pipeline then only needs to send ONE representative log per cluster
to the LLM, and fans the result out to all jobs in that cluster.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from log_parser import ParsedLog

# ---------------------------------------------------------------------------
# Run-health thresholds
# ---------------------------------------------------------------------------
MASS_FAILURE_THRESHOLD = 0.70       # ≥70% jobs failed
PARTIAL_FAILURE_THRESHOLD = 0.20    # ≥20% jobs failed
# Below 20% = isolated failures


@dataclass
class RunHealth:
    """High-level health classification of a test run."""
    total_jobs: int                 # total jobs on the run page
    failed_jobs: int                # jobs that failed / dead / error / hung
    passed_jobs: int                # jobs that passed
    failure_ratio: float            # failed / total  (0.0 – 1.0)
    classification: str             # "mass_failure" | "partial_failure" | "isolated"
    hint_for_llm: str               # One-line context injected into the LLM prompt

    @property
    def pct(self) -> str:
        return f"{self.failure_ratio:.0%}"


def classify_run_health(
    total_jobs: int,
    failed_jobs: int,
) -> RunHealth:
    """Classify the run based on the failure ratio."""
    passed = max(0, total_jobs - failed_jobs)
    ratio = failed_jobs / total_jobs if total_jobs > 0 else 0.0

    if ratio >= MASS_FAILURE_THRESHOLD:
        classification = "mass_failure"
        hint = (
            f"MASS FAILURE: {failed_jobs}/{total_jobs} jobs failed ({ratio:.0%}). "
            "This is almost certainly a systemic/infrastructure issue — "
            "look for a common root cause (bad image, provisioning failure, "
            "network outage, broken dependency) rather than individual test bugs."
        )
    elif ratio >= PARTIAL_FAILURE_THRESHOLD:
        classification = "partial_failure"
        hint = (
            f"PARTIAL FAILURE: {failed_jobs}/{total_jobs} jobs failed ({ratio:.0%}). "
            "This suggests a regression or flaky subsystem affecting a "
            "significant portion of tests.  Look for a shared component "
            "or configuration change that could explain the pattern."
        )
    else:
        classification = "isolated"
        hint = (
            f"ISOLATED FAILURES: {failed_jobs}/{total_jobs} jobs failed ({ratio:.0%}). "
            "Most jobs passed — these are likely individual test bugs, "
            "known flakes, or one-off infrastructure glitches."
        )

    return RunHealth(
        total_jobs=total_jobs,
        failed_jobs=failed_jobs,
        passed_jobs=passed,
        failure_ratio=ratio,
        classification=classification,
        hint_for_llm=hint,
    )


# ---------------------------------------------------------------------------
# Error fingerprinting
# ---------------------------------------------------------------------------
_FINAL_EXCEPTION_RE = re.compile(
    r"^(\w+(?:\.\w+)*(?:Error|Exception|Failure|Warning)[^\n]{0,200})",
    re.MULTILINE,
)
_COMMAND_FAILED_RE = re.compile(
    r"(CommandFailedError[^\n]{0,200})", re.IGNORECASE,
)
_TIMEOUT_RE = re.compile(
    r"((?:timed?\s*out|timeout|deadline exceeded)[^\n]{0,150})",
    re.IGNORECASE,
)
_FATAL_RE = re.compile(
    r"((?:FATAL|CRITICAL|segmentation fault|core dumped|panic)[^\n]{0,150})",
    re.IGNORECASE,
)

# Lines to strip from fingerprints (timestamps, PIDs, paths that vary)
_NOISE_RE = re.compile(
    r"(?:"
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"  # timestamps
    r"|0x[0-9a-f]{6,}"                           # hex addresses
    r"|\bpid[= ]\d+"                             # PIDs
    r"|/tmp/[^\s]+"                               # temp paths
    r"|\b\d{10,}\b"                               # large numbers (job IDs etc.)
    r")",
    re.IGNORECASE,
)

# Similarity threshold for merging near-duplicate clusters.
# Two clusters with Jaccard token similarity >= this value are merged.
CLUSTER_SIMILARITY_THRESHOLD = 0.70


def _normalize_for_fingerprint(text: str) -> str:
    """Strip volatile parts (timestamps, PIDs, paths) so that
    structurally identical errors produce the same fingerprint."""
    return _NOISE_RE.sub("", text).strip()


def _tokenize_for_similarity(text: str) -> set[str]:
    """Extract meaningful tokens (length > 2) for Jaccard comparison."""
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', text)
    return {t.lower() for t in tokens}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_error_signature(parsed: ParsedLog) -> str:
    """Extract a short, stable "error signature" string from a parsed log.

    Priority order:
    1. Last exception line from the last traceback
    2. CommandFailedError line
    3. Timeout / fatal line
    4. First error block's key line
    5. First 200 chars of condensed text
    """
    signatures: List[str] = []

    # 1. Last traceback's final exception line
    if parsed.tracebacks:
        last_tb = parsed.tracebacks[-1]
        matches = _FINAL_EXCEPTION_RE.findall(last_tb)
        if matches:
            signatures.append(matches[-1])

    # 2. CommandFailedError
    cmd_match = _COMMAND_FAILED_RE.search(parsed.condensed_text)
    if cmd_match:
        signatures.append(cmd_match.group(1))

    # 3. Timeout / fatal
    for pat in (_TIMEOUT_RE, _FATAL_RE):
        m = pat.search(parsed.condensed_text)
        if m:
            signatures.append(m.group(1))
            break

    # 4. First error block key line
    if parsed.error_blocks and not signatures:
        first_block = parsed.error_blocks[0].text
        for line in first_block.splitlines():
            stripped = line.strip()
            if stripped and re.search(
                r"\b(ERROR|FAIL|Exception|Traceback)\b", stripped, re.IGNORECASE
            ):
                signatures.append(stripped[:200])
                break

    # 5. Fallback
    if not signatures:
        signatures.append(parsed.condensed_text[:200])

    # Combine top signals, normalize, and produce a single signature
    combined = " | ".join(signatures[:3])
    return _normalize_for_fingerprint(combined)


def _fingerprint_hash(signature: str) -> str:
    """Produce a short hex hash of the error signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cluster data class
# ---------------------------------------------------------------------------
@dataclass
class FailureCluster:
    """A group of jobs that share the same error fingerprint."""
    cluster_id: str                     # short hex hash
    signature: str                      # human-readable error signature
    job_ids: List[str] = field(default_factory=list)
    representative_job_id: str = ""     # the one we send to the LLM
    representative_parsed: Optional[ParsedLog] = None

    @property
    def size(self) -> int:
        return len(self.job_ids)


# ---------------------------------------------------------------------------
# Fuzzy merge pass
# ---------------------------------------------------------------------------
def _merge_similar_clusters(
    clusters: List[FailureCluster],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> List[FailureCluster]:
    """Merge clusters whose error signatures are similar above *threshold*.

    Uses Jaccard similarity on token sets extracted from each signature.
    Runs iteratively until no more merges are possible (typically 1-2 passes).
    """
    if len(clusters) <= 1:
        return clusters

    changed = True
    while changed:
        changed = False
        token_sets = [_tokenize_for_similarity(c.signature) for c in clusters]
        absorbed: set[int] = set()
        merged_clusters: List[FailureCluster] = []

        for i in range(len(clusters)):
            if i in absorbed:
                continue
            ci = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in absorbed:
                    continue
                sim = _jaccard_similarity(token_sets[i], token_sets[j])
                if sim >= threshold:
                    cj = clusters[j]
                    ci.job_ids.extend(cj.job_ids)
                    # Keep the representative with the longest condensed text
                    if (cj.representative_parsed is not None
                            and ci.representative_parsed is not None
                            and len(cj.representative_parsed.condensed_text)
                            > len(ci.representative_parsed.condensed_text)):
                        ci.representative_job_id = cj.representative_job_id
                        ci.representative_parsed = cj.representative_parsed
                        ci.signature = cj.signature
                        ci.cluster_id = cj.cluster_id
                    absorbed.add(j)
                    changed = True
            merged_clusters.append(ci)

        clusters = merged_clusters
        clusters.sort(key=lambda c: c.size, reverse=True)

    return clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cluster_parsed_logs(
    parsed_logs: List[ParsedLog],
) -> Tuple[List[FailureCluster], Dict[str, str]]:
    """
    Group parsed logs into failure clusters by error fingerprint.

    Two-phase approach:
    1. Exact fingerprint hash grouping.
    2. Fuzzy merge pass using Jaccard token similarity.

    Returns
    -------
    clusters : list[FailureCluster]
        Sorted by cluster size descending (biggest cluster first).
    job_to_cluster : dict[str, str]
        Mapping from job_id → cluster_id for reverse lookup.
    """
    # Build fingerprint → jobs mapping
    fp_to_jobs: Dict[str, List[Tuple[str, ParsedLog]]] = {}
    fp_to_sig: Dict[str, str] = {}

    for pl in parsed_logs:
        sig = _extract_error_signature(pl)
        fp = _fingerprint_hash(sig)

        if fp not in fp_to_jobs:
            fp_to_jobs[fp] = []
            fp_to_sig[fp] = sig
        fp_to_jobs[fp].append((pl.job_id, pl))

    # Build cluster objects
    clusters: List[FailureCluster] = []

    for fp, job_list in fp_to_jobs.items():
        best_job_id, best_parsed = max(
            job_list, key=lambda x: len(x[1].condensed_text)
        )

        cluster = FailureCluster(
            cluster_id=fp,
            signature=fp_to_sig[fp],
            job_ids=[jid for jid, _ in job_list],
            representative_job_id=best_job_id,
            representative_parsed=best_parsed,
        )
        clusters.append(cluster)

    # Sort: biggest clusters first
    clusters.sort(key=lambda c: c.size, reverse=True)

    # Phase 2: fuzzy merge near-duplicate clusters
    clusters = _merge_similar_clusters(clusters)

    # Rebuild reverse lookup after potential merges
    job_to_cluster: Dict[str, str] = {}
    for cluster in clusters:
        for jid in cluster.job_ids:
            job_to_cluster[jid] = cluster.cluster_id

    return clusters, job_to_cluster
