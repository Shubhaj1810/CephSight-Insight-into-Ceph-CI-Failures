"""Tests for cluster module."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from log_parser import ParsedLog, ErrorBlock
from cluster import (
    classify_run_health, cluster_parsed_logs, _extract_error_signature,
    _fingerprint_hash, _normalize_for_fingerprint, _tokenize_for_similarity,
    _jaccard_similarity, _merge_similar_clusters, FailureCluster,
)


# ---------------------------------------------------------------------------
# Run health classification
# ---------------------------------------------------------------------------
class TestClassifyRunHealth:
    def test_mass_failure(self):
        rh = classify_run_health(total_jobs=100, failed_jobs=80)
        assert rh.classification == "mass_failure"
        assert rh.failure_ratio == 0.8

    def test_partial_failure(self):
        rh = classify_run_health(total_jobs=100, failed_jobs=30)
        assert rh.classification == "partial_failure"

    def test_isolated(self):
        rh = classify_run_health(total_jobs=100, failed_jobs=5)
        assert rh.classification == "isolated"

    def test_zero_total(self):
        rh = classify_run_health(total_jobs=0, failed_jobs=0)
        assert rh.failure_ratio == 0.0

    def test_all_failed(self):
        rh = classify_run_health(total_jobs=50, failed_jobs=50)
        assert rh.classification == "mass_failure"

    def test_pct_property(self):
        rh = classify_run_health(total_jobs=200, failed_jobs=100)
        assert rh.pct == "50%"


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
class TestFingerprinting:
    def test_normalize_strips_timestamps(self):
        text = "2026-02-27T10:30:00 ERROR: something failed pid=12345"
        normalized = _normalize_for_fingerprint(text)
        assert "2026-02-27" not in normalized
        assert "12345" not in normalized

    def test_normalize_strips_hex_addresses(self):
        text = "crash at 0xdeadbeef in function foo"
        normalized = _normalize_for_fingerprint(text)
        assert "0xdeadbeef" not in normalized

    def test_fingerprint_deterministic(self):
        sig = "RuntimeError: connection refused"
        h1 = _fingerprint_hash(sig)
        h2 = _fingerprint_hash(sig)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_sigs_different_hashes(self):
        h1 = _fingerprint_hash("RuntimeError: A")
        h2 = _fingerprint_hash("RuntimeError: B")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
class TestSimilarity:
    def test_tokenize(self):
        tokens = _tokenize_for_similarity("RuntimeError: connection_refused in ceph_osd")
        assert "runtimeerror" in tokens
        assert "connection_refused" in tokens
        assert "ceph_osd" in tokens

    def test_identical_sets(self):
        a = {"foo", "bar", "baz"}
        assert _jaccard_similarity(a, a) == 1.0

    def test_disjoint_sets(self):
        a = {"foo", "bar"}
        b = {"baz", "qux"}
        assert _jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"foo", "bar", "baz"}
        b = {"bar", "baz", "qux"}
        sim = _jaccard_similarity(a, b)
        assert 0.4 < sim < 0.6

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), set()) == 1.0
        assert _jaccard_similarity({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Cluster merging
# ---------------------------------------------------------------------------
def _make_parsed(job_id: str, condensed: str) -> ParsedLog:
    return ParsedLog(
        job_id=job_id,
        condensed_text=condensed,
        total_lines=1,
        total_chars=len(condensed),
    )


class TestClusterMerging:
    def test_merge_similar_clusters(self):
        c1 = FailureCluster(
            cluster_id="aaa",
            signature="RuntimeError connection_refused in ceph_osd module component handler",
            job_ids=["1", "2"],
            representative_job_id="1",
            representative_parsed=_make_parsed("1", "log text 1"),
        )
        c2 = FailureCluster(
            cluster_id="bbb",
            signature="RuntimeError connection_refused in ceph_osd module component service",
            job_ids=["3"],
            representative_job_id="3",
            representative_parsed=_make_parsed("3", "log text very long 3"),
        )
        c3 = FailureCluster(
            cluster_id="ccc",
            signature="completely different timeout error in rados gateway",
            job_ids=["4"],
            representative_job_id="4",
            representative_parsed=_make_parsed("4", "log text 4"),
        )
        merged = _merge_similar_clusters([c1, c2, c3])
        assert len(merged) == 2
        sizes = sorted([c.size for c in merged])
        assert sizes == [1, 3]

    def test_no_merge_when_dissimilar(self):
        c1 = FailureCluster(
            cluster_id="aaa",
            signature="RuntimeError: network unreachable",
            job_ids=["1"],
            representative_job_id="1",
            representative_parsed=_make_parsed("1", "t1"),
        )
        c2 = FailureCluster(
            cluster_id="bbb",
            signature="OSD crash segmentation fault core dump",
            job_ids=["2"],
            representative_job_id="2",
            representative_parsed=_make_parsed("2", "t2"),
        )
        merged = _merge_similar_clusters([c1, c2])
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# Full clustering pipeline
# ---------------------------------------------------------------------------
class TestClusterParsedLogs:
    def test_identical_errors_cluster(self):
        logs = [
            _make_parsed("1", "RuntimeError: boom"),
            _make_parsed("2", "RuntimeError: boom"),
            _make_parsed("3", "RuntimeError: boom"),
        ]
        for pl in logs:
            pl.tracebacks = ["Traceback (most recent call last):\nRuntimeError: boom"]
        clusters, mapping = cluster_parsed_logs(logs)
        assert len(clusters) == 1
        assert clusters[0].size == 3
        assert all(jid in mapping for jid in ["1", "2", "3"])

    def test_different_errors_separate(self):
        p1 = _make_parsed("1", "TimeoutError: connection timed out in rados")
        p1.tracebacks = ["Traceback (most recent call last):\nTimeoutError: connection timed out"]
        p2 = _make_parsed("2", "AssertionError: expected True but got False in test_rgw")
        p2.tracebacks = ["Traceback (most recent call last):\nAssertionError: expected True"]
        clusters, _ = cluster_parsed_logs([p1, p2])
        assert len(clusters) == 2

    def test_empty_input(self):
        clusters, mapping = cluster_parsed_logs([])
        assert clusters == []
        assert mapping == {}
