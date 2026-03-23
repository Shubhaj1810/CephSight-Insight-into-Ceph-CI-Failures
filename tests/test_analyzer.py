"""Tests for analyzer module — JSON parsing, validation, heuristics, patterns."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from analyzer import (
    _parse_json_response, _validate_json_schema,
    _normalize_severity, _normalize_failure_type,
    _infer_failure_type, _apply_heuristics, _build_analysis_result,
    _concrete_pattern_match, _extract_teuthology_failure_reason,
    _as_text, _as_str_list,
    AnalysisResult, ALLOWED_SEVERITIES, ALLOWED_FAILURE_TYPES,
    _PROMPT_VERSION,
)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
class TestParseJsonResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "root_cause": "disk full",
            "severity": "high",
            "failure_type": "resource",
            "confidence": 0.9,
        })
        parsed = _parse_json_response(raw)
        assert parsed["root_cause"] == "disk full"
        assert parsed["confidence"] == 0.9

    def test_json_with_code_fences(self):
        raw = '```json\n{"root_cause": "test", "severity": "low", "failure_type": "test_bug", "confidence": 0.5}\n```'
        parsed = _parse_json_response(raw)
        assert parsed["root_cause"] == "test"

    def test_json_embedded_in_text(self):
        raw = 'Here is the analysis:\n{"root_cause": "crash", "severity": "critical", "failure_type": "crash", "confidence": 0.95}\nDone.'
        parsed = _parse_json_response(raw)
        assert parsed["root_cause"] == "crash"

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        parsed = _parse_json_response(raw)
        assert parsed == {}

    def test_empty_string(self):
        parsed = _parse_json_response("")
        assert parsed == {}

    def test_trailing_comma_fix(self):
        raw = '{"root_cause": "test", "severity": "low", "failure_type": "crash", "confidence": 0.5,}'
        parsed = _parse_json_response(raw)
        assert parsed.get("root_cause") == "test"


# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------
class TestValidateJsonSchema:
    def test_valid_schema(self):
        parsed = {
            "root_cause": "test",
            "severity": "high",
            "failure_type": "crash",
            "confidence": 0.8,
        }
        assert _validate_json_schema(parsed) is True

    def test_missing_required_key(self):
        parsed = {"root_cause": "test", "severity": "high"}
        assert _validate_json_schema(parsed) is False

    def test_non_numeric_confidence(self):
        parsed = {
            "root_cause": "test", "severity": "high",
            "failure_type": "crash", "confidence": "not a number",
        }
        assert _validate_json_schema(parsed) is False

    def test_empty_dict(self):
        assert _validate_json_schema({}) is False

    def test_empty_severity(self):
        parsed = {
            "root_cause": "test", "severity": "",
            "failure_type": "crash", "confidence": 0.5,
        }
        assert _validate_json_schema(parsed) is False


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
class TestNormalization:
    def test_severity_aliases(self):
        assert _normalize_severity("sev1") == "critical"
        assert _normalize_severity("sev4") == "low"

    def test_severity_unknown_defaults_medium(self):
        assert _normalize_severity("banana") == "medium"
        assert _normalize_severity(None) == "medium"

    def test_failure_type_aliases(self):
        assert _normalize_failure_type("connection") == "network"
        assert _normalize_failure_type("infrastructure") == "infra"
        assert _normalize_failure_type("oom") == "resource"

    def test_failure_type_unknown_defaults(self):
        assert _normalize_failure_type("banana") == "unknown"
        assert _normalize_failure_type(None) == "unknown"


# ---------------------------------------------------------------------------
# Teuthology-specific heuristic inference
# ---------------------------------------------------------------------------
class TestInferFailureType:
    def test_timeout(self):
        assert _infer_failure_type("operation timed out after 60s") == "timeout"

    def test_network(self):
        assert _infer_failure_type("ConnectionError: connection refused") == "network"

    def test_permission(self):
        assert _infer_failure_type("PermissionError: permission denied") == "permission"

    def test_resource(self):
        assert _infer_failure_type("Cannot allocate memory") == "resource"

    def test_crash(self):
        assert _infer_failure_type("segmentation fault (core dumped)") == "crash"

    def test_config(self):
        assert _infer_failure_type("invalid config value for ceph.conf") == "config"

    def test_infra(self):
        assert _infer_failure_type("qa-proxy returned 502") == "infra"

    def test_test_bug(self):
        assert _infer_failure_type("workunit test_rgw failed") == "test_bug"

    def test_unknown(self):
        assert _infer_failure_type("something happened") == "unknown"

    def test_no_module_named(self):
        assert _infer_failure_type("No module named 'boto.vendored.six.moves'") == "config"

    def test_yum_install_failure(self):
        text = "Command failed on trial053 with status 1: 'sudo yum -y install erlang'"
        assert _infer_failure_type(text) == "infra"

    def test_dnf_install_failure(self):
        text = "Command failed on trial138 with status 1: 'sudo dnf -y install ceph-radosgw'"
        assert _infer_failure_type(text) == "infra"

    def test_shaman_fetch_failure(self):
        text = "Failed to fetch package version from https://shaman.ceph.com/api/search/"
        assert _infer_failure_type(text) == "infra"

    def test_reimage_failure(self):
        text = "Error reimaging machines: Reimage of trial010 failed"
        assert _infer_failure_type(text) == "infra"

    def test_openstack_failure(self):
        text = 'Command failed on trial164 with status 1: "openstack project create"'
        assert _infer_failure_type(text) == "infra"

    def test_valgrind_error(self):
        text = "valgrind error: SyscallParam __libc_sendmsg sendmsg"
        assert _infer_failure_type(text) == "test_bug"

    def test_workunit_failure(self):
        text = "Command failed (workunit test rgw/run-datacache.sh) on trial"
        assert _infer_failure_type(text) == "test_bug"

    def test_ragweed_failure(self):
        text = 'Command failed (ragweed tests against rgw) on trial050'
        assert _infer_failure_type(text) == "test_bug"


# ---------------------------------------------------------------------------
# Teuthology failure reason extraction
# ---------------------------------------------------------------------------
class TestExtractFailureReason:
    def test_no_module_named(self):
        text = "blah\nNo module named 'boto.vendored.six.moves'\nblah"
        reason = _extract_teuthology_failure_reason(text)
        assert reason is not None
        assert "boto" in reason

    def test_command_failed(self):
        text = "Command failed (ragweed tests against rgw) on trial050 with status 2: 'some cmd'"
        reason = _extract_teuthology_failure_reason(text)
        assert reason is not None
        assert "ragweed" in reason

    def test_valgrind_error(self):
        text = "valgrind error: SyscallParam __libc_sendmsg sendmsg"
        reason = _extract_teuthology_failure_reason(text)
        assert reason is not None
        assert "valgrind" in reason

    def test_no_match(self):
        text = "Everything is fine\nnothing to see here"
        reason = _extract_teuthology_failure_reason(text)
        assert reason is None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_as_text(self):
        assert _as_text(None) == ""
        assert _as_text(None, "default") == "default"
        assert _as_text("  hello  ") == "hello"
        assert _as_text(42) == "42"

    def test_as_str_list(self):
        assert _as_str_list(["a", "b"]) == ["a", "b"]
        assert _as_str_list("a, b, c") == ["a", "b", "c"]
        assert _as_str_list("single") == ["single"]
        assert _as_str_list(None) == []
        assert _as_str_list([]) == []


# ---------------------------------------------------------------------------
# Build analysis result
# ---------------------------------------------------------------------------
class TestBuildAnalysisResult:
    def test_valid_llm_response(self):
        raw = json.dumps({
            "root_cause": "OSD crashed due to disk failure",
            "severity": "critical",
            "failure_type": "crash",
            "confidence": 0.95,
            "explanation": "The OSD process segfaulted.",
            "fix_suggestions": ["Replace disk", "Check other OSDs"],
            "recommended_action": "Replace the failed disk",
            "affected_components": ["OSD", "ceph-volume"],
        })
        result = _build_analysis_result("12345", raw, "condensed log text")
        assert result.success is True
        assert result.severity == "critical"
        assert result.failure_type == "crash"
        assert result.confidence == 0.95
        assert "disk" in result.root_cause.lower()

    def test_invalid_json_with_infra_log(self):
        text = "Command failed on trial053 with status 1: 'sudo dnf -y install erlang'"
        result = _build_analysis_result("12345", "not json", text, json_valid=False)
        assert result.success is True
        assert result.failure_type == "infra"

    def test_invalid_json_with_test_bug_log(self):
        text = "Command failed (ragweed tests against rgw) on trial050 with status 2"
        result = _build_analysis_result("12345", "not json", text, json_valid=False)
        assert result.success is True
        assert result.failure_type == "test_bug"


# ---------------------------------------------------------------------------
# Prompt version
# ---------------------------------------------------------------------------
class TestPromptVersion:
    def test_prompt_version_exists(self):
        assert isinstance(_PROMPT_VERSION, str)
        assert len(_PROMPT_VERSION) == 12


# ---------------------------------------------------------------------------
# Apply heuristics
# ---------------------------------------------------------------------------
class TestApplyHeuristics:
    def test_upgrades_severity_for_crash(self):
        result = AnalysisResult(job_id="1", failure_type="crash", severity="medium")
        result = _apply_heuristics(result, "some crash log")
        assert result.severity == "high"

    def test_fills_missing_fields(self):
        result = AnalysisResult(
            job_id="1", failure_type="unknown", severity="medium",
            root_cause="", explanation="", recommended_action="",
        )
        result = _apply_heuristics(result, "Command failed on trial053 with status 1: 'sudo yum -y install erlang'")
        assert result.failure_type == "infra"
        assert result.recommended_action != ""
        assert result.confidence > 0

    def test_concrete_pattern_upgrades_severity(self):
        result = AnalysisResult(job_id="1", failure_type="unknown", severity="low")
        result = _apply_heuristics(result, "segmentation fault (core dumped)")
        assert result.failure_type == "crash"
        assert result.severity == "critical"

    def test_resource_severity_not_low(self):
        result = AnalysisResult(job_id="1", failure_type="resource", severity="low")
        result = _apply_heuristics(result, "some log")
        assert result.severity == "high"

    def test_root_cause_from_teuthology_reason(self):
        result = AnalysisResult(
            job_id="1", failure_type="unknown", severity="medium",
            root_cause="",
        )
        text = "blah\nNo module named 'boto.vendored.six.moves'\nblah"
        result = _apply_heuristics(result, text)
        assert "boto" in result.root_cause


# ---------------------------------------------------------------------------
# Concrete pattern matching
# ---------------------------------------------------------------------------
class TestConcretePatterns:
    def test_segfault(self):
        match = _concrete_pattern_match("SIGSEGV received, dumping core")
        assert match is not None
        ft, sev = match
        assert ft == "crash"
        assert sev == "critical"

    def test_oom(self):
        match = _concrete_pattern_match("oom-killer invoked, killed process")
        assert match is not None
        ft, sev = match
        assert ft == "resource"
        assert sev == "critical"

    def test_no_module(self):
        match = _concrete_pattern_match("No module named 'boto.vendored'")
        assert match is not None
        ft, _ = match
        assert ft == "config"

    def test_yum_install(self):
        match = _concrete_pattern_match(
            "Command failed on trial053 with status 1: 'sudo yum -y install erlang'"
        )
        assert match is not None
        ft, _ = match
        assert ft == "infra"

    def test_no_match(self):
        match = _concrete_pattern_match("everything is fine")
        assert match is None

    def test_highest_severity_wins(self):
        text = "Connection refused and also SIGSEGV core dumped"
        match = _concrete_pattern_match(text)
        assert match is not None
        _, sev = match
        assert sev == "critical"
