#!/usr/bin/env python3
"""
AI log analyzer using Ollama.

Sends condensed log excerpts to a local Ollama instance and returns
structured analysis results.  Optimized for llama3:8b's context window
and instruction-following capability.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Structured analysis of a single job's log."""
    job_id: str
    root_cause: str = "Unknown"
    severity: str = "medium"
    error_category: str = "Unknown"
    failure_type: str = "unknown"
    confidence: float = 0.0
    explanation: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    recommended_action: str = ""
    affected_components: List[str] = field(default_factory=list)
    raw_llm_response: str = ""
    success: bool = True
    error_message: str = ""
    cached: bool = False
    run_user: str = ""
    run_name: str = ""


# ---------------------------------------------------------------------------
# Prompts — compact for llama3:8b (fits in ~800 tokens)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Ceph/Teuthology CI failure analyst. Given a failed test log, \
return ONLY valid JSON (no markdown, no commentary) with these fields:

{
  "root_cause": "<one-line summary citing the specific error>",
  "severity": "<critical|high|medium|low>",
  "error_category": "<e.g. package install failure, boto import error, SSH failure>",
  "failure_type": "<network|crash|timeout|config|test_bug|infra|permission|resource|unknown>",
  "confidence": <0.0-1.0>,
  "explanation": "<2-3 sentences explaining what failed and why>",
  "fix_suggestions": ["<fix 1>", "<fix 2>"],
  "recommended_action": "<most important fix>",
  "affected_components": ["<component>"]
}

severity: critical=data loss/cluster crash, high=product bug, medium=flaky/infra, low=cosmetic.
failure_type meanings: network=connectivity, crash=segfault/assert, timeout=hung op, \
config=bad config, test_bug=test logic, infra=provisioning/packages/artifacts, \
permission=auth/caps, resource=OOM/disk full.
Return ONLY the JSON object."""

USER_PROMPT_TEMPLATE = """\
Analyze this failed teuthology job {job_id}. Return JSON only.

{condensed_text}"""

RETRY_PROMPT = """\
Your response was not valid JSON. Return ONLY a JSON object with these keys: \
root_cause, severity, error_category, failure_type, confidence, explanation, \
fix_suggestions, recommended_action, affected_components.

Job {job_id} log:
{condensed_text}"""

CLUSTER_USER_PROMPT_TEMPLATE = """\
Analyze this failed teuthology job. It represents {cluster_size} jobs with the same error.
Cluster signature: {signature}
Job IDs: {job_ids}
Representative job: {job_id}
{run_health_hint}

{condensed_text}"""

SUMMARY_PROMPT_TEMPLATE = """\
You are a Ceph CI engineer. Write an executive summary (3-5 paragraphs).

{run_health_context}

{cluster_count} failure clusters found:
{analyses_json}
{artifact_note}
Cover: 1) overall health 2) top clusters by impact 3) root causes \
4) priority actions 5) infra issues. Be specific. No JSON."""

_PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + CLUSTER_USER_PROMPT_TEMPLATE
     + RETRY_PROMPT).encode()
).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Normalization & heuristics
# ---------------------------------------------------------------------------
ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
ALLOWED_FAILURE_TYPES = {
    "network", "crash", "timeout", "config",
    "test_bug", "infra", "permission", "resource", "unknown",
}

REQUIRED_JSON_KEYS = {"root_cause", "severity", "failure_type", "confidence"}

SEVERITY_ALIASES = {
    "sev1": "critical", "sev2": "high", "sev3": "medium", "sev4": "low",
}

FAILURE_TYPE_ALIASES = {
    "connection": "network", "connectivity": "network", "dns": "network",
    "socket": "network", "http": "network",
    "command_failure": "infra", "commandfailederror": "infra",
    "infra_issue": "infra", "infrastructure": "infra",
    "configuration": "config",
    "permission_error": "permission", "auth": "permission",
    "authorization": "permission",
    "oom": "resource", "memory": "resource", "disk": "resource",
    "flaky": "test_bug", "test_failure": "test_bug",
}

# Lines from our own metadata headers that should NOT be treated as errors
_METADATA_PREFIX_RE = re.compile(
    r"^(===|---|\s*Job ID:|\s*Total lines:|\s*Total characters:|\s*Tracebacks|\s*Block #)",
    re.IGNORECASE,
)

ERROR_LINE_RE = re.compile(
    r"\b(ERROR|FAIL|FAILED|Traceback|Exception|CRITICAL|assert|timed out|"
    r"CommandFailedError|ConnectionError)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Teuthology-specific heuristic patterns
# ---------------------------------------------------------------------------
def _infer_failure_type(text: str) -> str:
    """Rule-based failure type inference with Teuthology-specific patterns."""
    t = text.lower()

    # Teuthology-specific patterns (check first — most precise)
    if "no module named" in t:
        return "config"
    if re.search(r"command failed on \S+ with status \d+.*sudo (yum|dnf|apt).*install", t):
        return "infra"
    if "failed to fetch package version" in t:
        return "infra"
    if re.search(r"error reimaging|reimage.*failed", t):
        return "infra"
    if re.search(r"command failed.*openstack", t):
        return "infra"
    if "valgrind error" in t:
        return "test_bug"
    if re.search(r"command failed \(workunit test", t):
        return "test_bug"
    if re.search(r"command failed \(ragweed tests", t):
        return "test_bug"
    if re.search(r"command failed \(bucket notification tests", t):
        return "test_bug"
    if re.search(r"command failed \(s3.* tests", t):
        return "test_bug"

    # Generic patterns
    if any(s in t for s in (
        "segmentation fault", "segfault", "core dumped", "sigsegv",
        "sigabrt", "ceph_assert", "fatal signal", "panic",
    )):
        return "crash"
    if any(s in t for s in ("timed out", "timeout", "deadline exceeded")):
        return "timeout"
    if any(s in t for s in (
        "connectionerror", "connection refused", "network is unreachable",
        "name or service not known", "no route to host",
        "connection reset by peer", "sslerror",
    )):
        return "network"
    if any(s in t for s in (
        "permission denied", "unauthorized", "forbidden", "eacces",
        "authentication failed", "keyring",
    )):
        return "permission"
    if any(s in t for s in (
        "out of memory", "cannot allocate memory", "no space left on device",
        "disk full", "oom",
    )):
        return "resource"
    if any(s in t for s in (
        "traceback", "assertionerror", "runtimeerror",
    )):
        return "crash"
    if any(s in t for s in (
        "invalid config", "parse error", "bad value", "unknown option",
    )):
        return "config"
    if any(s in t for s in (
        "qa-proxy", "node down", "ssh: connect", "artifact", "404",
        "commandfailederror",
    )):
        return "infra"
    if any(s in t for s in ("workunit", "test failed", "assert ", "expected")):
        return "test_bug"
    return "unknown"


def _extract_teuthology_failure_reason(text: str) -> Optional[str]:
    """Extract the Teuthology 'Failure Reason' or 'Command failed' line."""
    for line in text.splitlines():
        stripped = line.strip()
        # Teuthology failure reason from summary
        if stripped.lower().startswith("failure_reason:"):
            return stripped[len("failure_reason:"):].strip()[:300]
        # "Command failed ..." lines
        m = re.match(
            r".*?(Command failed\s*\(.*?\).*?with status \d+.*)",
            stripped, re.IGNORECASE,
        )
        if m:
            return m.group(1)[:300]
        # "No module named" standalone
        m2 = re.search(r"(No module named '[^']+')", stripped)
        if m2:
            return m2.group(1)
        # "Failed to fetch package version"
        if "failed to fetch package version" in stripped.lower():
            return stripped[:300]
        # "Error reimaging"
        if "error reimaging" in stripped.lower():
            return stripped[:300]
        # "valgrind error"
        m3 = re.search(r"(valgrind error:.*)", stripped, re.IGNORECASE)
        if m3:
            return m3.group(1)[:300]
    return None


def _default_recommended_action(failure_type: str) -> str:
    actions = {
        "network": "Check connectivity and DNS between test nodes.",
        "crash": "Inspect stack traces and identify the crashing component.",
        "timeout": "Identify the hung operation; increase timeout as workaround.",
        "config": "Fix the missing module/dependency or config error.",
        "test_bug": "Debug the failing test and open a tracker bug.",
        "infra": "Check teuthology infrastructure, packages, and provisioning.",
        "permission": "Verify credentials, keyrings, and caps.",
        "resource": "Check memory/disk pressure on test nodes.",
        "unknown": "Inspect the full raw log for the root error.",
    }
    return actions.get(failure_type, actions["unknown"])


def _first_signal_line(text: str) -> str:
    """Find the first meaningful error line, skipping metadata headers."""
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if _METADATA_PREFIX_RE.match(line_stripped):
            continue
        if ERROR_LINE_RE.search(line_stripped):
            return line_stripped[:300]
    return ""


def _concrete_pattern_match(condensed_text: str) -> Optional[tuple[str, str]]:
    """Regex-based cross-validation against known Ceph error signatures."""
    patterns = [
        (r"segmentation fault|segfault|SIGSEGV|core dumped", "crash", "critical"),
        (r"ceph_assert|FAILED ceph_assert", "crash", "critical"),
        (r"suicide timeout", "crash", "critical"),
        (r"bluefs.*out of space", "resource", "critical"),
        (r"out of memory|OOM|oom-killer|Cannot allocate memory", "resource", "critical"),
        (r"No space left on device", "resource", "high"),
        (r"No module named", "config", "high"),
        (r"Command failed on \S+ with status \d+.*sudo (yum|dnf|apt)", "infra", "high"),
        (r"Failed to fetch package version", "infra", "high"),
        (r"Error reimaging|reimage.*failed", "infra", "high"),
        (r"valgrind error", "test_bug", "medium"),
        (r"Command failed \(workunit test", "test_bug", "high"),
        (r"Command failed \(ragweed tests", "test_bug", "high"),
        (r"Command failed \(bucket notification tests", "test_bug", "high"),
        (r"Command failed.*openstack", "infra", "high"),
        (r"Connection refused|ConnectionRefusedError", "network", "medium"),
        (r"timed? ?out|Timeout|deadline exceeded", "timeout", "medium"),
        (r"Permission denied|EACCES|Forbidden", "permission", "high"),
        (r"ssh:.*connect|SSH.*timeout|ansible.*unreachable", "infra", "medium"),
    ]
    sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    best: Optional[tuple[str, str, int]] = None
    for pat, ft, sev in patterns:
        if re.search(pat, condensed_text, re.IGNORECASE):
            rank = sev_rank.get(sev, 99)
            if best is None or rank < best[2]:
                best = (ft, sev, rank)
    if best:
        return best[0], best[1]
    return None


def _apply_heuristics(result: AnalysisResult, condensed_text: str) -> AnalysisResult:
    """Post-LLM validation and correction."""
    # Cross-validate against concrete patterns
    pattern_match = _concrete_pattern_match(condensed_text)

    inferred_type = _infer_failure_type(condensed_text)
    if result.failure_type == "unknown" and inferred_type != "unknown":
        result.failure_type = inferred_type

    if pattern_match:
        pm_type, pm_sev = pattern_match
        sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        result_sev_rank = sev_rank.get(result.severity, 99)
        pattern_sev_rank = sev_rank.get(pm_sev, 99)
        if result.failure_type == "unknown":
            result.failure_type = pm_type
        if pattern_sev_rank < result_sev_rank:
            result.severity = pm_sev

    if result.severity not in ALLOWED_SEVERITIES:
        result.severity = "medium"

    if result.failure_type in {"crash", "config", "resource", "permission", "test_bug"}:
        if result.severity in {"medium", "low"}:
            result.severity = "high"

    if result.failure_type in {"network", "timeout", "infra"} and result.severity == "low":
        result.severity = "medium"

    if not result.error_category or result.error_category.lower() == "unknown":
        category_map = {
            "network": "network/connectivity issue",
            "crash": "component crash/exception",
            "timeout": "operation timeout",
            "config": "configuration/dependency error",
            "test_bug": "test logic failure",
            "infra": "infrastructure/provisioning issue",
            "permission": "permission/authentication issue",
            "resource": "resource exhaustion",
            "unknown": "unknown issue",
        }
        result.error_category = category_map.get(result.failure_type, "unknown issue")

    # Fix root cause — prefer Teuthology failure reason over generic lines
    if not result.root_cause or result.root_cause.lower() in {"unknown", "n/a", "none"}:
        teuth_reason = _extract_teuthology_failure_reason(condensed_text)
        if teuth_reason:
            result.root_cause = teuth_reason
        else:
            signal = _first_signal_line(condensed_text)
            if signal:
                result.root_cause = signal
            else:
                result.root_cause = f"Likely {result.failure_type} failure (heuristic)"

    if not result.explanation.strip():
        result.explanation = (
            f"Heuristic classification: {result.failure_type}. "
            f"Root cause: {result.root_cause[:200]}"
        )

    if not result.recommended_action.strip():
        result.recommended_action = _default_recommended_action(result.failure_type)

    if not result.fix_suggestions:
        result.fix_suggestions = [result.recommended_action]

    if result.confidence <= 0.0:
        result.confidence = 0.65 if result.failure_type != "unknown" else 0.4
    elif result.failure_type != "unknown" and result.confidence < 0.5:
        result.confidence = 0.5

    return result


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _normalize_severity(value: Any) -> str:
    if value is None:
        return "medium"
    sev = str(value).strip().lower()
    sev = SEVERITY_ALIASES.get(sev, sev)
    return sev if sev in ALLOWED_SEVERITIES else "medium"


def _normalize_failure_type(value: Any) -> str:
    if value is None:
        return "unknown"
    ft = str(value).strip().lower()
    ft = ft.replace("-", "_").replace(" ", "_")
    ft = re.sub(r"[^a-z0-9_]", "", ft)
    ft = FAILURE_TYPE_ALIASES.get(ft, ft)
    return ft if ft in ALLOWED_FAILURE_TYPES else "unknown"


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        if "," in txt:
            return [p.strip() for p in txt.split(",") if p.strip()]
        if "\n" in txt:
            return [p.strip("- ").strip() for p in txt.splitlines() if p.strip()]
        return [txt]
    return []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = 7 * 24 * 3600


class AnalysisCache:
    def __init__(self, cache_dir: str = ".analysis_cache") -> None:
        if os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
            raise ValueError(f"cache_dir is not a directory: {cache_dir}")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _hash_text(text: str, model: str = "",
                   prompt_version: str = _PROMPT_VERSION) -> str:
        blob = f"{model}\n{prompt_version}\n{text}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:32]

    def _cache_path(self, text_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{text_hash}.json")

    def get(self, condensed_text: str, model: str = "") -> Optional[AnalysisResult]:
        h = self._hash_text(condensed_text, model)
        path = self._cache_path(h)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("_cached_at", 0) > CACHE_TTL_SECONDS:
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None
            return AnalysisResult(
                job_id=data.get("job_id", ""),
                root_cause=data.get("root_cause", "Unknown"),
                severity=data.get("severity", "medium"),
                error_category=data.get("error_category", "Unknown"),
                failure_type=data.get("failure_type", "unknown"),
                confidence=data.get("confidence", 0.0),
                explanation=data.get("explanation", ""),
                fix_suggestions=data.get("fix_suggestions", []),
                recommended_action=data.get("recommended_action", ""),
                affected_components=data.get("affected_components", []),
                raw_llm_response=data.get("raw_llm_response", ""),
                success=True, cached=True,
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def put(self, condensed_text: str, result: AnalysisResult,
            model: str = "") -> None:
        if not result.success:
            return
        h = self._hash_text(condensed_text, model)
        path = self._cache_path(h)
        data = {
            "job_id": result.job_id, "model": model,
            "prompt_version": _PROMPT_VERSION,
            "_cached_at": time.time(),
            "root_cause": result.root_cause, "severity": result.severity,
            "error_category": result.error_category,
            "failure_type": result.failure_type,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "fix_suggestions": result.fix_suggestions,
            "recommended_action": result.recommended_action,
            "affected_components": result.affected_components,
            "raw_llm_response": result.raw_llm_response,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> int:
        count = 0
        if not os.path.isdir(self.cache_dir):
            return count
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                    count += 1
                except OSError:
                    pass
        return count


class NullAnalysisCache:
    def get(self, condensed_text: str, model: str = "") -> Optional[AnalysisResult]:
        return None
    def put(self, condensed_text: str, result: AnalysisResult,
            model: str = "") -> None:
        return None
    def clear(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# JSON parsing — aggressive extraction
# ---------------------------------------------------------------------------
def _parse_json_response(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM response."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the last JSON object (models sometimes prepend text)
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text))
    for match in reversed(matches):
        try:
            parsed = json.loads(match.group())
            if any(k in parsed for k in ("root_cause", "failure_type", "severity")):
                return parsed
        except json.JSONDecodeError:
            continue

    # Try greedy extraction
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            raw_json = match.group()
            raw_json = re.sub(r",\s*}", "}", raw_json)
            raw_json = re.sub(r",\s*]", "]", raw_json)
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                pass

    return {}


def _validate_json_schema(parsed: Dict[str, Any]) -> bool:
    if not parsed:
        return False
    if not REQUIRED_JSON_KEYS.issubset(parsed.keys()):
        return False
    try:
        float(parsed["confidence"])
    except (ValueError, TypeError):
        return False
    if not isinstance(parsed.get("severity"), str) or not parsed["severity"].strip():
        return False
    if not isinstance(parsed.get("failure_type"), str) or not parsed["failure_type"].strip():
        return False
    return True


def _build_analysis_result(job_id: str, raw: str, condensed_text: str,
                           json_valid: bool = True) -> AnalysisResult:
    parsed = _parse_json_response(raw)

    confidence = parsed.get("confidence", 0.0) if parsed else 0.0
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (ValueError, TypeError):
        confidence = 0.0

    result = AnalysisResult(
        job_id=job_id,
        root_cause=_as_text(parsed.get("root_cause") if parsed else None, "Unknown"),
        severity=_normalize_severity(parsed.get("severity") if parsed else None),
        error_category=_as_text(parsed.get("error_category") if parsed else None, "Unknown"),
        failure_type=_normalize_failure_type(parsed.get("failure_type") if parsed else None),
        confidence=confidence,
        explanation=_as_text(parsed.get("explanation") if parsed else None, ""),
        fix_suggestions=_as_str_list(parsed.get("fix_suggestions") if parsed else None),
        recommended_action=_as_text(parsed.get("recommended_action") if parsed else None, ""),
        affected_components=_as_str_list(parsed.get("affected_components") if parsed else None),
        raw_llm_response=raw,
        success=True,
    )

    if not parsed or not json_valid:
        result.explanation = ""

    return _apply_heuristics(result, condensed_text)


# ---------------------------------------------------------------------------
# Ollama Analyzer
# ---------------------------------------------------------------------------
class OllamaAnalyzer:
    """Ollama-based log analyzer optimized for llama3:8b."""

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_retries: int = 3,
        json_retries: int = 3,
        num_ctx: int = 0,
        cache_dir: str = ".analysis_cache",
        cache_enabled: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_retries = json_retries
        self.num_ctx = num_ctx
        self.cache = (
            AnalysisCache(cache_dir) if cache_enabled else NullAnalysisCache()
        )
        self._exec_summary_cache: Dict[str, str] = {}

    def _generate(self, prompt: str, system: str = "",
                  force_json: bool = False) -> str:
        url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": 0.1,
            "num_predict": 2048,
        }
        if self.num_ctx > 0:
            options["num_ctx"] = self.num_ctx

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
        }
        if force_json:
            payload["format"] = "json"

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except Exception as exc:
                last_err = exc
                wait = 2 ** attempt
                log.warning("ollama attempt %d failed: %s", attempt, exc)
                print(f"  [ollama] attempt {attempt} failed: {exc} — retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(
            f"Ollama request failed after {self.max_retries} attempts: {last_err}"
        )

    def check_ready(self) -> tuple[bool, str]:
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return (False, f"Ollama not reachable at {self.base_url}: {exc}")

        names: set[str] = set()
        for m in data.get("models", []):
            name = str(m.get("name", "")).strip()
            if not name:
                continue
            names.add(name)
            if ":" in name:
                names.add(name.split(":", 1)[0])

        if self.model not in names:
            return (False, f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        return (True, "ok")

    def _try_json_with_retries(self, prompt: str, condensed_text: str,
                                job_id: str) -> tuple[str, bool]:
        # First attempt WITHOUT force_json — llama3:8b returns {} when
        # format=json is set.  Let it respond naturally, then extract JSON.
        try:
            raw = self._generate(prompt, system=SYSTEM_PROMPT, force_json=False)
        except RuntimeError:
            return "", False

        first_raw = raw  # keep the natural-language response for the report

        parsed = _parse_json_response(raw)
        json_valid = _validate_json_schema(parsed)

        if not json_valid:
            # Retry WITH force_json — some models do better constrained
            for retry_num in range(1, self.json_retries + 1):
                print(f"  [retry {retry_num}/{self.json_retries}] "
                      f"re-prompting for JSON...", end=" ", flush=True)
                retry_prompt = RETRY_PROMPT.format(
                    job_id=job_id, condensed_text=condensed_text,
                )
                try:
                    raw = self._generate(
                        retry_prompt, system=SYSTEM_PROMPT, force_json=True,
                    )
                except RuntimeError:
                    print("failed")
                    continue
                parsed = _parse_json_response(raw)
                json_valid = _validate_json_schema(parsed)
                if json_valid:
                    print("OK")
                    break
                else:
                    print("still invalid")

        # If retries got valid JSON, combine: use parsed JSON for fields
        # but keep the original natural response for display
        if json_valid and first_raw != raw:
            raw = raw + "\n\n--- Initial model response ---\n" + first_raw
        elif not json_valid:
            # All retries failed — use the richer first response for display
            raw = first_raw

        return raw, json_valid

    def analyze_job(self, job_id: str, condensed_text: str) -> AnalysisResult:
        cached = self.cache.get(condensed_text, model=self.model)
        if cached is not None:
            cached.job_id = job_id
            return cached

        prompt = USER_PROMPT_TEMPLATE.format(
            job_id=job_id, condensed_text=condensed_text,
        )

        raw, json_valid = self._try_json_with_retries(
            prompt, condensed_text, job_id,
        )
        if not raw:
            # LLM completely failed — build from heuristics alone
            result = _build_analysis_result(job_id, "", condensed_text,
                                            json_valid=False)
            if result.success:
                self.cache.put(condensed_text, result, model=self.model)
            return result

        result = _build_analysis_result(job_id, raw, condensed_text,
                                        json_valid=json_valid)
        if result.success:
            self.cache.put(condensed_text, result, model=self.model)
        return result

    def analyze_cluster(
        self,
        representative_job_id: str,
        condensed_text: str,
        cluster_size: int,
        signature: str,
        job_ids: List[str],
        run_health_hint: str = "",
    ) -> AnalysisResult:
        cached = self.cache.get(condensed_text, model=self.model)
        if cached is not None:
            cached.job_id = representative_job_id
            return cached

        prompt = CLUSTER_USER_PROMPT_TEMPLATE.format(
            cluster_size=cluster_size,
            signature=signature,
            job_ids=", ".join(job_ids[:20]) + (
                f" ... (+{len(job_ids) - 20} more)" if len(job_ids) > 20 else ""
            ),
            job_id=representative_job_id,
            run_health_hint=run_health_hint,
            condensed_text=condensed_text,
        )

        raw, json_valid = self._try_json_with_retries(
            prompt, condensed_text, representative_job_id,
        )
        if not raw:
            result = _build_analysis_result(
                representative_job_id, "", condensed_text, json_valid=False,
            )
            if result.success:
                self.cache.put(condensed_text, result, model=self.model)
            return result

        result = _build_analysis_result(
            representative_job_id, raw, condensed_text, json_valid=json_valid,
        )
        if result.success:
            self.cache.put(condensed_text, result, model=self.model)
        return result

    def clear_cache(self) -> int:
        return self.cache.clear()

    def generate_executive_summary(
        self,
        results: List[AnalysisResult],
        artifact_unavailable_count: int = 0,
        run_health_context: str = "",
        cluster_count: int = 0,
    ) -> str:
        summaries = [{
            "job_id": r.job_id, "root_cause": r.root_cause,
            "severity": r.severity, "error_category": r.error_category,
            "failure_type": r.failure_type, "confidence": r.confidence,
            "recommended_action": r.recommended_action,
        } for r in results]

        summary_key = hashlib.sha256(
            json.dumps(summaries, sort_keys=True).encode()
        ).hexdigest()[:32]
        cached = self._exec_summary_cache.get(summary_key)
        if cached:
            return cached

        artifact_note = ""
        if artifact_unavailable_count > 0:
            artifact_note = (
                f"\nNote: {artifact_unavailable_count} job(s) had unavailable "
                "log artifacts (infrastructure issue).\n"
            )

        if not run_health_context:
            run_health_context = f"Total analyzed jobs: {len(results)}"
        if cluster_count <= 0:
            cluster_count = len(results)

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            run_health_context=run_health_context,
            cluster_count=cluster_count,
            analyses_json=json.dumps(summaries, indent=2),
            artifact_note=artifact_note,
        )

        try:
            result_text = self._generate(
                prompt, system="You are a Ceph CI engineer.",
            )
            self._exec_summary_cache[summary_key] = result_text
            return result_text
        except RuntimeError as exc:
            return f"(Could not generate executive summary: {exc})"
