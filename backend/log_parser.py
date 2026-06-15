#!/usr/bin/env python3
"""
Teuthology log preprocessor — multi-signal extraction pipeline.

Extracts ALL types of failure signals from teuthology.log files:
  - failure_reason from the YAML summary (single most important line)
  - stderr from failed remote commands (highest diagnostic value)
  - Failure context windows (30 lines before each failure)
  - Full Python tracebacks (including chained exceptions)
  - ERROR/WARNING/CRITICAL log lines with context
  - Health/timeout/connection/resource/permission signals
  - Deduplicates repeated patterns with frequency counting

Produces a priority-ordered condensed text for LLM analysis (24K chars max).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Patterns — comprehensive signal detection
# ---------------------------------------------------------------------------

# Traditional error keyword pattern (existing)
ERROR_KW_RE = re.compile(
    r"\b(ERROR|FAIL|FAILED|Traceback|Exception|assert|CRITICAL|"
    r"CommandFailedError|RuntimeError|ConnectionError|Timeout|Dead)\b",
    re.IGNORECASE,
)

TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
TRACEBACK_END_RE = re.compile(
    r"^(\w+(?:\.\w+)*(?:Error|Exception|Failure|Warning).*)", re.MULTILINE
)
CHAINED_TB_RE = re.compile(
    r"^(During handling of the above exception|"
    r"The above exception was the direct cause of|"
    r"Caused by:)",
    re.IGNORECASE,
)

# --- New signal patterns ---

# Teuthology remote stderr (the #1 signal for actual error messages)
STDERR_LINE_RE = re.compile(
    r"teuthology\.orchestra\.run\.(\w+)\.stderr:(.*)"
)

# CommandFailedError with host and status
CMD_FAILED_RE = re.compile(
    r"Command failed on (\w+) with status (\d+)"
)

# Non-zero remote exit code
EXIT_CODE_RE = re.compile(
    r"got remote process result: (\d+)"
)

# failure_reason from the YAML summary block
FAILURE_REASON_RE = re.compile(
    r"^failure_reason:\s*(.*)", re.MULTILINE
)

# Timeout patterns
TIMEOUT_RE = re.compile(
    r"\b(timed? ?out|timeout|deadline exceeded|hung|not responding)\b",
    re.IGNORECASE,
)

# Ceph health errors
HEALTH_ERROR_RE = re.compile(
    r"(\[ERR\]|\[WRN\]|\[SEC\]|HEALTH_ERR|HEALTH_WARN)",
)

# Connection/network failures
CONNECTION_RE = re.compile(
    r"\b(Connection refused|ConnectionRefusedError|unreachable|"
    r"ssh: connect|Network is unreachable|No route to host|"
    r"Name or service not known|connection reset)\b",
    re.IGNORECASE,
)

# Assertion / crash signals
CRASH_RE = re.compile(
    r"\b(ceph_assert|FAILED ceph_assert|segmentation fault|segfault|"
    r"SIGSEGV|SIGABRT|core dumped|fatal signal|suicide timeout)\b",
    re.IGNORECASE,
)

# Resource exhaustion
RESOURCE_RE = re.compile(
    r"\b(No space left on device|Cannot allocate memory|"
    r"out of memory|OOM|oom-killer|disk full|ENOSPC)\b",
    re.IGNORECASE,
)

# Permission errors
PERMISSION_RE = re.compile(
    r"\b(Permission denied|EACCES|Forbidden|unauthorized|"
    r"authentication failed)\b",
    re.IGNORECASE,
)

# Ansible failures
ANSIBLE_RE = re.compile(
    r"\b(fatal:|UNREACHABLE|MODULE FAILURE)\b",
)

# WARNING lines from teuthology tasks (often contain failure details)
WARNING_RE = re.compile(
    r"WARNING:tasks\.\w+:"
)

# Noise patterns to filter out (low-value lines that pollute extraction)
NOISE_RE = re.compile(
    r"(journalctl@ceph\.mon\.\w+\.\w+\.stdout:.*dispatch$|"
    r"journalctl@ceph\.mon\.\w+\.\w+\.stdout:.*Reconfiguring|"
    r"gzip -5 --verbose|"
    r"tar c -f - -C|"
    r"-- replaced with .+\.gz$|"
    r"^\s*\d+\.\d+% -- replaced|"
    r"INFO:teuthology\.orchestra\.run\.\w+\.stderr:\s*$|"
    r"INFO:teuthology\.orchestra\.run\.\w+\.stderr:\s*\d+\+\d+ records|"
    r"INFO:teuthology\.orchestra\.run\.\w+\.stderr:\s*\d+ bytes copied|"
    r"\.stderr:.*\d+\.\d+\.\d+\.\d+:\d+/\d+\s*>>|"
    r"\.stderr:.*conn\(0x[0-9a-f]+|"
    r"\.stderr:.*s=STATE_|"
    r"\.stderr:.*s=READY|"
    r"\.stderr:.*s=BANNER|"
    r"\.stderr:.*s=AUTH_|"
    r"\.stderr:.*s=NONE|"
    r"\.stderr:.*s=CLOSED|"
    r"\.stderr:.*\.mark_down|"
    r"\.stderr:.*\.stop$|"
    r"\.stderr:.*\.ready entity|"
    r"\.stderr:.*shutdown_connections|"
    r"\.stderr:.*wait complete|"
    r"\.stderr:.*Processor -- start|"
    r"\.stderr:.*start start$|"
    r"\.stderr:.*\.connect$|"
    r"\.stderr:.*mon_subscribe|"
    r"\.stderr:.*mon_getmap|"
    r"\.stderr:.*mon_command|"
    r"\.stderr:.*==== mon_|"
    r"\.stderr:.*==== config\(|"
    r"\.stderr:.*==== mgrmap|"
    r"\.stderr:.*==== osd_map|"
    r"\.stderr:.*==== mon_command_ack|"
    r"\.stderr:.*_handle_peer_banner|"
    r"\.stderr:.*Inferring config|"
    r"real\s+\d+m[\d.]+s$|"
    r"user\s+\d+m[\d.]+s$|"
    r"sys\s+\d+m[\d.]+s$)"
)

# Teuthology summary section markers
SUMMARY_HEADER_RE = re.compile(
    r"^(=+ SUMMARY =+|summary of|\*+ RESULTS \*+|^Results:)",
    re.IGNORECASE | re.MULTILINE,
)
SUMMARY_DATA_RE = re.compile(
    r"^(description:|duration:|failure_reason:|flavor:|owner:|status:|success:)",
    re.IGNORECASE,
)

# Token budget
MAX_CONDENSED_CHARS = 24_000
CONTEXT_LINES = 5
MAX_SUMMARY_LINES = 300


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ErrorBlock:
    """A contiguous block of lines centered on an error."""
    start_line: int
    end_line: int
    text: str


@dataclass
class StderrBlock:
    """Stderr output from a failed remote command."""
    host: str
    stderr_lines: List[str]
    command_line: str = ""
    exit_status: int = 0
    line_number: int = 0


@dataclass
class ParsedLog:
    """Preprocessed representation of a single teuthology.log."""
    job_id: str
    raw_log: str = ""
    error_blocks: List[ErrorBlock] = field(default_factory=list)
    tracebacks: List[str] = field(default_factory=list)
    summary_section: str = ""
    condensed_text: str = ""
    total_lines: int = 0
    total_chars: int = 0
    raw_log_path: Optional[str] = None

    def get_raw_log(self) -> str:
        if self.raw_log:
            return self.raw_log
        if self.raw_log_path:
            try:
                with open(self.raw_log_path, "r", encoding="utf-8",
                          errors="replace") as f:
                    return f.read()
            except OSError:
                pass
        return ""

    def release_raw_log(self) -> None:
        self.raw_log = ""


# ---------------------------------------------------------------------------
# Signal Extractors
# ---------------------------------------------------------------------------

def _extract_failure_reason(lines: List[str]) -> str:
    """Extract the failure_reason: line from the YAML summary block.
    This is the single most informative line in any teuthology log."""
    for i in range(len(lines) - 1, max(len(lines) - 500, -1), -1):
        if lines[i].strip().startswith("failure_reason:"):
            reason_parts = [lines[i].strip()]
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or lines[j].startswith("\t")):
                reason_parts.append(lines[j].strip())
                j += 1
            return " ".join(reason_parts)
    return ""


def _extract_command_stderr(lines: List[str]) -> List[StderrBlock]:
    """Extract stderr output from failed remote commands.

    Strategy: Find CommandFailedError or non-zero exit codes, then look
    backwards to collect .stderr: lines from the same host. Also capture
    the command that was executed.
    """
    stderr_blocks: List[StderrBlock] = []
    failure_indices: List[Tuple[int, str, int]] = []

    for i, line in enumerate(lines):
        m = CMD_FAILED_RE.search(line)
        if m:
            failure_indices.append((i, m.group(1), int(m.group(2))))
            continue
        m = EXIT_CODE_RE.search(line)
        if m and int(m.group(1)) > 0:
            host = ""
            for j in range(max(0, i - 3), i):
                hm = re.search(r"teuthology\.orchestra\.run\.(\w+)", lines[j])
                if hm:
                    host = hm.group(1)
                    break
            if host:
                failure_indices.append((i, host, int(m.group(1))))

    for fail_idx, host, status in failure_indices:
        stderr_lines: List[str] = []
        command_line = ""

        look_back = min(50, fail_idx)
        for j in range(fail_idx - 1, fail_idx - look_back - 1, -1):
            if j < 0:
                break
            line = lines[j]

            m = STDERR_LINE_RE.search(line)
            if m and m.group(1) == host:
                content = m.group(2).strip()
                if content and not NOISE_RE.search(line):
                    stderr_lines.insert(0, content)
            elif f"orchestra.run.{host}:>" in line or f"run.{host}:>" in line:
                cmd_match = re.search(r":>\s*(.+)", line)
                if cmd_match:
                    command_line = cmd_match.group(1).strip()
                break
            elif re.search(r"DEBUG:teuthology\.orchestra\.run\.\w+:>", line) and host not in line:
                if stderr_lines:
                    break

        if stderr_lines:
            stderr_blocks.append(StderrBlock(
                host=host,
                stderr_lines=stderr_lines,
                command_line=command_line,
                exit_status=status,
                line_number=fail_idx,
            ))

    return stderr_blocks


def _extract_failure_context_windows(lines: List[str]) -> List[str]:
    """Capture 30 lines before and 10 lines after each high-signal failure.

    High-signal failures: CommandFailedError, ERROR:teuthology, and
    WARNING:tasks lines that contain actual failure info.
    """
    failure_lines: List[int] = []

    for i, line in enumerate(lines):
        if CMD_FAILED_RE.search(line):
            failure_lines.append(i)
        elif "ERROR:teuthology" in line and "Saw exception" in line:
            failure_lines.append(i)
        elif WARNING_RE.search(line) and ("failed" in line.lower() or "error" in line.lower()):
            failure_lines.append(i)

    if not failure_lines:
        return []

    seen_ranges: List[Tuple[int, int]] = []
    for idx in failure_lines:
        start = max(0, idx - 30)
        end = min(len(lines) - 1, idx + 10)
        if seen_ranges and start <= seen_ranges[-1][1] + 5:
            seen_ranges[-1] = (seen_ranges[-1][0], end)
        else:
            seen_ranges.append((start, end))

    windows: List[str] = []
    for start, end in seen_ranges[:5]:
        window_lines = []
        for j in range(start, end + 1):
            if not NOISE_RE.search(lines[j]):
                window_lines.append(lines[j])
        if window_lines:
            windows.append("\n".join(window_lines))

    return windows


def _extract_failure_signals(lines: List[str]) -> List[Tuple[int, str, str]]:
    """Single pass to detect all high-value failure signals.

    Returns list of (line_index, signal_type, line_content).
    """
    signals: List[Tuple[int, str, str]] = []

    for i, line in enumerate(lines):
        if NOISE_RE.search(line):
            continue
        if TIMEOUT_RE.search(line) and "ERROR" in line.upper():
            signals.append((i, "timeout", line))
        elif HEALTH_ERROR_RE.search(line) and "grep" not in line:
            signals.append((i, "health", line))
        elif CONNECTION_RE.search(line):
            signals.append((i, "connection", line))
        elif CRASH_RE.search(line):
            signals.append((i, "crash", line))
        elif RESOURCE_RE.search(line):
            signals.append((i, "resource", line))
        elif PERMISSION_RE.search(line) and "grep" not in line and "audit" not in line:
            signals.append((i, "permission", line))
        elif ANSIBLE_RE.search(line):
            signals.append((i, "ansible", line))

    return signals


def _extract_tracebacks(lines: List[str]) -> List[str]:
    """Return full Python traceback strings, including chained exceptions."""
    tracebacks: List[str] = []
    i = 0
    while i < len(lines):
        if TRACEBACK_START_RE.search(lines[i]):
            tb_lines = [lines[i]]
            j = i + 1
            while j < len(lines):
                tb_lines.append(lines[j])
                stripped = lines[j].strip()
                if stripped and not stripped.startswith("File ") and not stripped.startswith("..."):
                    if TRACEBACK_END_RE.match(stripped):
                        k = j + 1
                        while k < len(lines) and not lines[k].strip():
                            k += 1
                        if k < len(lines) and (
                            CHAINED_TB_RE.match(lines[k].strip())
                            or TRACEBACK_START_RE.search(lines[k])
                        ):
                            while k < len(lines) and not TRACEBACK_START_RE.search(lines[k]):
                                tb_lines.append(lines[k])
                                k += 1
                            if k < len(lines) and TRACEBACK_START_RE.search(lines[k]):
                                j = k - 1
                            else:
                                break
                        else:
                            break
                    elif j + 1 < len(lines):
                        next_line = lines[j + 1]
                        if next_line.strip() and not next_line.startswith(" ") and not next_line.startswith("\t"):
                            if not TRACEBACK_START_RE.search(next_line) and not CHAINED_TB_RE.match(next_line.strip()):
                                break
                j += 1
            tracebacks.append("\n".join(tb_lines))
            i = j + 1
        else:
            i += 1
    return tracebacks


def _extract_error_blocks(
    lines: List[str], context: int = CONTEXT_LINES
) -> List[ErrorBlock]:
    """Return de-overlapped error blocks with surrounding context."""
    hit_indices: List[int] = []
    for idx, line in enumerate(lines):
        if NOISE_RE.search(line):
            continue
        if ERROR_KW_RE.search(line):
            hit_indices.append(idx)

    if not hit_indices:
        return []

    ranges: List[Tuple[int, int]] = []
    for idx in hit_indices:
        start = max(0, idx - context)
        end = min(len(lines) - 1, idx + context)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))

    blocks: List[ErrorBlock] = []
    for s, e in ranges:
        blocks.append(ErrorBlock(
            start_line=s, end_line=e,
            text="\n".join(lines[s: e + 1]),
        ))
    return blocks


def _extract_summary(lines: List[str]) -> str:
    """Extract the YAML summary block at the end of the log."""
    for i in range(len(lines) - 1, max(len(lines) - 500, -1), -1):
        if SUMMARY_DATA_RE.match(lines[i].strip()):
            start = i
            while start > 0 and SUMMARY_DATA_RE.match(lines[start - 1].strip()):
                start -= 1
            end = min(i + MAX_SUMMARY_LINES, len(lines))
            j = i + 1
            while j < end:
                stripped = lines[j].strip()
                if stripped and not stripped.startswith("-") and not stripped.startswith(" ") and not stripped.startswith("\t"):
                    if not SUMMARY_DATA_RE.match(stripped):
                        break
                j += 1
            return "\n".join(lines[start:j])

    for i in range(len(lines) - 1, max(len(lines) - 2000, -1), -1):
        if SUMMARY_HEADER_RE.search(lines[i]):
            end = min(i + MAX_SUMMARY_LINES, len(lines))
            return "\n".join(lines[i:end])

    return "\n".join(lines[-150:])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _fingerprint_error(text: str) -> str:
    """Normalize an error block for deduplication."""
    fp = text.strip()
    fp = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*\s*", "", fp)
    fp = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<IP>", fp)
    fp = re.sub(r"\b(trial|smithi|mira)\d+\b", "<HOST>", fp)
    fp = re.sub(r"\bpid[= ]\d+\b", "pid=<PID>", fp, flags=re.IGNORECASE)
    fp = re.sub(r"\b\d{5,}\b", "<ID>", fp)
    fp = re.sub(r"\s+", " ", fp)
    return fp[:200]


def _deduplicate_blocks(blocks: List[ErrorBlock]) -> List[ErrorBlock]:
    """Deduplicate error blocks by fingerprint with frequency counting."""
    seen: Dict[str, int] = {}
    unique: List[ErrorBlock] = []
    for b in blocks:
        fp = _fingerprint_error(b.text)
        if fp not in seen:
            seen[fp] = 1
            unique.append(b)
        else:
            seen[fp] += 1

    for b in unique:
        fp = _fingerprint_error(b.text)
        count = seen.get(fp, 1)
        if count > 1:
            b.text = f"[Repeated {count}x] {b.text}"

    return unique


# ---------------------------------------------------------------------------
# Condensed text assembly
# ---------------------------------------------------------------------------

def _clip_middle(text: str, max_chars: int) -> str:
    """Trim long text preserving both beginning and ending."""
    if len(text) <= max_chars:
        return text

    head_budget = max_chars * 2 // 3
    tail_budget = max_chars - head_budget
    marker = "\n\n... [middle truncated to fit model context] ...\n\n"

    head_end = text.rfind("\n", 0, head_budget)
    if head_end < 0:
        head_end = head_budget

    tail_start_search = max(0, len(text) - tail_budget)
    tail_start = text.find("\n", tail_start_search)
    if tail_start < 0:
        tail_start = tail_start_search

    return text[:head_end] + marker + text[tail_start:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_log(job_id: str, raw_log: str) -> ParsedLog:
    """
    Parse a raw teuthology log using multi-signal extraction.
    Priority order: failure_reason > stderr > context windows > tracebacks > signals > error blocks
    """
    lines = raw_log.splitlines()

    # --- Extract all signals ---
    failure_reason = _extract_failure_reason(lines)
    stderr_blocks = _extract_command_stderr(lines)
    context_windows = _extract_failure_context_windows(lines)
    tracebacks = _extract_tracebacks(lines)
    failure_signals = _extract_failure_signals(lines)
    error_blocks = _extract_error_blocks(lines)
    error_blocks = _deduplicate_blocks(error_blocks)
    summary = _extract_summary(lines)

    # --- Build condensed text (priority order) ---
    parts: List[str] = []

    # Priority 1: failure_reason (the single most important line)
    if failure_reason:
        parts.append("=== FAILURE REASON (from teuthology summary) ===")
        parts.append(failure_reason)
        parts.append("")

    # Priority 2: stderr from failed commands (deduplicated)
    if stderr_blocks:
        unique_stderr: List[StderrBlock] = []
        stderr_fps: Dict[str, int] = {}
        for sb in stderr_blocks:
            fp = _fingerprint_error("\n".join(sb.stderr_lines))
            if fp not in stderr_fps:
                stderr_fps[fp] = 1
                unique_stderr.append(sb)
            else:
                stderr_fps[fp] += 1

        parts.append(f"=== COMMAND STDERR ({len(stderr_blocks)} failed, {len(unique_stderr)} unique) ===")
        for idx, sb in enumerate(unique_stderr[:5], 1):
            fp = _fingerprint_error("\n".join(sb.stderr_lines))
            repeat = stderr_fps.get(fp, 1)
            repeat_note = f" [repeated {repeat}x]" if repeat > 1 else ""
            parts.append(f"--- Failed command #{idx} on {sb.host} (exit {sb.exit_status}){repeat_note} ---")
            if sb.command_line:
                parts.append(f"Command: {sb.command_line}")
            parts.append("Stderr output:")
            for sl in sb.stderr_lines[-20:]:
                parts.append(f"  {sl}")
            parts.append("")

    # Priority 3: failure context windows (30 lines before each failure)
    if context_windows:
        parts.append(f"=== FAILURE CONTEXT ({len(context_windows)} window(s)) ===")
        for idx, window in enumerate(context_windows[:3], 1):
            parts.append(f"--- Context window #{idx} ---")
            parts.append(window)
            parts.append("")

    # Priority 4: tracebacks (deduplicated)
    if tracebacks:
        unique_tbs: List[str] = []
        tb_fps: set = set()
        for tb in tracebacks:
            fp = _fingerprint_error(tb)
            if fp not in tb_fps:
                tb_fps.add(fp)
                unique_tbs.append(tb)
        if unique_tbs:
            tb_saved = len(tracebacks) - len(unique_tbs)
            parts.append(f"=== TRACEBACKS ({len(unique_tbs)} unique, {tb_saved} duplicates removed) ===")
            for idx, tb in enumerate(unique_tbs[:5], 1):
                parts.append(f"--- Traceback #{idx} ---")
                parts.append(tb)
                parts.append("")

    # Priority 5: other failure signals (timeout, health, connection, etc.)
    if failure_signals:
        sig_by_type: Dict[str, List[str]] = {}
        for _, sig_type, line in failure_signals:
            sig_by_type.setdefault(sig_type, []).append(line.strip())

        sig_parts: List[str] = []
        for sig_type, sig_lines in sig_by_type.items():
            unique_sigs = list(dict.fromkeys(sig_lines))[:5]
            for sl in unique_sigs:
                sig_parts.append(f"  [{sig_type}] {sl}")

        if sig_parts:
            parts.append(f"=== OTHER FAILURE SIGNALS ({len(failure_signals)} detected) ===")
            parts.extend(sig_parts[:15])
            parts.append("")

    # Priority 6: YAML summary block
    if summary and not failure_reason:
        parts.append("=== LOG SUMMARY ===")
        parts.append(summary)
        parts.append("")

    # Priority 7: generic error blocks (fill remaining budget)
    capped_blocks = error_blocks[:10]
    if capped_blocks:
        parts.append(f"=== ERROR BLOCKS ({len(capped_blocks)} of {len(error_blocks)} unique) ===")
        for idx, blk in enumerate(capped_blocks, 1):
            parts.append(f"--- Block #{idx} (lines {blk.start_line}-{blk.end_line}) ---")
            parts.append(blk.text)
            parts.append("")

    condensed = "\n".join(parts)

    if len(condensed) > MAX_CONDENSED_CHARS:
        condensed = _clip_middle(condensed, MAX_CONDENSED_CHARS)

    if not condensed.strip():
        tail = "\n".join(lines[-200:]) if len(lines) > 200 else raw_log
        condensed = "=== LOG TAIL (no specific errors extracted) ===\n" + tail
        if len(condensed) > MAX_CONDENSED_CHARS:
            condensed = _clip_middle(condensed, MAX_CONDENSED_CHARS)

    return ParsedLog(
        job_id=job_id,
        raw_log=raw_log,
        error_blocks=error_blocks,
        tracebacks=tracebacks,
        summary_section=summary,
        condensed_text=condensed,
        total_lines=len(lines),
        total_chars=len(raw_log),
    )
