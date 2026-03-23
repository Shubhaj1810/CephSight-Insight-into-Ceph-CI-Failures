#!/usr/bin/env python3
"""
Teuthology log preprocessor.

Extracts the most relevant sections from large teuthology.log files:
  - Error / failure lines with surrounding context
  - Full Python tracebacks (including chained exceptions)
  - The final summary block
  - Deduplicates repeated patterns

Produces a condensed text suitable for LLM analysis (~8 K tokens max).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
ERROR_KW_RE = re.compile(
    r"\b(ERROR|FAIL|FAILED|Traceback|Exception|assert|CRITICAL|"
    r"CommandFailedError|RuntimeError|ConnectionError|Timeout|Dead)\b",
    re.IGNORECASE,
)

TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
TRACEBACK_END_RE = re.compile(
    r"^(\w+(?:\.\w+)*(?:Error|Exception|Failure|Warning).*)", re.MULTILINE
)

# Chained exception markers produced by Python
CHAINED_TB_RE = re.compile(
    r"^(During handling of the above exception|"
    r"The above exception was the direct cause of|"
    r"Caused by:)",
    re.IGNORECASE,
)

# Teuthology prints a summary section near the end
SUMMARY_HEADER_RE = re.compile(
    r"^(=+ SUMMARY =+|summary:|\*+ RESULTS \*+|^Results:)", re.IGNORECASE | re.MULTILINE
)

# Token budget: llama3:8b has 8192 token context.  System prompt ~800 tokens,
# user prompt template ~100 tokens, response ~1500 tokens → ~5800 tokens for log.
# 1 token ≈ 4 chars → ~23K chars raw, but we leave headroom.
MAX_CONDENSED_CHARS = 12_000
CONTEXT_LINES = 5  # lines before/after each error line
# Maximum lines to keep after the summary header (avoids blowing up on
# very large logs where the header appears early).
MAX_SUMMARY_LINES = 200


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
class ParsedLog:
    """Preprocessed representation of a single teuthology.log."""
    job_id: str
    raw_log: str = ""
    error_blocks: List[ErrorBlock] = field(default_factory=list)
    tracebacks: List[str] = field(default_factory=list)
    summary_section: str = ""
    condensed_text: str = ""          # what we send to the LLM
    total_lines: int = 0
    total_chars: int = 0
    raw_log_path: Optional[str] = None

    def get_raw_log(self) -> str:
        """Get raw log text, loading from disk if released from memory."""
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
        """Free raw log from memory. Can be reloaded via raw_log_path."""
        self.raw_log = ""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def _extract_tracebacks(lines: List[str]) -> List[str]:
    """Return a list of full Python traceback strings, including chained
    exceptions (``During handling of ...`` / ``The above exception ...``)."""
    tracebacks: List[str] = []
    i = 0
    while i < len(lines):
        if TRACEBACK_START_RE.search(lines[i]):
            tb_lines = [lines[i]]
            j = i + 1
            # Collect until the exception line or we run out
            while j < len(lines):
                tb_lines.append(lines[j])
                stripped = lines[j].strip()
                if stripped and not stripped.startswith("File ") and not stripped.startswith("..."):
                    if TRACEBACK_END_RE.match(stripped):
                        # Check for chained exception continuation
                        # Look ahead for "During handling..." or another Traceback
                        k = j + 1
                        # Skip blank lines between chains
                        while k < len(lines) and not lines[k].strip():
                            k += 1
                        if k < len(lines) and (
                            CHAINED_TB_RE.match(lines[k].strip())
                            or TRACEBACK_START_RE.search(lines[k])
                        ):
                            # Include the chaining line(s) and continue
                            while k < len(lines) and not TRACEBACK_START_RE.search(lines[k]):
                                tb_lines.append(lines[k])
                                k += 1
                            if k < len(lines) and TRACEBACK_START_RE.search(lines[k]):
                                j = k - 1  # will be incremented below
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
        if ERROR_KW_RE.search(line):
            hit_indices.append(idx)

    if not hit_indices:
        return []

    # Merge overlapping ranges
    ranges: List[tuple[int, int]] = []
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
            start_line=s,
            end_line=e,
            text="\n".join(lines[s : e + 1]),
        ))
    return blocks


def _extract_summary(lines: List[str]) -> str:
    """Extract the summary / results section at the end of the log.

    Caps the returned text to ``MAX_SUMMARY_LINES`` lines after the
    header to avoid blowing up on logs where the header appears far
    from the end.
    """
    # Search from the end — scan up to 2000 lines back for the summary header
    for i in range(len(lines) - 1, max(len(lines) - 2000, -1), -1):
        if SUMMARY_HEADER_RE.search(lines[i]):
            end = min(i + MAX_SUMMARY_LINES, len(lines))
            return "\n".join(lines[i:end])
    # Fallback: last 60 lines often contain useful status info
    return "\n".join(lines[-60:])


def _deduplicate_blocks(blocks: List[ErrorBlock]) -> List[ErrorBlock]:
    """Remove error blocks whose text is identical (keep first)."""
    seen: set[str] = set()
    unique: List[ErrorBlock] = []
    for b in blocks:
        sig = b.text.strip()
        if sig not in seen:
            seen.add(sig)
            unique.append(b)
    return unique


def _clip_middle(text: str, max_chars: int) -> str:
    """
    Trim long text while preserving both beginning and ending context.

    Splits at line boundaries so we never cut in the middle of a
    traceback or error message.
    """
    if len(text) <= max_chars:
        return text

    head_budget = max_chars * 2 // 3
    tail_budget = max_chars - head_budget
    marker = "\n\n... [middle truncated to fit model context] ...\n\n"

    # Find the last newline within the head budget
    head_end = text.rfind("\n", 0, head_budget)
    if head_end < 0:
        head_end = head_budget  # no newline found, hard cut

    # Find the first newline from (end - tail_budget)
    tail_start_search = max(0, len(text) - tail_budget)
    tail_start = text.find("\n", tail_start_search)
    if tail_start < 0:
        tail_start = tail_start_search  # no newline found, hard cut

    return text[:head_end] + marker + text[tail_start:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_log(job_id: str, raw_log: str) -> ParsedLog:
    """
    Parse a raw teuthology log and produce a ParsedLog with all
    extracted sections and a condensed LLM-ready text.
    """
    lines = raw_log.splitlines()

    tracebacks = _extract_tracebacks(lines)
    error_blocks = _extract_error_blocks(lines)
    error_blocks = _deduplicate_blocks(error_blocks)
    summary = _extract_summary(lines)

    # ---- Build condensed text ----
    parts: List[str] = []

    # Summary first — this is the single most informative section for
    # Teuthology logs, containing the failure reason.
    if summary:
        parts.append("=== LOG SUMMARY / TAIL ===")
        parts.append(summary)
        parts.append("")

    # Tracebacks are strong signal for classification
    if tracebacks:
        parts.append(f"=== TRACEBACKS ({len(tracebacks)}) ===")
        for idx, tb in enumerate(tracebacks, 1):
            parts.append(f"--- Traceback #{idx} ---")
            parts.append(tb)
            parts.append("")

    # Error blocks provide surrounding context — limit count to avoid
    # blowing the token budget on noisy logs with hundreds of blocks.
    capped_blocks = error_blocks[:15]
    if capped_blocks:
        parts.append(f"=== RELEVANT LOG SECTIONS ({len(capped_blocks)}) ===")
        for idx, blk in enumerate(capped_blocks, 1):
            parts.append(f"--- Block #{idx} (lines {blk.start_line}-{blk.end_line}) ---")
            parts.append(blk.text)
            parts.append("")

    condensed = "\n".join(parts)

    # Truncate if still too large, keeping both head and tail context.
    if len(condensed) > MAX_CONDENSED_CHARS:
        condensed = _clip_middle(condensed, MAX_CONDENSED_CHARS)

    # If we found nothing, fall back to the tail of the log
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
