"""Tests for log_parser module."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from log_parser import (
    parse_log, _extract_tracebacks, _extract_error_blocks,
    _extract_summary, _deduplicate_blocks, _clip_middle,
    ErrorBlock, ParsedLog, CONTEXT_LINES,
)


# ---------------------------------------------------------------------------
# Traceback extraction
# ---------------------------------------------------------------------------
class TestExtractTracebacks:
    def test_simple_traceback(self):
        log = (
            "some preamble\n"
            "Traceback (most recent call last):\n"
            '  File "foo.py", line 10, in <module>\n'
            "    do_stuff()\n"
            '  File "bar.py", line 5, in do_stuff\n'
            "    raise RuntimeError('boom')\n"
            "RuntimeError: boom\n"
            "some epilog\n"
        )
        tbs = _extract_tracebacks(log.splitlines())
        assert len(tbs) == 1
        assert "Traceback (most recent call last):" in tbs[0]
        assert "RuntimeError" in tbs[0]

    def test_chained_traceback(self):
        log = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1, in f\n'
            "ValueError: bad value\n"
            "\n"
            "During handling of the above exception, another exception occurred:\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "b.py", line 2, in h\n'
            "RuntimeError: wrapped\n"
        )
        tbs = _extract_tracebacks(log.splitlines())
        assert len(tbs) >= 1
        all_text = "\n".join(tbs)
        assert "ValueError" in all_text
        assert "RuntimeError" in all_text

    def test_no_traceback(self):
        log = "INFO all good\nDEBUG nothing here\n"
        tbs = _extract_tracebacks(log.splitlines())
        assert tbs == []

    def test_multiple_tracebacks(self):
        log = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1\n'
            "ValueError: one\n"
            "gap line\n"
            "Traceback (most recent call last):\n"
            '  File "b.py", line 2\n'
            "TypeError: two\n"
        )
        tbs = _extract_tracebacks(log.splitlines())
        assert len(tbs) == 2


# ---------------------------------------------------------------------------
# Error block extraction
# ---------------------------------------------------------------------------
class TestExtractErrorBlocks:
    def test_basic_error(self):
        lines = ["ok", "ok", "ERROR: something failed", "ok", "ok"]
        blocks = _extract_error_blocks(lines)
        assert len(blocks) == 1
        assert "ERROR: something failed" in blocks[0].text

    def test_context_lines(self):
        lines = [f"line{i}" for i in range(20)]
        lines[10] = "FAIL: test broke"
        blocks = _extract_error_blocks(lines)
        assert len(blocks) == 1
        assert blocks[0].start_line == 10 - CONTEXT_LINES
        assert blocks[0].end_line == 10 + CONTEXT_LINES

    def test_overlapping_blocks_merged(self):
        lines = [f"line{i}" for i in range(20)]
        lines[5] = "ERROR: first"
        lines[7] = "ERROR: second"
        blocks = _extract_error_blocks(lines)
        assert len(blocks) == 1

    def test_no_errors(self):
        lines = ["all fine", "nothing wrong"]
        blocks = _extract_error_blocks(lines)
        assert blocks == []


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------
class TestExtractSummary:
    def test_summary_header(self):
        lines = [f"line{i}" for i in range(100)]
        lines[90] = "======= SUMMARY ======="
        lines[91] = "passed: 10"
        lines[92] = "failed: 2"
        summary = _extract_summary(lines)
        assert "SUMMARY" in summary
        assert "passed: 10" in summary

    def test_fallback_tail(self):
        lines = [f"line{i}" for i in range(100)]
        summary = _extract_summary(lines)
        assert "line99" in summary

    def test_deep_summary(self):
        """Summary header 1500 lines from end should still be found."""
        lines = [f"line{i}" for i in range(2000)]
        lines[500] = "======= SUMMARY ======="
        summary = _extract_summary(lines)
        assert "SUMMARY" in summary


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplicateBlocks:
    def test_duplicates_removed(self):
        blocks = [
            ErrorBlock(0, 2, "error A"),
            ErrorBlock(5, 7, "error A"),
            ErrorBlock(10, 12, "error B"),
        ]
        unique = _deduplicate_blocks(blocks)
        assert len(unique) == 2

    def test_no_duplicates(self):
        blocks = [
            ErrorBlock(0, 2, "error A"),
            ErrorBlock(5, 7, "error B"),
        ]
        unique = _deduplicate_blocks(blocks)
        assert len(unique) == 2


# ---------------------------------------------------------------------------
# Clip middle
# ---------------------------------------------------------------------------
class TestClipMiddle:
    def test_short_text_unchanged(self):
        text = "short text"
        assert _clip_middle(text, 100) == text

    def test_long_text_clipped(self):
        text = "a\n" * 10000
        clipped = _clip_middle(text, 200)
        assert len(clipped) < len(text)
        assert "truncated" in clipped


# ---------------------------------------------------------------------------
# Full parse_log integration
# ---------------------------------------------------------------------------
class TestParseLog:
    def test_basic_parse(self):
        raw = (
            "INFO starting test\n"
            "DEBUG doing stuff\n"
            "Traceback (most recent call last):\n"
            '  File "test.py", line 42, in test_foo\n'
            "    assert False\n"
            "AssertionError\n"
            "ERROR: test_foo failed\n"
            "======= SUMMARY =======\n"
            "1 failed, 0 passed\n"
        )
        pl = parse_log("12345", raw)
        assert pl.job_id == "12345"
        assert pl.total_lines == 9
        assert len(pl.tracebacks) >= 1
        assert len(pl.error_blocks) >= 1
        assert "SUMMARY" in pl.summary_section
        assert len(pl.condensed_text) > 0

    def test_empty_log(self):
        pl = parse_log("99999", "")
        assert pl.job_id == "99999"
        assert pl.condensed_text != ""

    def test_lazy_raw_log(self, tmp_path):
        raw = "ERROR: test data for lazy loading\n"
        log_file = tmp_path / "test.log"
        log_file.write_text(raw)

        pl = parse_log("11111", raw)
        pl.raw_log_path = str(log_file)
        pl.release_raw_log()

        assert pl.raw_log == ""
        reloaded = pl.get_raw_log()
        assert "test data for lazy loading" in reloaded
