#!/usr/bin/env python3
"""
HTML report generator for the Teuthology AI Log Analyzer.

Produces a single self-contained HTML file with embedded CSS and JS.
No external dependencies — uses only Python builtins.

Interactive features:
- Search bar to filter jobs by text
- Severity / failure-type filter buttons
- Sort options (severity, confidence, job ID)
- JSON / CSV export
- Light / dark mode toggle
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from analyzer import AnalysisResult
from cluster import FailureCluster, RunHealth
from log_parser import ParsedLog

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "critical": "#e74c3c",
    "high":     "#e67e22",
    "medium":   "#f1c40f",
    "low":      "#2ecc71",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

MAX_EMBEDDED_RAW_LOG_CHARS = 50_000
MAX_EMBEDDED_LLM_RESPONSE_CHARS = 30_000


def _sev_badge(severity: str) -> str:
    sev = severity.lower()
    color = SEVERITY_COLORS.get(sev, "#95a5a6")
    return (
        f'<span class="badge" style="background:{color};">'
        f'{html.escape(sev.upper())}</span>'
    )


# ---------------------------------------------------------------------------
# CSS (with light mode support)
# ---------------------------------------------------------------------------
CSS = """\
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --fg: #c9d1d9;
  --fg2: #8b949e;
  --accent: #58a6ff;
  --border: #30363d;
  --red: #e74c3c;
  --orange: #e67e22;
  --yellow: #f1c40f;
  --green: #2ecc71;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  --mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}
html.light {
  --bg: #ffffff;
  --bg2: #f6f8fa;
  --fg: #1f2328;
  --fg2: #656d76;
  --accent: #0969da;
  --border: #d0d7de;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
h1 { color: var(--accent); margin-bottom: 0.25rem; font-size: 1.8rem; }
h2 { color: var(--accent); margin: 1.5rem 0 0.75rem; font-size: 1.3rem; }
h3 { color: var(--fg); margin: 1rem 0 0.5rem; font-size: 1.1rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.meta { color: var(--fg2); font-size: 0.9rem; margin-bottom: 1.5rem; }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 700;
  color: #fff;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.summary-box {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  white-space: pre-wrap;
  font-size: 0.95rem;
}
.stats {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin-bottom: 1.5rem;
}
.stat-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.5rem;
  min-width: 140px;
  text-align: center;
}
.stat-card .num { font-size: 2rem; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 0.85rem; color: var(--fg2); }
.stat-card .sub { font-size: 0.8rem; color: var(--fg2); margin-top: 0.25rem; }
.panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin-bottom: 1.25rem;
}
.warning-box {
  background: rgba(231, 76, 60, 0.10);
  border: 1px solid rgba(231, 76, 60, 0.35);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.75rem;
}
.good-box {
  background: rgba(46, 204, 113, 0.10);
  border: 1px solid rgba(46, 204, 113, 0.35);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.75rem;
}
details {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 1rem;
}
details > summary {
  cursor: pointer;
  padding: 0.75rem 1rem;
  font-weight: 600;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
details > summary::before {
  content: '\\25B6';
  font-size: 0.7rem;
  transition: transform 0.2s;
}
details[open] > summary::before {
  transform: rotate(90deg);
}
details > .content {
  padding: 0 1rem 1rem;
}
.field-label {
  color: var(--fg2);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.75rem;
  margin-bottom: 0.25rem;
}
.prose {
  white-space: pre-wrap;
}
.fix-list { padding-left: 1.25rem; }
.fix-list li { margin-bottom: 0.35rem; }
.component-tag {
  display: inline-block;
  background: var(--border);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  margin: 2px 4px 2px 0;
}
pre.log-block {
  background: #010409;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  overflow-x: auto;
  max-height: 500px;
  overflow-y: auto;
  font-family: var(--mono);
  font-size: 0.78rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--fg2);
}
html.light pre.log-block { background: #f0f0f0; }
.error-highlight {
  background: rgba(231, 76, 60, 0.15);
  border-left: 3px solid var(--red);
  padding-left: 0.5rem;
  margin: 0.5rem 0;
}
.status-pill {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 700;
  border-radius: 999px;
  padding: 2px 8px;
  border: 1px solid var(--border);
}
.status-ok { color: var(--green); border-color: rgba(46, 204, 113, 0.5); }
.status-warn { color: var(--orange); border-color: rgba(230, 126, 34, 0.5); }
.simple-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.5rem;
}
.simple-table th, .simple-table td {
  border-bottom: 1px solid var(--border);
  padding: 0.4rem 0.5rem;
  text-align: left;
  font-size: 0.9rem;
}
.simple-table th {
  color: var(--fg2);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--fg2);
  font-size: 0.8rem;
  text-align: center;
}
.toolbar {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.toolbar input[type="text"] {
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--fg);
  font-size: 0.9rem;
  min-width: 200px;
  flex: 1;
}
.toolbar select, .toolbar button {
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--fg);
  font-size: 0.85rem;
  cursor: pointer;
}
.toolbar button:hover { background: var(--border); }
.toolbar .spacer { flex: 1; }
.hidden { display: none !important; }
.user-tag {
  font-size: 0.8em;
  color: var(--accent);
  font-weight: 500;
  margin-left: 0.3em;
}
"""

# ---------------------------------------------------------------------------
# JavaScript for interactivity
# ---------------------------------------------------------------------------
JS = """\
(function() {
  var searchInput = document.getElementById('job-search');
  var userSearch = document.getElementById('user-search');
  var sevFilter = document.getElementById('sev-filter');
  var typeFilter = document.getElementById('type-filter');
  var sortSelect = document.getElementById('sort-select');
  var jobContainer = document.getElementById('job-sections');
  var themeBtn = document.getElementById('theme-toggle');

  function getJobSections() {
    return Array.from(jobContainer.querySelectorAll(':scope > details[data-job-id]'));
  }

  function applyFilters() {
    var query = (searchInput.value || '').toLowerCase();
    var userQuery = userSearch ? (userSearch.value || '').toLowerCase().trim() : '';
    var sev = sevFilter.value;
    var ftype = typeFilter.value;
    var sections = getJobSections();
    var visible = 0;

    var userList = [];
    if (userQuery) {
      userList = userQuery.split(/[,;\\s]+/).filter(function(u) { return u.length > 0; });
    }

    sections.forEach(function(el) {
      var text = el.textContent.toLowerCase();
      var elSev = el.getAttribute('data-severity') || '';
      var elType = el.getAttribute('data-failure-type') || '';
      var elUser = (el.getAttribute('data-user') || '').toLowerCase();
      var show = true;
      if (query && text.indexOf(query) === -1) show = false;
      if (sev && elSev !== sev) show = false;
      if (ftype && elType !== ftype) show = false;
      if (userList.length > 0) {
        var matchUser = userList.some(function(u) { return elUser.indexOf(u) !== -1; });
        if (!matchUser) show = false;
      }
      el.classList.toggle('hidden', !show);
      if (show) visible++;
    });
    var counter = document.getElementById('visible-count');
    if (counter) counter.textContent = visible + ' of ' + sections.length + ' jobs shown';
  }

  function applySort() {
    var sections = getJobSections();
    var order = sortSelect.value;
    var sevOrder = {critical:0, high:1, medium:2, low:3};
    sections.sort(function(a, b) {
      if (order === 'severity') {
        return (sevOrder[a.getAttribute('data-severity')] || 9)
             - (sevOrder[b.getAttribute('data-severity')] || 9);
      } else if (order === 'confidence') {
        return parseFloat(b.getAttribute('data-confidence') || 0)
             - parseFloat(a.getAttribute('data-confidence') || 0);
      } else if (order === 'job-id') {
        return (a.getAttribute('data-job-id') || '').localeCompare(
               b.getAttribute('data-job-id') || '', undefined, {numeric: true});
      } else if (order === 'user') {
        var cmp = (a.getAttribute('data-user') || '').localeCompare(
                   b.getAttribute('data-user') || '');
        if (cmp !== 0) return cmp;
        return (sevOrder[a.getAttribute('data-severity')] || 9)
             - (sevOrder[b.getAttribute('data-severity')] || 9);
      }
      return 0;
    });
    sections.forEach(function(el) { jobContainer.appendChild(el); });
  }

  if (searchInput) searchInput.addEventListener('input', applyFilters);
  if (userSearch) userSearch.addEventListener('input', applyFilters);
  if (sevFilter) sevFilter.addEventListener('change', applyFilters);
  if (typeFilter) typeFilter.addEventListener('change', applyFilters);
  if (sortSelect) sortSelect.addEventListener('change', function() {
    applySort(); applyFilters();
  });

  if (themeBtn) {
    themeBtn.addEventListener('click', function() {
      document.documentElement.classList.toggle('light');
      themeBtn.textContent = document.documentElement.classList.contains('light')
        ? 'Dark Mode' : 'Light Mode';
    });
  }

  var exportJsonBtn = document.getElementById('export-json');
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', function() {
      var dataEl = document.getElementById('report-data');
      if (!dataEl) return;
      var blob = new Blob([dataEl.textContent], {type: 'application/json'});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url; a.download = 'analysis_report.json';
      a.click(); URL.revokeObjectURL(url);
    });
  }

  var exportCsvBtn = document.getElementById('export-csv');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', function() {
      var dataEl = document.getElementById('report-data');
      if (!dataEl) return;
      try {
        var data = JSON.parse(dataEl.textContent);
        var rows = [['job_id','severity','failure_type','confidence','error_category','root_cause','recommended_action']];
        data.forEach(function(r) {
          rows.push([r.job_id, r.severity, r.failure_type, r.confidence,
                      r.error_category, '"'+r.root_cause.replace(/"/g,'""')+'"',
                      '"'+r.recommended_action.replace(/"/g,'""')+'"']);
        });
        var csv = rows.map(function(r){return r.join(',');}).join('\\n');
        var blob = new Blob([csv], {type: 'text/csv'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'analysis_report.csv';
        a.click(); URL.revokeObjectURL(url);
      } catch(e) { alert('Export failed: ' + e); }
    });
  }
})();
"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def _esc(text) -> str:
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text)


def _truncate_for_html(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    suffix = (
        "\n\n... [truncated for HTML report size] ...\n"
        f"(showing first {max_chars:,} of {len(text):,} chars)"
    )
    return text[:max_chars] + suffix, True


def _status_pill(analysis: AnalysisResult) -> str:
    ec = (analysis.error_category or "").lower()
    if "artifact availability" in ec or "log fetch failure" in ec:
        return '<span class="status-pill status-warn">fallback analysis</span>'
    if analysis.success:
        return '<span class="status-pill status-ok">analysis available</span>'
    return '<span class="status-pill status-warn">log unavailable</span>'


def _render_breakdown_table(title: str, counts: Dict[str, int]) -> str:
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"<h3>{_esc(title)}</h3>"]
    parts.append('<table class="simple-table">')
    parts.append("<thead><tr><th>Category</th><th>Count</th></tr></thead>")
    parts.append("<tbody>")
    if rows:
        for key, value in rows:
            parts.append(f"<tr><td>{_esc(key)}</td><td>{value}</td></tr>")
    else:
        parts.append("<tr><td>none</td><td>0</td></tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _render_job_section(
    analysis: AnalysisResult,
    parsed: Optional[ParsedLog],
) -> str:
    """Render a single job's collapsible section with data attributes for filtering."""
    sev = analysis.severity.lower()
    badge = _sev_badge(sev)
    status = _status_pill(analysis)

    run_user = getattr(analysis, "run_user", "") or ""
    run_name = getattr(analysis, "run_name", "") or ""

    parts: List[str] = []
    open_attr = " open" if sev in {"critical", "high"} else ""
    parts.append(
        f'<details id="job-{_esc(analysis.job_id)}" '
        f'data-job-id="{_esc(analysis.job_id)}" '
        f'data-severity="{_esc(sev)}" '
        f'data-failure-type="{_esc(analysis.failure_type)}" '
        f'data-confidence="{analysis.confidence:.2f}" '
        f'data-user="{_esc(run_user)}" '
        f'data-run="{_esc(run_name)}"'
        f'{open_attr}>'
    )
    user_label = f' <span class="user-tag">[{_esc(run_user)}]</span>' if run_user else ""
    parts.append(
        f"<summary>{badge} Job <strong>{_esc(analysis.job_id)}</strong>"
        f"{user_label}"
        f" {status}"
        f" — {_esc(analysis.error_category)}"
        f" — {_esc(analysis.root_cause)}</summary>"
    )
    parts.append('<div class="content">')

    confidence = getattr(analysis, 'confidence', 0.0)
    failure_type = getattr(analysis, 'failure_type', 'unknown')
    recommended_action = getattr(analysis, 'recommended_action', '')
    is_cached = getattr(analysis, 'cached', False)

    conf_pct = int(confidence * 100)
    conf_color = "#2ecc71" if conf_pct >= 75 else "#f1c40f" if conf_pct >= 50 else "#e74c3c"
    parts.append('<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem;">')
    parts.append(
        f'<span class="component-tag">Type: <strong>{_esc(failure_type)}</strong></span>'
    )
    parts.append(
        f'<span class="component-tag">Confidence: '
        f'<strong style="color:{conf_color}">{conf_pct}%</strong></span>'
    )
    parts.append(
        f'<span class="component-tag">Severity: <strong>{_esc(analysis.severity.upper())}</strong></span>'
    )
    if is_cached:
        parts.append('<span class="component-tag">CACHED</span>')
    parts.append('</div>')

    if not analysis.success:
        parts.append(
            '<div class="warning-box"><strong>Log could not be analyzed.</strong> '
            'This job appears in the failed list, but its log artifact was unavailable.</div>'
        )

    parts.append('<div class="field-label">Root Cause</div>')
    parts.append(f'<div class="prose">{_esc(analysis.root_cause)}</div>')

    if recommended_action:
        parts.append('<div class="field-label">Recommended Action</div>')
        parts.append(
            f'<div class="error-highlight" style="border-left-color:#2ecc71;">'
            f'<strong>{_esc(recommended_action)}</strong></div>'
        )

    if analysis.explanation:
        parts.append('<div class="field-label">Detailed Explanation</div>')
        parts.append(f'<div class="prose">{_esc(analysis.explanation)}</div>')

    if analysis.fix_suggestions:
        parts.append('<div class="field-label">Fix Suggestions</div>')
        parts.append('<ul class="fix-list">')
        for s in analysis.fix_suggestions:
            parts.append(f"  <li>{_esc(s)}</li>")
        parts.append("</ul>")

    if analysis.affected_components:
        parts.append('<div class="field-label">Affected Components</div>')
        parts.append("<div>")
        for c in analysis.affected_components:
            parts.append(f'<span class="component-tag">{_esc(c)}</span>')
        parts.append("</div>")

    if analysis.success and parsed and parsed.error_blocks:
        parts.append('<div class="field-label">Extracted Error Blocks</div>')
        for blk in parsed.error_blocks[:10]:
            parts.append('<div class="error-highlight">')
            parts.append(f'<pre class="log-block">{_esc(blk.text)}</pre>')
            parts.append("</div>")

    if analysis.success and parsed and parsed.condensed_text:
        parts.append("<details>")
        cc = len(parsed.condensed_text)
        parts.append(
            f"<summary>Condensed Log Sent to LLM ({cc:,} chars)</summary>"
        )
        parts.append(f'<pre class="log-block">{_esc(parsed.condensed_text)}</pre>')
        parts.append("</details>")

    # Full raw log — use get_raw_log() for lazy loading from disk
    if analysis.success and parsed:
        raw_log_text = parsed.get_raw_log()
        if raw_log_text:
            raw_for_report, was_trimmed = _truncate_for_html(
                raw_log_text, MAX_EMBEDDED_RAW_LOG_CHARS
            )
            parts.append("<details>")
            lc = parsed.total_lines
            cc = parsed.total_chars
            parts.append(
                f"<summary>Full Raw Log ({lc:,} lines, {cc:,} chars)</summary>"
            )
            if was_trimmed:
                parts.append(
                    '<p class="meta">Raw log is truncated in this HTML to keep the '
                    "report usable. Full log is available on disk.</p>"
                )
            parts.append(f'<pre class="log-block">{_esc(raw_for_report)}</pre>')
            parts.append("</details>")

    if analysis.raw_llm_response:
        llm_for_report, llm_trimmed = _truncate_for_html(
            analysis.raw_llm_response, MAX_EMBEDDED_LLM_RESPONSE_CHARS
        )
        parts.append("<details>")
        parts.append("<summary>Raw LLM Response</summary>")
        if llm_trimmed:
            parts.append(
                '<p class="meta">Raw LLM response is truncated for readability.</p>'
            )
        parts.append(
            f'<pre class="log-block">{_esc(llm_for_report)}</pre>'
        )
        parts.append("</details>")

    parts.append("</div>")  # .content
    parts.append("</details>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_html_report(
    run_url: str,
    analyses: List[AnalysisResult],
    parsed_logs: List[ParsedLog],
    executive_summary: str = "",
    model_name: str = "",
    output_path: str = "report.html",
    clusters: Optional[List[FailureCluster]] = None,
    cluster_results: Optional[Dict[str, AnalysisResult]] = None,
    run_health: Optional[RunHealth] = None,
    all_run_urls: Optional[List[str]] = None,
) -> str:
    if clusters is None:
        clusters = []
    if cluster_results is None:
        cluster_results = {}

    parsed_map = {p.job_id: p for p in parsed_logs}

    sorted_analyses = sorted(
        analyses,
        key=lambda a: SEVERITY_ORDER.get(a.severity.lower(), 99),
    )

    total = len(analyses)
    analyzed_ok = [a for a in analyses if a.success]
    unavailable = [a for a in analyses if not a.success]
    fallback_rows = [
        a for a in analyses
        if "artifact availability" in (a.error_category or "").lower()
        or "log fetch failure" in (a.error_category or "").lower()
    ]
    low_conf = [a for a in analyzed_ok if a.confidence < 0.6]
    unknown_type = [a for a in analyzed_ok if a.failure_type == "unknown"]

    by_sev: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for a in analyses:
        s = a.severity.lower()
        by_sev[s] = by_sev.get(s, 0) + 1
        t = a.failure_type or "unknown"
        by_type[t] = by_type.get(t, 0) + 1

    # Collect unique failure types for the filter dropdown
    all_failure_types = sorted({a.failure_type or "unknown" for a in analyses})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Collect unique users for the filter
    all_users = sorted({
        getattr(a, "run_user", "") or "" for a in analyses
    } - {""})
    is_multi_run = bool(all_run_urls and len(all_run_urls) > 1) or len(all_users) > 1

    # Build JSON data for export
    export_data = []
    for a in sorted_analyses:
        entry = {
            "job_id": a.job_id,
            "severity": a.severity,
            "failure_type": a.failure_type,
            "confidence": round(a.confidence, 2),
            "error_category": a.error_category,
            "root_cause": a.root_cause,
            "explanation": a.explanation,
            "recommended_action": a.recommended_action,
            "fix_suggestions": a.fix_suggestions,
            "affected_components": a.affected_components,
        }
        if is_multi_run:
            entry["user"] = getattr(a, "run_user", "") or ""
            entry["run"] = getattr(a, "run_name", "") or ""
        export_data.append(entry)

    html_parts: List[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append('<html lang="en">')
    html_parts.append("<head>")
    html_parts.append('<meta charset="utf-8">')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    html_parts.append("<title>Teuthology Log Analysis Report</title>")
    html_parts.append(f"<style>{CSS}</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Hidden data element for JSON export
    html_parts.append(
        f'<script id="report-data" type="application/json">'
        f'{json.dumps(export_data)}</script>'
    )

    # Header
    html_parts.append("<h1>Teuthology Log Analysis Report</h1>")
    if is_multi_run:
        html_parts.append(f'<p class="meta">Suite: <a href="{_esc(run_url)}">{_esc(run_url)}</a><br>')
        html_parts.append(f"Runs combined: <strong>{len(all_run_urls or [])}</strong> "
                          f"| Users: <strong>{', '.join(all_users) if all_users else 'unknown'}</strong><br>")
    else:
        html_parts.append(f'<p class="meta">Run: <a href="{_esc(run_url)}">{_esc(run_url)}</a><br>')
    if model_name:
        html_parts.append(f"Model: <strong>{_esc(model_name)}</strong><br>")
    html_parts.append(f"Generated: {now}<br>")
    html_parts.append(f"Total failed jobs discovered: {total}</p>")

    # Stat cards
    html_parts.append('<div class="stats">')
    html_parts.append(
        f'<div class="stat-card"><div class="num">{total}</div>'
        f'<div class="label">Failed Jobs (total)</div></div>'
    )
    html_parts.append(
        f'<div class="stat-card"><div class="num">{len(analyzed_ok)}</div>'
        f'<div class="label">Logs Analyzed</div></div>'
    )
    html_parts.append(
        f'<div class="stat-card"><div class="num">{len(unavailable)}</div>'
        f'<div class="label">Logs Unavailable</div></div>'
    )
    html_parts.append(
        f'<div class="stat-card"><div class="num">{len(fallback_rows)}</div>'
        f'<div class="label">Fallback Analyses</div></div>'
    )
    html_parts.append(
        f'<div class="stat-card"><div class="num">{len(low_conf)}</div>'
        f'<div class="label">Low Confidence (&lt;60%)</div></div>'
    )
    if clusters:
        html_parts.append(
            f'<div class="stat-card"><div class="num">{len(clusters)}</div>'
            f'<div class="label">Failure Clusters</div>'
            f'<div class="sub">{total} jobs grouped</div></div>'
        )
    for sev in ("critical", "high", "medium", "low"):
        cnt = by_sev.get(sev, 0)
        if cnt:
            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            html_parts.append(
                f'<div class="stat-card">'
                f'<div class="num" style="color:{color};">{cnt}</div>'
                f'<div class="label">{sev.upper()}</div></div>'
            )
    html_parts.append("</div>")

    # Run Health Dashboard
    if run_health is not None:
        health_color = {
            "mass_failure": "#e74c3c",
            "partial_failure": "#e67e22",
            "isolated": "#2ecc71",
        }.get(run_health.classification, "#95a5a6")
        health_label = run_health.classification.replace("_", " ").upper()

        html_parts.append('<div class="panel">')
        html_parts.append("<h2>Run Health</h2>")
        html_parts.append(
            f'<div style="display:flex;align-items:center;gap:1.5rem;'
            f'margin-bottom:0.75rem;">'
        )
        html_parts.append(
            f'<span class="badge" style="background:{health_color};'
            f'font-size:0.9rem;padding:4px 16px;">{health_label}</span>'
        )
        html_parts.append(
            f'<span style="font-size:1.3rem;font-weight:700;color:{health_color};">'
            f'{run_health.pct}</span>'
            f'<span style="color:var(--fg2);font-size:0.9rem;">'
            f'failure ratio ({run_health.failed_jobs}/{run_health.total_jobs} jobs)</span>'
        )
        html_parts.append("</div>")
        html_parts.append(
            f'<p style="color:var(--fg2);font-size:0.9rem;">'
            f'{_esc(run_health.hint_for_llm)}</p>'
        )
        html_parts.append("</div>")

    # Failure Cluster Overview
    if clusters:
        html_parts.append('<div class="panel">')
        html_parts.append(
            f"<h2>Failure Clusters ({len(clusters)} patterns "
            f"&rarr; {total} jobs)</h2>"
        )
        if len(clusters) == 1 and clusters[0].size > 5:
            html_parts.append(
                '<div class="warning-box"><strong>All failures share the same '
                'error signature.</strong> This strongly suggests a single systemic '
                'root cause (infra, bad image, or environment issue).</div>'
            )
        elif len(clusters) <= 3 and total > 20:
            html_parts.append(
                '<div class="warning-box"><strong>Only '
                f'{len(clusters)} distinct failure pattern(s)</strong> across '
                f'{total} jobs — likely 1-3 systemic issues.</div>'
            )

        html_parts.append('<table class="simple-table">')
        html_parts.append(
            "<thead><tr>"
            "<th>#</th><th>Jobs</th><th>Severity</th>"
            "<th>Type</th><th>Confidence</th>"
            "<th>Root Cause / Signature</th>"
            "</tr></thead>"
        )
        html_parts.append("<tbody>")
        for ci, cluster in enumerate(clusters, start=1):
            cr = cluster_results.get(cluster.cluster_id)
            if cr and cr.success:
                sev = cr.severity.upper()
                sev_color = SEVERITY_COLORS.get(cr.severity.lower(), "#95a5a6")
                ftype = cr.failure_type
                conf = f"{int(cr.confidence * 100)}%"
                root = cr.root_cause
            else:
                sev = "?"
                sev_color = "#95a5a6"
                ftype = "—"
                conf = "—"
                root = cluster.signature[:120]

            job_links = ", ".join(
                f'<a href="#job-{_esc(jid)}">{_esc(jid)}</a>'
                for jid in cluster.job_ids[:10]
            )
            if cluster.size > 10:
                job_links += f" &hellip; (+{cluster.size - 10} more)"

            html_parts.append(
                "<tr>"
                f'<td>{ci}</td>'
                f'<td><strong>{cluster.size}</strong>'
                f'<div style="font-size:0.75rem;color:var(--fg2);">'
                f'{job_links}</div></td>'
                f'<td><span class="badge" style="background:{sev_color};">'
                f'{sev}</span></td>'
                f'<td>{_esc(ftype)}</td>'
                f'<td>{conf}</td>'
                f'<td style="max-width:400px;overflow:hidden;'
                f'text-overflow:ellipsis;">{_esc(root)}</td>'
                "</tr>"
            )
        html_parts.append("</tbody></table>")
        html_parts.append("</div>")

    # Snapshot & breakdowns
    html_parts.append('<div class="panel">')
    html_parts.append("<h2>Run Snapshot</h2>")
    if unavailable:
        html_parts.append(
            f'<div class="warning-box">{len(unavailable)} job(s) could not be analyzed '
            "because their logs were unavailable from qa-proxy.</div>"
        )
    elif fallback_rows:
        html_parts.append(
            f'<div class="warning-box">{len(fallback_rows)} job(s) were analyzed using '
            "fallback artifact metadata because full logs were unavailable.</div>"
        )
    else:
        html_parts.append(
            '<div class="good-box">All discovered failed jobs had downloadable logs.</div>'
        )
    if unknown_type:
        html_parts.append(
            f'<div class="warning-box">{len(unknown_type)} analyzed job(s) still have '
            "failure_type=unknown; review these first.</div>"
        )
    html_parts.append("</div>")

    html_parts.append('<div class="panel">')
    html_parts.append(_render_breakdown_table("Failure Type Breakdown", by_type))
    html_parts.append("</div>")

    # Needs Attention
    fallback_ids = {a.job_id for a in fallback_rows}
    attention_rows = [
        a for a in sorted_analyses
        if (not a.success)
        or (a.job_id in fallback_ids)
        or (a.confidence < 0.6)
        or (a.failure_type == "unknown")
    ]
    if attention_rows:
        html_parts.append('<div class="panel">')
        html_parts.append("<h2>Needs Attention</h2>")
        html_parts.append('<table class="simple-table">')
        html_parts.append(
            "<thead><tr><th>Job</th><th>Status</th><th>Type</th>"
            "<th>Confidence</th><th>Root Cause</th></tr></thead>"
        )
        html_parts.append("<tbody>")
        for a in attention_rows[:30]:
            if not a.success:
                attn_status = "log unavailable"
            elif a.job_id in fallback_ids:
                attn_status = "fallback metadata"
            else:
                attn_status = "low-confidence"
            html_parts.append(
                "<tr>"
                f"<td><a href=\"#job-{_esc(a.job_id)}\">{_esc(a.job_id)}</a></td>"
                f"<td>{_esc(attn_status)}</td>"
                f"<td>{_esc(a.failure_type)}</td>"
                f"<td>{int(a.confidence * 100)}%</td>"
                f"<td>{_esc(a.root_cause)}</td>"
                "</tr>"
            )
        html_parts.append("</tbody></table>")
        html_parts.append("</div>")

    # Executive summary
    if executive_summary:
        html_parts.append("<h2>Executive Summary</h2>")
        html_parts.append(f'<div class="summary-box">{_esc(executive_summary)}</div>')

    # Interactive toolbar
    html_parts.append("<h2>Per-Job Analysis</h2>")
    html_parts.append('<div class="toolbar">')
    html_parts.append(
        '<input type="text" id="job-search" placeholder="Search jobs...">'
    )
    html_parts.append(
        '<input type="text" id="user-search" '
        'placeholder="Filter by user(s)..." '
        'title="Enter one or more usernames separated by commas or spaces">'
    )
    html_parts.append(
        '<select id="sev-filter">'
        '<option value="">All Severities</option>'
        '<option value="critical">Critical</option>'
        '<option value="high">High</option>'
        '<option value="medium">Medium</option>'
        '<option value="low">Low</option>'
        '</select>'
    )
    type_options = '<option value="">All Types</option>'
    for ft in all_failure_types:
        type_options += f'<option value="{_esc(ft)}">{_esc(ft)}</option>'
    html_parts.append(f'<select id="type-filter">{type_options}</select>')
    sort_options = (
        '<select id="sort-select">'
        '<option value="severity">Sort: Severity</option>'
        '<option value="confidence">Sort: Confidence</option>'
        '<option value="job-id">Sort: Job ID</option>'
    )
    if is_multi_run:
        sort_options += '<option value="user">Sort: User</option>'
    sort_options += '</select>'
    html_parts.append(sort_options)
    html_parts.append('<span id="visible-count" class="meta" '
                      f'style="margin:0;">{total} of {total} jobs shown</span>')
    html_parts.append('<span class="spacer"></span>')
    html_parts.append('<button id="export-json">Export JSON</button>')
    html_parts.append('<button id="export-csv">Export CSV</button>')
    html_parts.append('<button id="theme-toggle">Light Mode</button>')
    html_parts.append("</div>")

    # Per-job sections (wrapped in a container for JS)
    html_parts.append('<div id="job-sections">')
    for a in sorted_analyses:
        parsed = parsed_map.get(a.job_id)
        html_parts.append(_render_job_section(a, parsed))
    html_parts.append("</div>")

    # Footer
    html_parts.append("<footer>")
    model_note = f" ({_esc(model_name)})" if model_name else ""
    html_parts.append(
        f"Generated by <strong>Teuthology AI Log Analyzer</strong> "
        f"using LLM{model_note} &middot; {now}"
    )
    html_parts.append("</footer>")

    # Inline JS (at end of body for DOM readiness)
    html_parts.append(f"<script>{JS}</script>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    full_html = "\n".join(html_parts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return os.path.abspath(output_path)
