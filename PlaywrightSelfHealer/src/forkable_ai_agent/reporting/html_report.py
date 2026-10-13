"""Run reports.

One HTML file with inline CSS and no external requests - a report that needs a
CDN is useless on the machine this agent is built for. A sibling JSON file
carries the same data for CI consumers.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from ..schema import RunResult

_CSS = """
:root{--ink:#0f1724;--muted:#657089;--line:#e2e6ee;--ok:#0f6e5c;--bad:#a3132b;--warn:#8a5b00;--heal:#3b3ba8}
*{box-sizing:border-box}
body{margin:0;background:#f7f8fa;color:var(--ink);
 font:15px/1.5 "Inter",system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:var(--ink);color:#eef2f7;padding:22px 28px}
header h1{margin:0;font-size:19px;letter-spacing:-.01em}
header p{margin:4px 0 0;color:#9dabc2;font-size:13px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
main{max-width:1000px;margin:26px auto;padding:0 24px 60px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px}
.card{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card b{display:block;font-size:22px;letter-spacing:-.02em}
.card span{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13.5px}
th{background:#eff2f6;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
tr:last-child td{border-bottom:0}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;letter-spacing:.04em}
.passed{background:#e4f4ef;color:var(--ok)}
.failed{background:#fce9ec;color:var(--bad)}
.warned{background:#fdf3e0;color:var(--warn)}
.healed{background:#e9e9fb;color:var(--heal)}
h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:30px 0 10px}
.panel{background:#fff;border:1px solid var(--line);border-left:3px solid var(--bad);border-radius:8px;padding:14px 16px}
.panel dt{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-top:8px}
.panel dd{margin:2px 0 0}
.panel.note{border-left-color:var(--warn)}
.panel ul{margin:4px 0 0;padding-left:18px}
.panel li{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;word-break:break-word}
footer{color:var(--muted);font-size:12px;margin-top:34px}
"""


def _tag(status: str) -> str:
    return f'<span class="tag {html.escape(status)}">{html.escape(status)}</span>'


def render_html(result: RunResult, memory: Any = None) -> str:
    plan = result.plan
    rows = []
    for step in result.steps:
        healed = ' <span class="tag healed">healed</span>' if step.healed else ""
        detail = html.escape(step.message or "")
        if step.screenshot:
            detail += f'<br><span class="mono">{html.escape(Path(step.screenshot).name)}</span>'
        rows.append(
            "<tr>"
            f"<td class='mono'>{step.index}</td>"
            f"<td><code>{html.escape(step.step.action)}</code></td>"
            f"<td>{html.escape(step.step.target or '-')}</td>"
            f"<td>{html.escape(step.step.value or '-')}</td>"
            f"<td class='mono'>{html.escape(step.selector or '-')}<br>"
            f"<span style='color:#657089'>{html.escape(step.strategy or '')}</span></td>"
            f"<td>{_tag(step.status)}{healed}</td>"
            f"<td class='mono'>{step.duration_ms:.0f} ms</td>"
            f"<td>{detail}</td>"
            "</tr>"
        )

    diagnosis_block = ""
    if result.diagnosis:
        d = result.diagnosis
        citations = ", ".join(html.escape(c) for c in d.citations) or "none"
        diagnosis_block = f"""
        <h2>Automatic failure analysis</h2>
        <div class="panel">
          <dl>
            <dt>Category</dt><dd><code>{html.escape(d.category)}</code>
              &nbsp;confidence {d.confidence:.2f} &nbsp;via {html.escape(d.source)}</dd>
            <dt>Summary</dt><dd>{html.escape(d.summary)}</dd>
            <dt>Likely cause</dt><dd>{html.escape(d.likely_cause)}</dd>
            <dt>Suggested fix</dt><dd>{html.escape(d.suggested_fix)}</dd>
            <dt>Retrieved from</dt><dd class="mono">{citations}</dd>
          </dl>
        </div>"""

    diagnostics = getattr(result, "diagnostics", None) or {}
    labels = {
        "page_errors": "Uncaught JavaScript errors",
        "console_errors": "Console errors",
        "failed_requests": "Failed requests",
        "http_errors": "HTTP error responses",
    }
    diag_sections = "".join(
        f"<dt>{labels.get(key, html.escape(key))} ({len(items)})</dt><dd><ul>"
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in items[:10])
        + "</ul></dd>"
        for key, items in diagnostics.items() if items
    )
    # Page evidence matters most when something failed or the page threw; on a
    # clean pass it is noise, so it only appears when there is a reason to look.
    page_block = ""
    if diag_sections and (result.status == "failed" or diagnostics.get("page_errors")):
        page_block = f"""
        <h2>What the page reported</h2>
        <div class="panel note"><dl>{diag_sections}</dl></div>"""

    ambiguous_steps = [
        s for s in result.steps if "uniquely" in (s.message or "") or "elements matched" in (s.message or "")
    ]
    ambiguity_block = ""
    if ambiguous_steps:
        items = "".join(
            f"<li>step {s.index}: {html.escape(s.step.target or s.step.action)} - {html.escape(s.message)}</li>"
            for s in ambiguous_steps
        )
        ambiguity_block = f"""
        <h2>Ambiguous selectors</h2>
        <div class="panel note"><dl><dt>These steps matched more than one element</dt>
        <dd><ul>{items}</ul></dd></dl></div>"""

    trace = (getattr(result, "artifacts", None) or {}).get("trace", "")
    trace_block = ""
    if trace:
        trace_block = f"""
        <h2>Trace</h2>
        <div class="panel note"><dl><dt>Playwright trace</dt>
        <dd class="mono">{html.escape(trace)}</dd>
        <dt>Open with</dt><dd class="mono">python -m playwright show-trace {html.escape(trace)}</dd></dl></div>"""

    proposals_block = ""
    proposals = list(getattr(memory, "proposals", []) or [])[-10:] if memory is not None else []
    if proposals:
        body = "".join(
            "<tr>"
            f"<td class='mono'>{html.escape(str(p.get('scope', '')))}</td>"
            f"<td>{html.escape(str(p.get('description', '')))}</td>"
            f"<td class='mono'>{html.escape(str(p.get('key', '')))}</td>"
            f"<td>{html.escape(str(p.get('reason', '')))}</td>"
            "</tr>"
            for p in reversed(proposals)
        )
        proposals_block = f"""
        <h2>Heals held for review (not trusted)</h2>
        <table><thead><tr><th>Scope</th><th>Description</th><th>Locator</th><th>Why held</th></tr></thead>
        <tbody>{body}</tbody></table>"""

    healing_block = ""
    if memory is not None:
        healed_rows = [r for r in memory.healing_report() if r["healed"]][:15]
        if healed_rows:
            body = "".join(
                "<tr>"
                f"<td class='mono'>{html.escape(r['scope'])}</td>"
                f"<td>{html.escape(r['description'])}</td>"
                f"<td class='mono'>{html.escape(r['locator'])}</td>"
                f"<td class='mono'>{r['hits']}/{r['hits'] + r['misses']}</td>"
                "</tr>"
                for r in healed_rows
            )
            healing_block = f"""
        <h2>Learned locators</h2>
        <table><thead><tr><th>Scope</th><th>Description</th><th>Locator</th><th>Hits</th></tr></thead>
        <tbody>{body}</tbody></table>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlaywrightSelfHealer - {html.escape(plan.name)}</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>PlaywrightSelfHealer run report</h1>
  <p>{html.escape(plan.name)} &middot; plan source: {html.escape(plan.source)}
     &middot; {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.started_at))}</p>
</header>
<main>
  <div class="cards">
    <div class="card"><b>{html.escape(result.status)}</b><span>result</span></div>
    <div class="card"><b>{len(result.steps)}</b><span>steps</span></div>
    <div class="card"><b>{result.healed_count}</b><span>healed</span></div>
    <div class="card"><b>{result.duration_s:.1f}s</b><span>duration</span></div>
    <div class="card"><b>{html.escape(plan.base_url)}</b><span>target</span></div>
  </div>
  <h2>Steps</h2>
  <table><thead><tr>
    <th>#</th><th>Action</th><th>Target</th><th>Value</th><th>Bound selector</th>
    <th>Status</th><th>Time</th><th>Detail</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
  {diagnosis_block}
  {page_block}
  {ambiguity_block}
  {trace_block}
  {healing_block}
  {proposals_block}
  <footer>Generated offline by PlaywrightSelfHealer. No external resources are referenced by this file.</footer>
</main></body></html>"""


def write_report(settings, result: RunResult, memory: Any | None = None) -> Path:
    directory = settings.path(settings.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(result.started_at))
    base = directory / f"{result.plan.name}-{stamp}"
    html_path = base.with_suffix(".html")
    html_path.write_text(render_html(result, memory), encoding="utf-8")
    base.with_suffix(".json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return html_path
