"""Automatic failure analysis.

Classification is rule-first: a regex table maps the error text to a category
with a default cause and fix. The knowledge base then supplies project-specific
remediation, and a local model - if one is loaded - writes the human summary
grounded in those retrieved passages. Without a model the diagnosis is still
useful, just terser.
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import Diagnosis, RunResult, StepResult

ANALYZE_SYSTEM = (
    "You are a senior SDET triaging a failed Playwright test. Use the retrieved "
    "notes as ground truth. Reply with three short lines: CAUSE:, FIX:, CONFIDENCE: "
    "(0-1). No preamble."
)

# (pattern, category, cause, fix)
_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    # Must precede selector_not_found: our own AmbiguousMatch message and
    # Playwright's strict-mode error both say "could not locate" / "strict mode
    # violation", and a broader pattern listed first used to swallow them, so
    # this category could never be produced.
    (re.compile(r"strict mode violation.*resolved to \d+ elements|AmbiguousMatch", re.I | re.S),
     "ambiguous_selector",
     "The selector matched several elements, so it does not identify one control.",
     "Narrow the description or scope the locator to a container; add a data-testid "
     "to the intended element."),
    (re.compile(r"could not locate|ElementNotFound|no element matches", re.I),
     "selector_not_found",
     "The element description did not bind to any element on the page.",
     "Check the page actually rendered the control, then let the healer re-bind it; "
     "add a data-testid to make the binding permanent."),
    (re.compile(r"Timeout .*exceeded|TimeoutError|waiting for", re.I),
     "timeout",
     "The action timed out waiting for the element or navigation.",
     "Confirm the app is serving on the expected port; raise browser.default_timeout_ms "
     "for genuinely slow flows; prefer web-first assertions over sleeps."),
    (re.compile(r"net::ERR_CONNECTION_REFUSED|ECONNREFUSED|Connection refused", re.I),
     "app_unreachable",
     "Nothing was listening on the target URL.",
     "Start the bundled demo app with 'forkable serve', or point --base-url at a running app."),
    (re.compile(r"offline mode: refused", re.I),
     "offline_guard",
     "Code attempted a non-loopback connection while the offline guard was armed.",
     "Keep all traffic on loopback, or unset FORKABLE_OFFLINE if the call is legitimate."),
    (re.compile(r"net::ERR_NAME_NOT_RESOLVED|getaddrinfo", re.I),
     "dns",
     "A hostname could not be resolved.",
     "Air-gapped runs must target 127.0.0.1; add a hosts entry if a name is required."),
    (re.compile(r"expected url|url mismatch|expect_url", re.I),
     "assertion_url",
     "Navigation ended somewhere other than the expected URL.",
     "Verify the redirect chain and any auth guard; assert on a path fragment rather than a full URL."),
    (re.compile(r"expected text|text not found|expect_text", re.I),
     "assertion_text",
     "The expected copy was not present.",
     "Copy may have changed or the state did not update; assert on a role/testid instead of prose."),
    (re.compile(r"visual diff|pixel", re.I),
     "visual_regression",
     "The rendered page drifted from its baseline beyond tolerance.",
     "Inspect the diff image; if the change is intended, refresh the baseline."),
    (re.compile(r"playwright is not installed|could not launch|Executable doesn't exist", re.I),
     "browser_missing",
     "Playwright or its browser bundle is not installed.",
     "Run 'python -m playwright install chromium' once with network access, or seed "
     "PLAYWRIGHT_BROWSERS_PATH from an offline bundle."),
]


def classify(error_text: str) -> tuple[str, str, str]:
    for pattern, category, cause, fix in _PATTERNS:
        if pattern.search(error_text or ""):
            return category, cause, fix
    return ("unknown", "Unrecognised failure mode.",
            "Inspect the screenshot and step log; re-run with --headed to watch it live.")


class FailureAnalyzer:
    def __init__(self, settings, knowledge: Any = None, llm: Any = None) -> None:
        self.settings = settings
        self.knowledge = knowledge
        self.llm = llm

    # ------------------------------------------------------------------
    def analyze_text(self, error_text: str, context: str = "") -> Diagnosis:
        category, cause, fix = classify(error_text)
        citations: list[str] = []
        retrieved = ""

        if self.knowledge is not None:
            try:
                hits = self.knowledge.search(f"{category} {error_text[:200]}", top_k=3)
                citations = [h.citation for h in hits]
                retrieved = self.knowledge.context_block(hits, max_chars=2000)
            except Exception:
                pass

        summary = f"{category.replace('_', ' ')}: {error_text.strip()[:240]}"
        confidence = 0.55 if category != "unknown" else 0.25
        source = "rules"

        if self.llm is not None and getattr(self.llm, "name", "") != "rules":
            prompt = (
                "# task: analyze\n"
                f"Category: {category}\n"
                f"Error:\n{error_text[:1500]}\n\n"
                f"Step context:\n{context[:800]}\n\n"
                f"Retrieved notes:\n{retrieved[:2000]}\n"
            )
            try:
                response = self.llm.generate(prompt, system=ANALYZE_SYSTEM)
                parsed = _parse_analysis(response.text)
                if parsed:
                    cause = parsed.get("cause") or cause
                    fix = parsed.get("fix") or fix
                    confidence = parsed.get("confidence", confidence)
                    source = "llm"
            except Exception:
                pass

        return Diagnosis(
            category=category,
            summary=summary,
            likely_cause=cause,
            suggested_fix=fix,
            confidence=round(float(confidence), 2),
            citations=citations,
            source=source,
        )

    # ------------------------------------------------------------------
    def analyze_run(self, result: RunResult) -> Diagnosis | None:
        failed: StepResult | None = next(
            (s for s in result.steps if s.status == "failed"), None
        )
        if failed is None:
            return None
        context = (
            f"step {failed.index}: {failed.step.action} target={failed.step.target!r} "
            f"value={failed.step.value!r} selector={failed.selector} strategy={failed.strategy}"
        )
        evidence = describe_page_evidence(getattr(result, "diagnostics", None) or {})
        if evidence:
            context += f"\nThe page itself reported: {evidence}"
        diagnosis = self.analyze_text(failed.message, context)
        if evidence:
            diagnosis.likely_cause = f"{diagnosis.likely_cause} The page itself reported: {evidence}".strip()
            if (getattr(result, "diagnostics", None) or {}).get("page_errors"):
                diagnosis.suggested_fix = (
                    "Fix the uncaught JavaScript error first - a control that never rendered is "
                    f"usually a symptom of it. {diagnosis.suggested_fix}"
                ).strip()
        return diagnosis


def describe_page_evidence(diagnostics: dict[str, list[str]], limit: int = 2) -> str:
    """One line summarising what the page reported while the plan ran.

    A test that fails with "could not locate" is often reporting a symptom: the
    app threw, a request failed, and the control never rendered. Putting that in
    the diagnosis saves the reader from re-deriving it from a screenshot.
    """
    labels = (
        ("page_errors", "uncaught JavaScript error"),
        ("console_errors", "console error"),
        ("failed_requests", "failed request"),
        ("http_errors", "HTTP error response"),
    )
    parts: list[str] = []
    for key, label in labels:
        items = diagnostics.get(key) or []
        if not items:
            continue
        shown = "; ".join(str(i)[:140] for i in items[:limit])
        more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
        parts.append(f"{len(items)} {label}{'s' if len(items) != 1 else ''}: {shown}{more}")
    return " | ".join(parts)


def _parse_analysis(text: str) -> dict:
    out: dict = {}
    for line in (text or "").splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("cause:"):
            out["cause"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("fix:"):
            out["fix"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("confidence:"):
            match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if match:
                try:
                    out["confidence"] = max(0.0, min(1.0, float(match.group(1))))
                except ValueError:
                    pass
    return out
