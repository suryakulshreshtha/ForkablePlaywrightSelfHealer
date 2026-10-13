#!/usr/bin/env python3
"""Measure healing accuracy against a labelled set.

`RESULTS.md` can claim the agent survives a refactor. This turns that claim
into a number that CI can fail on, and it is what makes the thresholds in
`config/agent.toml` tunable rather than guessed.

    python scripts/benchmark_healing.py                  # report
    python scripts/benchmark_healing.py --min-accuracy 0.83 --max-false-heals 0
    python scripts/benchmark_healing.py --sweep           # threshold tuning

Metrics
-------
accuracy      correct element / positive cases. The headline number.
refusals      positive cases the agent gave up on. An honest failure.
false heals   negative cases it "healed" anyway. The dangerous number -- a
              confident wrong answer is worse than no answer, so the default
              budget for these is zero.
ambiguous     heals that matched more than one element. Counted separately
              because .first silently choosing is a correctness risk even
              when the chosen element happens to be right.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from forkable_ai_agent.agent.healer import ElementNotFound, LocatorResolver  # noqa: E402
from forkable_ai_agent.agent.memory import Memory  # noqa: E402
from forkable_ai_agent.config import load_settings  # noqa: E402
from tests.support.fake_page import Document, FakePage  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "healing_benchmark.json"


@dataclass
class Report:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    refused: int = 0
    ambiguous: int = 0
    false_heals: int = 0
    negatives: int = 0
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def refusal_rate(self) -> float:
        return self.refused / self.total if self.total else 0.0


def _page(html: str) -> FakePage:
    page = FakePage()
    page.document = Document(html)
    page.url = "http://127.0.0.1:8799/benchmark"
    return page


def _resolver(heuristic: float | None = None, ask_llm: float | None = None):
    settings = load_settings()
    settings.root = tempfile.mkdtemp(prefix="forkable-bench-")
    # Static pages: there is nothing to wait for, so do not.
    settings.healer.settle_ms = 0
    settings.healer.dom_quiet_ms = 0
    if heuristic is not None:
        settings.healer.heuristic_threshold = heuristic
    if ask_llm is not None:
        settings.healer.ask_llm_below = ask_llm
    memory = Memory(Path(settings.root) / "memory.json")
    # No LLM: the benchmark measures the deterministic scorer, which is the
    # part that must never regress. A model on top can only be judged fairly
    # once the floor beneath it is fixed.
    return LocatorResolver(settings, memory, llm=None)


def run(data: dict, heuristic: float | None = None, ask_llm: float | None = None) -> Report:
    report = Report()

    for case in data["cases"]:
        report.total += 1
        resolver = _resolver(heuristic, ask_llm)
        page = _page(case["html"])
        try:
            resolution = resolver.resolve(page, case["description"], case.get("action", ""))
        except ElementNotFound:
            report.refused += 1
            report.failures.append(f"REFUSED  {case['id']:<32} {case['description']!r}")
            continue

        node = resolution.locator._node() if hasattr(resolution.locator, "_node") else None
        got = node.get("id") if node is not None else ""
        if got == case["expect"]:
            report.correct += 1
            if resolution.ambiguous:
                report.ambiguous += 1
        else:
            report.wrong += 1
            report.failures.append(
                f"WRONG    {case['id']:<32} want #{case['expect']} got #{got or '?'}"
            )

    for case in data.get("negatives", []):
        if case.get("id") is None:
            continue
        report.negatives += 1
        resolver = _resolver(heuristic, ask_llm)
        try:
            resolver.resolve(_page(case["html"]), case["description"], case.get("action", ""))
        except ElementNotFound:
            continue  # correct: refusing is the right answer
        report.false_heals += 1
        report.failures.append(
            f"FALSE    {case['id']:<32} healed something that is not there"
        )

    return report


def _sweep(data: dict) -> None:
    print(f"{'heuristic':>10} {'ask_llm':>8} {'accuracy':>9} {'refused':>8} {'false':>6}")
    for heuristic in (0.20, 0.26, 0.34, 0.42, 0.50, 0.60):
        r = run(data, heuristic=heuristic)
        print(
            f"{heuristic:>10.2f} {'-':>8} {r.accuracy:>8.1%} "
            f"{r.refused:>8} {r.false_heals:>6}"
        )
    print("\nPick the highest threshold that keeps false heals at 0 and accuracy flat.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="fail below this accuracy (CI gate)")
    parser.add_argument("--max-false-heals", type=int, default=None,
                        help="fail above this many confident wrong answers")
    parser.add_argument("--sweep", action="store_true",
                        help="report metrics across candidate thresholds")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))

    if args.sweep:
        _sweep(data)
        return 0

    report = run(data)

    if args.json:
        print(json.dumps({
            "accuracy": round(report.accuracy, 4),
            "correct": report.correct, "wrong": report.wrong,
            "refused": report.refused, "ambiguous": report.ambiguous,
            "false_heals": report.false_heals, "total": report.total,
        }, indent=2))
    else:
        print("Healing benchmark (deterministic scorer, no model)")
        print("-" * 52)
        print(f"  cases            {report.total}")
        print(f"  correct          {report.correct}  ({report.accuracy:.1%})")
        print(f"  wrong element    {report.wrong}")
        print(f"  refused          {report.refused}  ({report.refusal_rate:.1%})")
        print(f"  ambiguous heals  {report.ambiguous}")
        print(f"  false heals      {report.false_heals} / {report.negatives} negatives")
        if report.failures:
            print("\n  " + "\n  ".join(report.failures))

    failed = False
    if args.min_accuracy is not None and report.accuracy < args.min_accuracy:
        print(f"\nFAIL accuracy {report.accuracy:.1%} < floor {args.min_accuracy:.1%}",
              file=sys.stderr)
        failed = True
    if args.max_false_heals is not None and report.false_heals > args.max_false_heals:
        print(f"FAIL {report.false_heals} false heals > budget {args.max_false_heals}",
              file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
