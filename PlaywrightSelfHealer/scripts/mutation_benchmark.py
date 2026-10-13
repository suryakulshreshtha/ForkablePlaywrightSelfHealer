#!/usr/bin/env python3
"""Generated mutation benchmark for the healer.

``benchmark_healing.py`` scores a small hand-labelled set. This one *generates*
hundreds of cases: take a realistic screen, mutate the element under test the
way real front-end changes do (renamed ids, churned classes, wrapper divs,
reworded copy, decoys that share its words), and check the resolver either
finds the right element or - when the element was removed - refuses.

    python scripts/mutation_benchmark.py
    python scripts/mutation_benchmark.py --min-accuracy 0.85 --max-false-heals 0
    python scripts/mutation_benchmark.py --list-failures

The number that matters is **false heals**: a removed control that the healer
"found" anyway is how a green run hides a real regression. Accuracy can be
traded against refusals; false heals cannot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from forkable_ai_agent.agent.healer import ElementNotFound, LocatorResolver  # noqa: E402
from forkable_ai_agent.agent.memory import Memory  # noqa: E402
from forkable_ai_agent.config import load_settings  # noqa: E402
from tests.support.fake_page import Document, FakePage  # noqa: E402

MARK = "data-bm"


# ---------------------------------------------------------------------------
# Screens: the element under test is described the way a plan would name it.
# ---------------------------------------------------------------------------
@dataclass
class El:
    key: str
    desc: str
    action: str
    kind: str                    # button | link | input | password | checkbox | select
    text: str = ""               # visible text (button/link) or label (fields)
    placeholder: str = ""
    ident: str = ""              # original id / name
    synonym: str = ""            # reworded copy used by the "reword" mutation


SCREENS: dict[str, list[El]] = {
    "login": [
        El("user", "username field", "fill", "input", "Username", "Enter username", "username", "User name"),
        El("pass", "password field", "fill", "password", "Password", "Enter password", "password", "Passphrase"),
        El("remember", "remember me checkbox", "check", "checkbox", "Remember me", "", "remember", "Keep me signed in"),
        El("login", "log in button", "click", "button", "Log in", "", "login", "Sign in"),
        El("forgot", "forgot password link", "click", "link", "Forgot password", "", "forgot", "Reset password"),
    ],
    "search": [
        El("query", "search box", "fill", "input", "Search", "Search products", "query", "Find"),
        El("go", "search button", "click", "button", "Search", "", "search", "Find"),
        El("sort", "sort dropdown", "select", "select", "Sort by", "", "sort", "Order by"),
        El("clear", "clear filters button", "click", "button", "Clear filters", "", "clear", "Reset filters"),
    ],
    "checkout": [
        El("card", "card number field", "fill", "input", "Card number", "1234 5678", "card", "Card no."),
        El("promo", "promo code field", "fill", "input", "Promo code", "Enter code", "promo", "Discount code"),
        El("terms", "accept terms checkbox", "check", "checkbox", "Accept terms", "", "terms", "I agree"),
        El("pay", "pay now button", "click", "button", "Pay now", "", "pay", "Place order"),
    ],
    "settings": [
        El("email", "email field", "fill", "input", "Email", "you@example.com", "email", "E-mail address"),
        El("notify", "notifications checkbox", "check", "checkbox", "Notifications", "", "notify", "Email alerts"),
        El("save", "save changes button", "click", "button", "Save changes", "", "save", "Apply"),
        El("delete", "delete account button", "click", "button", "Delete account", "", "delete", "Close account"),
    ],
}


def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:6]


def render(el: El, *, ident: str | None = None, testid: bool = True, text: str | None = None,
           cls: str | None = None, mark: bool = False, placeholder: str | None = None) -> str:
    """HTML for one element. ``mark`` tags the target so the check cannot be fooled by ids."""
    ident = el.ident if ident is None else ident
    text = el.text if text is None else text
    placeholder = el.placeholder if placeholder is None else placeholder
    attrs = []
    if ident:
        attrs.append(f'id="{ident}"')
    if testid:
        attrs.append(f'data-testid="{el.ident}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if mark:
        attrs.append(f'{MARK}="target"')
    a = " ".join(attrs)

    if el.kind == "button":
        return f"<button {a}>{text}</button>"
    if el.kind == "link":
        return f'<a href="/{el.ident}" {a}>{text}</a>'
    # A field with no id is labelled by wrapping, exactly as real markup does it.
    def labelled(control: str) -> str:
        if ident:
            return f'<label for="{ident}">{text}</label>{control}'
        return f"<label>{text} {control}</label>"

    if el.kind == "select":
        return labelled(f"<select {a}><option>A</option><option>B</option></select>")
    if el.kind == "checkbox":
        return labelled(f'<input type="checkbox" {a}>')
    typ = "password" if el.kind == "password" else "text"
    ph = f' placeholder="{placeholder}"' if placeholder else ""
    return labelled(f'<input type="{typ}" {a}{ph}>')


# ---------------------------------------------------------------------------
# Mutations. Each returns (rendered target html, extra html before, extra after).
# ---------------------------------------------------------------------------
def m_baseline(el):
    return render(el, mark=True), "", ""


def m_rename_id(el):
    return render(el, ident=f"x_{_h(el.key, 'id')}", testid=False, mark=True), "", ""


def m_strip_hooks(el):
    return render(el, ident="", testid=False, mark=True), "", ""


def m_class_churn(el):
    return render(el, ident=f"c_{_h(el.key, 'cls')}", testid=False, cls=f"css-{_h(el.key, 'a')} sc-{_h(el.key, 'b')}", mark=True), "", ""


def m_wrapper(el):
    inner = render(el, ident="", testid=False, mark=True)
    return f'<div class="row"><div class="col"><span>{inner}</span></div></div>', "", ""


def m_reword(el):
    return render(el, ident=f"x_{_h(el.key, 'rw')}", testid=False, text=el.synonym,
                  placeholder=el.synonym, mark=True), "", ""


def m_testid_only(el):
    return render(el, ident="", testid=True, text=el.text, mark=True), "", ""


def m_decoys(el):
    """The real control keeps no hooks; look-alikes with the same words surround it."""
    words = el.text or el.desc
    decoys = (f'<h2>{words}</h2><p>Please {words.lower()} to continue</p>'
              f'<a href="/help" class="help">{words} help</a>')
    return render(el, ident=f"x_{_h(el.key, 'dc')}", testid=False, mark=True), decoys, ""


MUTATIONS = {
    "baseline": m_baseline,
    "rename_id": m_rename_id,
    "strip_hooks": m_strip_hooks,
    "class_churn": m_class_churn,
    "wrapper": m_wrapper,
    "reword": m_reword,
    "testid_only": m_testid_only,
    "decoys": m_decoys,
}


def _others(screen: str, skip: str) -> str:
    return "".join(
        f"<div>{render(e, ident=e.ident, testid=True)}</div>" for e in SCREENS[screen] if e.key != skip
    )


def build_cases() -> tuple[list[dict], list[dict]]:
    positives: list[dict] = []
    negatives: list[dict] = []
    for screen, elements in SCREENS.items():
        for el in elements:
            others = _others(screen, el.key)
            for name, mutate in MUTATIONS.items():
                target, before, after = mutate(el)
                html = f"<html><body><form>{before}{others}<div>{target}</div>{after}</form></body></html>"
                positives.append({
                    "id": f"{screen}/{el.key}/{name}", "mutation": name, "screen": screen,
                    "description": el.desc, "action": el.action, "html": html,
                })
            # Removed: the element is gone. Variants differ in what is left behind.
            words = el.text or el.desc
            leftovers = {
                "removed": "",
                "removed+heading": f"<h2>{words}</h2>",
                "removed+help-link": f'<a href="/help">{words} help</a>',
                "removed+copy": f"<p>Please {words.lower()} to continue</p>",
            }
            for name, extra in leftovers.items():
                html = f"<html><body><form>{others}{extra}</form></body></html>"
                negatives.append({
                    "id": f"{screen}/{el.key}/{name}", "mutation": name, "screen": screen,
                    "description": el.desc, "action": el.action, "html": html,
                })
    return positives, negatives


# ---------------------------------------------------------------------------
@dataclass
class Report:
    positives: int = 0
    correct: int = 0
    wrong: int = 0
    refused: int = 0
    negatives: int = 0
    false_heals: int = 0
    failures: list[str] = field(default_factory=list)
    by_mutation: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @property
    def accuracy(self) -> float:
        return self.correct / self.positives if self.positives else 0.0

    @property
    def precision(self) -> float:
        answered = self.correct + self.wrong
        return self.correct / answered if answered else 1.0


def _resolver() -> LocatorResolver:
    settings = load_settings()
    settings.root = tempfile.mkdtemp(prefix="forkable-mut-")
    settings.healer.settle_ms = 0
    settings.healer.dom_quiet_ms = 0
    return LocatorResolver(settings, Memory(Path(settings.root) / "memory.json"), llm=None)


def _page(html: str) -> FakePage:
    page = FakePage()
    page.document = Document(html)
    page.url = "http://127.0.0.1:8799/benchmark"
    return page


def run() -> Report:
    positives, negatives = build_cases()
    report = Report()
    for case in positives:
        report.positives += 1
        try:
            res = _resolver().resolve(_page(case["html"]), case["description"], case["action"])
        except ElementNotFound:
            report.refused += 1
            report.by_mutation[case["mutation"]]["refused"] += 1
            report.failures.append(f"REFUSED {case['id']:<34} {case['description']!r}")
            continue
        node = res.locator._node()
        if node.get(MARK) == "target":
            report.correct += 1
            report.by_mutation[case["mutation"]]["correct"] += 1
        else:
            report.wrong += 1
            report.by_mutation[case["mutation"]]["wrong"] += 1
            report.failures.append(
                f"WRONG   {case['id']:<34} picked <{node.tag}> {node.text.strip()[:24]!r}")
    for case in negatives:
        report.negatives += 1
        try:
            res = _resolver().resolve(_page(case["html"]), case["description"], case["action"])
        except ElementNotFound:
            report.by_mutation[case["mutation"]]["refused_ok"] += 1
            continue
        node = res.locator._node()
        report.false_heals += 1
        report.by_mutation[case["mutation"]]["false_heal"] += 1
        report.failures.append(
            f"FALSE   {case['id']:<34} healed to <{node.tag}> {node.text.strip()[:24]!r}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-accuracy", type=float)
    parser.add_argument("--max-false-heals", type=int)
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--list-failures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = run()
    summary = {
        "positives": r.positives, "correct": r.correct, "wrong": r.wrong, "refused": r.refused,
        "accuracy": round(r.accuracy, 4), "precision": round(r.precision, 4),
        "negatives": r.negatives, "false_heals": r.false_heals,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Mutation benchmark (deterministic scorer, no model)")
        print("-" * 56)
        print(f"  positive cases   {r.positives}")
        print(f"  correct          {r.correct}  ({r.accuracy:.1%})")
        print(f"  wrong element    {r.wrong}   (precision when it answers: {r.precision:.1%})")
        print(f"  refused          {r.refused}")
        print(f"  false heals      {r.false_heals} / {r.negatives} removed-element cases")
        print("\n  by mutation:")
        for name, c in sorted(r.by_mutation.items()):
            print(f"    {name:<18} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
        if args.list_failures and r.failures:
            print("\n  " + "\n  ".join(r.failures))

    failed = False
    if args.min_accuracy is not None and r.accuracy < args.min_accuracy:
        print(f"FAIL accuracy {r.accuracy:.1%} < {args.min_accuracy:.1%}", file=sys.stderr)
        failed = True
    if args.min_precision is not None and r.precision < args.min_precision:
        print(f"FAIL precision {r.precision:.1%} < {args.min_precision:.1%}", file=sys.stderr)
        failed = True
    if args.max_false_heals is not None and r.false_heals > args.max_false_heals:
        print(f"FAIL {r.false_heals} false heals > {args.max_false_heals}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
