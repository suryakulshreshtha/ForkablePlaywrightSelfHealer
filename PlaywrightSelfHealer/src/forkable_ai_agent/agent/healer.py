"""Locator resolution and self-healing.

Resolution walks the candidate ladder from :mod:`browser.locators`. When every
candidate misses - because a developer renamed ``#login-btn`` to
``#signin-btn`` overnight - the healer takes a compact DOM snapshot and scores
each element against the human description.

The scorer is deterministic and runs first; a local LLM is consulted only to
break ties or when the best heuristic score is weak. That ordering keeps the
common case fast and offline-pure, and keeps the model's role to judgement
rather than plumbing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..browser.locators import (
    LocatorCandidate,
    build_candidates,
    core,
    explicit_role,
    guess_role,
    qualifier_clash,
    shape_ok,
    tokens,
    variants,
)
from ..browser.snapshot import ElementInfo, capture, inspect_locator, render
from .memory import Memory, scope_for

log = logging.getLogger(__name__)

INPUT_TAGS = {"input", "textarea", "select"}
CLICK_TAGS = {"button", "a", "summary", "label", "div", "span", "li"}

#: Element assertions. A missing element here is a finding, not a locator to
#: repair, so these never heal unless ``healer.heal_assertions`` is set.
ASSERTION_ACTIONS = {"expect_visible", "expect_value"}

#: Actions that cannot work on the wrong kind of element (you cannot ``fill`` a
#: heading), so a hit of the wrong shape is a miss.
SHAPE_ACTIONS = {"fill", "check", "uncheck", "select", "expect_value"}

# Resolves once the DOM has gone quiet (no mutations for `quiet` ms), or after
# `cap` ms regardless, so a page that animates forever cannot hang the heal.
DOM_QUIET_JS = r"""
({quiet, cap}) => new Promise((resolve) => {
  let timer = null;
  const finish = () => { observer.disconnect(); clearTimeout(timer); clearTimeout(hard); resolve(true); };
  const arm = () => { clearTimeout(timer); timer = setTimeout(finish, quiet); };
  const observer = new MutationObserver(arm);
  const hard = setTimeout(finish, cap);
  observer.observe(document, {subtree: true, childList: true, attributes: true, characterData: true});
  arm();
})
"""

HEAL_SYSTEM = (
    "You repair broken UI test selectors. You are given a target description and a "
    "numbered list of elements from the live page. Reply with JSON only: "
    '{"index": <number>, "confidence": <0-1>, "reason": "<short>"}. '
    'If nothing matches, use {"index": -1, "confidence": 0, "reason": "no match"}.'
)


class ElementNotFound(RuntimeError):
    """No candidate and no healing strategy could bind the description."""


class AmbiguousMatch(ElementNotFound):
    """The best selector matched several elements and the policy forbids guessing.

    A subclass of :class:`ElementNotFound` on purpose: for the caller, "could not
    pick one" and "found none" are both a refusal to act on a guess. The message
    deliberately uses Playwright's own phrasing so the analyzer classifies both
    the same way.
    """


@dataclass
class Resolution:
    locator: Any
    candidate: LocatorCandidate
    strategy: str
    healed: bool = False
    attempts: int = 0
    confidence: float = 1.0
    #: How many elements the winning selector matched. Anything above 1 means
    #: ``.first`` made a choice on the caller's behalf.
    match_count: int = 1

    @property
    def ambiguous(self) -> bool:
        """True when the selector was not unique.

        Asserting against the wrong element while reporting a successful heal
        is the worst failure mode this tool has; callers and the HTML report
        surface this so it is visible rather than silent.
        """
        return self.match_count > 1


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def token_similarity(a: str, b: str) -> float:
    """Fuzzy 0..1 similarity between two identifier tokens.

    Renames are rarely total: ``password`` becomes ``passphrase``, ``username``
    becomes ``user name``. Exact set overlap misses all of those, so scoring
    falls back through substring, shared prefix and trigram overlap.
    """
    if a == b:
        return 1.0
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return 0.75
    prefix = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix += 1
    if prefix >= 4:
        return 0.6
    grams_a, grams_b = _trigrams(a), _trigrams(b)
    union = grams_a | grams_b
    if not union:
        return 0.0
    jaccard = len(grams_a & grams_b) / len(union)
    return 0.55 * jaccard if jaccard > 0.25 else 0.0


def _match_strength(desc_tokens: Sequence[str], value: str) -> float:
    """Sum of the best per-token similarity against *value*'s tokens."""
    value_tokens = tokens(value)
    if not value_tokens:
        return 0.0
    return sum(max(token_similarity(t, v) for v in value_tokens) for t in desc_tokens)


def _extra_wording_penalty(description: str, element: ElementInfo) -> float:
    """Penalty for an element whose visible words say *more* than the description.

    "Log in help" contains every word of "log in button" and still is not the
    log-in button. When the only evidence is human-readable text (no id, name or
    test id matched), a full match plus extra words is a different control that
    merely shares a phrase. A stable identifier overrides this: ``id="login"``
    with the text "Log in to continue" is still the log-in button.
    """
    wanted = tokens(core(description)) or tokens(description)
    if not wanted:
        return 0.0
    desc_tokens = tokens(description)
    if any(_match_strength(desc_tokens, v) for v in (element.testid, element.id, element.name) if v):
        return 0.0
    best: float | None = None
    for value in (element.text, element.label, element.aria):
        words = tokens(value or "")
        if not words:
            continue
        covered = all(any(token_similarity(w, t) >= 0.8 for t in words) for w in wanted)
        if not covered:
            continue
        extra = [t for t in words if not any(token_similarity(t, w) >= 0.6 for w in wanted)]
        penalty = min(0.45, 0.35 + 0.1 * (len(extra) - 1)) if extra else 0.0
        best = penalty if best is None else min(best, penalty)
    return best or 0.0


def score_element(description: str, element: ElementInfo, action: str = "") -> float:
    """Heuristic 0..1 similarity between a description and a live element."""
    desc_tokens = tokens(description)
    if not desc_tokens:
        return 0.0

    fields = (
        (element.testid, 4.0),
        (element.id, 3.2),
        (element.name, 3.2),
        (element.aria, 2.8),
        (element.label, 2.8),
        (element.placeholder, 2.4),
        (element.text, 2.2),
        (element.title, 1.4),
        (element.type, 1.0),
        (element.role, 1.0),
        (element.cls, 0.5),
    )

    best_possible = 4.0 * len(desc_tokens)
    total = 0.0
    spellings = set(variants(description))
    for value, weight in fields:
        if not value:
            continue
        strength = _match_strength(desc_tokens, value)
        if strength:
            total += weight * strength
        # an exact identifier match is the strongest signal available
        normalised = value.strip().lower()
        if normalised in spellings:
            total += weight * len(desc_tokens) * 0.5
        elif spellings.intersection(tokens(normalised)):
            # A spelling appearing as one *token* of the attribute is nearly as
            # strong: "#signin-submit", "#btn-login" and "#login_button" all
            # name the login button, but none of them equals a spelling of it.
            # Scored below an exact match so a whole-attribute hit still wins.
            total += weight * len(desc_tokens) * 0.35

    score = total / best_possible if best_possible else 0.0

    # action/shape compatibility
    if action in {"fill", "press", "expect_value"}:
        if element.tag in INPUT_TAGS:
            score += 0.12
        else:
            score -= 0.25
    elif action in {"click", "hover"}:
        if element.tag in CLICK_TAGS or element.role in {"button", "link", "tab"}:
            score += 0.10
        if element.tag == "input" and element.type in {"submit", "button", "checkbox", "radio"}:
            score += 0.10
    elif action in {"check", "uncheck"}:
        score += 0.15 if element.type in {"checkbox", "radio"} else -0.2
    elif action == "select":
        score += 0.15 if element.tag == "select" else -0.15

    score -= _extra_wording_penalty(description, element)
    if qualifier_clash(description, " ".join(filter(None, (element.text, element.label, element.aria)))):
        score -= 0.5

    # A description that ends in a UI noun ("log in button") names a kind of
    # element. Text that matches on the wrong kind (a heading reading "Log in")
    # is a decoy, not a candidate.
    named = explicit_role(description)
    if named and not shape_ok(named, "", {
        "tag": element.tag, "type": element.type, "role": element.role,
        "editable": element.editable,
    }):
        score -= 0.35

    if not element.visible:
        score -= 0.3
    return max(0.0, min(1.0, score))


def rank_elements(
    description: str, elements: Sequence[ElementInfo], action: str = ""
) -> list[tuple[ElementInfo, float]]:
    ranked = [(el, score_element(description, el, action)) for el in elements]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def candidate_from_element(element: ElementInfo, action: str = "", description: str = "") -> LocatorCandidate:
    """Pick the most durable selector that identifies *element*."""
    if element.testid:
        return LocatorCandidate("testid", f'[data-testid="{element.testid}"]', strategy="healed-testid")
    if element.id:
        return LocatorCandidate("css", f"#{element.id}", strategy="healed-id")
    if element.name:
        return LocatorCandidate("css", f'[name="{element.name}"]', strategy="healed-name")
    if element.aria:
        role = element.role or guess_role(description or element.aria, action)
        return LocatorCandidate("role", element.aria, role=role, strategy="healed-aria")
    if element.label:
        return LocatorCandidate("label", element.label, strategy="healed-label")
    if element.placeholder:
        return LocatorCandidate("placeholder", element.placeholder, strategy="healed-placeholder")
    if element.text and element.tag in CLICK_TAGS:
        return LocatorCandidate("text", element.text[:40], strategy="healed-text")
    return LocatorCandidate("css", element.css, strategy="healed-csspath")


def _parse_llm_choice(text: str) -> tuple[int, float, str]:
    if not text:
        return -1, 0.0, ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return -1, 0.0, ""
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return -1, 0.0, ""
    try:
        index = int(data.get("index", -1))
    except (TypeError, ValueError):
        index = -1
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return index, confidence, str(data.get("reason", ""))[:200]


@dataclass
class _Hit:
    candidate: LocatorCandidate
    locator: Any
    count: int
    rung: int  # 1-based position in the ladder


@dataclass
class _Scan:
    unique: _Hit | None = None
    ambiguous: _Hit | None = None
    missed: frozenset = frozenset()  # 1-based rungs that matched nothing


class LocatorResolver:
    """Binds natural-language element descriptions to live Playwright locators."""

    def __init__(
        self,
        settings,
        memory: Memory,
        llm: Any = None,
        heuristic_threshold: float | None = None,
        ask_llm_below: float | None = None,
        candidate_timeout_ms: int | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.llm = llm
        self.namespace = getattr(settings, "memory_namespace", "")

        # Thresholds live in settings.healer so they can be tuned and
        # version-controlled; explicit arguments still win, which keeps the
        # existing tests and any caller that pins them working.
        healer_cfg = getattr(settings, "healer", None)

        def _cfg(name: str, override: Any, fallback: Any) -> Any:
            if override is not None:
                return override
            return getattr(healer_cfg, name, fallback)

        self.heuristic_threshold = _cfg("heuristic_threshold", heuristic_threshold, 0.34)
        self.ask_llm_below = _cfg("ask_llm_below", ask_llm_below, 0.62)
        self.candidate_timeout_ms = _cfg("candidate_timeout_ms", candidate_timeout_ms, 250)
        self.warn_on_ambiguous = getattr(healer_cfg, "warn_on_ambiguous", True)
        self.on_ambiguous = str(getattr(healer_cfg, "on_ambiguous", "warn"))
        self.settle_ms = int(getattr(healer_cfg, "settle_ms", 1500))
        self.settle_interval_ms = int(getattr(healer_cfg, "settle_interval_ms", 150))
        self.dom_quiet_ms = int(getattr(healer_cfg, "dom_quiet_ms", 150))
        self.heal_assertions = bool(getattr(healer_cfg, "heal_assertions", False))
        self.commit_policy = str(getattr(healer_cfg, "commit_policy", "on_pass"))
        self.events: list[dict] = []
        self._staging = False

    # ------------------------------------------------------------------
    # Run lifecycle. A heal is only trusted once the run it happened in passed.
    # ------------------------------------------------------------------
    def begin_run(self) -> None:
        """Start holding fresh heals instead of saving them straight away."""
        self._staging = self.commit_policy == "on_pass"

    def end_run(self, passed: bool) -> None:
        """Commit staged heals if the run passed; otherwise park them for review."""
        if not self._staging:
            return
        self._staging = False
        if passed:
            count = self.memory.commit_pending()
            kind = "heals_committed"
        else:
            count = self.memory.discard_pending("run did not pass")
            kind = "heals_discarded"
        if count:
            self.events.append({"kind": kind, "count": count})

    def _remember_heal(self, description: str, scope: str, candidate: LocatorCandidate) -> None:
        if self._staging:
            self.memory.stage_heal(description, scope, candidate.key, candidate.strategy)
        else:
            self.memory.record_success(
                description, scope, candidate.key, candidate.strategy, healed=True,
            )

    # ------------------------------------------------------------------
    def _count(self, locator: Any) -> int:
        """Count matches under a short budget.

        Playwright's ``count()`` takes no timeout, but the underlying query
        inherits the page's default (5s here). Walking an eight-rung ladder for
        an element that has genuinely gone means eight of those before healing
        even begins. A candidate that is not on the page is never *slowly* not
        on the page, so a short explicit budget costs nothing and bounds the
        worst case.
        """
        page = getattr(locator, "page", None)
        setter = getattr(page, "set_default_timeout", None)
        if setter is None:  # test harness, or a Locator without a page handle
            return int(locator.count())
        previous = getattr(self.settings, "browser", None)
        previous_ms = getattr(previous, "default_timeout_ms", 5000)
        try:
            setter(self.candidate_timeout_ms)
            return int(locator.count())
        finally:
            with contextlib.suppress(Exception):
                setter(previous_ms)

    def _try_candidate(
        self,
        page: Any,
        candidate: LocatorCandidate,
        *,
        action: str = "",
        named: str = "",
        require_visible: bool = True,
        description: str = "",
    ) -> tuple[Any, int] | None:
        """Return ``(locator, match_count)`` for a candidate that binds, else None.

        ``named`` is the role a trailing UI noun asked for ("button"); a hit of
        the wrong kind is treated as a miss. An element that cannot be inspected
        is given the benefit of the doubt - a flaky inspection must not fail a test.
        """
        try:
            locator = candidate.build(page)
            count = self._count(locator)
        except Exception:
            return None
        if count == 0:
            return None
        target = locator.first
        if require_visible:
            try:
                if not target.is_visible():
                    return None
            except Exception:
                return None
        info = inspect_locator(target)
        if info is not None:
            if (named or action in SHAPE_ACTIONS) and not shape_ok(named, action, info):
                return None
            # "Log in help" matches the text "log in" and is still not the button.
            if description and qualifier_clash(description, str(info.get("text") or "")):
                return None
        return target, count

    # ------------------------------------------------------------------
    def _note_ambiguity(self, description: str, scope: str, candidate: LocatorCandidate,
                        count: int) -> None:
        if not (self.warn_on_ambiguous and count > 1):
            return
        self.events.append({
            "description": description,
            "scope": scope,
            "selector": candidate.describe(),
            "kind": "ambiguous",
            "match_count": count,
            "reason": (
                f"{count} elements matched; .first was used. A unique selector is "
                f"required before this assertion can be trusted."
            ),
        })

    def _policy_for(self, action: str) -> str:
        """``fail`` | ``warn`` | ``first`` for a selector that matched several."""
        if self.on_ambiguous == "first":
            return "first"
        if action in ASSERTION_ACTIONS or self.on_ambiguous == "fail":
            return "fail"
        return "warn"

    def _refuse_ambiguous(
        self, description: str, scope: str, candidate: LocatorCandidate, count: int
    ) -> AmbiguousMatch:
        reason = (
            f"strict mode violation: {candidate.describe()} resolved to {count} elements"
        )
        self.events.append({
            "description": description, "scope": scope, "selector": candidate.describe(),
            "kind": "ambiguous_refused", "match_count": count, "reason": reason,
        })
        return AmbiguousMatch(
            f"could not locate {description!r} uniquely: {reason}; "
            "narrow the description or scope it to a container"
        )

    # ------------------------------------------------------------------
    def _scan(self, page: Any, candidates: Sequence[LocatorCandidate], action: str,
              named: str, description: str = "") -> _Scan:
        """Walk the ladder once. Prefer the first *unique* binding.

        An earlier rung that matches several elements is remembered but does not
        stop the walk: a later rung that pins exactly one element is a better
        answer than the first of many.
        """
        scan = _Scan()
        missed: set[int] = set()
        for rung, candidate in enumerate(candidates, start=1):
            found = self._try_candidate(
                page, candidate, action=action, named=named, description=description)
            if found is None:
                missed.add(rung)
                continue
            locator, count = found
            hit = _Hit(candidate, locator, count, rung)
            if count == 1:
                scan.unique = hit
                break
            if scan.ambiguous is None:
                scan.ambiguous = hit
        scan.missed = frozenset(missed)
        return scan

    def _settle(self, page: Any, candidates: Sequence[LocatorCandidate], action: str,
                named: str, description: str = "") -> _Scan:
        """Re-walk the ladder until something binds or the settle budget is spent.

        Healing is for elements that *changed*, not elements that have not
        rendered yet. Without this wait, an async page misses every rung, the
        healer ranks a half-built DOM, and the wrong binding gets cached.
        """
        deadline = time.monotonic() + max(0, self.settle_ms) / 1000.0
        while True:
            scan = self._scan(page, candidates, action, named, description)
            if scan.unique or scan.ambiguous or time.monotonic() >= deadline:
                return scan
            time.sleep(max(10, self.settle_interval_ms) / 1000.0)

    def _wait_dom_quiet(self, page: Any) -> None:
        if self.dom_quiet_ms <= 0:
            return
        cap = max(self.dom_quiet_ms, min(self.settle_ms, 3000))
        with contextlib.suppress(Exception):
            page.evaluate(DOM_QUIET_JS, {"quiet": self.dom_quiet_ms, "cap": cap})

    # ------------------------------------------------------------------
    def _missing(self, page: Any, description: str, action: str,
                 candidates: Sequence[LocatorCandidate]) -> ElementNotFound:
        """Explain why an assertion's element is missing, without healing it."""
        named = explicit_role(description)
        for candidate in candidates:
            found = self._try_candidate(
                page, candidate, action=action, named=named, require_visible=False, description=description,
            )
            if found is not None:
                return ElementNotFound(
                    f"could not locate {description!r}: it is in the page but not visible "
                    f"({candidate.describe()}). {action} checks what a user can see; "
                    "assertions do not heal"
                )
        nearest = ""
        elements = capture(page, limit=140)
        if elements:
            best, score = rank_elements(description, elements, action)[0]
            nearest = (
                f" Nearest element: {best.line()[:120]!r} (score {score:.2f}); "
                "not used, because assertions never heal."
            )
        return ElementNotFound(
            f"could not locate {description!r} for {action}: no element on the page matches.{nearest}"
        )

    # ------------------------------------------------------------------
    def resolve(self, page: Any, description: str, action: str = "") -> Resolution:
        scope = scope_for(getattr(page, "url", "") or "", self.namespace)
        known = self.memory.known_keys(description, scope)
        candidates = build_candidates(description, action, known=known)
        named = explicit_role(description)

        scan = self._settle(page, candidates, action, named, description)
        hit = scan.unique or scan.ambiguous

        if hit is None:
            for rung in scan.missed:
                if candidates[rung - 1].strategy == "memory-cache":
                    self.memory.record_failure(description, scope, candidates[rung - 1].key)
            if action in ASSERTION_ACTIONS and not self.heal_assertions:
                raise self._missing(page, description, action, candidates)
            return self.heal(page, description, action, attempts=len(candidates))

        for rung in scan.missed:
            if rung < hit.rung and candidates[rung - 1].strategy == "memory-cache":
                self.memory.record_failure(description, scope, candidates[rung - 1].key)

        if hit.count > 1:
            policy = self._policy_for(action)
            if policy == "fail":
                raise self._refuse_ambiguous(description, scope, hit.candidate, hit.count)
            if policy == "warn":
                self._note_ambiguity(description, scope, hit.candidate, hit.count)

        # A cache hit is not a heal: the repair happened on an earlier run, and
        # reporting it as fresh would inflate the healing count on every
        # subsequent run. A heal still staged in *this* run is not yet proven,
        # so a reuse must not promote it behind the transaction's back.
        fresh_heal = hit.candidate.strategy.startswith("healed-")
        if not self.memory.is_pending(description, scope, hit.candidate.key):
            self.memory.record_success(
                description, scope, hit.candidate.key, hit.candidate.strategy, healed=fresh_heal,
            )
        return Resolution(
            locator=hit.locator,
            candidate=hit.candidate,
            strategy=hit.candidate.strategy,
            healed=fresh_heal,
            attempts=hit.rung,
            match_count=hit.count,
        )

    # ------------------------------------------------------------------
    def heal(self, page: Any, description: str, action: str = "", attempts: int = 0) -> Resolution:
        self._wait_dom_quiet(page)
        elements = capture(page, limit=140)
        if not elements:
            raise ElementNotFound(
                f"could not locate {description!r}: the page exposed no interactive elements"
            )

        # Score only elements the action can physically be performed on. A text
        # input called "terms" is not a checkbox however well its name matches,
        # and a soft score penalty is not enough to stop an exact-name match.
        named = explicit_role(description)
        if named or action in SHAPE_ACTIONS:
            usable = [
                el for el in elements
                if shape_ok(named, action, {
                    "tag": el.tag, "type": el.type, "role": el.role, "editable": el.editable,
                })
            ]
            if not usable:
                raise ElementNotFound(
                    f"could not locate {description!r}: no element on the page can be used "
                    f"for {action or 'this step'} ({len(elements)} elements inspected)"
                )
            elements = usable

        ranked = rank_elements(description, elements, action)
        best, best_score = ranked[0]
        chosen, confidence, reason, source = best, best_score, "heuristic token overlap", "heuristic"

        if self.llm is not None and getattr(self.llm, "name", "") != "rules" and best_score < self.ask_llm_below:
            shortlist = [el for el, _ in ranked[:25]]
            prompt = (
                "# task: heal\n"
                f"Target description: {description}\n"
                f"Intended action: {action or 'interact'}\n\n"
                "Elements:\n"
                f"{render(shortlist, limit=25)}\n"
            )
            # A dead model must never break healing -- but it must not be
            # invisible either. Previously a bare `except Exception: pass` made
            # "Ollama is not running", "the model replied with prose" and "there
            # is a bug in _parse_llm_choice" indistinguishable, and none of them
            # reached the report. Now the fallback still happens and the reason
            # is recorded, so the report can answer "was the model consulted?".
            try:
                response = self.llm.generate(prompt, system=HEAL_SYSTEM, json_mode=True)
                index, llm_conf, llm_reason = _parse_llm_choice(response.text)
                if 0 <= index < len(shortlist) and llm_conf >= 0.4:
                    chosen = shortlist[index]
                    confidence = max(best_score, llm_conf)
                    reason = llm_reason or "model selection"
                    source = "llm"
                else:
                    self.events.append({
                        "description": description, "kind": "llm_declined",
                        "reason": f"model returned index={index} confidence={llm_conf:.2f}",
                    })
            except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                # Transport died, or the model produced something unusable.
                self.events.append({
                    "description": description,
                    "kind": "llm_error",
                    "reason": f"{type(exc).__name__}: {exc}"[:200],
                })
                log.warning("healing LLM unavailable for %r, using heuristic: %s", description, exc)

        if confidence < self.heuristic_threshold:
            raise ElementNotFound(
                f"could not locate {description!r}; best candidate "
                f"{chosen.line()[:120]!r} scored {confidence:.2f}"
            )

        candidate = candidate_from_element(chosen, action, description)
        found = self._try_candidate(page, candidate)
        if found is not None and found[1] > 1:
            # The durable-looking selector is shared with other elements. The
            # snapshot's own path points at exactly this one, so prefer it.
            pinned = LocatorCandidate("css", chosen.css, strategy="healed-csspath")
            narrower = self._try_candidate(page, pinned)
            if narrower is not None and narrower[1] == 1:
                candidate, found = pinned, narrower
        if found is None:
            fallback = LocatorCandidate("css", chosen.css, strategy="healed-csspath")
            found = self._try_candidate(page, fallback)
            if found is None:
                raise ElementNotFound(
                    f"healing picked an element for {description!r} but no selector bound to it"
                )
            candidate = fallback
        locator, match_count = found

        scope = scope_for(getattr(page, "url", "") or "", self.namespace)
        policy = self._policy_for(action)
        if match_count > 1 and policy == "fail":
            raise self._refuse_ambiguous(description, scope, candidate, match_count)

        self._remember_heal(description, scope, candidate)
        self.events.append({
            "description": description,
            "scope": scope,
            "selector": candidate.describe(),
            "confidence": round(confidence, 3),
            "source": source,
            "reason": reason,
            "match_count": match_count,
            # A heal that resolved to one of several matches is not a heal you
            # can trust; surfacing it is the whole point.
            "ambiguous": match_count > 1,
            "staged": self._staging,
        })
        if policy == "warn":
            self._note_ambiguity(description, scope, candidate, match_count)
        return Resolution(
            locator=locator,
            candidate=candidate,
            strategy=candidate.strategy,
            healed=True,
            attempts=attempts + 1,
            confidence=confidence,
            match_count=match_count,
        )
