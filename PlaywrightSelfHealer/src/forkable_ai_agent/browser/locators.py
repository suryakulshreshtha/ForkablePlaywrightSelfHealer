"""Turning "the log in button" into a Playwright locator.

The agent never hard-codes a selector in a plan. Instead every element step
carries a human description, and this module expands that description into an
ordered ladder of candidate locators, most-stable first:

``data-testid`` -> ARIA role+name -> label -> placeholder -> id/name -> text -> css

A candidate knows three things: how to build a live ``Locator``, how to render
itself as Python source for generated tests, and how to serialise to a cache
key so a healed selector survives across runs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RAW_PREFIXES = ("css=", "xpath=", "text=", "id=", "data-testid=", "role=", "//", "#", ".", "[")

_ROLE_HINTS: list[tuple[str, str]] = [
    (r"\bbutton\b|\bsubmit\b|\bcta\b", "button"),
    (r"\blink\b|\banchor\b", "link"),
    (r"\bcheckbox\b|\btick box\b", "checkbox"),
    (r"\bradio\b", "radio"),
    (r"\bdropdown\b|\bselect\b|\bcombobox\b", "combobox"),
    (r"\btab\b", "tab"),
    (r"\bheading\b|\btitle\b|\bheader\b", "heading"),
    (r"\balert\b|\berror\b|\bbanner\b|\bmessage\b|\btoast\b", "alert"),
    (r"\bsearch\b", "searchbox"),
    (r"\bpassword\b|\bfield\b|\binput\b|\btextbox\b|\bemail\b|\busername\b|\btext box\b", "textbox"),
]

# Kept deliberately small: dropping "in" would collapse "log in" to "log" and
# lose the most useful spelling of all - "login".
_STOP = {"the", "a", "an", "please", "element"}

# Attributes commonly used as stable test hooks, in order of preference.
TESTID_ATTRS = ("data-testid", "data-test-id", "data-test", "data-qa", "data-cy")

#: Phrases that mean the same control but share no characters, so no amount of
#: fuzzy string matching connects them. Symmetric: each key expands to its
#: alternates. Small and hand-audited on purpose -- every entry added here is a
#: chance to heal to the wrong element, so a new one should come with a case in
#: tests/fixtures/healing_benchmark.json.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "log in": ("sign in", "login", "signin", "auth"),
    "login": ("sign in", "signin", "log in", "auth"),
    "sign in": ("log in", "login", "signin", "auth"),
    "signin": ("log in", "login", "sign in", "auth"),
    "log out": ("sign out", "logout", "signout"),
    "logout": ("sign out", "signout", "log out"),
    "sign up": ("register", "signup", "create account"),
    "signup": ("register", "sign up", "create account"),
    "register": ("sign up", "signup", "create account"),
    "delete": ("remove", "destroy"),
    "remove": ("delete", "destroy"),
    "submit": ("send", "confirm", "save"),
    "cancel": ("dismiss", "close", "abort"),
    "search": ("query", "find", "q"),
}


def normalise(description: str) -> str:
    return re.sub(r"\s+", " ", description.strip().lower())


def tokens(description: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", description.lower()) if t not in _STOP]


#: Nouns people append to name a control ("username field", "log in button").
#: They belong in the description, but never in the accessible name.
UI_NOUNS = {"button", "field", "input", "box", "textbox", "link", "icon", "tab",
            "menu", "checkbox", "dropdown", "label", "text", "control", "form"}


def core(description: str) -> str:
    """"username field" -> "username"; "log in button" -> "log in"."""
    parts = tokens(description)
    while len(parts) > 1 and parts[-1] in UI_NOUNS:
        parts.pop()
    return " ".join(parts) if parts else normalise(description)


def variants(description: str) -> list[str]:
    """Identifier spellings a developer might have used for this element."""
    parts = tokens(core(description))
    if not parts:
        return []
    joined = "".join(parts)
    out = [
        "-".join(parts),
        "_".join(parts),
        joined,
        parts[0] + "".join(p.capitalize() for p in parts[1:]),
    ]
    if len(parts) > 1:
        out.append(parts[-1])
        out.append(parts[0])

    # Synonym families. A rename from "log in" to "sign in" is the single most
    # common auth-button change there is, and no amount of character-level
    # fuzziness connects the two - they share no prefix, no substring and no
    # trigram. Kept deliberately small and hand-audited; this is a place where
    # a large list would start healing to the wrong element.
    for phrase, alternates in _SYNONYMS.items():
        if phrase == joined or phrase == " ".join(parts):
            for alt in alternates:
                alt_parts = alt.split()
                out.extend([
                    "-".join(alt_parts), "_".join(alt_parts), "".join(alt_parts),
                ])

    seen: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen


def guess_role(description: str, action: str = "") -> str:
    text = description.lower()
    for pattern, role in _ROLE_HINTS:
        if re.search(pattern, text):
            return role
    if action in {"fill", "press", "expect_value"}:
        return "textbox"
    if action in {"check", "uncheck"}:
        return "checkbox"
    if action == "select":
        return "combobox"
    return "button"


def _escape(value: str) -> str:
    """Escape for the inside of a double-quoted literal. Kept for callers that
    already wrap the quotes themselves; new code should use :func:`py_literal`."""
    return py_literal(value)[1:-1]


def py_literal(value: str) -> str:
    """A complete, double-quoted Python string literal for *value*.

    ``json.dumps`` escapes every control character (newlines included), so a
    value can never end the line or the string it sits in. The previous
    implementation escaped only backslashes and quotes, which let a newline in
    a ``css=...`` target or a note break out of the emitted statement.
    """
    return json.dumps(str(value), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Element *kind* - "log in button" must not bind to an <h1> that contains the
# words. A description that ends in a UI noun names a kind of element; a hit of
# the wrong kind is a miss, however well its text matches.
# ---------------------------------------------------------------------------
_NOUN_ROLE = {
    "button": "button", "btn": "button",
    "link": "link",
    "checkbox": "checkbox",
    "radio": "radio",
    "field": "textbox", "input": "textbox", "textbox": "textbox",
    "dropdown": "combobox", "select": "combobox", "combobox": "combobox",
    "tab": "tab",
    "heading": "heading",
}
_BUTTON_INPUT_TYPES = {"submit", "button", "reset", "image"}
_NON_TEXT_INPUT_TYPES = _BUTTON_INPUT_TYPES | {"checkbox", "radio", "hidden", "file", "range", "color"}
_TEXTBOX_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
#: Elements that are never what a user means by "the X button/link/tab". Used as
#: a reject-list rather than an allow-list because clickable <div>/<span> with no
#: role are common, and refusing them would break UIs that work fine today.
_NOT_CLICKABLE_TAGS = _HEADING_TAGS | {
    "p", "label", "ul", "ol", "table", "tr", "form", "fieldset", "textarea", "select", "img",
}


def _clickable_kind(tag: str, kind: str, role: str) -> bool:
    if role in {"button", "link", "tab", "menuitem"}:
        return True
    if tag == "input":
        return kind in _BUTTON_INPUT_TYPES  # a text box is not a button
    return tag not in _NOT_CLICKABLE_TAGS


#: Words that turn "X" into a *different* control about X: "Log in help" is not
#: the log-in button, "Privacy policy" is not "Privacy". Deliberately short; a
#: longer list starts rejecting legitimate labels.
QUALIFIER_WORDS = frozenset({
    "help", "faq", "learn", "policy", "privacy", "tutorial", "guide", "docs", "documentation",
})


def qualifier_clash(description: str, visible_text: str) -> bool:
    """True when the element's words add a qualifier the description never asked for."""
    wanted = set(tokens(description))
    extra = {t for t in tokens(visible_text or "") if t in QUALIFIER_WORDS}
    return bool(extra - wanted)


def explicit_role(description: str) -> str:
    """Role named by a trailing UI noun, or ``""`` when the description names none.

    Only an explicit noun counts. Inferring a role from content words ("message",
    "password") is what :func:`guess_role` is for; using that here would reject
    legitimate matches on a hunch.
    """
    parts = tokens(description)
    if len(parts) < 2:
        return ""
    if parts[-2:] == ["text", "box"]:
        return "textbox"
    if parts[-2:] == ["check", "box"]:
        return "checkbox"
    return _NOUN_ROLE.get(parts[-1], "")


def _role_matches(expected: str, info: Mapping[str, Any]) -> bool:
    tag = str(info.get("tag") or "")
    kind = str(info.get("type") or "")
    role = str(info.get("role") or "")
    editable = bool(info.get("editable"))
    if expected in {"button", "link", "tab"}:
        # <a class="btn"> is a button to every user, and so is a clickable div.
        return _clickable_kind(tag, kind, role)
    if expected == "checkbox":
        return (tag == "input" and kind == "checkbox") or role in {"checkbox", "switch"}
    if expected == "radio":
        return (tag == "input" and kind == "radio") or role == "radio"
    if expected == "textbox":
        return (tag == "input" and kind not in _NON_TEXT_INPUT_TYPES) or tag == "textarea" \
            or editable or role in _TEXTBOX_ROLES
    if expected == "combobox":
        return tag == "select" or role in {"combobox", "listbox"}
    if expected == "heading":
        return tag in _HEADING_TAGS or role == "heading"
    return True


def _action_matches(action: str, info: Mapping[str, Any]) -> bool:
    tag = str(info.get("tag") or "")
    kind = str(info.get("type") or "")
    role = str(info.get("role") or "")
    if action in {"fill", "expect_value"}:
        return tag in {"input", "textarea", "select"} or bool(info.get("editable")) \
            or role in _TEXTBOX_ROLES
    if action in {"check", "uncheck"}:
        return (tag == "input" and kind in {"checkbox", "radio"}) or role in {"checkbox", "radio", "switch"}
    if action == "select":
        return tag == "select" or role in {"combobox", "listbox"}
    return True


def shape_ok(expected_role: str, action: str, info: Mapping[str, Any]) -> bool:
    """True when an element of shape *info* can serve this description and action."""
    return _role_matches(expected_role, info) and _action_matches(action, info)


@dataclass
class LocatorCandidate:
    kind: str                  # testid | role | label | placeholder | text | css | xpath
    value: str
    role: str = ""
    strategy: str = ""
    exact: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # -- runtime -------------------------------------------------------
    def build(self, page: Any) -> Any:
        if self.kind == "testid":
            return page.locator(self.value)
        if self.kind == "role":
            return page.get_by_role(self.role or "button", name=self.value, exact=self.exact)
        if self.kind == "label":
            return page.get_by_label(self.value, exact=self.exact)
        if self.kind == "placeholder":
            return page.get_by_placeholder(self.value, exact=self.exact)
        if self.kind == "text":
            return page.get_by_text(self.value, exact=self.exact)
        if self.kind == "xpath":
            return page.locator(f"xpath={self.value}")
        return page.locator(self.value)

    # -- codegen -------------------------------------------------------
    def code(self, var: str = "page") -> str:
        if self.kind == "role":
            exact = ", exact=True" if self.exact else ""
            return f"{var}.get_by_role({py_literal(self.role or 'button')}, name={py_literal(self.value)}{exact})"
        if self.kind in {"label", "placeholder", "text"}:
            method = {"label": "get_by_label", "placeholder": "get_by_placeholder", "text": "get_by_text"}[self.kind]
            exact = ", exact=True" if self.exact else ""
            return f"{var}.{method}({py_literal(self.value)}{exact})"
        if self.kind == "xpath":
            return f"{var}.locator({py_literal('xpath=' + self.value)})"
        return f"{var}.locator({py_literal(self.value)})"

    # -- persistence ---------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.kind}|{self.role}|{int(self.exact)}|{self.value}"

    @classmethod
    def from_key(cls, key: str, strategy: str = "cache") -> LocatorCandidate:
        kind, role, exact, value = key.split("|", 3)
        return cls(kind=kind, value=value, role=role, exact=bool(int(exact)), strategy=strategy)

    def describe(self) -> str:
        return f"{self.kind}({self.role + ':' if self.role else ''}{self.value})"


def raw_candidate(description: str) -> LocatorCandidate | None:
    """Allow escape hatches: a plan may still carry a literal selector."""
    text = description.strip()
    if text.startswith("xpath=") or text.startswith("//"):
        return LocatorCandidate("xpath", text.removeprefix("xpath="), strategy="literal-xpath")
    if text.startswith("css="):
        return LocatorCandidate("css", text.removeprefix("css="), strategy="literal-css")
    if text.startswith(("#", ".", "[")) and " " not in text:
        return LocatorCandidate("css", text, strategy="literal-css")
    return None


def build_candidates(
    description: str,
    action: str = "",
    known: Sequence[str] = (),
) -> list[LocatorCandidate]:
    """Ordered locator ladder for *description*.

    ``known`` holds cache keys previously proven for this description; they are
    tried first so a healed selector costs nothing on the next run.
    """
    out: list[LocatorCandidate] = []
    for key in known:
        try:
            out.append(LocatorCandidate.from_key(key, strategy="memory-cache"))
        except ValueError:
            continue

    literal = raw_candidate(description)
    if literal:
        out.append(literal)
        return out

    text = core(description)
    full = normalise(description)
    role = guess_role(description, action)
    spellings = variants(description)

    for spelling in spellings[:4]:
        for attr in TESTID_ATTRS[:3]:
            out.append(
                LocatorCandidate("testid", f'[{attr}="{spelling}"]', strategy=f"{attr}")
            )

    out.append(LocatorCandidate("role", text, role=role, strategy="aria-role-name"))
    if full != text:
        out.append(LocatorCandidate("role", full, role=role, strategy="aria-role-name-verbose"))
    out.append(LocatorCandidate("label", text, strategy="label"))
    if role in {"textbox", "searchbox", "combobox"}:
        out.append(LocatorCandidate("placeholder", text, strategy="placeholder"))

    for spelling in spellings[:4]:
        out.append(LocatorCandidate("css", f"#{spelling}", strategy="id"))
        out.append(LocatorCandidate("css", f'[name="{spelling}"]', strategy="name-attr"))
        out.append(LocatorCandidate("css", f'[aria-label*="{spelling}" i]', strategy="aria-label"))

    out.append(LocatorCandidate("text", text, strategy="visible-text"))
    if role in {"textbox", "searchbox"}:
        for spelling in spellings[:3]:
            out.append(
                LocatorCandidate("css", f'input[type="{spelling}"]', strategy="input-type")
            )

    deduped: list[LocatorCandidate] = []
    seen = set()
    for candidate in out:
        if candidate.key not in seen:
            seen.add(candidate.key)
            deduped.append(candidate)
    return deduped
