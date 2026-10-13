"""A tiny stand-in for ``playwright.sync_api.expect`` over :class:`FakePage`.

It exists so the *generated* pytest modules can be executed offline and their
verdicts compared with the in-process executor's. It implements only the
assertions the generator emits, with Playwright's semantics for each (substring
and ``ignore_case`` text matching, regex *search* for URLs, strict single-element
visibility).
"""

from __future__ import annotations

import re
import types
from typing import Any

from tests.support.fake_page import FakeLocator, FakePage


def _text_of(target: Any) -> str:
    if isinstance(target, FakeLocator):
        if target.description == "body":
            return target.page.document.body_text
        return " ".join(n.text for n in target.nodes)
    raise TypeError(f"cannot read text from {target!r}")


class _Assertion:
    def __init__(self, target: Any, negate: bool = False) -> None:
        self.target = target
        self.negate = negate

    @property
    def not_(self) -> _Assertion:
        return _Assertion(self.target, not self.negate)

    def _check(self, ok: bool, message: str) -> None:
        if ok == self.negate:
            raise AssertionError(message)

    # -- page ----------------------------------------------------------
    def to_have_url(self, expected: Any, **_: Any) -> None:
        url = self.target.url
        ok = bool(expected.search(url)) if hasattr(expected, "search") else url == expected
        self._check(ok, f"URL {url!r} does not match {expected!r}")

    def to_have_title(self, expected: Any, **_: Any) -> None:
        title = self.target.title()
        ok = bool(expected.search(title)) if hasattr(expected, "search") else title == expected
        self._check(ok, f"title {title!r} does not match {expected!r}")

    # -- locator -------------------------------------------------------
    def to_contain_text(self, expected: str, ignore_case: bool = False, **_: Any) -> None:
        haystack = _text_of(self.target)
        needle = expected
        if ignore_case:
            haystack, needle = haystack.lower(), needle.lower()
        self._check(needle in haystack, f"text {expected!r} not found")

    def not_to_contain_text(self, expected: str, ignore_case: bool = False, **_: Any) -> None:
        haystack = _text_of(self.target)
        needle = expected
        if ignore_case:
            haystack, needle = haystack.lower(), needle.lower()
        if needle in haystack:
            raise AssertionError(f"text {expected!r} unexpectedly present")

    def to_be_visible(self, **_: Any) -> None:
        count = self.target.count()
        if count > 1:
            raise AssertionError(f"strict mode violation: resolved to {count} elements")
        self._check(count == 1 and self.target.is_visible(), "element is not visible")

    def to_have_value(self, expected: Any, **_: Any) -> None:
        value = self.target.input_value()
        ok = bool(expected.search(value)) if hasattr(expected, "search") else value == expected
        self._check(ok, f"value {value!r} != {expected!r}")


def expect(target: Any) -> _Assertion:
    return _Assertion(target)


def install_fake_playwright(monkeypatch) -> None:
    """Make ``from playwright.sync_api import Page, expect`` resolve to the fakes."""
    module = types.ModuleType("playwright.sync_api")
    module.Page = FakePage  # type: ignore[attr-defined]
    module.expect = expect  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", module)


def run_generated(code: str, page: FakePage, monkeypatch) -> str:
    """Execute a generated module's single test against ``page``.

    Returns ``"passed"`` or ``"failed"`` the way the executor would label it.
    """
    install_fake_playwright(monkeypatch)
    namespace: dict[str, Any] = {"__name__": "generated_under_test"}
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - validated, test-only
    tests = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    assert len(tests) == 1, tests
    try:
        tests[0](page)
    except AssertionError:
        return "failed"
    except Exception as exc:
        # Locator misses etc. - a failure, but let the cause show in the output.
        return f"failed ({type(exc).__name__}: {exc})"
    return "passed"


__all__ = ["expect", "install_fake_playwright", "run_generated", "re"]
