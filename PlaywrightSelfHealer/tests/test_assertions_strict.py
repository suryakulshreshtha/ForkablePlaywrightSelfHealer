"""Assertions must never be satisfied by a different element than the one named.

A healer that quietly rebinds ``should see Welcome back`` to whatever is nearest
turns a failing check into a passing one, which is the one outcome a test suite
exists to prevent.
"""

from __future__ import annotations

import pytest

from forkable_ai_agent.agent.healer import AmbiguousMatch, ElementNotFound, LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from tests.support.fake_page import FakePage, page_from_html


def _resolver(settings, **overrides):
    for key, value in overrides.items():
        setattr(settings.healer, key, value)
    return LocatorResolver(settings, Memory(settings.path(settings.memory_path)))


def test_assertion_on_missing_element_fails_instead_of_healing(settings):
    page = page_from_html('<a id="help" href="/h">Login help</a>')
    resolver = _resolver(settings)
    with pytest.raises(ElementNotFound) as err:
        resolver.resolve(page, "log in button", "expect_visible")
    assert "assertions never heal" in str(err.value) or "assertions do not heal" in str(err.value)
    assert not [e for e in resolver.events if "kind" not in e], "an assertion produced a heal"


def test_assertion_failure_names_the_nearest_element(settings):
    page = page_from_html('<a id="help" href="/h">Login help</a>')
    with pytest.raises(ElementNotFound) as err:
        _resolver(settings).resolve(page, "log in button", "expect_visible")
    assert "Nearest element" in str(err.value)


def test_assertions_can_opt_in_to_healing(settings, demo_app, variant_v2):
    """On the refactored login page no rung binds the username input."""
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolution = _resolver(settings, heal_assertions=True).resolve(page, "username", "expect_value")
    assert resolution.healed is True


def test_assertion_does_not_heal_by_default_on_the_same_page(settings, demo_app, variant_v2):
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "username", "expect_value")


def test_assertion_on_ambiguous_selector_always_fails(settings):
    page = page_from_html(
        '<button class="a">Delete</button><button class="a">Delete</button>'
    )
    resolver = _resolver(settings)
    with pytest.raises(AmbiguousMatch):
        resolver.resolve(page, "delete", "expect_visible")
    assert any(e.get("kind") == "ambiguous_refused" for e in resolver.events)


def test_actions_warn_on_ambiguity_by_default_but_still_act(settings):
    page = page_from_html(
        '<button class="a">Delete</button><button class="a">Delete</button>'
    )
    resolution = _resolver(settings).resolve(page, "delete", "click")
    assert resolution.match_count == 2 and resolution.ambiguous


def test_actions_can_be_made_strict_for_destructive_flows(settings):
    page = page_from_html(
        '<button class="a">Delete</button><button class="a">Delete</button>'
    )
    with pytest.raises(AmbiguousMatch):
        _resolver(settings, on_ambiguous="fail").resolve(page, "delete", "click")


def test_a_unique_rung_beats_an_earlier_ambiguous_one(settings):
    """Two buttons share a label, but only one carries the test id."""
    page = page_from_html(
        '<button data-testid="save">Save</button><button>Save</button>'
    )
    resolution = _resolver(settings, on_ambiguous="fail").resolve(page, "save", "click")
    assert resolution.match_count == 1
    assert resolution.strategy.startswith("testid") or "testid" in resolution.candidate.kind


# -- role-aware binding ------------------------------------------------------
def test_fill_never_binds_to_a_button(settings):
    page = page_from_html('<button id="u">Username</button>')
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "username field", "fill")


def test_check_requires_a_checkbox(settings):
    page = page_from_html('<input id="terms" name="terms" type="text" placeholder="terms">')
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "terms checkbox", "check")


def test_button_noun_rejects_a_heading_with_the_same_words(settings):
    page = page_from_html('<h2 id="t">Log in</h2>')
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "log in button", "click")


def test_button_noun_accepts_a_styled_link(settings):
    page = page_from_html('<a class="btn" href="/login">Log in</a>')
    assert _resolver(settings).resolve(page, "log in button", "click").match_count == 1
