"""Locator resolution and self-healing against the real demo markup."""

from __future__ import annotations

import pytest

from forkable_ai_agent.agent.healer import (
    ElementNotFound,
    LocatorResolver,
    candidate_from_element,
    rank_elements,
    score_element,
    token_similarity,
)
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.browser.locators import build_candidates, core, variants
from forkable_ai_agent.browser.snapshot import ElementInfo
from tests.support.fake_page import Document, FakePage


def _resolver(settings):
    return LocatorResolver(settings, Memory(settings.path(settings.memory_path)))


# -- unit ---------------------------------------------------------------
def test_token_similarity_survives_renames():
    assert token_similarity("password", "password") == 1.0
    assert token_similarity("password", "passphrase") > 0.5   # shared prefix
    assert token_similarity("username", "user") > 0.5         # substring
    assert token_similarity("username", "cheese") == 0.0


def test_core_strips_ui_nouns():
    assert core("username field") == "username"
    assert core("log in button") == "log in"
    assert "login" in variants("log in")


def test_candidate_ladder_is_ordered_most_stable_first():
    kinds = [c.kind for c in build_candidates("log in", "click")]
    assert kinds[0] == "testid"
    assert "role" in kinds
    assert kinds.index("role") < kinds.index("text")


def test_scoring_prefers_the_right_shape():
    field = ElementInfo(tag="input", type="password", id="pw_9x77", label="Passphrase")
    button = ElementInfo(tag="button", text="Log in")
    assert score_element("password", field, "fill") > score_element("password", button, "fill")
    assert score_element("log in", button, "click") > score_element("log in", field, "click")


def test_healed_candidate_prefers_durable_attributes():
    assert candidate_from_element(ElementInfo(testid="login", id="a")).kind == "testid"
    assert candidate_from_element(ElementInfo(id="a")).value == "#a"
    assert candidate_from_element(ElementInfo(css="form > input:nth-of-type(2)")).kind == "css"


# -- integration against the live demo app ------------------------------
def test_resolves_without_healing_on_the_stable_ui(settings, demo_app, variant_v1):
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolver = _resolver(settings)

    resolution = resolver.resolve(page, "username", "fill")
    assert not resolution.healed
    assert resolution.candidate.kind == "testid"
    assert resolver.events == []


@pytest.mark.parametrize("description,action,expected_id", [
    ("username", "fill", "usr_1a2b"),
    ("password", "fill", "pw_9x77"),
    ("log in", "click", ""),
])
def test_still_binds_the_right_element_on_the_refactored_ui(
    settings, demo_app, variant_v2, description, action, expected_id
):
    """Every stable hook is gone in v2, yet each description still binds."""
    page = FakePage(demo_app.base_url)
    page.goto("/login")

    # the hooks a hand-written test would have used are genuinely absent
    assert page.locator(f'[data-testid="{description}"]').count() == 0

    resolution = _resolver(settings).resolve(page, description, action)
    node = resolution.locator.nodes[0]
    if expected_id:
        assert node.get("id") == expected_id
    else:
        assert node.tag == "button"


def test_username_field_needs_the_healer_on_v2(settings, demo_app, variant_v2):
    """No rung of the static ladder matches the renamed username input."""
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolver = _resolver(settings)

    resolution = resolver.resolve(page, "username", "fill")

    assert resolution.healed
    assert resolution.locator.nodes[0].get("id") == "usr_1a2b"
    assert resolver.events[-1]["source"] == "heuristic"
    assert resolver.events[-1]["confidence"] > 0.34


def test_password_survives_on_a_lower_rung_without_healing(settings, demo_app, variant_v2):
    """Healing is the last resort: input[type=password] still matches, so use it."""
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolver = _resolver(settings)

    resolution = resolver.resolve(page, "password", "fill")

    assert not resolution.healed
    assert resolution.strategy == "input-type"
    assert resolution.locator.nodes[0].get("id") == "pw_9x77"
    assert resolver.events == []


def test_healed_locator_is_remembered_and_reused(settings, demo_app, variant_v2):
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    memory = Memory(settings.path(settings.memory_path))
    first = LocatorResolver(settings, memory)
    healed = first.resolve(page, "username", "fill")
    assert healed.healed
    memory.save()

    # a fresh resolver, fresh page, same persisted memory
    second_page = FakePage(demo_app.base_url)
    second_page.goto("/login")
    second = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    reused = second.resolve(second_page, "username", "fill")

    assert reused.strategy == "memory-cache"
    assert reused.attempts == 1, "the proven locator should be tried first"
    assert not reused.healed, "reusing a cached locator is not a fresh heal"
    assert second.events == [], "no second healing pass was needed"


def test_stale_cache_entry_is_demoted_then_rehealed(settings, demo_app, variant_v1):
    memory = Memory(settings.path(settings.memory_path))
    memory.record_success("username", "/login", 'css||0|#gone-forever', "id")
    page = FakePage(demo_app.base_url)
    page.goto("/login")

    resolution = LocatorResolver(settings, memory).resolve(page, "username", "fill")

    assert resolution.locator.nodes[0].get("id") == "username"
    stale = [r for r in memory.locators["/login::username"] if "gone-forever" in r.key][0]
    assert stale.misses == 1


def test_unknown_element_raises_rather_than_guessing(settings, demo_app, variant_v1):
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "quarterly revenue chart", "click")


def test_ranking_puts_the_human_choice_first(settings, demo_app, variant_v2):
    from forkable_ai_agent.browser.snapshot import ElementInfo as EI

    page = FakePage(demo_app.base_url)
    page.goto("/login")
    elements = [EI.from_dict(d) for d in page.evaluate("snapshot", 120)]
    best, score = rank_elements("password", elements, "fill")[0]
    assert best.id == "pw_9x77"
    assert score > 0.34


# -- resolution safety --------------------------------------------------
# The three behaviours below distinguish a heal you can trust from one you
# cannot. Each fails against the pre-fix resolver.


def test_ambiguous_match_is_reported_not_silently_first(settings):
    """A selector matching several elements must be flagged.

    Resolving to `.first` and reporting a clean heal is how a self-healing
    suite ends up asserting confidently against the wrong element.
    """
    page = FakePage()
    page.document = Document(
        "<html><body>"
        '<button class="action">Delete</button>'
        '<button class="action">Delete</button>'
        '<button class="action">Delete</button>'
        "</body></html>"
    )
    resolver = _resolver(settings)
    resolution = resolver.resolve(page, "delete", "click")

    assert resolution.match_count == 3
    assert resolution.ambiguous is True
    flagged = [e for e in resolver.events if e.get("kind") == "ambiguous"]
    assert flagged, "an ambiguous resolution produced no event"
    assert "3 elements matched" in flagged[0]["reason"]


def test_unique_match_is_not_flagged(settings):
    page = FakePage()
    page.document = Document(
        '<html><body><button id="delete-btn">Delete</button></body></html>'
    )
    resolution = _resolver(settings).resolve(page, "delete", "click")
    assert resolution.match_count == 1
    assert resolution.ambiguous is False


def test_candidate_timeout_is_bounded_and_restored(settings):
    """The ladder must not inherit the page's full default timeout.

    Eight candidates at the 5s browser default is 40s of dead wait for one
    missing element, before healing has even started.
    """
    seen: list[int] = []
    page = FakePage()
    page.document = Document(
        '<html><body><input id="username" name="username"></body></html>'
    )
    page.set_default_timeout = seen.append  # type: ignore[attr-defined]

    resolver = _resolver(settings)
    resolver.resolve(page, "username", "fill")

    assert seen, "the resolver never bounded the per-candidate timeout"
    assert seen[0] == resolver.candidate_timeout_ms <= 500
    # ...and it puts the page's own default back afterwards.
    assert seen[-1] == settings.browser.default_timeout_ms


def test_thresholds_come_from_settings(settings):
    settings.healer.heuristic_threshold = 0.9
    settings.healer.ask_llm_below = 0.95
    settings.healer.candidate_timeout_ms = 111
    resolver = _resolver(settings)
    assert (resolver.heuristic_threshold, resolver.ask_llm_below) == (0.9, 0.95)
    assert resolver.candidate_timeout_ms == 111
    # An explicit argument still wins, so existing callers keep working.
    assert LocatorResolver(
        settings, Memory(settings.path(settings.memory_path)), heuristic_threshold=0.1
    ).heuristic_threshold == 0.1


def test_dead_model_falls_back_but_leaves_a_trace(settings):
    """A dead Ollama must not break healing -- nor vanish from the report."""

    class DeadModel:
        name = "ollama"

        def generate(self, *_a, **_k):
            raise OSError("connection refused")

    # Two anonymous fields: the heuristic cannot decide between them, which is
    # exactly the case the model is there to break. Its death must degrade to
    # the heuristic rather than propagate.
    page = FakePage()
    page.document = Document(
        '<html><body><input id="field-a" name="field-a">'
        '<input id="field-b" name="field-b"></body></html>'
    )
    settings.healer.heuristic_threshold = 0.05
    resolver = LocatorResolver(
        settings, Memory(settings.path(settings.memory_path)), llm=DeadModel()
    )
    resolution = resolver.heal(page, "password", "fill")

    assert resolution.healed is True, "a dead model broke healing"
    errors = [e for e in resolver.events if e.get("kind") == "llm_error"]
    assert errors, "the model failure was swallowed silently"
    assert "OSError" in errors[0]["reason"]


def test_synonym_family_bridges_login_and_signin():
    """"log in" -> "sign in" is the most common auth rename there is.

    No character-level metric connects them: no shared prefix, no substring,
    no trigram overlap. Without the synonym family this scores ~0.28 and the
    heal is refused.
    """
    assert "signin" in variants("log in button")
    assert "login" in variants("sign in")
    element = ElementInfo(tag="button", id="signin-submit", text="Sign in")
    assert score_element("log in button", element, "click") > 0.4


def test_spelling_matching_a_token_of_an_identifier_scores():
    """`#signin-submit` names the login button; whole-attribute equality misses it."""
    for identifier in ("signin-submit", "btn-login", "login_button"):
        element = ElementInfo(tag="button", id=identifier)
        assert score_element("log in button", element, "click") > 0.3, identifier
