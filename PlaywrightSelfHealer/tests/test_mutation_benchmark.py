"""The generated mutation benchmark doubles as a regression gate.

Floors are the measured values when they were set, minus a small margin. Raise
them when the scorer improves; never lower them to make a change pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from forkable_ai_agent.agent.healer import ElementNotFound, LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.browser.locators import qualifier_clash
from tests.support.fake_page import page_from_html

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mutation_benchmark.py"


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("mutation_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_benchmark"] = module
    spec.loader.exec_module(module)
    return module.run()


def test_never_heals_onto_a_removed_element(report):
    assert report.false_heals == 0, "\n".join(f for f in report.failures if f.startswith("FALSE"))


def test_never_picks_the_wrong_element_when_it_answers(report):
    assert report.wrong == 0, "\n".join(f for f in report.failures if f.startswith("WRONG"))


def test_accuracy_floor(report):
    assert report.accuracy >= 0.94, f"accuracy fell to {report.accuracy:.1%}"


def test_benchmark_is_large_enough_to_mean_something(report):
    assert report.positives >= 100 and report.negatives >= 50


# -- the qualifier rule on its own ---------------------------------------------
@pytest.mark.parametrize("description, text, clash", [
    ("log in button", "Log in help", True),
    ("privacy", "Privacy policy", True),
    ("log in button", "Log in", False),
    ("help button", "Help", False),
    ("help link", "Help centre", False),
    ("contact support", "Contact support", False),
])
def test_qualifier_clash(description, text, clash):
    assert qualifier_clash(description, text) is clash


def test_help_link_is_not_the_login_button(settings):
    page = page_from_html('<a href="/h">Log in help</a>')
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    with pytest.raises(ElementNotFound):
        resolver.resolve(page, "log in button", "click")


def test_stable_id_beats_extra_wording(settings):
    """A control with the right id and a longer label is still the control."""
    page = page_from_html('<button id="login-btn" class="x">Log in to continue</button>')
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    assert resolver.resolve(page, "log in button", "click").match_count == 1
