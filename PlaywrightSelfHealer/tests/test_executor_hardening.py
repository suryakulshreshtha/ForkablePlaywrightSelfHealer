"""Executor, analyzer and wait semantics added in the hardening pass."""

from __future__ import annotations

import pytest

from forkable_ai_agent.agent.analyzer import FailureAnalyzer, classify, describe_page_evidence
from forkable_ai_agent.agent.executor import Executor
from forkable_ai_agent.agent.healer import LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.agent.planner import Planner
from forkable_ai_agent.schema import MAX_WAIT_SECONDS, Step, TestPlan, wait_seconds
from tests.support.fake_page import FakeSession


# -- wait semantics -----------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("2", 2.0), ("0.5", 0.5), ("99", 99.0 if 99.0 <= MAX_WAIT_SECONDS else MAX_WAIT_SECONDS),
    ("500", 0.5), ("1500", 1.5), ("", 1.0), (None, 1.0), ("abc", 1.0), ("-3", 1.0),
    ("nan", 1.0), ("9999999", MAX_WAIT_SECONDS),
])
def test_wait_seconds(raw, expected):
    assert wait_seconds(raw) == pytest.approx(expected)


# -- analyzer -----------------------------------------------------------------
def test_ambiguous_selector_is_actually_reachable():
    category, _, _ = classify(
        "could not locate 'delete' uniquely: strict mode violation: role=button resolved to 3 elements"
    )
    assert category == "ambiguous_selector"


def test_plain_miss_is_still_selector_not_found():
    assert classify("could not locate 'login': no element matches")[0] == "selector_not_found"


def test_page_evidence_is_summarised_and_capped():
    text = describe_page_evidence({
        "page_errors": ["TypeError: x is undefined", "ReferenceError: y", "third"],
        "http_errors": ["500 GET /api/jobs"],
    })
    assert "uncaught JavaScript error" in text and "(+1 more)" in text
    assert "HTTP error response" in text
    assert describe_page_evidence({}) == ""


class _EvidenceSession(FakeSession):
    def __init__(self, base_url, diagnostics):
        super().__init__(base_url)
        self._diag = diagnostics
        self.marked = None

    def collect_diagnostics(self):
        return self._diag

    def mark_result(self, passed):
        self.marked = passed


def test_failed_run_carries_page_evidence_into_the_diagnosis(settings, demo_app, variant_v1):
    plan = Planner(settings, None).plan(
        "Test: evidence\ngo to the login page\nclick the missing gizmo button\n", demo_app.base_url)
    session = _EvidenceSession(demo_app.base_url, {"page_errors": ["TypeError: boom is not a function"]})
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    result = Executor(settings, resolver, FailureAnalyzer(settings)).run(plan, session)

    assert result.status == "failed"
    assert result.diagnostics == {"page_errors": ["TypeError: boom is not a function"]}
    assert "boom is not a function" in result.diagnosis.likely_cause
    assert result.diagnosis.suggested_fix.startswith("Fix the uncaught JavaScript error first")
    assert session.marked is False
    assert result.to_dict()["diagnostics"]["page_errors"]


def test_session_is_told_when_a_run_passes(settings, demo_app, variant_v1):
    plan = Planner(settings, None).plan(
        "Test: ok\ngo to the login page\nshould see Sign in\n", demo_app.base_url)
    session = _EvidenceSession(demo_app.base_url, {})
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    result = Executor(settings, resolver, FailureAnalyzer(settings)).run(plan, session)
    assert result.status == "passed" and session.marked is True


# -- statuses -----------------------------------------------------------------
def test_warned_visual_check_is_not_overwritten_to_passed(settings, demo_app, variant_v1):
    plan = TestPlan(name="visual", base_url=demo_app.base_url,
                    steps=[Step("goto", "/login"), Step("visual_check", "", "login")])
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    executor = Executor(settings, resolver, FailureAnalyzer(settings), visual=None)
    result = executor.run(plan, FakeSession(demo_app.base_url))
    visual = [s for s in result.steps if s.step.action == "visual_check"]
    assert visual and visual[0].status == "warned"
    assert result.status == "passed"


def test_ambiguous_assertion_fails_the_step_with_a_clear_message(settings, demo_app, variant_v1):
    from tests.support.fake_page import Document

    session = FakeSession(demo_app.base_url)
    session.page.document = Document(
        '<html><body><button class="a">Delete</button><button class="a">Delete</button></body></html>')
    session.page.goto = lambda *a, **k: None  # keep the hand-built DOM
    plan = Planner(settings, None).plan("Test: amb\ndelete button should be visible\n", demo_app.base_url)
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    result = Executor(settings, resolver, FailureAnalyzer(settings)).run(plan, session)
    assert result.status == "failed"
    assert "uniquely" in result.steps[-1].message
    assert result.diagnosis.category == "ambiguous_selector"
