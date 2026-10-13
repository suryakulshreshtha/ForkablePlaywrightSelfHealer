"""The generated pytest module and the in-process executor must agree.

They are two implementations of the same plan. When they drift, a suite that is
green under ``forkable run`` goes red under plain ``pytest`` (or, worse, the
reverse), and nobody can tell which one is telling the truth.
"""

from __future__ import annotations

import pytest

from forkable_ai_agent.agent.analyzer import FailureAnalyzer
from forkable_ai_agent.agent.executor import Executor
from forkable_ai_agent.agent.generator import TestGenerator
from forkable_ai_agent.agent.healer import LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.agent.planner import Planner
from tests.support.fake_expect import run_generated
from tests.support.fake_page import FakePage, FakeSession

LOGIN = """go to the login page
fill username with demo
fill password with secret123
click log in
"""

CASES = {
    "happy_login": (LOGIN + "should be redirected to /dashboard\nshould see Welcome\n", "passed"),
    "case_insensitive_text": (LOGIN + "should see WELCOME\n", "passed"),
    "absent_text": (LOGIN + "should see This text never appears\n", "failed"),
    "negative_text_present": (LOGIN + "should not see Welcome\n", "failed"),
    "negative_text_absent": (LOGIN + "should not see Invalid username\n", "passed"),
    "wrong_url": (LOGIN + "should be redirected to /nowhere\n", "failed"),
    "bad_password": (
        "go to the login page\nfill username with demo\n"
        "fill password with definitely-wrong\nclick log in\n"
        "should see Invalid username or password\n", "passed"),
    "api_after_login": (
        LOGIN + "call GET /api/jobs\nthe response status should be 200\n"
        "the response count should be 3\n", "passed"),
    "api_wrong_status": (
        LOGIN + "call GET /api/jobs\nthe response status should be 500\n", "failed"),
    "api_unauthenticated": (
        "call GET /api/jobs\nthe response status should be 401\n", "passed"),
}


def _both(settings, demo_app, monkeypatch, body: str):
    spec = f"Test: differential_case\n{body}"
    plan = Planner(settings, None).plan(spec, demo_app.base_url)

    memory = Memory(settings.path(settings.memory_path))
    resolver = LocatorResolver(settings, memory)
    executed = Executor(settings, resolver, FailureAnalyzer(settings)).run(
        plan, FakeSession(demo_app.base_url)).status
    executed = "passed" if executed in {"passed", "warned"} else "failed"

    code = TestGenerator(settings).generate(plan).code
    generated = run_generated(code, FakePage(demo_app.base_url), monkeypatch)
    return executed, generated


@pytest.mark.parametrize("name", sorted(CASES))
def test_executor_and_generated_code_agree(name, settings, demo_app, variant_v1, monkeypatch):
    body, expected = CASES[name]
    executed, generated = _both(settings, demo_app, monkeypatch, body)
    assert executed == expected, f"executor said {executed}"
    assert generated.split(" ")[0] == expected, f"generated code said {generated}"
