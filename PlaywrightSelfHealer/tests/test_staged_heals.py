"""A heal is only trusted after the run it happened in has passed."""

from __future__ import annotations

from forkable_ai_agent.agent.analyzer import FailureAnalyzer
from forkable_ai_agent.agent.executor import Executor
from forkable_ai_agent.agent.healer import LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.agent.planner import Planner
from tests.support.fake_page import FakePage, FakeSession

PASSING = """Test: staged_pass
go to the login page
fill username with demo
fill password with secret123
click log in
should be redirected to /dashboard
"""

FAILING = """Test: staged_fail
go to the login page
fill username with demo
fill password with secret123
click log in
should see Text that never appears
"""


def _run(settings, demo_app, spec):
    plan = Planner(settings, None).plan(spec, demo_app.base_url)
    memory = Memory(settings.path(settings.memory_path))
    resolver = LocatorResolver(settings, memory)
    result = Executor(settings, resolver, FailureAnalyzer(settings)).run(
        plan, FakeSession(demo_app.base_url))
    return result, resolver, memory


def test_heal_is_committed_when_the_run_passes(settings, demo_app, variant_v2):
    result, resolver, memory = _run(settings, demo_app, PASSING)
    assert result.status in {"passed", "warned"}, result.steps
    assert result.healed_count >= 1
    assert memory.pending == []
    assert memory.known_keys("username", "/login"), "a proven heal was not remembered"
    assert any(e.get("kind") == "heals_committed" for e in resolver.events)


def test_heal_from_a_failed_run_is_parked_not_trusted(settings, demo_app, variant_v2):
    result, resolver, memory = _run(settings, demo_app, FAILING)
    assert result.status == "failed"
    assert memory.pending == []
    assert memory.known_keys("username", "/login") == []
    assert memory.proposals, "the discarded heal left no reviewable record"
    assert memory.proposals[0]["reason"]
    assert any(e.get("kind") == "heals_discarded" for e in resolver.events)


def test_legacy_policy_commits_immediately(settings, demo_app, variant_v2):
    settings.healer.commit_policy = "always"
    result, _, memory = _run(settings, demo_app, FAILING)
    assert result.status == "failed"
    assert memory.known_keys("username", "/login")


def test_staged_heal_is_reused_within_the_same_run(settings, demo_app, variant_v2):
    memory = Memory(settings.path(settings.memory_path))
    resolver = LocatorResolver(settings, memory)
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolver.begin_run()
    first = resolver.resolve(page, "username", "fill")
    second = resolver.resolve(page, "username", "fill")
    assert first.healed and not second.healed
    assert memory.is_pending("username", "/login", first.candidate.key)
    resolver.end_run(False)
    assert memory.known_keys("username", "/login") == []


def test_direct_resolve_outside_a_run_commits_immediately(settings, demo_app, variant_v2):
    memory = Memory(settings.path(settings.memory_path))
    resolver = LocatorResolver(settings, memory)
    page = FakePage(demo_app.base_url)
    page.goto("/login")
    resolver.resolve(page, "username", "fill")
    assert memory.known_keys("username", "/login")


def test_proposals_survive_a_save_and_reload(settings, demo_app, variant_v2):
    _, _, memory = _run(settings, demo_app, FAILING)
    memory.save()
    assert Memory(settings.path(settings.memory_path)).proposals
