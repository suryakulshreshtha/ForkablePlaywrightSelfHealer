"""Real-Chromium checks for the code paths the DOM harness cannot exercise.

The harness stands in for Playwright, so anything that depends on a real
``Locator.evaluate``, the DOM-quiet script, tracing or page events has to be
proven against the real thing.
"""

from __future__ import annotations

import pytest

from forkable_ai_agent.agent.healer import AmbiguousMatch, ElementNotFound, LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from forkable_ai_agent.browser import BrowserSession, browsers_installed, playwright_installed

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not playwright_installed(), reason="playwright package not installed"),
    pytest.mark.skipif(not browsers_installed("chromium"), reason="chromium bundle not installed"),
]


@pytest.fixture
def session(settings):
    with BrowserSession(settings) as s:
        yield s


def _resolver(settings):
    return LocatorResolver(settings, Memory(settings.path(settings.memory_path)))


def test_role_noun_rejects_a_heading_but_accepts_a_styled_link(settings, session):
    page = session.page
    page.set_content("<h2>Log in</h2>")
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(page, "log in button", "click")
    page.set_content('<a class="btn" href="#x">Log in</a>')
    assert _resolver(settings).resolve(page, "log in button", "click").match_count == 1


def test_help_link_is_not_the_login_button(settings, session):
    session.page.set_content('<a href="#h">Log in help</a>')
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(session.page, "log in button", "click")


def test_fill_refuses_a_non_input(settings, session):
    session.page.set_content('<button id="u">Username</button>')
    with pytest.raises(ElementNotFound):
        _resolver(settings).resolve(session.page, "username field", "fill")


def test_contenteditable_counts_as_a_textbox(settings, session):
    session.page.set_content('<div contenteditable="true" aria-label="Message">hi</div>')
    assert _resolver(settings).resolve(session.page, "message field", "fill").match_count == 1


def test_assertion_on_two_identical_controls_fails(settings, session):
    session.page.set_content("<button>Delete</button><button>Delete</button>")
    with pytest.raises(AmbiguousMatch):
        _resolver(settings).resolve(session.page, "delete", "expect_visible")


def test_slow_render_is_awaited_not_healed(settings, session):
    settings.healer.settle_ms = 4000
    session.page.set_content(
        "<h1>loading</h1><script>setTimeout(()=>{"
        "const b=document.createElement('button');b.dataset.testid='go';b.textContent='Go';"
        "document.body.appendChild(b);},700)</script>"
    )
    resolution = _resolver(settings).resolve(session.page, "go button", "click")
    assert resolution.healed is False


def test_heal_waits_for_a_churning_dom_and_still_finishes(settings, session):
    settings.healer.dom_quiet_ms = 150
    settings.healer.settle_ms = 0
    session.page.set_content(
        '<input id="usr_9" name="usr_9" placeholder="Your username">'
        "<script>let n=0;const t=setInterval(()=>{document.body.appendChild("
        "document.createElement('i'));if(++n>5)clearInterval(t)},40)</script>"
    )
    resolution = _resolver(settings).resolve(session.page, "username", "fill")
    assert resolution.locator.count() == 1


def test_page_errors_and_failed_requests_are_captured(settings, session, demo_app):
    session.page.goto(f"{demo_app.base_url}/login")
    session.page.evaluate("console.error('visible problem')")
    session.page.set_content("<script>setTimeout(()=>{throw new Error('boom')},0)</script>")
    session.page.wait_for_timeout(200)
    session.page.goto(f"{demo_app.base_url}/definitely-missing")
    diag = session.collect_diagnostics()
    assert any("visible problem" in m for m in diag.get("console_errors", []))
    assert any("boom" in m for m in diag.get("page_errors", []))
    assert any(m.startswith("404 ") for m in diag.get("http_errors", []))


def test_trace_is_written_for_a_failed_run_only(settings, demo_app):
    with BrowserSession(settings) as failed:
        failed.page.goto(f"{demo_app.base_url}/login")
        failed.mark_result(False)
    assert failed.trace_path and __import__("os").path.getsize(failed.trace_path) > 0

    with BrowserSession(settings) as ok:
        ok.page.goto(f"{demo_app.base_url}/login")
        ok.mark_result(True)
    assert ok.trace_path == ""


def test_two_sessions_never_share_an_artifact_folder(settings):
    # Sync Playwright cannot nest two sessions in one thread, so run them in turn.
    folders = []
    for _ in range(2):
        with BrowserSession(settings) as s:
            assert s.screenshot("x")
            folders.append(s.run_dir)
    assert folders[0] != folders[1]
    assert all(f.exists() for f in folders)
