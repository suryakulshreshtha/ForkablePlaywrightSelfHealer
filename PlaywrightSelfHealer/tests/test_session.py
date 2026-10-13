"""BrowserSession lifecycle, driven by a fake Playwright object graph.

No browser is launched; the point is the policy around it - when the sandbox is
disabled, when a trace is kept, what diagnostics are captured, and that two
sessions never share an artifact folder.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from forkable_ai_agent.browser import session as session_module
from forkable_ai_agent.browser.session import BrowserSession, should_disable_sandbox


class _Tracing:
    def __init__(self):
        self.started = False
        self.stopped_with: list = []

    def start(self, **kw):
        self.started = True

    def stop(self, path=None):
        self.stopped_with.append(path)


class _Page:
    def __init__(self):
        self.handlers: dict = {}

    def on(self, event, handler):
        self.handlers[event] = handler

    def screenshot(self, path="", full_page=False):
        open(path, "wb").write(b"png")


class _Context:
    def __init__(self):
        self.tracing = _Tracing()
        self.page = _Page()
        self.kwargs: dict = {}
        self.closed = False

    def set_default_timeout(self, ms):
        pass

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class _Browser:
    def __init__(self, ctx):
        self.ctx = ctx

    def new_context(self, **kw):
        self.ctx.kwargs = kw
        return self.ctx

    def close(self):
        pass


class _Launcher:
    def __init__(self, ctx):
        self.ctx = ctx
        self.launch_kwargs: dict = {}

    def launch(self, **kw):
        self.launch_kwargs = kw
        return _Browser(self.ctx)


class _Playwright:
    def __init__(self, ctx):
        self.chromium = _Launcher(ctx)

    def stop(self):
        pass


@pytest.fixture
def fake_pw(monkeypatch):
    ctx = _Context()
    pw = _Playwright(ctx)
    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = lambda: types.SimpleNamespace(start=lambda: pw)
    monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", module)
    return types.SimpleNamespace(ctx=ctx, pw=pw)


# -- sandbox policy -------------------------------------------------------------
def test_sandbox_modes(monkeypatch):
    assert should_disable_sandbox("always") is True
    assert should_disable_sandbox("never") is False
    monkeypatch.setattr(session_module.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("container", raising=False)
    monkeypatch.setattr(session_module.Path, "exists", lambda self: False)
    assert should_disable_sandbox("auto") is False
    monkeypatch.setenv("CI", "true")
    assert should_disable_sandbox("auto") is True
    monkeypatch.delenv("CI")
    monkeypatch.setattr(session_module.os, "geteuid", lambda: 0, raising=False)
    assert should_disable_sandbox("auto") is True


def test_sandbox_stays_on_when_told_never(settings, fake_pw):
    settings.browser.no_sandbox = "never"
    with BrowserSession(settings):
        pass
    assert fake_pw.pw.chromium.launch_kwargs["args"] is None


def test_sandbox_flags_passed_when_forced(settings, fake_pw):
    settings.browser.no_sandbox = "always"
    with BrowserSession(settings):
        pass
    assert "--no-sandbox" in fake_pw.pw.chromium.launch_kwargs["args"]


# -- TLS ------------------------------------------------------------------------
def test_tls_errors_are_not_ignored_by_default(settings, fake_pw):
    with BrowserSession(settings):
        pass
    assert fake_pw.ctx.kwargs["ignore_https_errors"] is False


def test_tls_errors_can_be_ignored_explicitly(settings, fake_pw):
    settings.browser.ignore_https_errors = True
    with BrowserSession(settings):
        pass
    assert fake_pw.ctx.kwargs["ignore_https_errors"] is True


# -- trace policy ---------------------------------------------------------------
def test_trace_is_kept_for_a_failed_run(settings, fake_pw):
    settings.browser.trace = False
    settings.browser.trace_on_failure = True
    with BrowserSession(settings) as session:
        session.mark_result(False)
    assert fake_pw.ctx.tracing.stopped_with[0] is not None
    assert session.trace_path.endswith("trace.zip")


def test_trace_is_discarded_for_a_passing_run(settings, fake_pw):
    settings.browser.trace = False
    settings.browser.trace_on_failure = True
    with BrowserSession(settings) as session:
        session.mark_result(True)
    assert fake_pw.ctx.tracing.stopped_with == [None]
    assert session.trace_path == ""


def test_trace_is_kept_when_the_outcome_is_unknown(settings, fake_pw):
    """A crash never calls mark_result; that is exactly when the trace matters."""
    with BrowserSession(settings) as session:
        pass
    assert session.trace_path


def test_no_tracing_when_both_switches_are_off(settings, fake_pw):
    settings.browser.trace = False
    settings.browser.trace_on_failure = False
    with BrowserSession(settings):
        pass
    assert fake_pw.ctx.tracing.started is False


# -- diagnostics ----------------------------------------------------------------
def test_page_diagnostics_are_captured_and_capped(settings, fake_pw):
    with BrowserSession(settings) as session:
        handlers = fake_pw.ctx.page.handlers
        handlers["console"](types.SimpleNamespace(type="error", text="bad thing"))
        handlers["console"](types.SimpleNamespace(type="log", text="ignored"))
        handlers["pageerror"](RuntimeError("uncaught"))
        handlers["requestfailed"](types.SimpleNamespace(method="GET", url="http://x/y", failure="net::ERR"))
        handlers["response"](types.SimpleNamespace(
            status=500, url="http://x/api", request=types.SimpleNamespace(method="GET")))
        handlers["response"](types.SimpleNamespace(
            status=200, url="http://x/ok", request=types.SimpleNamespace(method="GET")))
        for i in range(200):
            handlers["pageerror"](RuntimeError(f"e{i}"))
        diag = session.collect_diagnostics()

    assert diag["console_errors"] == ["bad thing"]
    assert diag["http_errors"] == ["500 GET http://x/api"]
    assert diag["failed_requests"][0].startswith("GET http://x/y")
    assert len(diag["page_errors"]) == session_module.DIAGNOSTICS_CAP


def test_a_broken_handler_never_raises_into_the_page(settings, fake_pw):
    with BrowserSession(settings):
        fake_pw.ctx.page.handlers["response"](object())  # missing attributes


def test_diagnostics_can_be_switched_off(settings, fake_pw):
    settings.browser.capture_diagnostics = False
    with BrowserSession(settings):
        pass
    assert fake_pw.ctx.page.handlers == {}


# -- artifacts ------------------------------------------------------------------
def test_each_session_gets_its_own_artifact_folder(settings, fake_pw):
    a, b = BrowserSession(settings), BrowserSession(settings)
    assert a.run_dir != b.run_dir
    with a:
        shot = a.screenshot("../../escape me.png")
    assert Path(shot).parent == a.run_dir


def test_context_is_closed_on_exit(settings, fake_pw):
    with BrowserSession(settings):
        pass
    assert fake_pw.ctx.closed
