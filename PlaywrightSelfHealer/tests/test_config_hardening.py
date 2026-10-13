"""New hardening settings load from config and from the environment."""

from __future__ import annotations

from forkable_ai_agent.config import load_settings


def test_defaults_are_the_safe_ones(monkeypatch):
    for name in ("FORKABLE_IGNORE_HTTPS_ERRORS", "FORKABLE_NO_SANDBOX"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_settings()
    assert cfg.browser.ignore_https_errors is False
    assert cfg.browser.no_sandbox == "auto"
    assert cfg.browser.trace_on_failure is True
    assert cfg.healer.heal_assertions is False
    assert cfg.healer.commit_policy == "on_pass"
    assert cfg.healer.on_ambiguous == "warn"


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("FORKABLE_IGNORE_HTTPS_ERRORS", "1")
    monkeypatch.setenv("FORKABLE_NO_SANDBOX", "never")
    cfg = load_settings()
    assert cfg.browser.ignore_https_errors is True
    assert cfg.browser.no_sandbox == "never"
