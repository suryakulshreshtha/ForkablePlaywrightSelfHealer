"""Healing is for elements that changed, not elements that have not rendered yet."""

from __future__ import annotations

import threading
import time

from forkable_ai_agent.agent.healer import LocatorResolver
from forkable_ai_agent.agent.memory import Memory
from tests.support.fake_page import Document, page_from_html


def test_slow_element_is_awaited_not_healed(settings):
    settings.healer.settle_ms = 3000
    settings.healer.settle_interval_ms = 50
    page = page_from_html('<h1>loading</h1>')
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))

    def render_later() -> None:
        time.sleep(0.4)
        page.document = Document(
            '<html><body><button data-testid="go">Go</button></body></html>'
        )

    threading.Thread(target=render_later, daemon=True).start()
    resolution = resolver.resolve(page, "go button", "click")

    assert resolution.healed is False
    assert resolution.strategy.startswith("testid") or resolution.candidate.kind == "testid"
    assert not [e for e in resolver.events if "kind" not in e]


def test_settle_budget_is_bounded(settings):
    settings.healer.settle_ms = 300
    settings.healer.settle_interval_ms = 50
    page = page_from_html('<p>nothing here</p>')
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    started = time.monotonic()
    try:
        resolver.resolve(page, "go button", "click")
    except Exception:
        pass
    assert time.monotonic() - started < 3.0


def test_no_settle_means_heal_immediately(settings):
    settings.healer.settle_ms = 0
    page = page_from_html('<p>nothing here</p>')
    resolver = LocatorResolver(settings, Memory(settings.path(settings.memory_path)))
    started = time.monotonic()
    try:
        resolver.resolve(page, "go button", "click")
    except Exception:
        pass
    assert time.monotonic() - started < 1.0
