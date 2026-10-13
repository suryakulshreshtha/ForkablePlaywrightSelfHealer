"""Shared fixtures.

The demo app is started once per session on an ephemeral port so tests never
collide with a developer's own `forkable serve`.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forkable_ai_agent.config import load_settings  # noqa: E402
from forkable_ai_agent.testapp import DemoApp  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def demo_app():
    app = DemoApp(host="127.0.0.1", port=_free_port()).start()
    yield app
    app.stop()


@pytest.fixture
def settings(tmp_path, demo_app):
    cfg = load_settings()
    cfg.root = str(ROOT)
    cfg.app.port = demo_app.port
    cfg.memory_path = str(tmp_path / "memory.json")
    cfg.rag.index_dir = str(tmp_path / "index")
    cfg.report_dir = str(tmp_path / "reports")
    cfg.visual.baseline_dir = str(tmp_path / "baselines")
    cfg.visual.diff_dir = str(tmp_path / "diffs")
    cfg.browser.artifacts_dir = str(tmp_path / "artifacts")
    # The demo pages are static, so waiting for them to "settle" only slows the
    # suite down. Tests that exercise the wait set these explicitly.
    cfg.healer.settle_ms = 0
    cfg.healer.dom_quiet_ms = 0
    return cfg


@pytest.fixture
def variant_v1():
    os.environ["FORKABLE_UI_VARIANT"] = "v1"
    yield "v1"
    os.environ.pop("FORKABLE_UI_VARIANT", None)


@pytest.fixture
def variant_v2():
    os.environ["FORKABLE_UI_VARIANT"] = "v2"
    yield "v2"
    os.environ.pop("FORKABLE_UI_VARIANT", None)
