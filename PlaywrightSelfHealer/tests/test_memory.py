"""Locator memory: scoring, persistence and reporting."""

from __future__ import annotations

import json
import logging
import threading

import pytest

from forkable_ai_agent.agent import memory as memory_module
from forkable_ai_agent.agent.memory import Memory, scope_for


def test_scope_ignores_host_port_and_query():
    assert scope_for("http://127.0.0.1:8799/login?ui=v2") == "/login"
    assert scope_for("http://localhost:9000/login") == "/login"
    assert scope_for("") == "*"


def test_successes_outrank_failures(tmp_path):
    memory = Memory(tmp_path / "m.json")
    memory.record_success("login", "/login", "css||0|#good", "id")
    memory.record_success("login", "/login", "css||0|#good", "id")
    memory.record_success("login", "/login", "css||0|#meh", "id")
    memory.record_failure("login", "/login", "css||0|#meh")
    memory.record_failure("login", "/login", "css||0|#meh")

    assert memory.known_keys("login", "/login")[0] == "css||0|#good"


def test_round_trips_through_disk(tmp_path):
    path = tmp_path / "m.json"
    first = Memory(path)
    first.record_success("username", "/login", "testid||0|[data-testid=\"u\"]", "data-testid", healed=True)
    first.record_run({"name": "login", "status": "passed"})
    first.save()

    second = Memory(path)
    assert second.known_keys("username", "/login") == ['testid||0|[data-testid="u"]']
    assert second.runs[-1]["status"] == "passed"


def test_healing_report_surfaces_healed_entries_first(tmp_path):
    memory = Memory(tmp_path / "m.json")
    memory.record_success("plain", "/a", "css||0|#p", "id")
    memory.record_success("fixed", "/a", "css||0|#f", "healed-id", healed=True)
    assert memory.healing_report()[0]["description"] == "fixed"


def test_unknown_description_returns_nothing(tmp_path):
    assert Memory(tmp_path / "m.json").known_keys("nope", "/x") == []


def test_namespace_keeps_environments_apart(tmp_path):
    """Two DOMs at one path must not share learned selectors."""
    memory = Memory(tmp_path / "m.json")
    memory.record_success("username", scope_for("http://x/login", "v1"), "css||0|#username", "id")
    memory.record_success("username", scope_for("http://x/login", "v2"), "css||0|#usr_1a2b", "healed-id")

    assert memory.known_keys("username", scope_for("http://x/login", "v1")) == ["css||0|#username"]
    assert memory.known_keys("username", scope_for("http://x/login", "v2")) == ["css||0|#usr_1a2b"]


def test_scope_without_a_namespace_is_unchanged():
    assert scope_for("http://x/login") == "/login"
    assert scope_for("http://x/login", "staging") == "staging:/login"


# --------------------------------------------------------------------------
# Durability. These three cover the failure mode that used to lose every
# learned locator without a word: a partially written or unreadable file.
# --------------------------------------------------------------------------


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "m.json"
    memory = Memory(path)
    memory.record_success("login", "/login", "css||0|#a", "id")
    memory.save()
    memory.record_success("login", "/login", "css||0|#b", "id")
    memory.save()

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files survived a successful save: {leftovers}"
    # The lock file is expected; the payload must still be valid JSON.
    assert json.loads(path.read_text(encoding="utf-8"))["version"] >= 2


def test_crash_midway_through_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "m.json"
    memory = Memory(path)
    memory.record_success("login", "/login", "css||0|#original", "id")
    memory.save()
    good = path.read_text(encoding="utf-8")

    # Simulate the process dying after the temp file is written but before the
    # atomic swap. Under the old write_text() implementation the live file was
    # already truncated at this point.
    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(memory_module.os, "replace", boom)
    memory.record_success("login", "/login", "css||0|#newer", "id")
    with pytest.raises(KeyboardInterrupt):
        memory.save()

    assert path.read_text(encoding="utf-8") == good, "a failed save damaged the live file"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_corrupt_file_is_quarantined_and_reported(tmp_path, caplog):
    path = tmp_path / "m.json"
    path.write_text('{"locators": {"a": [', encoding="utf-8")  # truncated mid-write

    with caplog.at_level(logging.WARNING):
        memory = Memory(path)

    assert memory.locators == {}, "a corrupt file must not be loaded"
    assert any("unreadable" in r.message for r in caplog.records), "corruption was silent"
    quarantined = list(tmp_path.glob("m.json.corrupt-*"))
    assert quarantined, "the corrupt file was discarded instead of quarantined"
    # The evidence is preserved verbatim for debugging.
    assert quarantined[0].read_text(encoding="utf-8") == '{"locators": {"a": ['


def test_concurrent_writers_do_not_interleave(tmp_path):
    """Two processes saving at once must leave one whole, parseable file."""
    path = tmp_path / "m.json"

    def writer(tag: str) -> None:
        mem = Memory(path)
        for i in range(25):
            mem.record_success(f"{tag}-{i}", "/login", f"css||0|#{tag}{i}", "id")
            mem.save()

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    payload = json.loads(path.read_text(encoding="utf-8"))  # must not raise
    assert payload["version"] >= 2
