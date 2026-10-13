"""Two writers must not lose each other's updates."""

from __future__ import annotations

import json
import threading

from forkable_ai_agent.agent.memory import Memory


def test_second_writer_does_not_erase_the_first(tmp_path):
    path = tmp_path / "m.json"
    a, b = Memory(path), Memory(path)
    a.record_success("username", "/login", "css||0|#a", "id", healed=True)
    b.record_success("password", "/login", "css||0|#b", "id", healed=True)
    a.save()
    b.save()  # the stale snapshot must not wipe out a's entry

    merged = Memory(path)
    assert merged.known_keys("username", "/login") == ["css||0|#a"]
    assert merged.known_keys("password", "/login") == ["css||0|#b"]


def test_counters_merge_as_deltas_not_overwrites(tmp_path):
    path = tmp_path / "m.json"
    seed = Memory(path)
    seed.record_success("login", "/login", "css||0|#x", "id")
    seed.save()

    a, b = Memory(path), Memory(path)
    for _ in range(3):
        a.record_success("login", "/login", "css||0|#x", "id")
    for _ in range(2):
        b.record_success("login", "/login", "css||0|#x", "id")
    a.save()
    b.save()

    record = Memory(path).locators["/login::login"][0]
    assert record.hits == 1 + 3 + 2


def test_many_threads_saving_never_corrupt_the_file(tmp_path):
    path = tmp_path / "m.json"
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            mem = Memory(path)
            mem.record_success(f"el{n}", "/p", f"css||0|#e{n}", "id")
            mem.save()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert not errors
    data = json.loads(path.read_text())
    assert len(data["locators"]) == 12


def test_runs_from_both_writers_survive(tmp_path):
    path = tmp_path / "m.json"
    a, b = Memory(path), Memory(path)
    a.record_run({"name": "one", "status": "passed"})
    b.record_run({"name": "two", "status": "failed"})
    a.save()
    b.save()
    names = {r["name"] for r in Memory(path).runs}
    assert names == {"one", "two"}
