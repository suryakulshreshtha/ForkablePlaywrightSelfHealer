"""Agent memory: what worked, where, and how often.

Self-healing is only useful if the repair is remembered. Every successful
resolution is scored and written to a JSON file keyed by *(page scope,
element description)*, so the next run tries the proven locator first and the
expensive healing path stays cold. The same file is the audit trail that
answers "which selectors has the app broken lately?".
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MEMORY_VERSION = 2

log = logging.getLogger(__name__)

# Advisory locking, stdlib only. The agent's stated design rule is a single
# runtime dependency, so `filelock` is deliberately not pulled in for this.
try:  # POSIX
    import fcntl

    def _lock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

except ImportError:  # pragma: no cover - Windows
    try:
        import msvcrt

        def _lock_fd(fd: int) -> None:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

        def _unlock_fd(fd: int) -> None:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

    except ImportError:  # pragma: no cover - no locking primitive available

        def _lock_fd(fd: int) -> None:
            return None

        def _unlock_fd(fd: int) -> None:
            return None


@contextlib.contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Serialise writers on *lock_path*.

    Two pytest-xdist workers healing the same page both try to persist memory.
    Without this they interleave and the loser's records are silently lost.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _lock_fd(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            _unlock_fd(fd)
        os.close(fd)


def scope_for(url: str, namespace: str = "") -> str:
    """Page scope for a URL.

    Path only, so ports and query strings do not fragment the cache, optionally
    prefixed by a namespace so distinct environments keep distinct memories.
    """
    if not url:
        return f"{namespace}:*" if namespace else "*"
    parsed = urlparse(url)
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{namespace}:{path}" if namespace else path


@dataclass
class LocatorRecord:
    key: str
    strategy: str = ""
    hits: int = 0
    misses: int = 0
    healed: bool = False
    last_seen: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Memory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.locators: dict[str, list[LocatorRecord]] = {}
        self.runs: list[dict[str, Any]] = []
        #: Heals made during a run that has not finished yet. They are usable
        #: inside the run but are not part of what gets saved until the run
        #: passes (see :meth:`commit_pending`).
        self.pending: list[dict[str, Any]] = []
        #: Heals from runs that failed. Kept for review, never trusted.
        self.proposals: list[dict[str, Any]] = []
        #: (entry, key) -> (hits, misses) as of the last load/save. ``save``
        #: merges *deltas* against this so parallel writers do not clobber
        #: each other.
        self._baseline: dict[tuple[str, str], tuple[int, int]] = {}
        self.load()

    # ------------------------------------------------------------------
    @staticmethod
    def _entry_key(description: str, scope: str) -> str:
        return f"{scope}::{description.strip().lower()}"

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Previously a bare `return`: a truncated or unreadable file threw
            # away every learned locator and the agent silently reverted to
            # cold healing, with nothing in the logs to explain the sudden
            # drop in cache hits. Quarantine it instead, and say so.
            quarantine = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                self.path.rename(quarantine)
            except OSError:
                quarantine = self.path  # left in place; at least warn about it
            log.warning(
                "memory at %s is unreadable (%s); quarantined to %s and starting empty",
                self.path, exc, quarantine,
            )
            return
        for key, records in (data.get("locators") or {}).items():
            self.locators[key] = [
                LocatorRecord(**{k: v for k, v in r.items() if k in LocatorRecord.__dataclass_fields__})
                for r in records
            ]
        self.runs = list(data.get("runs") or [])
        self.proposals = list(data.get("proposals") or [])
        self._reset_baseline()

    def _reset_baseline(self) -> None:
        self._baseline = {
            (entry, r.key): (r.hits, r.misses)
            for entry, records in self.locators.items()
            for r in records
        }

    def _read_disk(self) -> dict[str, Any] | None:
        """Current on-disk state, or None. Never quarantines: ``save`` is about
        to replace the file anyway, and a corrupt one was already reported by
        ``load``."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _merge_from_disk(self, disk: dict[str, Any]) -> None:
        """Fold another writer's changes into memory before saving.

        Holding a lock around ``write_text`` stops two writers *corrupting* the
        file, but each still dumps its own whole picture, so the later save
        silently erases the earlier one's records. Counters are therefore merged
        as deltas against what this instance loaded: disk value + what *we* did.
        """
        for entry, raw_records in (disk.get("locators") or {}).items():
            for raw in raw_records:
                fields = {k: v for k, v in raw.items() if k in LocatorRecord.__dataclass_fields__}
                if "key" not in fields:
                    continue
                mine = self._find(entry, fields["key"])
                if mine is None:
                    self.locators[entry].append(LocatorRecord(**fields))
                    continue
                base_hits, base_misses = self._baseline.get((entry, mine.key), (0, 0))
                mine.hits = int(fields.get("hits", 0)) + (mine.hits - base_hits)
                mine.misses = int(fields.get("misses", 0)) + (mine.misses - base_misses)
                mine.last_seen = max(mine.last_seen, float(fields.get("last_seen", 0.0)))
                mine.healed = mine.healed or bool(fields.get("healed", False))
                mine.strategy = mine.strategy or str(fields.get("strategy", ""))

        merged: dict[tuple[Any, Any], dict[str, Any]] = {}
        for run in list(disk.get("runs") or []) + self.runs:  # ours override on a tie
            merged[(run.get("name"), run.get("at"))] = run
        self.runs = sorted(merged.values(), key=lambda r: r.get("at") or 0.0)

        seen = {(p.get("description"), p.get("scope"), p.get("key"), p.get("at")) for p in self.proposals}
        for p in disk.get("proposals") or []:
            ident = (p.get("description"), p.get("scope"), p.get("key"), p.get("at"))
            if ident not in seen:
                self.proposals.append(p)
                seen.add(ident)

    def save(self) -> None:
        """Persist memory atomically.

        The previous implementation called ``write_text`` straight onto the
        live file. That truncates first and writes second, so a crash, a
        ``Ctrl-C`` or an out-of-disk between the two left a half-written
        ``memory.json`` - and ``load`` then discarded the lot. Two xdist
        workers writing concurrently interleaved into the same corruption.

        Write to a sibling temp file, fsync it, then ``os.replace`` - which is
        atomic on POSIX and on NTFS. A reader sees either the whole old file
        or the whole new one, never a partial write.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _exclusive(self._lock_path):
            disk = self._read_disk()
            if disk is not None:
                self._merge_from_disk(disk)
            payload = {
                "version": MEMORY_VERSION,
                "updated_at": time.time(),
                "locators": {k: [r.to_dict() for r in v] for k, v in self.locators.items()},
                "runs": self.runs[-100:],
                "proposals": self.proposals[-200:],
            }
            serialised = json.dumps(payload, indent=2)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(serialised)
                    handle.flush()
                    os.fsync(handle.fileno())  # survive a power cut, not just a crash
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise

            # Durably record the rename itself, so the directory entry cannot
            # outlive the data it points at.
            if hasattr(os, "O_DIRECTORY"):  # pragma: no branch - POSIX
                with contextlib.suppress(OSError):
                    dir_fd = os.open(self.path.parent, os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)

            self._reset_baseline()

    # -- staged heals ---------------------------------------------------
    def stage_heal(self, description: str, scope: str, key: str, strategy: str = "") -> None:
        """Hold a fresh heal until the run it happened in has passed.

        A heal that merely *worked* (the click did not throw) is not a heal that
        was *right*. Committing it immediately means a wrong binding is cached
        and then reinforced on every later run. Staged heals are visible to the
        rest of this run, so later steps reuse them, but they are only saved
        once :meth:`commit_pending` is called.
        """
        entry = self._entry_key(description, scope)
        if self.is_pending(description, scope, key):
            return
        self.pending.append({
            "entry": entry, "description": description, "scope": scope,
            "key": key, "strategy": strategy, "at": time.time(),
        })

    def is_pending(self, description: str, scope: str, key: str) -> bool:
        entry = self._entry_key(description, scope)
        return any(p["entry"] == entry and p["key"] == key for p in self.pending)

    def pending_keys(self, description: str, scope: str) -> list[str]:
        entry = self._entry_key(description, scope)
        return [p["key"] for p in reversed(self.pending) if p["entry"] == entry]

    def commit_pending(self) -> int:
        """Promote staged heals to proven locators. Returns how many."""
        staged, self.pending = self.pending, []
        for p in staged:
            self.record_success(p["description"], p["scope"], p["key"], p["strategy"], healed=True)
        return len(staged)

    def discard_pending(self, reason: str = "run did not pass") -> int:
        """Drop staged heals, keeping a reviewable record of what they were."""
        staged, self.pending = self.pending, []
        for p in staged:
            self.proposals.append({
                "description": p["description"], "scope": p["scope"], "key": p["key"],
                "strategy": p["strategy"], "at": p["at"], "reason": reason,
            })
        self.proposals = self.proposals[-200:]
        return len(staged)

    # ------------------------------------------------------------------
    def known_keys(self, description: str, scope: str = "*", min_confidence: float = 0.34) -> list[str]:
        """Proven locator keys for this description, best first.

        Heals staged earlier in the current run come first, so a second step
        targeting the same element reuses the binding instead of healing again.
        """
        staged = self.pending_keys(description, scope)
        records = list(self.locators.get(self._entry_key(description, scope), []))
        if scope != "*":
            records += list(self.locators.get(self._entry_key(description, "*"), []))
        good = [r for r in records if r.confidence >= min_confidence or r.hits > 0]
        good.sort(key=lambda r: (r.confidence, r.hits, r.last_seen), reverse=True)
        seen: set[str] = set(staged)
        out: list[str] = list(staged)
        for record in good:
            if record.key not in seen:
                seen.add(record.key)
                out.append(record.key)
        return out

    def _find(self, entry: str, key: str) -> LocatorRecord | None:
        for record in self.locators.setdefault(entry, []):
            if record.key == key:
                return record
        return None

    def record_success(
        self, description: str, scope: str, key: str, strategy: str = "", healed: bool = False
    ) -> None:
        entry = self._entry_key(description, scope)
        record = self._find(entry, key)
        if record is None:
            record = LocatorRecord(key=key, strategy=strategy, healed=healed)
            self.locators[entry].append(record)
        record.hits += 1
        record.strategy = strategy or record.strategy
        record.healed = record.healed or healed
        record.last_seen = time.time()

    def record_failure(self, description: str, scope: str, key: str) -> None:
        entry = self._entry_key(description, scope)
        record = self._find(entry, key)
        if record is None:
            return  # never proven, nothing to demote
        record.misses += 1
        record.last_seen = time.time()

    def record_run(self, summary: dict[str, Any]) -> None:
        self.runs.append(summary)

    # ------------------------------------------------------------------
    def healing_report(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry, records in sorted(self.locators.items()):
            scope, _, description = entry.partition("::")
            for record in records:
                rows.append({
                    "scope": scope,
                    "description": description,
                    "locator": record.key,
                    "strategy": record.strategy,
                    "hits": record.hits,
                    "misses": record.misses,
                    "confidence": round(record.confidence, 3),
                    "healed": record.healed,
                })
        rows.sort(key=lambda r: (not r["healed"], -r["hits"]))
        return rows
