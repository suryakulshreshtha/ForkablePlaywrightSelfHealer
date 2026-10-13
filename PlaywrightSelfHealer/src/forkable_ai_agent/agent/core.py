"""The façade: one object that owns the whole pipeline.

``ForkableAgent`` is intentionally lazy. Building a plan should not start a
browser; asking a knowledge question should not probe Ollama twice. Every
sub-component is constructed on first use, which keeps the CLI fast and makes
the class cheap to instantiate inside pytest fixtures.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import Settings, load_settings
from ..llm import build_llm
from ..net_guard import enforce_offline, is_enforced
from ..schema import RunResult, TestPlan
from .analyzer import FailureAnalyzer
from .executor import Executor
from .generator import GeneratedTest, TestGenerator
from .healer import LocatorResolver
from .memory import Memory
from .planner import Planner


def port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


class ForkableAgent:
    def __init__(self, settings: Settings | None = None, probe_llm: bool = True) -> None:
        self.settings = settings or load_settings()
        if self.settings.offline and not is_enforced():
            enforce_offline()
        self._probe_llm = probe_llm
        self._llm: Any = None
        self._knowledge: Any = None
        self._memory: Memory | None = None
        self._visual: Any = None
        self._app: Any = None

    # -- lazy components ------------------------------------------------
    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm(self.settings, probe=self._probe_llm)
        return self._llm

    @property
    def knowledge(self) -> Any:
        if self._knowledge is None:
            from ..rag import KnowledgeBase

            self._knowledge = KnowledgeBase(self.settings, self.llm)
            self._knowledge.load()
        return self._knowledge

    @property
    def memory(self) -> Memory:
        if self._memory is None:
            self._memory = Memory(self.settings.path(self.settings.memory_path))
        return self._memory

    @property
    def visual(self) -> Any:
        if self._visual is None:
            from ..visual import VisualValidator

            validator = VisualValidator(self.settings)
            self._visual = validator if validator.available() else None
        return self._visual

    @property
    def planner(self) -> Planner:
        return Planner(self.settings, self.llm)

    @property
    def resolver(self) -> LocatorResolver:
        return LocatorResolver(self.settings, self.memory, self.llm)

    @property
    def analyzer(self) -> FailureAnalyzer:
        return FailureAnalyzer(self.settings, self.knowledge, self.llm)

    @property
    def generator(self) -> TestGenerator:
        return TestGenerator(self.settings, self.memory)

    # -- demo app lifecycle --------------------------------------------
    def ensure_app(self, base_url: str = "") -> bool:
        """Start the bundled demo target if the configured URL is not answering."""
        url = base_url or self.settings.app.base_url
        parsed = urlparse(url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
        if port_open(host, port):
            return False
        from ..testapp import DemoApp

        self._app = DemoApp(host=host, port=port).start()
        for _ in range(50):
            if port_open(host, port):
                break
            time.sleep(0.05)
        return True

    def stop_app(self) -> None:
        if self._app is not None:
            self._app.stop()
            self._app = None

    # -- pipeline -------------------------------------------------------
    def plan(self, spec: str, base_url: str = "") -> TestPlan:
        return self.planner.plan(spec, base_url or self.settings.app.base_url)

    def generate(self, spec_or_plan: str | TestPlan, out_dir: str = "tests/generated") -> GeneratedTest:
        plan = spec_or_plan if isinstance(spec_or_plan, TestPlan) else self.plan(spec_or_plan)
        generated = self.generator.generate(plan)
        self.generator.write(generated, self.settings.path(out_dir))
        return generated

    def run(
        self,
        spec_or_plan: str | TestPlan,
        base_url: str = "",
        headed: bool = False,
        autostart_app: bool = True,
    ) -> RunResult:
        from ..browser import BrowserSession

        plan = spec_or_plan if isinstance(spec_or_plan, TestPlan) else self.plan(spec_or_plan, base_url)
        if base_url:
            plan.base_url = base_url
        plan.base_url = plan.base_url or self.settings.app.base_url

        started = self.ensure_app(plan.base_url) if autostart_app else False
        resolver = self.resolver
        executor = Executor(self.settings, resolver, self.analyzer, self.visual)
        # Held by reference: ``Memory.save`` merges in other writers' runs, so
        # "the last entry" is not necessarily this one when tests run in parallel.
        summary: dict[str, Any] = {"name": plan.name, "at": time.time(), "source": plan.source}
        session: Any = None
        try:
            with BrowserSession(self.settings, headless=not headed) as session:
                result = executor.run(plan, session)
        finally:
            # Heals were already committed or parked by Executor.run; this only
            # persists them. A run that crashed before then parks them here.
            resolver.end_run(False)
            self.memory.record_run(summary)
            self.memory.save()
            if started:
                self.stop_app()

        # Only genuine repairs count; ambiguity, model errors and commit notices
        # are events too, but they are not heals.
        result.artifacts["healing_events"] = str(sum(1 for e in resolver.events if "kind" not in e))
        ambiguous = sum(1 for e in resolver.events if e.get("kind") in {"ambiguous", "ambiguous_refused"})
        if ambiguous:
            result.artifacts["ambiguous_matches"] = str(ambiguous)
        trace = getattr(session, "trace_path", "")
        if trace:
            result.artifacts["trace"] = trace
        summary.update({"status": result.status, "healed": result.healed_count})
        self.memory.save()
        return result

    def ask(self, question: str) -> tuple[str, list[Any]]:
        return self.knowledge.answer(question)

    def index(self) -> dict[str, Any]:
        from ..rag import KnowledgeBase

        kb = KnowledgeBase(self.settings, self.llm)
        manifest = kb.index()
        self._knowledge = kb
        return manifest

    def report(self, result: RunResult) -> Path:
        from ..reporting import write_report

        return write_report(self.settings, result, self.memory)

    # -- diagnostics ----------------------------------------------------
    def doctor(self) -> list[Check]:
        from ..browser import browsers_installed, playwright_installed
        from ..visual import PIL_AVAILABLE

        checks: list[Check] = [
            Check("offline guard", is_enforced() or not self.settings.offline,
                  "socket layer restricted to loopback" if is_enforced() else "not armed"),
            Check("python package", True, "forkable_ai_agent importable"),
            Check("playwright package", playwright_installed(),
                  "pip install playwright" if not playwright_installed() else "installed"),
            Check("browser binaries", browsers_installed(self.settings.browser.engine),
                  f"{self.settings.browser.engine} bundle under {Path.home() / '.cache/ms-playwright'}"),
        ]

        llm = self.llm
        live = getattr(llm, "name", "") == "ollama"
        models = llm.list_models() if live else []
        checks.append(Check(
            "ollama daemon", live,
            f"{self.settings.llm.base_url} -> {', '.join(models[:4]) or 'no models pulled'}"
            if live else "not reachable; deterministic rule engine in use",
            required=False,
        ))
        checks.append(Check(
            "rag index", bool(self.knowledge.store.docs),
            f"{len(self.knowledge.store.docs)} chunks "
            f"({self.knowledge.store.manifest.get('embedder', 'n/a')})"
            if self.knowledge.store.docs else "run 'forkable index'",
        ))
        checks.append(Check("visual diffing", PIL_AVAILABLE,
                            "Pillow present" if PIL_AVAILABLE else "pip install pillow", required=False))
        checks.append(Check(
            "demo app port", True,
            f"{self.settings.app.base_url} "
            f"({'listening' if port_open(self.settings.app.host, self.settings.app.port) else 'free, will autostart'})",
            required=False,
        ))
        return checks
