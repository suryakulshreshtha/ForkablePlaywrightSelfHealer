"""Playwright lifecycle management.

Playwright is imported lazily so the rest of the agent - planning, RAG,
codegen, reporting - stays importable and testable on a machine where browsers
were never installed. That matters for air-gapped setups where the browser
bundle arrives separately.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

#: Most entries kept per diagnostics bucket; a runaway page cannot fill memory.
DIAGNOSTICS_CAP = 50


class BrowserUnavailable(RuntimeError):
    """Playwright or its browser binaries are missing."""


def playwright_installed() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


def browsers_path() -> Path:
    custom = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if custom and custom not in {"0"}:
        return Path(custom)
    home = Path.home()
    if os.name == "nt":  # pragma: no cover
        return home / "AppData" / "Local" / "ms-playwright"
    if os.uname().sysname == "Darwin":  # pragma: no cover
        return home / "Library" / "Caches" / "ms-playwright"
    return home / ".cache" / "ms-playwright"


def browsers_installed(engine: str = "chromium") -> bool:
    root = browsers_path()
    if not root.exists():
        return False
    return any(p.name.startswith(engine) for p in root.iterdir())


def should_disable_sandbox(mode: str = "auto") -> bool:
    """Whether to launch Chromium with ``--no-sandbox``.

    The sandbox is a security boundary, so it stays on wherever it can work.
    ``auto`` turns it off only where Chromium cannot use it anyway: running as
    root, inside a container, or on CI. ``always`` / ``never`` force the answer.
    """
    mode = (mode or "auto").strip().lower()
    if mode in {"always", "true", "1", "yes", "on"}:
        return True
    if mode in {"never", "false", "0", "no", "off"}:
        return False
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and geteuid() == 0:
        return True
    return (
        Path("/.dockerenv").exists()
        or bool(os.environ.get("CI"))
        or bool(os.environ.get("container"))
    )


class BrowserSession:
    """Context manager owning a Playwright instance, browser, context and page."""

    def __init__(self, settings, engine: str | None = None, headless: bool | None = None) -> None:
        self.settings = settings
        self.engine = engine or settings.browser.engine
        self.headless = settings.browser.headless if headless is None else headless
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self.page: Any = None
        self.artifacts_dir = settings.path(settings.browser.artifacts_dir)
        #: Each session writes into its own folder, so two tests running at once
        #: (xdist, or two terminals) never overwrite each other's evidence.
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_dir = self.artifacts_dir / self.run_id
        self.trace_path: str = ""
        self._tracing = False
        self._passed: bool | None = None
        self._diagnostics: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    def start(self) -> BrowserSession:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on install
            raise BrowserUnavailable(
                "playwright is not installed. Run: pip install playwright"
            ) from exc

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        launcher = getattr(self._pw, self.engine, None)
        if launcher is None:
            raise BrowserUnavailable(f"unknown browser engine {self.engine!r}")

        args = None
        if self.engine == "chromium" and should_disable_sandbox(self.settings.browser.no_sandbox):
            args = ["--no-sandbox", "--disable-dev-shm-usage"]
        try:
            self._browser = launcher.launch(
                headless=self.headless,
                slow_mo=self.settings.browser.slow_mo_ms or 0,
                args=args,
            )
        except Exception as exc:
            self.stop()
            raise BrowserUnavailable(
                f"could not launch {self.engine}: {exc}. "
                "Install browsers once with 'python -m playwright install chromium', "
                "or point PLAYWRIGHT_BROWSERS_PATH at a pre-seeded bundle."
            ) from exc

        self._context = self._browser.new_context(
            viewport={
                "width": self.settings.browser.viewport_width,
                "height": self.settings.browser.viewport_height,
            },
            ignore_https_errors=bool(self.settings.browser.ignore_https_errors),
        )
        self._context.set_default_timeout(self.settings.browser.default_timeout_ms)

        browser_cfg = self.settings.browser
        if browser_cfg.trace or browser_cfg.trace_on_failure:
            self._context.tracing.start(screenshots=True, snapshots=True, sources=False)
            self._tracing = True

        self.page = self._context.new_page()
        if browser_cfg.capture_diagnostics:
            self._attach_diagnostics(self.page)
        return self

    # -- diagnostics ---------------------------------------------------
    def _add(self, bucket: str, text: str) -> None:
        items = self._diagnostics.setdefault(bucket, [])
        if len(items) < DIAGNOSTICS_CAP:
            items.append(text[:300])

    def _attach_diagnostics(self, page: Any) -> None:
        """Record what the page itself says went wrong.

        Every handler is wrapped: diagnostics are evidence, and a bug in
        collecting evidence must never fail the test being observed.
        """

        def safe(handler):
            def wrapper(*args: Any) -> None:
                try:
                    handler(*args)
                except Exception:
                    pass
            return wrapper

        page.on("console", safe(
            lambda msg: self._add("console_errors", msg.text) if msg.type == "error" else None
        ))
        page.on("pageerror", safe(lambda err: self._add("page_errors", str(err))))
        page.on("requestfailed", safe(
            lambda req: self._add("failed_requests", f"{req.method} {req.url} - {req.failure}")
        ))
        page.on("response", safe(
            lambda resp: self._add("http_errors", f"{resp.status} {resp.request.method} {resp.url}")
            if resp.status >= 400 else None
        ))

    def collect_diagnostics(self) -> dict[str, list[str]]:
        """A copy of everything captured so far, omitting empty buckets."""
        return {k: list(v) for k, v in self._diagnostics.items() if v}

    def mark_result(self, passed: bool) -> None:
        """Tell the session how the run ended; decides whether the trace is kept."""
        self._passed = bool(passed)

    # ------------------------------------------------------------------
    def screenshot(self, name: str, full_page: bool = False) -> str:
        if self.page is None:
            return ""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).lstrip(".") or "shot"
        path = self.run_dir / f"{safe}.png"
        try:
            self.page.screenshot(path=str(path), full_page=full_page)
        except Exception:
            return ""
        return str(path)

    def _finish_trace(self) -> None:
        """Save the trace if it is worth keeping, otherwise discard it.

        ``trace = true`` keeps every trace. ``trace_on_failure`` keeps it only
        when the run did not pass (or its outcome is unknown, e.g. a crash).
        """
        if not self._tracing or self._context is None:
            return
        self._tracing = False
        keep = bool(self.settings.browser.trace) or self._passed is not True
        try:
            if keep:
                target = self.run_dir / "trace.zip"
                self._context.tracing.stop(path=str(target))
                self.trace_path = str(target)
            else:
                self._context.tracing.stop()
        except Exception:
            pass

    def stop(self) -> None:
        self._finish_trace()
        for closer in (self._context, self._browser):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self.page = None

    def __enter__(self) -> BrowserSession:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
