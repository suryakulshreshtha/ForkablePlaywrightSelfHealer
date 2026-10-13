"""Plan execution.

Each step is resolved (with healing), acted on, timed and recorded. Failures
capture a screenshot and stop the run unless the step is marked optional. The
executor deliberately implements its own polling assertions rather than using
``expect`` so that every wait, retry and message is reportable data rather than
a raised exception with an opaque string.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin

from ..schema import (
    API_ACTIONS,
    ELEMENT_ACTIONS,
    RunResult,
    Step,
    StepResult,
    TestPlan,
    wait_seconds,
)
from .healer import LocatorResolver


class StepFailed(RuntimeError):
    pass


def _poll(check: Callable[[], bool], timeout_ms: int, interval_ms: int = 100) -> bool:
    deadline = time.time() + timeout_ms / 1000.0
    while True:
        try:
            if check():
                return True
        except Exception:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(interval_ms / 1000.0)


class Executor:
    def __init__(
        self,
        settings,
        resolver: LocatorResolver,
        analyzer: Any = None,
        visual: Any = None,
        on_step: Callable[[StepResult], None] | None = None,
    ) -> None:
        self.settings = settings
        self.resolver = resolver
        self.analyzer = analyzer
        self.visual = visual
        self.on_step = on_step
        #: Last API response in this run, shared across api_request ->
        #: expect_status -> expect_json steps.
        self.last_response: Any = None
        self.last_body: Any = None

    # ------------------------------------------------------------------
    def _url_for(self, plan: TestPlan, target: str) -> str:
        base = plan.base_url or self.settings.app.base_url
        if target.startswith(("http://", "https://")):
            return target
        return urljoin(base.rstrip("/") + "/", target.lstrip("/"))

    # ------------------------------------------------------------------
    def run(self, plan: TestPlan, session: Any) -> RunResult:
        page = session.page
        result = RunResult(plan=plan)
        timeout = self.settings.browser.default_timeout_ms

        # Heals are a transaction: the resolver holds them while the run is in
        # flight, saves them if it passes and parks them for review if it fails.
        begin_run = getattr(self.resolver, "begin_run", None)
        end_run = getattr(self.resolver, "end_run", None)
        if begin_run is not None:
            begin_run()
        passed = False
        try:
            for index, step in enumerate(plan.steps, start=1):
                started = time.perf_counter()
                record = StepResult(index=index, step=step)
                try:
                    # _execute may downgrade the record itself (e.g. "warned");
                    # a clean return must not overwrite that.
                    self._execute(plan, page, session, step, record, timeout)
                except Exception as exc:
                    record.message = f"{type(exc).__name__}: {exc}"
                    record.status = "warned" if step.optional else "failed"
                    record.screenshot = session.screenshot(f"fail_step_{index}")
                record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
                result.steps.append(record)
                if self.on_step:
                    self.on_step(record)
                if record.status == "failed":
                    result.status = "failed"
                    break

            result.finished_at = time.time()
            if result.status != "failed":
                result.status = "passed"
            passed = result.status == "passed"

            collect = getattr(session, "collect_diagnostics", None)
            if collect is not None:
                result.diagnostics = collect()
            mark = getattr(session, "mark_result", None)
            if mark is not None:
                mark(passed)
            if result.status == "failed" and self.analyzer is not None:
                result.diagnosis = self.analyzer.analyze_run(result)
        finally:
            if end_run is not None:
                end_run(passed)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _dig(payload: Any, path: str) -> Any:
        """Walk a dotted path like ``jobs.0.status`` through parsed JSON."""
        current = payload
        for part in [p for p in path.split(".") if p]:
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    raise StepFailed(f"no index {part!r} in list of {len(current)}") from None
            elif isinstance(current, dict):
                if part not in current:
                    raise StepFailed(f"no key {part!r} in {sorted(current)[:8]}")
                current = current[part]
            else:
                raise StepFailed(f"cannot descend into {type(current).__name__} at {part!r}")
        return current

    def _execute_api(
        self,
        plan: TestPlan,
        page: Any,
        step: Step,
        record: StepResult,
        timeout: int,
    ) -> None:
        if step.action == "api_request":
            method, _, path = step.target.partition(" ")
            url = self._url_for(plan, path)
            # page.request shares the browser's cookie jar, so a UI login
            # authenticates the API call without re-authenticating.
            context = page.request
            kwargs: dict = {"timeout": timeout}
            if step.value:
                try:
                    kwargs["data"] = json.loads(step.value)
                except json.JSONDecodeError:
                    kwargs["data"] = step.value
            self.last_response = getattr(context, method.lower())(url, **kwargs)
            self.last_body = None
            record.selector = f"{method} {url}"
            record.message = f"HTTP {self.last_response.status}"
            return

        if self.last_response is None:
            raise StepFailed(f"{step.action} needs an api_request step before it")

        if step.action == "expect_status":
            expected = int(step.value or 200)
            actual = self.last_response.status
            record.selector = str(actual)
            if actual != expected:
                raise StepFailed(f"expected status {expected}, got {actual}")
            return

        if step.action == "expect_json":
            if self.last_body is None:
                try:
                    self.last_body = self.last_response.json()
                except Exception as exc:
                    raise StepFailed(f"response body is not JSON: {exc}") from exc
            value = self._dig(self.last_body, step.target)
            record.selector = f"{step.target}={value!r}"
            if step.value and str(step.value).lower() not in str(value).lower():
                raise StepFailed(
                    f"expected {step.target} to contain {step.value!r}, got {value!r}"
                )
            return

    def _execute(
        self,
        plan: TestPlan,
        page: Any,
        session: Any,
        step: Step,
        record: StepResult,
        timeout: int,
    ) -> None:
        action = step.action

        if action == "goto":
            page.goto(self._url_for(plan, step.target or "/"), wait_until="domcontentloaded")
            record.selector = self._url_for(plan, step.target or "/")
            return

        if action in API_ACTIONS:
            self._execute_api(plan, page, step, record, timeout)
            return

        if action == "wait":
            time.sleep(wait_seconds(step.value))
            return

        if action == "screenshot":
            record.screenshot = session.screenshot(step.value or f"step_{record.index}")
            return

        if action == "visual_check":
            if self.visual is None:
                record.message = "visual comparison unavailable (Pillow not installed)"
                record.status = "warned"
                return
            name = step.value or f"{plan.name}_{record.index}"
            outcome = self.visual.check(page, name)
            record.message = outcome.summary
            record.screenshot = outcome.diff_path or outcome.actual_path
            if not outcome.passed:
                raise StepFailed(f"visual diff {outcome.summary}")
            return

        if action == "expect_url":
            expected = step.value
            if not _poll(lambda: expected in (page.url or ""), timeout):
                raise StepFailed(f"expected url to contain {expected!r}, got {page.url!r}")
            record.selector = page.url
            return

        if action == "expect_title":
            expected = step.value
            if not _poll(lambda: expected.lower() in (page.title() or "").lower(), timeout):
                raise StepFailed(f"expected title to contain {expected!r}, got {page.title()!r}")
            return

        if action in {"expect_text", "expect_no_text"}:
            expected = step.value
            want = action == "expect_text"

            def seen() -> bool:
                try:
                    body = page.inner_text("body")
                except Exception:
                    body = page.content()
                return (expected.lower() in (body or "").lower()) == want

            if not _poll(seen, timeout):
                verb = "expected text" if want else "expected NO text"
                raise StepFailed(f"{verb} {expected!r} on {page.url}")
            return

        # ---- element actions -----------------------------------------
        if action in ELEMENT_ACTIONS:
            resolution = self.resolver.resolve(page, step.target, action)
            record.selector = resolution.candidate.describe()
            record.strategy = resolution.strategy
            record.healed = resolution.healed
            if resolution.ambiguous:
                record.message = (
                    f"ambiguous: {resolution.match_count} elements matched "
                    f"{resolution.candidate.describe()}; the first was used"
                )
            locator = resolution.locator

            if action == "click":
                locator.click(timeout=timeout)
            elif action == "fill":
                locator.fill(step.value or "", timeout=timeout)
            elif action == "press":
                locator.press(step.value or "Enter", timeout=timeout)
            elif action == "check":
                locator.check(timeout=timeout)
            elif action == "uncheck":
                locator.uncheck(timeout=timeout)
            elif action == "hover":
                locator.hover(timeout=timeout)
            elif action == "select":
                locator.select_option(step.value, timeout=timeout)
            elif action == "expect_visible":
                if not locator.is_visible():
                    raise StepFailed(f"{step.target!r} is not visible")
            elif action == "expect_value":
                actual = locator.input_value(timeout=timeout)
                if step.value.lower() not in (actual or "").lower():
                    raise StepFailed(f"expected value {step.value!r}, got {actual!r}")
            return

        raise StepFailed(f"unsupported action {action!r}")
