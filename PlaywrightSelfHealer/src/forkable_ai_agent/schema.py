"""Typed contracts shared by the planner, executor, generator and reporter.

A ``TestPlan`` is intentionally *semantic*: a step targets "password field",
not ``#pwd``. Binding a description to a real selector happens at run time,
which is precisely what makes self-healing possible.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

ACTIONS = {
    "goto",
    "click",
    "fill",
    "press",
    "check",
    "uncheck",
    "select",
    "hover",
    "wait",
    "screenshot",
    "visual_check",
    "expect_text",
    "expect_no_text",
    "expect_visible",
    "expect_url",
    "expect_title",
    "expect_value",
    # API-side steps. They share the browser's cookie jar, so a UI login and a
    # subsequent API assertion are the same session - which is the whole point
    # of doing API and UI in one plan rather than two suites.
    "api_request",
    "expect_status",
    "expect_json",
}

#: Steps that operate on the last API response rather than the DOM.
API_ACTIONS = {"api_request", "expect_status", "expect_json"}

#: Actions that need an element on the page (and therefore locator resolution).
ELEMENT_ACTIONS = {
    "click",
    "fill",
    "press",
    "check",
    "uncheck",
    "select",
    "hover",
    "expect_visible",
    "expect_value",
}


class PlanError(ValueError):
    """Raised when a plan cannot be parsed or validated."""


#: Longest a single ``wait`` step may pause. A plan is not a place to park a run.
MAX_WAIT_SECONDS = 30.0


def wait_seconds(value: str | float | None) -> float:
    """Seconds a ``wait`` step pauses for.

    Below 100 the value is seconds; from 100 up it is milliseconds ("500" means
    half a second). Unparseable or negative input falls back to one second, and
    the result is capped at :data:`MAX_WAIT_SECONDS`. The executor and the
    generator both call this, so a plan sleeps for the same time in a run and in
    the pytest file generated from it (they used to disagree by 1000x).
    """
    try:
        seconds = float(value if value not in (None, "") else 1.0)
    except (TypeError, ValueError):
        return 1.0
    if seconds != seconds or seconds < 0:  # NaN or negative
        return 1.0
    if seconds >= 100:
        seconds /= 1000.0
    return min(seconds, MAX_WAIT_SECONDS)


@dataclass
class Step:
    action: str
    target: str = ""              # natural-language element description or URL path
    value: str = ""               # text to type, expected text, option value...
    optional: bool = False        # failure downgrades to a warning
    note: str = ""

    def __post_init__(self) -> None:
        self.action = self.action.strip().lower()
        if self.action not in ACTIONS:
            raise PlanError(
                f"unknown action {self.action!r}; supported: {sorted(ACTIONS)}"
            )
        if self.action in ELEMENT_ACTIONS and not self.target:
            raise PlanError(f"action {self.action!r} requires a target description")
        if self.action == "api_request":
            parts = self.target.split()
            if len(parts) != 2 or parts[0].upper() not in {
                "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD",
            }:
                raise PlanError(
                    f"api_request target must be '<METHOD> <path>', got {self.target!r}"
                )
            self.target = f"{parts[0].upper()} {parts[1]}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(
            action=str(data.get("action", "")),
            target=str(data.get("target", "") or ""),
            value="" if data.get("value") is None else str(data.get("value")),
            optional=bool(data.get("optional", False)),
            note=str(data.get("note", "") or ""),
        )


@dataclass
class TestPlan:
    __test__ = False  # a plan is data, not a pytest test case

    name: str
    steps: list[Step]
    base_url: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = "llm"           # llm | rules | file

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [s.to_dict() for s in self.steps]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestPlan:
        if not isinstance(data, dict):
            raise PlanError("plan must be a JSON object")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanError("plan needs a non-empty 'steps' array")
        name = str(data.get("name") or "generated_test").strip()
        steps = [Step.from_dict(s) for s in raw_steps if isinstance(s, dict)]
        if not steps:
            raise PlanError("no valid steps in plan")
        return cls(
            name=slugify(name),
            steps=steps,
            base_url=str(data.get("base_url", "") or ""),
            description=str(data.get("description", "") or ""),
            tags=[str(t) for t in data.get("tags", []) if isinstance(t, (str, int))],
            source=str(data.get("source", "llm")),
        )

    @classmethod
    def from_json(cls, text: str) -> TestPlan:
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise PlanError(f"plan is not valid JSON: {exc}") from exc


@dataclass
class StepResult:
    index: int
    step: Step
    status: str = "passed"        # passed | failed | warned | skipped
    selector: str = ""            # selector actually used
    strategy: str = ""            # how the selector was found
    healed: bool = False
    duration_ms: float = 0.0
    message: str = ""
    screenshot: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["step"] = self.step.to_dict()
        return data


@dataclass
class RunResult:
    plan: TestPlan
    steps: list[StepResult] = field(default_factory=list)
    status: str = "passed"
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    diagnosis: Diagnosis | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    #: What the page itself reported while the plan ran: console errors,
    #: uncaught exceptions, failed requests and HTTP >= 400 responses. Empty
    #: when nothing went wrong or capture is off.
    diagnostics: dict[str, list[str]] = field(default_factory=dict)

    @property
    def healed_count(self) -> int:
        return sum(1 for s in self.steps if s.healed)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.finished_at or time.time()) - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "healed_count": self.healed_count,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "artifacts": self.artifacts,
            "diagnostics": self.diagnostics,
        }


@dataclass
class Diagnosis:
    category: str
    summary: str
    likely_cause: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)
    source: str = "rules"         # rules | llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(text: str, fallback: str = "generated_test") -> str:
    cleaned = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if slug and slug[0].isdigit():
        slug = "t_" + slug
    return slug[:60] or fallback
