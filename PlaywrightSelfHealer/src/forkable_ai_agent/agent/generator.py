"""Plan -> runnable pytest + Playwright source.

The generator prefers locators the agent has already *proven* on a live page
(from :class:`Memory`), falling back to the top of the static candidate ladder.
That is the difference between generated code that works and generated code
that merely looks plausible.

Plans are untrusted input: they come from a language model or from a file, and
their text ends up inside source code that somebody will run. Two defences, both
required:

* **Nothing is interpolated raw.** Every value reaches the emitted source as a
  ``json.dumps``-style literal (which escapes newlines and quotes) or, for text
  that lands in a comment or docstring, is flattened to one safe line first.
* **The emitted module is validated against an allowlist** - the only syntax,
  names, imports and method calls the generator can legitimately produce - before
  it is written. A denylist of "dangerous" calls is always one ``open()`` short
  of useless; an allowlist fails closed.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..browser.locators import LocatorCandidate, build_candidates, py_literal
from ..schema import ELEMENT_ACTIONS, Step, TestPlan, wait_seconds
from .memory import Memory, scope_for

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
ALLOWED_IMPORTS = {"re", "pytest", "playwright.sync_api"}
ALLOWED_FROM_NAMES = {"playwright.sync_api": {"Page", "expect"}}

#: Kept for callers that import them; the allowlist below is what actually gates.
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "input", "breakpoint", "open", "getattr",
                "setattr", "delattr", "globals", "locals", "vars", "dir"}
BANNED_ATTRS = {"system", "popen", "spawn", "spawnl", "check_output", "call", "run"}
BANNED_MODULES = {"subprocess", "socket", "shutil", "requests", "urllib", "httpx", "ctypes", "pickle",
                  "os", "pathlib", "sys", "importlib", "builtins"}

#: The only syntax the generator emits.
ALLOWED_NODES: tuple[type, ...] = (
    ast.Module, ast.Expr, ast.Constant, ast.Import, ast.ImportFrom, ast.alias, ast.Assign, ast.Assert,
    ast.FunctionDef, ast.arguments, ast.arg, ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Call,
    ast.keyword, ast.Subscript, ast.Compare, ast.In, ast.Eq, ast.JoinedStr, ast.FormattedValue,
    ast.BinOp, ast.Add, ast.Dict, ast.List, ast.Tuple, ast.UnaryOp, ast.USub, ast.Not, ast.Pass,
)

#: The only bare names that may appear (True/False/None are Constants).
ALLOWED_NAMES = {"page", "response", "re", "pytest", "expect", "str", "Page", "BASE_URL"}

#: The only attributes that may be read or called. Every one is a Playwright /
#: pytest / re / str API the generator itself emits.
ALLOWED_ATTRS = {
    "mark", "forkable", "request", "status", "I", "IGNORECASE",
    "get_by_role", "get_by_label", "get_by_placeholder", "get_by_text", "locator",
    "click", "fill", "press", "check", "uncheck", "hover", "select_option",
    "goto", "wait_for_timeout", "screenshot",
    "get", "post", "put", "patch", "delete", "head", "json", "text", "lower", "compile",
    "to_be_visible", "to_have_value", "to_have_url", "to_have_title",
    "to_contain_text", "not_to_contain_text",
}
ALLOWED_CALL_NAMES = {"expect", "str"}


class UnsafeGeneratedCode(ValueError):
    """The generated module failed validation and was not written to disk."""


@dataclass
class GeneratedTest:
    __test__ = False

    name: str
    code: str
    path: Path | None = None
    locators: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.locators is None:
            self.locators = []


def validate_code(code: str) -> None:
    """Parse generated source and check it against the allowlist.

    Raises :class:`UnsafeGeneratedCode`. Anything the generator does not itself
    emit - another import, ``open()``, an attribute walk to ``__class__`` - is
    refused, whatever it would have done.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeGeneratedCode(f"generated code does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS or alias.asname:
                    raise UnsafeGeneratedCode(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            allowed = ALLOWED_FROM_NAMES.get(module)
            if node.level or allowed is None:
                raise UnsafeGeneratedCode(f"import from {module!r} is not allowed")
            for alias in node.names:
                if alias.name not in allowed or alias.asname:
                    raise UnsafeGeneratedCode(f"import of {module}.{alias.name} is not allowed")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in ALLOWED_CALL_NAMES:
                    raise UnsafeGeneratedCode(f"call to {func.id}() is not allowed")
        elif isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                raise UnsafeGeneratedCode(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr not in ALLOWED_ATTRS:
                raise UnsafeGeneratedCode(f"attribute {node.attr!r} is not allowed")
        elif not isinstance(node, ALLOWED_NODES):
            raise UnsafeGeneratedCode(f"syntax element {type(node).__name__} is not allowed")

        # ALLOWED_NODES covers the generic cases above; the specific branches
        # still need the node-type gate for the types they handle.
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeGeneratedCode(f"syntax element {type(node).__name__} is not allowed")


# ---------------------------------------------------------------------------
# Safe emission helpers
# ---------------------------------------------------------------------------
def _py_str(value: Any) -> str:
    """A complete, double-quoted Python string literal. Newlines cannot escape it."""
    return py_literal(str(value))


def _inline(text: Any, limit: int = 200) -> str:
    """Flatten arbitrary text to one safe line for a comment or docstring.

    Control characters and every kind of line break collapse to a space, and the
    characters that could close a string or start an escape are neutralised, so
    the text cannot end the comment (new line) or the docstring (``\"\"\"``).
    """
    printable = "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in str(text or ""))
    flat = " ".join(printable.split())
    flat = flat.replace("\\", "/").replace('"', "'")
    return flat[:limit]


_SAFE_PATH = re.compile(r"^[A-Za-z0-9/_\-.~%?=&#:@!$()*+,;]*$")


def _url_expr(path: str) -> str:
    """Source for ``BASE_URL`` + *path* that cannot be broken out of."""
    path = path if path.startswith("/") else "/" + path
    if _SAFE_PATH.match(path):
        return f'f"{{BASE_URL}}{path}"'
    return f"BASE_URL + {_py_str(path)}"


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def _data_kwarg(value: str) -> str:
    """``, data=<literal>`` for an API body, or ``""`` when there is none.

    JSON is parsed and re-emitted as a *Python* literal, so ``true`` becomes
    ``True`` and ``null`` becomes ``None``. Pasting the JSON text in raw (as this
    used to) produced ``NameError`` at test time while still passing validation.
    """
    if not value:
        return ""
    try:
        parsed: Any = json.loads(value, parse_constant=_reject_constant)
    except ValueError:  # includes JSONDecodeError and NaN/Infinity
        parsed = value
    return f", data={parsed!r}"


def _test_name(raw: str) -> str:
    """A name usable in both ``def test_<name>`` and a filename.

    ``TestPlan.from_dict`` already slugifies, but a plan built in code does not,
    and the name lands in a ``def`` line and a file path.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(raw or "")).strip("_") or "generated_test"


def _artifact_name(raw: str, fallback: str) -> str:
    """A filename that cannot climb out of the artifacts directory."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", raw or "").lstrip(".")
    return cleaned or fallback


class TestGenerator:
    __test__ = False  # generates tests, is not one

    def __init__(self, settings, memory: Memory | None = None) -> None:
        self.settings = settings
        self.memory = memory
        self.namespace = getattr(settings, "memory_namespace", "")

    # ------------------------------------------------------------------
    def _locator_code(self, description: str, action: str, scope: str) -> tuple[str, str]:
        known: Sequence[str] = ()
        if self.memory is not None:
            known = self.memory.known_keys(description, scope)
        candidates = build_candidates(description, action, known=known)
        if not candidates:
            candidates = [LocatorCandidate("text", description, strategy="visible-text")]

        if known:
            # Proven on a real page during a run: emit exactly that.
            best = candidates[0]
        else:
            # Nothing proven yet. A speculative [data-testid="..."] is a coin
            # flip, while role/label/placeholder are derived from the words the
            # author already used, so they are far likelier to bind.
            best = next(
                (c for c in candidates if c.kind in {"role", "label", "placeholder", "text"}),
                candidates[0],
            )
        return best.code("page"), f"{best.strategy}:{best.describe()}"

    # ------------------------------------------------------------------
    def generate(self, plan: TestPlan) -> GeneratedTest:
        base = plan.base_url or self.settings.app.base_url
        name = _test_name(plan.name)
        lines: list[str] = []
        used: list[str] = []
        scope = "/"

        header = [
            '"""Generated by PlaywrightSelfHealer - do not edit by hand.',
            "",
            f"Plan: {_inline(plan.name, 80)}",
            f"Source: {_inline(plan.source, 40)}",
            f"Spec: {_inline(plan.description, 150)}",
            "",
            "Regenerate with:  forkable generate --spec <file>",
            '"""',
            "",
            "import re",
            "",
            "import pytest",
            "from playwright.sync_api import Page, expect",
            "",
            f"BASE_URL = {_py_str(base)}",
            "",
            "",
            "@pytest.mark.forkable",
            f"def test_{name}(page: Page) -> None:",
        ]

        for step in plan.steps:
            code_lines, scope, locator_note = self._step_code(step, scope)
            lines.extend(code_lines)
            if locator_note:
                used.append(locator_note)

        if len(lines) == 0:
            lines.append("    pass")

        code = "\n".join(header + lines) + "\n"
        validate_code(code)
        return GeneratedTest(name=name, code=code, locators=used)

    # ------------------------------------------------------------------
    def _step_code(self, step: Step, scope: str) -> tuple[list[str], str, str]:
        action = step.action
        note = ""
        out: list[str] = []
        if step.note:
            out.append(f"    # {_inline(step.note)}")

        if action == "goto":
            target = step.target or "/"
            scope = scope_for(target, self.namespace)
            if target.startswith(("http://", "https://")):
                out.append(f"    page.goto({_py_str(target)})")
            else:
                out.append(f"    page.goto({_url_expr(target)})")
            return out, scope, note

        if action == "wait":
            out.append(f"    page.wait_for_timeout({int(wait_seconds(step.value) * 1000)})")
            return out, scope, note

        if action == "screenshot":
            name = _artifact_name(step.value, "screenshot")
            out.append(f"    page.screenshot(path={_py_str(f'.forkable/artifacts/{name}.png')})")
            return out, scope, note

        if action == "visual_check":
            # Build the path first: nested same-quote f-strings are a syntax
            # error before Python 3.12.
            name = _artifact_name(step.value, "page")
            out.append(f"    # visual baseline check '{_inline(name, 60)}' runs through the agent runtime")
            out.append(f"    page.screenshot(path={_py_str(f'.forkable/artifacts/{name}.png')})")
            return out, scope, note

        if action == "api_request":
            method, _, path = step.target.partition(" ")
            url = _py_str(path) if path.startswith("http") else _url_expr(path)
            out.append(f"    response = page.request.{method.lower()}({url}{_data_kwarg(step.value)})")
            return out, scope, note

        if action == "expect_status":
            out.append(f"    assert response.status == {int(step.value or 200)}, response.text()")
            return out, scope, note

        if action == "expect_json":
            path = "".join(
                f"[{int(part)}]" if part.isdigit() else f"[{_py_str(part)}]"
                for part in step.target.split(".") if part
            )
            out.append(f"    assert {_py_str(step.value)}.lower() in str(response.json(){path}).lower()")
            return out, scope, note

        if action == "expect_url":
            out.append(f"    expect(page).to_have_url(re.compile({_py_str(re.escape(step.value))}))")
            return out, scope, note

        if action == "expect_title":
            out.append(f"    expect(page).to_have_title(re.compile({_py_str(re.escape(step.value))}, re.I))")
            return out, scope, note

        # The executor compares text case-insensitively, so the generated test must
        # too - otherwise a plan can pass in a run and fail in the file made from it.
        if action == "expect_text":
            out.append(
                f'    expect(page.locator("body")).to_contain_text({_py_str(step.value)}, ignore_case=True)'
            )
            return out, scope, note

        if action == "expect_no_text":
            out.append(
                f'    expect(page.locator("body")).not_to_contain_text({_py_str(step.value)}, ignore_case=True)'
            )
            return out, scope, note

        if action in ELEMENT_ACTIONS:
            locator, note = self._locator_code(step.target, action, scope)
            if action == "click":
                out.append(f"    {locator}.click()")
            elif action == "fill":
                out.append(f"    {locator}.fill({_py_str(step.value)})")
            elif action == "press":
                out.append(f"    {locator}.press({_py_str(step.value or 'Enter')})")
            elif action == "check":
                out.append(f"    {locator}.check()")
            elif action == "uncheck":
                out.append(f"    {locator}.uncheck()")
            elif action == "hover":
                out.append(f"    {locator}.hover()")
            elif action == "select":
                out.append(f"    {locator}.select_option({_py_str(step.value)})")
            elif action == "expect_visible":
                out.append(f"    expect({locator}).to_be_visible()")
            elif action == "expect_value":
                out.append(f"    expect({locator}).to_have_value(re.compile({_py_str(re.escape(step.value))}, re.I))")
            return out, scope, note

        out.append(f"    # unsupported action: {_inline(action, 40)}")
        return out, scope, note

    # ------------------------------------------------------------------
    def write(self, generated: GeneratedTest, out_dir: str | Path) -> Path:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"test_{generated.name}.py"
        path.write_text(generated.code, encoding="utf-8")
        generated.path = path
        return path
