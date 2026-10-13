"""Hostile or awkward plan text must never become executable generated code."""

from __future__ import annotations

import ast

import pytest

from forkable_ai_agent.agent.generator import TestGenerator, UnsafeGeneratedCode, validate_code
from forkable_ai_agent.schema import Step, TestPlan

NASTY = [
    'x"); __import__("os").system("touch /tmp/pwn"); ("',
    "line one\nimport os\nos.system('id')",
    "back\\slash and 'single' and \"double\" and \"\"\" triple",
    "unicode \u2028 separator \u0085 and \x00 nul",
    "{braces} {{double}} %s %d",
]


def _generate(settings, steps, **plan_kwargs):
    plan = TestPlan(name="t", steps=steps, base_url="http://127.0.0.1:1", **plan_kwargs)
    return TestGenerator(settings).generate(plan)


@pytest.mark.parametrize("payload", NASTY)
def test_hostile_values_stay_inside_string_literals(settings, payload):
    steps = [
        Step("goto", "/login"),
        Step("fill", "username", payload),
        Step("click", payload),
        Step("expect_text", "", payload),
        Step("expect_url", "", payload),
    ]
    generated = _generate(settings, steps)
    tree = ast.parse(generated.code)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 3  # re, pytest, playwright - and nothing smuggled in
    assert "os.system" not in "".join(
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    )


@pytest.mark.parametrize("field", ["description", "source"])
def test_plan_metadata_cannot_break_out_of_the_docstring(settings, field):
    kwargs = {"description": "ok", "source": "rules"}
    kwargs[field] = 'bad\n"""\nimport os\nos.system("id")\n"""'
    plan = TestPlan(name="t", steps=[Step("goto", "/")], base_url="http://127.0.0.1:1", **kwargs)
    generated = TestGenerator(settings).generate(plan)
    validate_code(generated.code)
    tree = ast.parse(generated.code)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "system"]


def test_plan_name_cannot_inject_code_or_climb_directories(settings, tmp_path):
    plan = TestPlan(name="x(page):\n    import os\n#../../evil", steps=[Step("goto", "/")],
                    base_url="http://127.0.0.1:1")
    generator = TestGenerator(settings)
    generated = generator.generate(plan)
    validate_code(generated.code)
    out = generator.write(generated, tmp_path / "out")
    assert out.parent == tmp_path / "out"
    assert ".." not in out.name and "/" not in out.name


def test_step_notes_cannot_inject_a_live_statement(settings):
    step = Step("goto", "/login", note="harmless\n    import os; os.system('id')")
    generated = _generate(settings, [step])
    tree = ast.parse(generated.code)
    assert len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]) == 3


def test_base_url_is_a_literal(settings):
    plan = TestPlan(name="t", steps=[Step("goto", "/")],
                    base_url='http://x"\nimport os\n#')
    generated = TestGenerator(settings).generate(plan)
    tree = ast.parse(generated.code)
    assert len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]) == 3


@pytest.mark.parametrize("code", [
    "import os\n",
    "import subprocess\n",
    "from os import system\n",
    "open('/etc/passwd')\n",
    "__import__('os')\n",
    "x = ().__class__.__bases__\n",
    "exec('1')\n",
    "eval('1')\n",
    "lambda: 1\n",
    "while True:\n    pass\n",
    "def f():\n    global x\n",
    "class A: pass\n",
    "with open('x') as f:\n    pass\n",
])
def test_validator_refuses_anything_the_generator_does_not_emit(code):
    with pytest.raises(UnsafeGeneratedCode):
        validate_code(code)


def test_validator_accepts_its_own_output(settings):
    generated = _generate(settings, [Step("goto", "/login"), Step("fill", "username", "demo"),
                                     Step("click", "log in"), Step("expect_text", "", "Welcome")])
    validate_code(generated.code)


def test_wait_step_is_clamped_like_the_executor(settings):
    generated = _generate(settings, [Step("wait", "", "9999999")])
    assert "wait_for_timeout" in generated.code
    assert "9999999000" not in generated.code
