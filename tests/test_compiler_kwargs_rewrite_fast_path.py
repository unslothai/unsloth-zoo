"""_rewrite_kwargs_param skips the tokenize round trip when no `def <name>` exists and the round trip is the identity."""

import io
import tokenize

import pytest

import unsloth_zoo.compiler as compiler


def _reference(source, func_name):
    # The full tokenize -> untokenize path, as before the fast path.
    saved = compiler._untokenize_round_trips
    compiler._untokenize_round_trips = lambda source: False
    try:
        return compiler._rewrite_kwargs_param(source, func_name)
    finally:
        compiler._untokenize_round_trips = saved


_CLASS_SOURCE = (
    "class UnslothSFTTrainer(SFTTrainer):\n"
    "    def __init__(self, model, args = None, **kwargs):\n"
    "        x = f\"{{literal}} {model}\"\n"
    "        y = 1 + \\\n"
    "            2\n"
    "        super().__init__(model, args, **kwargs)\n"
)


def test_class_source_does_not_tokenize(monkeypatch):
    calls = []
    real = tokenize.generate_tokens

    def counting(readline):
        calls.append(1)
        return real(readline)

    monkeypatch.setattr(compiler.tokenize, "generate_tokens", counting)
    out = compiler._rewrite_kwargs_param(_CLASS_SOURCE, "UnslothSFTTrainer")
    assert out == (_CLASS_SOURCE, None)
    if getattr(compiler, "_untokenize_round_trips", lambda source: True)(_CLASS_SOURCE):
        assert calls == []


@pytest.mark.parametrize(
    "source, name",
    [
        ("def f(a, kwargs, **kwargs):\n    return a\n", "f"),
        ("def  f (a, kwargs, **kwargs):\n    return a\n", "f"),
        ("def \\\nf(a, kwargs, **kwargs):\n    return a\n", "f"),
        ("async def f(kwargs, **kwargs):\n    pass\n", "f"),
        ("class A:\n    def f(self, kwargs, **kwargs):\n        pass\n", "f"),
        ("def g(kwargs, **kwargs):\n    pass\n", "f"),
        ("def f1(kwargs, **kwargs):\n    pass\n", "f"),
        ("s = 'def f(kwargs, **kwargs)'\n", "f"),
        ("def f(a,\tkwargs, **kwargs):\n    pass\n", "f"),
        ("def f(a, kwargs, **kwargs):\r\n    pass\r\n", "f"),
        ("x = 1 + \\\n    2\ndef h(**kwargs): pass\n", "f"),
        ("x = (1 +\n\t2)\n", "f"),
        ("y = f'{{a}} {b}'\n", "f"),
        ("z = 1 +\\\n\\\n 2\n", "f"),
        ("def f(\n    a,\n    kwargs,\n    **kwargs,\n):\n    '''doc'''\n    return a\n", "f"),
        ("def f(a, kwargs, **kwargs", "f"),
        ("\u00e9 = 1\ndef f\u0301(kwargs, **kwargs): pass\n", "f"),
        # No final newline: untokenize below 3.12 drops the trailing spaces, so no fast path.
        ("x = 1\n   ", "f"),
        ('x = f"""{{\n#}}{y}"""  ', "f"),
    ],
)
def test_matches_full_round_trip(source, name):
    assert compiler._rewrite_kwargs_param(source, name) == _reference(source, name)


def test_def_scan_is_linear_on_comment_runs():
    # A comment branch that can stop at any "#" made the def scan backtrack exponentially.
    import time

    source = (
        "def forward(self, hidden_states, **kwargs):\n"
        "    # see the reference kernel at commit 9a8b7c6d5e4f3a2b1c0def\n"
        "    " + "#" * 5000 + "\n"
        "    return self.mlp(hidden_states, **kwargs)\n"
    )
    start = time.perf_counter()
    result = compiler._rewrite_kwargs_param(source, "LlamaMLP")
    assert time.perf_counter() - start < 1.0
    assert result == _reference(source, "LlamaMLP")
