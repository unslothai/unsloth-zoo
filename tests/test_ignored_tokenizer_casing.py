# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test: fix_untrained_tokens' IGNORED_TOKENIZER_NAMES guard must match
case-insensitively.

The production list is exported (via UNSLOTH_IGNORED_TOKENIZER_NAMES) fully
lowercased, but callers may also provide canonical mixed-case IDs. Both the model
name and configured entries must therefore be normalized before comparison.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

_ZOO_ROOT = Path(__file__).resolve().parents[1]

_TEST_SCRIPT = textwrap.dedent(
    """
    import sys
    from types import SimpleNamespace

    class _BlockUnsloth:
        def find_spec(self, fullname, path = None, target = None):
            if fullname == "unsloth" or fullname.startswith("unsloth."):
                raise ModuleNotFoundError("unsloth blocked for zoo-only test")
            return None

    sys.meta_path.insert(0, _BlockUnsloth())

    from unsloth_zoo.tokenizer_utils import fix_untrained_tokens

    class _GuardPassed(Exception):
        pass

    class _SentinelWeight:
        @property
        def shape(self):
            raise _GuardPassed()

    def _model(name):
        emb = SimpleNamespace(weight = _SentinelWeight())
        return SimpleNamespace(
            get_input_embeddings = lambda: emb,
            get_output_embeddings = lambda: emb,
            config = SimpleNamespace(_name_or_path = name),
        )

    tokenizer = SimpleNamespace(chat_template = None)

    def _assert_skipped(model_name, ignored_name):
        result = fix_untrained_tokens(
            _model(model_name), tokenizer, None,
            IGNORED_TOKENIZER_NAMES = [ignored_name],
        )
        assert result is None

    model_name = "unsloth/Qwen2.5-Coder-7B-Instruct"
    _assert_skipped(model_name, "unsloth/qwen2.5-coder-7b-instruct")
    _assert_skipped(model_name, model_name)

    try:
        fix_untrained_tokens(
            _model("unsloth/Llama-3.1-8B-Instruct"), tokenizer, None,
            IGNORED_TOKENIZER_NAMES = ["unsloth/qwen2.5-coder-7b-instruct"],
        )
    except _GuardPassed:
        pass
    else:
        raise AssertionError("non-ignored model was skipped")

    print("IGNORED_TOKENIZER_CASING_OK")
    """
)


def test_ignored_tokenizer_casing_zoo_only():
    env = dict(os.environ)
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_ZOO_ROOT), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", _TEST_SCRIPT],
        cwd = _ZOO_ROOT,
        env = env,
        capture_output = True,
        text = True,
    )
    assert "IGNORED_TOKENIZER_CASING_OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
