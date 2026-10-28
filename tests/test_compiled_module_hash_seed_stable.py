# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Generated compiled modules are byte-identical under any PYTHONHASHSEED (CPU only).

The modeling import list used to follow set order, so gpt_oss, qwen3_next and
granitemoehybrid files were rewritten by every new process.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

# With set order, at least one pair of these seeds disagrees for each model type on CPython 3.13.
SEEDS = ("1", "2", "3", "6")
MODEL_TYPES = ("gpt_oss", "qwen3_next", "granitemoehybrid")

_CHILD = r'''
import os, sys, io, contextlib
from unsloth_zoo.compiler import unsloth_compile_transformers
for model_type in sys.argv[1].split(","):
    with contextlib.redirect_stdout(io.StringIO()):
        unsloth_compile_transformers(
            model_type = model_type, fast_lora_forwards = False, fullgraph = True,
            import_from_cache = False, disable = False, supports_sdpa = [None],
        )
'''


def _generate(tmp_path, seed, model_types):
    folder = tmp_path / f"seed_{seed}"
    folder.mkdir()
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(folder / "unsloth_compiled_cache")
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, ",".join(model_types)],
        cwd = str(folder), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    cache = folder / "unsloth_compiled_cache"
    return {
        name: (cache / name).read_bytes()
        for name in sorted(os.listdir(cache))
        if name.endswith(".py")
    }


def test_generated_modules_do_not_depend_on_hash_seed(tmp_path):
    model_types = []
    for model_type in MODEL_TYPES:
        try:
            __import__(f"transformers.models.{model_type}.modeling_{model_type}")
        except Exception:
            continue
        model_types.append(model_type)
    if not model_types:
        pytest.skip(reason = f"installed transformers has none of {MODEL_TYPES}")

    with ThreadPoolExecutor(max_workers = len(SEEDS)) as pool:
        outputs = list(pool.map(lambda seed: _generate(tmp_path, seed, model_types), SEEDS))

    reference = outputs[0]
    for model_type in model_types:
        assert f"unsloth_compiled_module_{model_type}.py" in reference, sorted(reference)
    for seed, output in zip(SEEDS[1:], outputs[1:]):
        assert sorted(output) == sorted(reference), seed
        differing = [name for name in reference if output[name] != reference[name]]
        assert differing == [], f"PYTHONHASHSEED={seed} vs {SEEDS[0]}: {differing}"
