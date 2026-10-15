# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bf16 and T4 (forced-float32 cast / fp16 + fp16 autocast) forward+backward on tiny
random shrinks of Gemma 4 E2B, Gemma 4 26B-A4B MoE, Gemma 4 12B Unified, Qwen3.5 (FLA),
Qwen3.6 MoE and Muse-Glimmer. CPU only, no downloads; one subprocess for all of them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ARCHS = ("gemma4_e2b", "gemma4_26b_a4b_moe", "gemma4_unified_12b", "qwen3_5_fla", "qwen3_6_moe", "muse_glimmer")
WORKER = Path(__file__).with_name("_fp16_path_tiny_worker.py")


@pytest.fixture(scope = "module")
def results():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "")
    # A flag leaked by another test would turn these forwards into hidden states / raw logits.
    for flag in ("UNSLOTH_RETURN_HIDDEN_STATES", "UNSLOTH_RETURN_LOGITS"):
        env.pop(flag, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(WORKER.parent.parent), env.get("PYTHONPATH")]))
    proc = subprocess.run([sys.executable, str(WORKER), *ARCHS], capture_output = True, text = True,
                          env = env, timeout = 600)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[len("RESULT "):])
            out[(r["mode"], r["arch"])] = r
    if not out:
        pytest.fail(f"worker produced no results (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    return out


@pytest.mark.parametrize("mode", ["bf16", "t4"])
@pytest.mark.parametrize("arch", ARCHS)
def test_tiny_model_trains(results, mode, arch):
    r = results.get((mode, arch))
    assert r is not None, f"worker skipped {mode}/{arch}"
    if (r.get("error") or "").startswith("UNAVAILABLE"):
        pytest.skip(r["error"])
    assert r["ok"], f"{mode}/{arch}: {r.get('error')}\n{r.get('trace', '')}\n{r}"
