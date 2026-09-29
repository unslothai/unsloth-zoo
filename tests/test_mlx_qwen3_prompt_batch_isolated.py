# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression: a mixed Qwen3 prompt batch keeps one visual row per input row.

Runs in a subprocess on purpose. Sibling MLX test modules install the
tests/mlx_simulation torch shim over `mlx.core` AND `mlx_vlm` for the rest of
the session, so an in-process version of this check can only skip once any of
them has run -- which is exactly what happens in the full suite. A subprocess
gets the real mlx-vlm merger every time, whatever ran before.
"""

import subprocess
import sys
import textwrap


_SCRIPT = textwrap.dedent(
    """
    import importlib, sys

    try:
        import mlx.core as mx
        import unsloth_zoo.mlx.compile as mc
        ar = importlib.import_module("mlx_vlm.generate.ar")
    except ImportError as error:
        # Only an absent optional dependency is a skip. An import-time
        # regression in the modules under test must fail the subprocess, or
        # this regression test reports success for the very break it exists
        # to catch.
        if (getattr(error, "name", "") or "").split(".")[0] not in (
            "mlx", "mlx_lm", "mlx_vlm",
        ):
            raise
        print("SKIP", type(error).__name__, error)
        raise SystemExit(0)

    if "mlx_simulation" in str(getattr(mx, "__file__", "")):
        print("SKIP simulation stub leaked into the subprocess")
        raise SystemExit(0)
    merge = getattr(ar, "_merge_prefill_prompt_kwargs", None)
    if merge is None:
        print("SKIP mlx-vlm without _merge_prefill_prompt_kwargs")
        raise SystemExit(0)
    merge = mc._qwen3_prompt_merge_adapter(merge)

    def visual_row(length, positions, base):
        flags = [False] * length
        for position in positions:
            flags[position] = True
        masks = mx.array([flags], dtype=mx.bool_)
        embeds = [mx.array([[float(base + i)] for i in range(len(positions))])]
        state, packed = mc._pack_qwen3_visual_state(masks, embeds)
        return {
            "inputs_embeds": mx.zeros((1, length, 4)),
            "visual_pos_masks": masks,
            mc._QWEN3_VISUAL_STATE_KEY: state,
            mc._QWEN3_VISUAL_POSITIONS_KEY: packed,
        }

    # Three rows of UNEQUAL length so mlx-vlm really left-pads, and a text-only
    # row first so a dropped row shifts every visual row onto the wrong entry.
    rows = [
        {"inputs_embeds": mx.zeros((1, 4, 4))},
        visual_row(7, [2, 3], 10),
        visual_row(6, [1, 2, 4], 20),
    ]
    input_ids = [[0] * 4, [0] * 7, [0] * 6]

    embeds, merged = merge(rows, input_ids)

    for key in ("visual_pos_masks",
                mc._QWEN3_VISUAL_STATE_KEY,
                mc._QWEN3_VISUAL_POSITIONS_KEY,
                mc._QWEN3_VISUAL_WIDTH_KEY):
        assert key in merged, f"{key} dropped out of the merged batch"
        assert merged[key].shape[0] == 3, (
            f"{key} came back with {merged[key].shape[0]} of 3 rows"
        )
    assert embeds.shape[:2] == merged["visual_pos_masks"].shape, (
        f"mask {merged['visual_pos_masks'].shape} does not match embeds {embeds.shape}"
    )

    # Left padding shifts each short row's mask and compact positions into the
    # same full-row coordinate system consumed by the prompt cache.
    mask_rows = merged["visual_pos_masks"].tolist()
    true_at = [[i for i, flag in enumerate(row) if flag] for row in mask_rows]
    assert true_at == [[], [2, 3], [2, 3, 5]], true_at
    assert merged[mc._QWEN3_VISUAL_POSITIONS_KEY].tolist() == [
        [-1, -1, -1], [2, 3, -1], [2, 3, 5]
    ]
    assert merged[mc._QWEN3_VISUAL_WIDTH_KEY].tolist() == [7, 7, 7]

    # Rebuild the window and scatter, exactly as the language call does.
    window_mask, layers = mc._qwen3_visual_window(
        merged["visual_pos_masks"],
        merged[mc._QWEN3_VISUAL_STATE_KEY],
        merged[mc._QWEN3_VISUAL_POSITIONS_KEY],
        mask_offsets=(0, 0, 0),
        position_widths=merged[mc._QWEN3_VISUAL_WIDTH_KEY],
        window=int(embeds.shape[1]),
    )
    assert int(window_mask.sum().item()) == layers[0].shape[0], (
        f"{int(window_mask.sum().item())} masked tokens vs {layers[0].shape[0]} features"
    )
    scattered = mc._add_visual_embeds(
        mx.zeros((window_mask.shape[0], window_mask.shape[1], 1)),
        window_mask,
        layers[0],
    ).reshape(window_mask.shape).tolist()
    assert scattered == [
        [0.0] * 7,
        [0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 20.0, 21.0, 0.0, 22.0, 0.0],
    ], scattered

    print("QWEN3_MIXED_PROMPT_BATCH_OK")
    """
)


def test_qwen3_mixed_prompt_batch_survives_a_three_row_unequal_batch():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    if "SKIP" in result.stdout:
        import pytest

        pytest.skip(result.stdout.strip())
    assert "QWEN3_MIXED_PROMPT_BATCH_OK" in result.stdout, (
        f"stdout={result.stdout}\n---\nstderr={result.stderr}"
    )


def test_import_regressions_fail_instead_of_reporting_a_skip(tmp_path):
    """Only an absent optional dependency may skip the isolated run.

    A blanket ``except Exception`` around the imports turns any import-time
    break in the modules under test into a successful skip, so the regression
    test above would report success for exactly the failure it exists to catch.
    """
    import os

    shim = tmp_path / "shim"
    (shim / "mlx").mkdir(parents=True)
    (shim / "mlx" / "__init__.py").write_text("")
    (shim / "mlx" / "core.py").write_text(
        "raise RuntimeError('simulated import-time regression')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0, f"stdout={result.stdout}"
    assert "SKIP" not in result.stdout, result.stdout
    assert "simulated import-time regression" in result.stderr, result.stderr
