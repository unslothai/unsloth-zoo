# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Writing safetensors back over the file they were loaded from.

mx.load() returns lazily file-backed arrays. Saving them to their own source path
truncates the file before they are read, and from mlx 0.32.1 that surfaces as
"RuntimeError: [read] Unable to read from file". It broke the Apple Silicon lane on
2026-08-18, the day 0.32.1 shipped, with no change on our side.

Re-saving into the directory you loaded from is the ordinary lifecycle for both
adapters (switch/merge) and optimizer state (resume, train, checkpoint), so each
writer materializes its arrays and writes through a temp file. These tests pin that
per writer, and pin the underlying mlx behaviour so a future version that makes
save-over-source safe again is visible rather than silently relied upon.
"""

import pytest

mx = pytest.importorskip("mlx.core")

# The Linux lane answers `import mlx.core` with the torch-backed simulation in
# tests/mlx_simulation/, which saves eagerly through safetensors.torch and so has no
# lazy file-backed arrays to protect. These tests are about real mlx's laziness, and
# asserting them against the simulation would only pin the simulation.
if mx.__name__ != "mlx.core":
    pytest.skip(
        "needs real mlx, not the torch-backed simulation", allow_module_level=True
    )


def _adapter_tensors():
    return {
        "layers.0.self_attn.q_proj.lora_a": mx.zeros((1, 4, 8)),
        "layers.0.self_attn.q_proj.lora_b": mx.ones((1, 8, 4)),
    }


def test_mlx_still_refuses_a_lazy_save_over_the_source(tmp_path):
    """The upstream behaviour the writers below defend against.

    If this starts failing, mlx has made save-over-source safe again. That is good
    news, not a reason to drop the temp-file writes: they are also what keeps a
    crash mid-write from leaving a truncated checkpoint.
    """
    path = str(tmp_path / "w.safetensors")
    mx.save_safetensors(path, {"x": mx.zeros((2, 3))})

    loaded = mx.load(path)
    lazy = {key: value.astype(mx.bfloat16) for key, value in loaded.items()}
    with pytest.raises(RuntimeError, match="Unable to read from file"):
        mx.save_safetensors(path, lazy)


def test_saving_adapters_back_over_their_source_survives(tmp_path, monkeypatch):
    """_save_adapter_artifacts must tolerate tensors backed by its own target."""
    from unsloth_zoo.mlx import utils as mlx_utils
    from unsloth_zoo.mlx.utils import _save_adapter_artifacts

    # Config enrichment reads the live model through mlx_lm; this test is about the
    # weights write, so keep it runnable on any lane that has mlx at all.
    monkeypatch.setattr(
        mlx_utils, "_enrich_mlx_adapter_config", lambda model, config: config
    )

    model = None
    _save_adapter_artifacts(model, tmp_path, _adapter_tensors())
    target = tmp_path / "adapters.safetensors"
    assert target.exists()

    # Lazy, still backed by target: exactly what a reload-then-resave produces.
    reloaded = mx.load(str(target))
    _save_adapter_artifacts(model, tmp_path, reloaded)

    round_tripped = mx.load(str(target))
    mx.eval(*round_tripped.values())
    assert set(round_tripped) == set(_adapter_tensors())
    assert not list(tmp_path.glob("*.tmp.safetensors")), "a temp file was left beside the adapter"


def test_saving_optimizer_state_back_over_its_source_survives(tmp_path):
    """save_optimizer_state after load_optimizer_state, into the same directory."""
    from unsloth_zoo.mlx.utils import load_optimizer_state, save_optimizer_state

    class _Optimizer:
        def __init__(self, state):
            self.state = state

    original = {"step": mx.array(3), "m": {"w": mx.zeros((2, 2))}}
    optimizer = _Optimizer(original)
    save_optimizer_state(optimizer, str(tmp_path))

    # Resume: state is now mx.load()'s file-backed arrays.
    load_optimizer_state(optimizer, str(tmp_path))
    save_optimizer_state(optimizer, str(tmp_path))

    reread = _Optimizer({})
    load_optimizer_state(reread, str(tmp_path))
    assert int(reread.state["step"].item()) == 3
    assert not list(tmp_path.glob("*.tmp.safetensors")), "a temp file was left beside the checkpoint"
