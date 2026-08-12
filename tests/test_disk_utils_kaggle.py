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

"""Kaggle detection, GGUF sizing and the /tmp redirect.

Both bugs under test are bugs of *degree* -- code that ran and returned an
answer, just the wrong one, never an exception:

1. `IS_KAGGLE_ENVIRONMENT` was "any env var starts with KAGGLE_", which is how
   the Kaggle CLI authenticates on an ordinary machine, so anyone who had ever
   used it had save paths rewritten to /tmp on their own laptop.
2. The free-disk estimate sized a GGUF export at the model twice over. The
   real peak is merge PLUS intermediate GGUF PLUS quants, since nothing is
   deleted in between.

The module is loaded straight from its file rather than through
`import unsloth_zoo`, so these run without torch, without a GPU and without
the package's import-time device detection. Nothing touches a real /tmp or
/kaggle: KAGGLE_TMP and KAGGLE_WORKING are pointed at a scratch tree.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "disk_utils.py"

GB = 1024**3


def _load_disk_utils():
    """Fresh module object per test, so monkeypatched globals never leak."""
    spec = importlib.util.spec_from_file_location("_unsloth_disk_utils_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def disk_utils():
    return _load_disk_utils()


@pytest.fixture
def kaggle_tree(tmp_path, disk_utils, monkeypatch):
    """A fake Kaggle filesystem: /kaggle/working and /tmp, both under tmp_path."""
    working = tmp_path / "kaggle" / "working"
    tmp = tmp_path / "tmp"
    working.mkdir(parents = True)
    tmp.mkdir(parents = True)
    monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(working))
    monkeypatch.setattr(disk_utils, "KAGGLE_TMP", str(tmp))
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("UNSLOTH_IS_KAGGLE", raising = False)
    monkeypatch.delenv("UNSLOTH_KAGGLE_USE_TMP", raising = False)
    monkeypatch.chdir(working)
    return working, tmp


def _fake_free(disk_utils, monkeypatch, sizes):
    """Report `sizes[path] = free bytes`, falling through to the real answer.

    Longest matching prefix wins, so a subdirectory inherits its parent's free
    space the way a real filesystem would report it.
    """
    real = disk_utils.free_bytes

    def fake(path):
        resolved = os.path.abspath(str(path))
        best = None
        for prefix, free in sizes.items():
            prefix = os.path.abspath(str(prefix))
            if (resolved == prefix or resolved.startswith(prefix + os.sep)) and (
                best is None or len(prefix) > len(best[0])
            ):
                best = (prefix, free)
        return best[1] if best is not None else real(path)

    monkeypatch.setattr(disk_utils, "free_bytes", fake)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


class TestKaggleDetection:
    def test_real_kernel_detected(self, disk_utils, kaggle_tree):
        assert disk_utils.is_kaggle_environment() is True

    def test_kaggle_cli_credentials_are_not_a_kernel(self, disk_utils, kaggle_tree, monkeypatch):
        """The exact false positive the old check had: a laptop with the Kaggle
        CLI configured exports KAGGLE_USERNAME and KAGGLE_KEY and has no
        KAGGLE_KERNEL_RUN_TYPE, yet `"\\nKAGGLE_" in keynames` said Kaggle."""
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
        monkeypatch.setenv("KAGGLE_USERNAME", "datadinosaur")
        monkeypatch.setenv("KAGGLE_KEY", "0" * 32)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/home/me/.kaggle")
        assert disk_utils.is_kaggle_environment() is False

    def test_kernel_run_type_without_kaggle_dir_is_not_a_kernel(
        self, disk_utils, tmp_path, monkeypatch
    ):
        """Half the signal is not the signal. Guards against a stray export."""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(tmp_path / "absent"))
        monkeypatch.delenv("UNSLOTH_IS_KAGGLE", raising = False)
        assert disk_utils.is_kaggle_environment() is False

    def test_kaggle_dir_without_run_type_is_not_a_kernel(self, disk_utils, tmp_path, monkeypatch):
        """The other half. A /kaggle/working left behind by a docker mount."""
        working = tmp_path / "kaggle" / "working"
        working.mkdir(parents = True)
        monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(working))
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
        monkeypatch.delenv("UNSLOTH_IS_KAGGLE", raising = False)
        assert disk_utils.is_kaggle_environment() is False

    def test_empty_run_type_is_not_a_kernel(self, disk_utils, kaggle_tree, monkeypatch):
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "   ")
        assert disk_utils.is_kaggle_environment() is False

    @pytest.mark.parametrize(
        "environ",
        [
            {},
            {"COLAB_GPU": "1", "COLAB_RELEASE_TAG": "x"},
            {"HOME": "/Users/me"},
            {"USERPROFILE": r"C:\Users\me"},
            {"WSL_DISTRO_NAME": "Ubuntu"},
        ],
        ids = ["linux", "colab", "mac", "windows", "wsl"],
    )
    def test_no_false_positive_off_kaggle(self, disk_utils, tmp_path, monkeypatch, environ):
        for key in list(os.environ):
            if key.startswith(("KAGGLE_", "UNSLOTH_IS_KAGGLE")):
                monkeypatch.delenv(key, raising = False)
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(tmp_path / "kaggle" / "working"))
        assert disk_utils.is_kaggle_environment() is False

    def test_colab_is_not_kaggle_even_with_a_kaggle_dir(self, disk_utils, kaggle_tree, monkeypatch):
        """Colab notebooks that pip install kaggle and export credentials."""
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        assert disk_utils.is_kaggle_environment() is False
        assert disk_utils.is_colab_environment() is True

    def test_env_override(self, disk_utils, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
        monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(tmp_path / "absent"))
        monkeypatch.setenv("UNSLOTH_IS_KAGGLE", "1")
        assert disk_utils.is_kaggle_environment() is True
        monkeypatch.setenv("UNSLOTH_IS_KAGGLE", "0")
        assert disk_utils.is_kaggle_environment() is False


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


class _FakeQuantState:
    def __init__(self, shape):
        self.shape = shape


class _FakeParam:
    """A bitsandbytes 4-bit weight: numel() is half the logical count."""

    def __init__(self, logical, quantized):
        self._logical = logical
        self.quant_state = _FakeQuantState((logical,)) if quantized else None

    def numel(self):
        return self._logical // 2 if self.quant_state is not None else self._logical


class _MXFP4Param:
    """A packed MXFP4 block or scale tensor: uint8, and no quant_state at all."""

    dtype = "torch.uint8"

    def __init__(self, packed):
        self._packed = packed

    def numel(self):
        return self._packed


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class _NamedModel:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return iter(self._named.items())

    def parameters(self):
        return iter(self._named.values())


class TestLogicalSize:
    def test_quantized_parameter_uses_quant_state_shape(self, disk_utils):
        param = _FakeParam(2_097_152, quantized = True)
        assert param.numel() == 1_048_576
        assert disk_utils.logical_numel(param) == 2_097_152

    def test_plain_parameter_uses_numel(self, disk_utils):
        assert disk_utils.logical_numel(_FakeParam(1000, quantized = False)) == 1000

    def test_model_16bit_bytes_counts_logical_parameters(self, disk_utils):
        model = _FakeModel(
            [_FakeParam(1_000_000, quantized = True), _FakeParam(100_000, quantized = False)]
        )
        # The bug: sum(p.numel()) would be 600_000, giving 1.2MB not 2.2MB.
        assert disk_utils.model_16bit_bytes(model) == 2 * 1_100_000

    def test_mxfp4_blocks_are_worth_twice_their_bytes(self, disk_utils):
        """gpt-oss kept packed has no quant_state, so numel() is the only signal
        and it is half the truth: `convert_moe_packed_tensors` turns (..., G, B)
        into (..., G, B * 2) and consumes the scales."""
        blocks = _MXFP4Param(90_000_000)
        assert (
            disk_utils.logical_numel(blocks, "model.layers.0.mlp.experts.gate_up_proj_blocks")
            == 180_000_000
        )
        assert (
            disk_utils.logical_numel(_MXFP4Param(5_625_000), "…experts.gate_up_proj_scales") == 0
        )
        # Nameless (the plain `parameters()` fallback) must not double anything.
        assert disk_utils.logical_numel(blocks) == 90_000_000

    def test_a_packed_mxfp4_model_is_not_under_counted_by_half(self, disk_utils):
        model = _NamedModel(
            {
                "model.layers.0.mlp.experts.gate_up_proj_blocks": _MXFP4Param(90_000_000),
                "model.layers.0.mlp.experts.gate_up_proj_scales": _MXFP4Param(5_625_000),
                "model.layers.0.self_attn.q_proj.weight": _FakeParam(4_000_000, quantized = False),
            }
        )
        # Believing numel() gives 99.6M and sizes the merge at half the disk.
        assert disk_utils.model_logical_numel(model) == 184_000_000

    def test_a_model_without_named_parameters_still_measures(self, disk_utils):
        model = _FakeModel([_FakeParam(1000, quantized = False)])
        assert disk_utils.model_logical_numel(model) == 1000

    def test_unmeasurable_model_is_zero_not_a_guess(self, disk_utils):
        class Broken:
            def parameters(self):
                raise RuntimeError("no")

        assert disk_utils.model_16bit_bytes(Broken()) == 0


class TestGGUFEstimate:
    def test_peak_is_merge_plus_intermediate_plus_quants(self, disk_utils):
        """The under-estimate, stated as arithmetic: a 7B to q4_k_m writes a
        14GB merge, a 14GB f16 GGUF and a ~4.3GB quant, and neither of the
        first two is deleted first. "Two copies" is 28GB; the floor is 32GB."""
        n = 7_000_000_000
        need = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q4_k_m"], first_conversion = "f16"
        )
        two_copies = 2 * n * 2
        assert need > two_copies
        expected = n * 2 + n * 2 + int(n * 4.9 / 8) + disk_utils.DISK_SLACK_BYTES
        assert need == expected

    def test_single_pass_export_skips_the_intermediate(self, disk_utils):
        """q8_0 alone converts straight through, so there is no f16 middle."""
        n = 1_000_000_000
        need = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q8_0"], first_conversion = "q8_0"
        )
        assert need == n * 2 + int(n * 8.5 / 8) + disk_utils.DISK_SLACK_BYTES

    def test_multiple_quants_each_cost_their_own_file(self, disk_utils):
        n = 1_000_000_000
        one = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q4_k_m"], first_conversion = "f16"
        )
        two = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q4_k_m", "q5_k_m"], first_conversion = "f16"
        )
        assert two - one == int(n * 5.7 / 8)

    def test_a_convert_only_export_still_counts_the_file_it_writes(self, disk_utils):
        """A GGUF has to be written before it can be deleted, so no export ever
        peaks below merge + first conversion."""
        n = 1_000_000_000
        need = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = [], first_conversion = "f16"
        )
        assert need == n * 2 + n * 2 + disk_utils.DISK_SLACK_BYTES

    def test_needs_merge_false_drops_the_merge(self, disk_utils):
        n = 1_000_000_000
        with_merge = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q8_0"], first_conversion = "q8_0"
        )
        without = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n,
            quantization_methods = ["q8_0"],
            first_conversion = "q8_0",
            needs_merge = False,
        )
        assert with_merge - without == n * 2

    def test_aliases_resolve(self, disk_utils):
        n = 1_000_000
        assert disk_utils.gguf_bits_per_weight("quantized") == disk_utils.gguf_bits_per_weight("q4_k_m")
        assert disk_utils.gguf_bits_per_weight("fast_quantized") == disk_utils.gguf_bits_per_weight("q8_0")
        assert disk_utils.gguf_bits_per_weight("Q4_K") == disk_utils.gguf_bits_per_weight("q4_k_m")
        assert disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["not_quantized"], first_conversion = "f16"
        ) == disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["f16"], first_conversion = "f16"
        )

    def test_unknown_quant_is_sized_high_not_low(self, disk_utils):
        """Guessing small is the failure this module exists to prevent."""
        assert disk_utils.gguf_bits_per_weight("q4_k_xxl_2027") == disk_utils.gguf_bits_per_weight("q8_0")

    def test_unmeasurable_returns_zero(self, disk_utils):
        assert disk_utils.estimate_gguf_export_bytes(n_parameters = 0) == 0
        assert disk_utils.estimate_gguf_export_bytes(model = None) == 0

    def test_base_cache_copy_is_a_fourth_copy(self, disk_utils):
        n = 1_000_000_000
        without = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q4_k_m"], first_conversion = "f16"
        )
        with_cache = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n,
            quantization_methods = ["q4_k_m"],
            first_conversion = "f16",
            base_cache_copy = True,
        )
        assert with_cache - without == n * 2

    def test_the_three_failing_rows_are_now_caught(self, disk_utils):
        """Gemma4 31B Vision: 174GB free, and it still did not fit.

        62GB went on pre-warming the Hugging Face cache with the base model,
        62GB on the merge, leaving 50GB for a GGUF needing 62GB - it died at
        48GB of a 65GB shard. Two copies (the old estimate) called that safe,
        and so did three; only counting the cached base too exceeds the free
        space, which is what lets the guard say no - or, better, drop the
        pre-warm and let the export through, which is what unsloth now does.
        """
        n = 31_000_000_000
        free = 174 * GB
        old_style = 2 * n * 2 + 2 * GB
        three = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n, quantization_methods = ["q8_0"], first_conversion = "f16"
        )
        four = disk_utils.estimate_gguf_export_bytes(
            n_parameters = n,
            quantization_methods = ["q8_0"],
            first_conversion = "f16",
            base_cache_copy = True,
        )
        assert old_style < free, "the old estimate let this through, which is the bug"
        assert three < free, "three copies still let it through"
        assert four > free, "the full peak has to be caught"
        # And the export does fit without the pre-warm, so refusing outright
        # would have been the wrong answer.
        assert three < free


# --------------------------------------------------------------------------
# The /tmp redirect
# --------------------------------------------------------------------------


class TestKaggleTmpRedirect:
    def test_relative_default_that_does_not_fit_moves_to_tmp(
        self, disk_utils, kaggle_tree, monkeypatch
    ):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 34 * GB)
        assert message is not None
        assert os.path.abspath(target).startswith(os.path.abspath(str(tmp)))
        assert target.endswith(os.path.join("unsloth_saves", "model"))
        assert os.path.isdir(target)

    def test_message_says_where_and_that_tmp_is_not_saved(
        self, disk_utils, kaggle_tree, monkeypatch
    ):
        """/tmp is scratch on Kaggle. A user who is told nothing loses the model."""
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 34 * GB)
        assert target in message
        assert "not saved as kernel output" in message.lower()

    def test_absolute_path_is_never_moved(self, disk_utils, kaggle_tree, monkeypatch):
        """A caller who named a directory gets that directory, full stop."""
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        explicit = str(working / "my_export")
        target, message = disk_utils.kaggle_tmp_redirect(explicit, need_bytes = 34 * GB)
        assert target == explicit
        assert message is None

    def test_path_outside_kaggle_working_is_never_moved(
        self, disk_utils, kaggle_tree, tmp_path, monkeypatch
    ):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        monkeypatch.chdir(tmp_path)
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 34 * GB)
        assert target == "model"
        assert message is None

    def test_it_fits_so_nothing_moves(self, disk_utils, kaggle_tree, monkeypatch):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 500 * GB, tmp: 1000 * GB})
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 34 * GB)
        assert target == "model"
        assert message is None

    def test_tmp_cannot_hold_it_either_so_nothing_moves(
        self, disk_utils, kaggle_tree, monkeypatch
    ):
        """Moving files somewhere that fails just the same helps nobody."""
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 20 * GB})
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 500 * GB)
        assert target == "model"
        assert message is None

    def test_off_kaggle_nothing_moves(self, disk_utils, tmp_path, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising = False)
        monkeypatch.delenv("UNSLOTH_IS_KAGGLE", raising = False)
        monkeypatch.setattr(disk_utils, "KAGGLE_WORKING", str(tmp_path / "absent"))
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 10**15)
        assert target == "model"
        assert message is None

    def test_kill_switch(self, disk_utils, kaggle_tree, monkeypatch):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        monkeypatch.setenv("UNSLOTH_KAGGLE_USE_TMP", "0")
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 34 * GB)
        assert target == "model"
        assert message is None

    def test_force_switch_moves_even_when_it_would_fit(
        self, disk_utils, kaggle_tree, monkeypatch
    ):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 500 * GB, tmp: 1000 * GB})
        monkeypatch.setenv("UNSLOTH_KAGGLE_USE_TMP", "1")
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 0)
        assert message is not None
        assert os.path.abspath(target).startswith(os.path.abspath(str(tmp)))

    def test_no_need_and_no_force_means_no_move(self, disk_utils, kaggle_tree, monkeypatch):
        """Without a size to compare against there is nothing to justify a move."""
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        target, message = disk_utils.kaggle_tmp_redirect("model", need_bytes = 0)
        assert target == "model"
        assert message is None

    def test_nested_relative_path_keeps_its_shape(self, disk_utils, kaggle_tree, monkeypatch):
        working, tmp = kaggle_tree
        _fake_free(disk_utils, monkeypatch, {working: 19 * GB, tmp: 1000 * GB})
        target, message = disk_utils.kaggle_tmp_redirect(
            os.path.join("outputs", "llama3"), need_bytes = 34 * GB
        )
        assert message is not None
        assert target.endswith(os.path.join("unsloth_saves", "outputs", "llama3"))


class TestFreeBytes:
    def test_nonexistent_path_walks_up_to_an_existing_parent(self, disk_utils, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        assert disk_utils.free_bytes(str(deep)) is not None

    def test_unmeasurable_is_none_not_zero(self, disk_utils, monkeypatch):
        """0 would read as "full", and every caller blocks when free < need."""
        import shutil

        def boom(path):
            raise OSError("nope")

        monkeypatch.setattr(shutil, "disk_usage", boom)
        assert disk_utils.free_bytes(str(Path.cwd())) is None
