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

"""Extended upstream-API pins for ``accelerate.*`` / ``safetensors.*``
/ ``bitsandbytes.*`` / ``triton.*`` / ``datasets.*`` /
``huggingface_hub.*`` / ``xformers.*`` references in
``unsloth_zoo/**/*.py`` not already covered by
``test_upstream_pinned_symbols_*.py`` or
``test_zoo_source_upstream_refs.py``.

DRIFT-DETECTED framing: each test cites its zoo callsite. Failure ->
``pytest.fail("DRIFT DETECTED: ...")``; never SKIP when zoo references
the symbol unconditionally.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Shared helpers (intentional copies of test_zoo_source_upstream_refs.py
# helpers so this file is grep-self-contained).
# ---------------------------------------------------------------------------


def _resolve(dotted: str) -> object:
    """``importlib.import_module`` + ``getattr`` chain. Any failure is
    surfaced as AssertionError tagged DRIFT DETECTED."""
    parts = dotted.split(".")
    obj: object = None
    consumed: list[str] = []
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            continue
        try:
            obj = importlib.import_module(mod_name)
            consumed = parts[:i]
            break
        except ImportError as exc:
            raise AssertionError(
                f"DRIFT DETECTED: `{mod_name}` exists but its imports "
                f"fail on this install ({type(exc).__name__}: {exc})."
            )
    if obj is None:
        raise AssertionError(
            f"DRIFT DETECTED: could not locate any module prefix of "
            f"`{dotted}`; zoo references this dotted path unconditionally."
        )
    for attr in parts[len(consumed):]:
        if not hasattr(obj, attr):
            walked = ".".join(consumed + [attr])
            raise AssertionError(
                f"DRIFT DETECTED: `{walked}` missing on installed upstream "
                f"(walked from `{dotted}`); zoo callsite cited in test "
                "docstring will ImportError/AttributeError at runtime."
            )
        obj = getattr(obj, attr)
        consumed.append(attr)
    return obj


def _resolve_all(dotted_paths: Iterable[str]) -> None:
    missing: list[str] = []
    for d in dotted_paths:
        try:
            _resolve(d)
        except AssertionError as e:
            missing.append(f"  - {d}\n      ({e})")
    assert not missing, "DRIFT DETECTED: missing upstream symbols:\n" + "\n".join(missing)


def _require_module(name: str):
    return pytest.importorskip(name)


# ===========================================================================
# accelerate
# ===========================================================================


def test_accelerate_init_empty_weights_signature_shape():
    """unsloth_zoo/empty_model.py:238, 322 -- ``with init_empty_weights():``
    as zero-arg context manager (default include_buffers=None). Pre-3.0
    flipped this to a required positional; pin the modern shape."""
    _require_module("accelerate")
    fn = _resolve("accelerate.init_empty_weights")
    sig = inspect.signature(fn)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if required:
        pytest.fail(
            "DRIFT DETECTED: accelerate.init_empty_weights now requires "
            f"positional args {required}; both zoo callsites use it as "
            "a zero-arg context manager (empty_model.py:238, 322) and "
            "will TypeError on the meta-model construction path."
        )


def test_accelerate_init_empty_weights_is_context_manager():
    """Return value must be a context manager. A regression that flips
    it to a plain function silently constructs a real-CPU state dict
    instead of meta tensors and OOMs on Llama-70B-class empty-model builds."""
    _require_module("accelerate")
    accelerate = importlib.import_module("accelerate")
    cm = accelerate.init_empty_weights()
    if not (hasattr(cm, "__enter__") and hasattr(cm, "__exit__")):
        pytest.fail(
            "DRIFT DETECTED: accelerate.init_empty_weights() no longer "
            "returns a context manager; empty_model.py:238/322 "
            "`with init_empty_weights():` will TypeError at runtime."
        )
    cm.__enter__()
    cm.__exit__(None, None, None)


def test_accelerate_utils_imports_module_surface():
    """Pin the module-path surface so accelerate.utils restructuring
    surfaces here, not at zoo import time."""
    _require_module("accelerate")
    _resolve_all([
        "accelerate.utils",
        "accelerate.utils.imports",
    ])


# ===========================================================================
# safetensors
# ===========================================================================


def test_safetensors_safe_open_top_level_exists():
    """unsloth_zoo/saving_utils.py:153 -- ``from safetensors import
    safe_open``. Streamed-read entry point for every LoRA save / merge
    / GGUF export."""
    _resolve("safetensors.safe_open")


def test_safetensors_safe_open_signature_shape():
    """saving_utils.py uses ``safe_open(path, framework='pt',
    device=...)``. Pin filename-first."""
    _require_module("safetensors")
    safe_open = _resolve("safetensors.safe_open")
    # ``safe_open`` is a Rust-backed class; either inspect.signature
    # works OR the class has an __init__ we can probe.
    try:
        sig = inspect.signature(safe_open)
    except (TypeError, ValueError):
        if not hasattr(safe_open, "__init__"):
            pytest.fail(
                "DRIFT DETECTED: safetensors.safe_open has no inspectable "
                "signature AND no __init__; saving_utils.py:153 cannot "
                "validate its 2-positional + device-kwarg call shape."
            )
        return
    params = list(sig.parameters)
    if not params or params[0] not in ("filename", "path"):
        pytest.fail(
            f"DRIFT DETECTED: safetensors.safe_open first parameter is "
            f"{params[:1]!r}; saving_utils.py:153 expects filename-first."
        )


def test_safetensors_torch_save_file_top_level():
    """saving_utils.py:154 -- ``from safetensors.torch import save_file``.
    Canonical write path for sharded LoRA / merged-model output."""
    _resolve("safetensors.torch.save_file")


def test_safetensors_torch_save_file_signature():
    """save_file is called with ``(tensors, filename, metadata=...)``."""
    _require_module("safetensors")
    save_file = _resolve("safetensors.torch.save_file")
    sig = inspect.signature(save_file)
    expected = {"tensors", "filename", "metadata"}
    missing = expected - set(sig.parameters)
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: safetensors.torch.save_file lost parameters "
            f"{sorted(missing)}; saving_utils.py:154 + sharded-write "
            "callsites will TypeError at runtime."
        )


def test_safetensors_torch_types_mapping_present():
    """saving_utils.py:512 -- ``SAFETENSORS_DTYPES = safetensors.torch._TYPES``.
    Fallback silently mis-types BF16/FP8 weights in sharded save; pin
    the upstream-provided mapping."""
    _require_module("safetensors")
    st_torch = importlib.import_module("safetensors.torch")
    if not hasattr(st_torch, "_TYPES"):
        pytest.fail(
            "DRIFT DETECTED: safetensors.torch._TYPES private mapping is "
            "gone; saving_utils.py:512 falls back to a hardcoded dtype "
            "table that silently mis-types BF16/FP8 weights in sharded "
            "save."
        )
    types_map = st_torch._TYPES
    string_keys = {str(k).lower() for k in types_map.keys()}
    for needed in ("bf16", "f16", "f32"):
        if not any(needed in k for k in string_keys):
            pytest.fail(
                f"DRIFT DETECTED: safetensors.torch._TYPES dropped {needed} "
                "coverage; sharded save dtype probe will miss tensors."
            )


def test_safetensors_torch_load_file_present():
    """saving_utils.py's shard-merge codepaths call ``safetensors.torch
    .load_file``. Pin the round-trip needed for delta-LoRA dequant verify."""
    _require_module("safetensors")
    _resolve("safetensors.torch.load_file")


# ===========================================================================
# bitsandbytes
# ===========================================================================


def test_bnb_top_level_import_and_version_attr():
    """device_type.py:260, saving_utils.py:65, every temporary_patches
    callsite reads ``bitsandbytes.__version__``. Drives HIP / pre-0.46
    gate cascades."""
    bnb = _require_module("bitsandbytes")
    if not hasattr(bnb, "__version__"):
        pytest.fail(
            "DRIFT DETECTED: bitsandbytes.__version__ missing; device_type.py:"
            "260 (HIP gate) and patching_utils.py:40 (0.46 gate) "
            "AttributeError at zoo import."
        )


def test_bnb_nn_linear4bit_top_level():
    """patching_utils.py:309 + saving_utils.py:88 isinstance-check.
    Both import paths must resolve because zoo reaches both."""
    _require_module("bitsandbytes")
    _resolve_all([
        "bitsandbytes.nn.Linear4bit",
        "bitsandbytes.nn.modules.Linear4bit",
    ])


def test_bnb_linear4bit_constructor_kwargs_preserved():
    """temporary_patches/bitsandbytes.py + vllm_utils.py:484 pass
    ``compute_dtype=...``. Regression silently disables
    UNSLOTH_bnb_4bit_compute_dtype overrides."""
    _require_module("bitsandbytes")
    Linear4bit = _resolve("bitsandbytes.nn.Linear4bit")
    sig = inspect.signature(Linear4bit.__init__)
    if "compute_dtype" not in sig.parameters:
        pytest.fail(
            "DRIFT DETECTED: bitsandbytes.nn.Linear4bit.__init__ lost "
            "`compute_dtype` kwarg; vllm_utils.py:484 + temporary_patches "
            "compute-dtype override break silently."
        )


def test_bnb_nn_modules_params4bit_present():
    """device_type.py:257 + temporary_patches/bitsandbytes.py:47 import
    Params4bit. HIP-gate blocksize source-inspection lives on the
    class' source."""
    _require_module("bitsandbytes")
    _resolve("bitsandbytes.nn.modules.Params4bit")


def test_bnb_fix_4bit_weight_quant_state_from_module_present():
    """temporary_patches/bitsandbytes.py:48, 73 -- repacks bnb-weight
    quant_state after transformers 5.x's ``weight.shape[-1] == 1``
    deferred-pack path. Rename takes the whole Linear4bit forward
    replacement down."""
    _require_module("bitsandbytes")
    _resolve(
        "bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module",
    )


def test_bnb_matmul_4bit_top_level_and_signature():
    """temporary_patches/bitsandbytes.py:106, moe_bnb.py:245,
    vllm_utils.py:190 -- ``bnb.matmul_4bit`` is the 4-bit GEMM kernel
    zoo replaces Linear4bit.forward with. Pin ``(A, B, quant_state, bias)``."""
    _require_module("bitsandbytes")
    matmul_4bit = _resolve("bitsandbytes.matmul_4bit")
    sig = inspect.signature(matmul_4bit)
    expected = {"A", "B", "quant_state", "bias"}
    missing = expected - set(sig.parameters)
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: bitsandbytes.matmul_4bit lost parameters "
            f"{sorted(missing)}; the patched Linear4bit.forward in "
            "temporary_patches/bitsandbytes.py:106 cannot bind its "
            "positional args."
        )


def test_bnb_functional_dequantize_4bit_present():
    """temporary_patches/moe_bnb.py:44 -- pre-dequantizes expert weights
    via dequantize_4bit. Rename takes out MoE-on-bnb training."""
    _require_module("bitsandbytes")
    _resolve("bitsandbytes.functional.dequantize_4bit")


def test_bnb_functional_quantstate_present_and_from_dict():
    """vllm_utils.py:495 -- monkeys ``QuantState.from_dict`` onto
    ``bitsandbytes.functional.QuantState``. The classmethod must exist
    pre-patch so the override is well-formed."""
    _require_module("bitsandbytes")
    QuantState = _resolve("bitsandbytes.functional.QuantState")
    if not hasattr(QuantState, "from_dict"):
        pytest.fail(
            "DRIFT DETECTED: bitsandbytes.functional.QuantState.from_dict "
            "removed; vllm_utils.py:495 monkey-rebind has nothing to "
            "shadow."
        )


def test_bnb_utils_pack_unpack_tensor_dict_present():
    """vllm_utils.py:421 -- ``pack_dict_to_tensor`` and
    ``unpack_tensor_to_dict`` are the matched pair the vLLM 4-bit
    serialization path uses."""
    _require_module("bitsandbytes")
    _resolve_all([
        "bitsandbytes.utils.pack_dict_to_tensor",
        "bitsandbytes.utils.unpack_tensor_to_dict",
    ])


def test_bnb_functional_module_exists():
    """vllm_utils.py:420 -- ``import bitsandbytes.functional``. 0.50-dev
    wheels rearranged to ``bnb._functional`` and re-exported under the
    old name; zoo depends on the re-export."""
    _require_module("bitsandbytes")
    _resolve("bitsandbytes.functional")


def test_bnb_nn_modules_module_path_present():
    """vllm_utils.py:481 -- ``import bitsandbytes.nn.modules`` for
    in-place class swap (Linear4bit -> custom subclass)."""
    _require_module("bitsandbytes")
    _resolve("bitsandbytes.nn.modules")


# ===========================================================================
# triton
# ===========================================================================


def test_triton_top_level_version_attr():
    """compiler.py:95 -- ``Version(triton.__version__) < Version("3.0.0")``.
    loss_utils.py:24 takes the same dunder."""
    triton_mod = _require_module("triton")
    if not hasattr(triton_mod, "__version__"):
        pytest.fail(
            "DRIFT DETECTED: triton.__version__ missing; compiler.py:95 "
            "version gate AttributeError at zoo import."
        )


def test_triton_runtime_autotuner_class_present():
    """compiler.py:3030 -- ``from triton.runtime.autotuner import
    Autotuner``. Introspects autotuner-decorated kernels for the
    compile-rewriter."""
    _require_module("triton")
    _resolve("triton.runtime.autotuner.Autotuner")


def test_triton_set_allocator_top_level_present():
    """temporary_patches/moe_utils.py:217 -- ``triton.set_allocator(
    persistent_alloc_fn)``. Required for persistent-allocator MoE
    expert-merge fast path."""
    triton_mod = _require_module("triton")
    if not hasattr(triton_mod, "set_allocator"):
        pytest.fail(
            "DRIFT DETECTED: triton.set_allocator removed/renamed; "
            "temporary_patches/moe_utils.py:217 AttributeError when MoE "
            "merge-allocator hook fires."
        )


def test_triton_set_allocator_signature_accepts_one_arg():
    """A regression to ``set_allocator(name, fn)`` form (proposed in
    triton 3.x main) breaks the single-arg callsite immediately."""
    triton_mod = _require_module("triton")
    set_alloc = triton_mod.set_allocator
    sig = inspect.signature(set_alloc)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(required) != 1:
        pytest.fail(
            f"DRIFT DETECTED: triton.set_allocator now requires "
            f"{len(required)} positional args (was 1); "
            "temporary_patches/moe_utils.py:217 single-arg call breaks."
        )


def test_triton_language_namespace_present():
    """compiler.py + compiled-cache codegen reference ``triton.language``
    for ``tl.constexpr``."""
    _require_module("triton")
    _resolve("triton.language")


def test_triton_jit_decorator_present():
    """``triton.jit`` is the decorator zoo's MoE / CCE / RoPE kernels
    depend on at codegen time."""
    triton_mod = _require_module("triton")
    if not callable(getattr(triton_mod, "jit", None)):
        pytest.fail(
            "DRIFT DETECTED: triton.jit removed/renamed; every Unsloth "
            "Triton kernel in unsloth_compiled_cache/* fails to compile."
        )


# ===========================================================================
# datasets
# ===========================================================================


def test_datasets_load_dataset_top_level():
    """tokenizer_utils.py + training_utils.py instantiate datasets via
    ``datasets.load_dataset``."""
    _require_module("datasets")
    _resolve("datasets.load_dataset")


def test_datasets_iterable_dataset_classmethod_for_isinstance():
    """tokenizer_utils.py:294 + training_utils.py:50 isinstance-check
    against ``datasets.IterableDataset``. A rename to
    ``datasets.iterable.IterableDataset`` (proposed in 4.x) silently
    drops the streaming path."""
    datasets = _require_module("datasets")
    ID = getattr(datasets, "IterableDataset", None)
    if ID is None:
        pytest.fail(
            "DRIFT DETECTED: datasets.IterableDataset missing on top "
            "level; tokenizer_utils.py:294, training_utils.py:50 "
            "isinstance returns False -> streaming-mode SFT path dropped."
        )
    if not isinstance(ID, type):
        pytest.fail(
            f"DRIFT DETECTED: datasets.IterableDataset is now "
            f"{type(ID).__name__}, not a class; isinstance() callsites "
            "raise TypeError."
        )


def test_datasets_dataset_dict_top_level():
    """dataset_utils.py walks DatasetDict-shaped train/eval pairs in
    multi-split SFT via duck-typed ``.column_names``."""
    _require_module("datasets")
    _resolve("datasets.DatasetDict")


def test_datasets_torchcodec_audio_decoder_present_or_absent_cleanly():
    """dataset_utils.py:873 -- ``from datasets.features._torchcodec
    import AudioDecoder``. Wrapped in try/except so absence is tolerated;
    if the module IS importable, AudioDecoder must be on it.

    ``torchcodec`` (separate package) is a legitimate optional
    transitive dep -- CI without audio support won't have it. Treat
    that environment as importorskip, NOT drift."""
    _require_module("datasets")
    spec = importlib.util.find_spec("datasets.features._torchcodec")
    if spec is None:
        return
    try:
        mod = importlib.import_module("datasets.features._torchcodec")
    except ModuleNotFoundError as exc:
        if "torchcodec" in str(exc):
            pytest.skip(
                f"`datasets.features._torchcodec` requires the optional "
                f"`torchcodec` package which isn't installed on this CI "
                f"runner ({exc}); zoo's call site is try/except wrapped."
            )
        pytest.fail(
            "DRIFT DETECTED: datasets.features._torchcodec exists but "
            f"fails to import ({exc!r}); dataset_utils.py:873 "
            "AudioDecoder patch silently no-ops on audio datasets."
        )
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: datasets.features._torchcodec exists but "
            f"fails to import ({exc!r}); dataset_utils.py:873 "
            "AudioDecoder patch silently no-ops on audio datasets."
        )
    if not hasattr(mod, "AudioDecoder"):
        pytest.fail(
            "DRIFT DETECTED: datasets.features._torchcodec lost "
            "AudioDecoder class; dataset_utils.py:873 patch site has no "
            "type to patch and audio datasets return raw decoders."
        )


# ===========================================================================
# huggingface_hub
# ===========================================================================


def test_hf_hub_top_level_save_path_symbols():
    """saving_utils.py:148-152 -- ``from huggingface_hub import (
    snapshot_download, hf_hub_download, HfFileSystem,)`` at MODULE TOP
    LEVEL (no try/except)."""
    _require_module("huggingface_hub")
    _resolve_all([
        "huggingface_hub.snapshot_download",
        "huggingface_hub.hf_hub_download",
        "huggingface_hub.HfFileSystem",
    ])


def test_hf_hub_hfapi_and_modelcard_top_level():
    """saving_utils.py:109 + mlx_utils.py:3152, 3195, 3248 -- HfApi and
    ModelCard referenced unguardedly."""
    _require_module("huggingface_hub")
    _resolve_all([
        "huggingface_hub.HfApi",
        "huggingface_hub.ModelCard",
    ])


def test_hf_hub_hfapi_method_surface():
    """saving_utils.py + mlx_utils.py call ``HfApi().create_repo /
    .upload_file / .upload_folder / .create_commit / .file_exists``."""
    _require_module("huggingface_hub")
    HfApi = _resolve("huggingface_hub.HfApi")
    expected = [
        "create_repo", "upload_file", "upload_folder",
        "create_commit", "file_exists", "snapshot_download",
    ]
    missing = [m for m in expected if not hasattr(HfApi, m)]
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: huggingface_hub.HfApi lost methods "
            f"{missing}; saving_utils.py + mlx_utils.py upload/commit "
            "paths fail with AttributeError on call."
        )


def test_hf_hub_get_token_top_level_or_utils_fallback():
    """saving_utils.py:67-73 try-cascade: ``get_token`` from top-level,
    then huggingface_hub.utils, then huggingface_hub.utils._token. At
    least one must resolve."""
    _require_module("huggingface_hub")
    found = False
    for path in (
        "huggingface_hub.get_token",
        "huggingface_hub.utils.get_token",
        "huggingface_hub.utils._token.get_token",
    ):
        try:
            _resolve(path)
            found = True
            break
        except AssertionError:
            continue
    if not found:
        pytest.fail(
            "DRIFT DETECTED: none of the three get_token cascade paths "
            "resolve; saving_utils.py:67-73 has no fallback left and "
            "uploads fail to authenticate."
        )


def test_hf_hub_split_state_dict_into_shards_factory_present_and_callable():
    """saving_utils.py:1602 -- ``split_state_dict_into_shards_factory``
    drives the core 5GB-shard path."""
    _require_module("huggingface_hub")
    fn = _resolve("huggingface_hub.split_state_dict_into_shards_factory")
    if not callable(fn):
        pytest.fail(
            "DRIFT DETECTED: huggingface_hub."
            "split_state_dict_into_shards_factory is no longer callable; "
            "saving_utils.py:1602 sharded-save factory builder breaks."
        )


def test_hf_hub_get_torch_storage_size_and_id_present():
    """saving_utils.py:1604-1605 -- underpin the LoRA delta dedup in
    sharded save."""
    _require_module("huggingface_hub")
    _resolve_all([
        "huggingface_hub.get_torch_storage_size",
        "huggingface_hub.get_torch_storage_id",
    ])


def test_hf_hub_serialization_base_parse_size_to_int():
    """saving_utils.py:1652 -- ``from huggingface_hub.serialization._base
    import parse_size_to_int``. Private module path; refactor breaks
    shard-size CLI string parser."""
    _require_module("huggingface_hub")
    _resolve("huggingface_hub.serialization._base.parse_size_to_int")


def test_hf_hub_errors_local_entry_not_found_error():
    """saving_utils.py:2365 -- imported INSIDE an except clause to
    re-classify a download failure."""
    _require_module("huggingface_hub")
    _resolve("huggingface_hub.errors.LocalEntryNotFoundError")


def test_hf_hub_constants_module_path():
    """``huggingface_hub.constants.HF_HUB_OFFLINE`` is re-injected by
    fix_huggingface_hub (import_fixes.py). Also covered in
    test_upstream_import_fixes_drift.py; double-pin so a typo in either
    test still catches drift."""
    _require_module("huggingface_hub")
    _resolve("huggingface_hub.constants.HF_HUB_OFFLINE")


def test_hf_hub_modelcard_load_method():
    """mlx_utils.py:3195 -- ``ModelCard.load(model_id)`` builds an
    MLX-derived card by inheriting the source card."""
    _require_module("huggingface_hub")
    ModelCard = _resolve("huggingface_hub.ModelCard")
    if not hasattr(ModelCard, "load"):
        pytest.fail(
            "DRIFT DETECTED: huggingface_hub.ModelCard.load classmethod "
            "removed; mlx_utils.py:3195 cannot rebuild the source model "
            "card for MLX-derived repos."
        )


def test_hf_hub_snapshot_download_signature_local_dir():
    """saving_utils.py + mlx_utils.py call ``snapshot_download(repo_id,
    local_dir=...)``. Regression to local_dir_use_symlinks-only would
    break every Unsloth offline workflow."""
    _require_module("huggingface_hub")
    fn = _resolve("huggingface_hub.snapshot_download")
    sig = inspect.signature(fn)
    if "local_dir" not in sig.parameters:
        pytest.fail(
            "DRIFT DETECTED: huggingface_hub.snapshot_download lost "
            "`local_dir` kwarg; saving_utils.py + mlx_utils.py offline "
            "download flows break."
        )


def test_hf_hub_hf_hub_download_signature_local_dir_and_repo_id():
    """saving_utils.py:3088 -- ``hf_hub_download(repo_id=..., filename=...,
    local_dir=...)``."""
    _require_module("huggingface_hub")
    fn = _resolve("huggingface_hub.hf_hub_download")
    sig = inspect.signature(fn)
    expected = {"repo_id", "filename", "local_dir"}
    missing = expected - set(sig.parameters)
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: huggingface_hub.hf_hub_download lost "
            f"parameters {sorted(missing)}; saving_utils.py:3088 "
            "TypeError at runtime."
        )


# ===========================================================================
# xformers
# ===========================================================================
# Zoo has no direct xformers imports; only re-exports if installed.
# transformers.modeling_utils probes xformers.ops at attention-kernel
# selection time -- a regression in xformers' module layout silently
# disables the memory-efficient attention dispatch zoo's patches rely on.


def test_xformers_ops_module_present_when_installed():
    """xformers.ops.memory_efficient_attention is the symbol
    transformers.modeling_utils probes at attention-kernel selection."""
    if importlib.util.find_spec("xformers") is None:
        pytest.skip("xformers not installed -- nothing to drift-check.")
    try:
        ops = importlib.import_module("xformers.ops")
    except Exception as exc:
        pytest.fail(
            f"DRIFT DETECTED: xformers installed but xformers.ops fails "
            f"to import ({exc!r}); attention kernel dispatch falls back "
            "to eager silently."
        )
    if not hasattr(ops, "memory_efficient_attention"):
        pytest.fail(
            "DRIFT DETECTED: xformers.ops.memory_efficient_attention "
            "removed/renamed; transformers attention selection drops "
            "the xformers backend silently."
        )


def test_xformers_components_module_present_when_installed():
    """xformers.components is the attention-block factory namespace
    some transformers vision encoders probe at compile time."""
    if importlib.util.find_spec("xformers") is None:
        pytest.skip("xformers not installed -- nothing to drift-check.")
    spec = importlib.util.find_spec("xformers.components")
    if spec is None:
        # Some xformers builds (CPU-only) ship without .components.
        return
    try:
        importlib.import_module("xformers.components")
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: xformers.components import-fails on this "
            f"install ({exc!r}); transformers attention block factory "
            "probes will mis-detect xformers availability."
        )
