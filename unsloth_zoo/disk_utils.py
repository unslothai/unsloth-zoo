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

"""Disk sizing helpers, plus the one place that decides "are we on Kaggle?".

Both problems here are the same one seen twice: an export needs more disk than
the filesystem it was pointed at can give.

1. Kaggle's working directory is ~20GB while the /tmp overlay on the same
   kernel is terabytes, so an export that cannot fit in one fits in the other.
2. An under-estimated export size is worse than no estimate: it reports safety
   it never checked. See `estimate_gguf_export_bytes` for the arithmetic.

Nothing here imports torch at module level, so it stays importable on hosts
where torch is broken or absent.
"""

import os
import shutil

__all__ = [
    "KAGGLE_TMP",
    "KAGGLE_WORKING",
    "GGUF_BITS_PER_WEIGHT",
    "is_kaggle_environment",
    "is_colab_environment",
    "free_bytes",
    "logical_numel",
    "model_16bit_bytes",
    "gguf_bits_per_weight",
    "estimate_gguf_export_bytes",
    "kaggle_tmp_redirect",
]

# Module level so tests can point these at a scratch tree, since a real /tmp is slow and may be forbidden on locked-down CI machines.
KAGGLE_TMP = "/tmp"
KAGGLE_WORKING = "/kaggle/working"

_FALSE = ("0", "false", "no", "off", "")
_TRUE = ("1", "true", "yes", "on")

# Slack for tokenizer files, the config, llama.cpp's build tree and the filesystem reserve, so a merge does not fail on the last megabyte.
DISK_SLACK_BYTES = 2 * 1024**3


def is_kaggle_environment():
    """True only in a running Kaggle kernel: needs both KAGGLE_KERNEL_RUN_TYPE and /kaggle/working; not any KAGGLE_*
    var, since Kaggle CLI credentials match it; not /kaggle alone, which Colab VMs have."""
    override = os.environ.get("UNSLOTH_IS_KAGGLE", None)
    if override is not None:
        return str(override).strip().lower() in _TRUE
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip():
        return False
    try:
        return os.path.isdir(KAGGLE_WORKING)
    except Exception:
        return False


def is_colab_environment():
    """True when any COLAB_* env var is set; Kaggle kernels export COLAB_* too, so it only means hosted notebook, which both callers treat alike."""
    for key in os.environ:
        if key.startswith("COLAB_"):
            return True
    return False


def free_bytes(path):
    """Free bytes on the filesystem holding `path`, or None if unmeasurable; None not 0, since 0 reads as a full disk
    and blocks every caller; walks up to the nearest existing ancestor."""
    try:
        probe = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        return None
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except Exception:
        return None


def _is_uint8(param):
    # str() rather than a torch import: this module stays importable without it.
    return str(getattr(param, "dtype", "")) == "torch.uint8"


def logical_numel(param, name = ""):
    """Logical parameter count, not numel() on quantized weights: bitsandbytes packs two 4-bit params per uint8 (real
    shape in quant_state); MXFP4 _blocks count 2x and _scales count 0."""
    if name and _is_uint8(param):
        if name.endswith("_scales"):
            return 0
        if name.endswith("_blocks"):
            try:
                return int(param.numel()) * 2
            except Exception:
                return 0
    try:
        shape = getattr(getattr(param, "quant_state", None), "shape", None)
        if shape is not None:
            n = 1
            for dim in shape:
                n *= int(dim)
            return n
    except Exception:
        pass
    try:
        return int(param.numel())
    except Exception:
        return 0


def model_logical_numel(model):
    """Total logical parameter count, or 0 if unmeasurable; iterates named_parameters() since MXFP4 packing is only identifiable by name."""
    try:
        return sum(logical_numel(p, n) for n, p in model.named_parameters())
    except Exception:
        pass
    try:
        return sum(logical_numel(p) for p in model.parameters())
    except Exception:
        return 0


def model_16bit_bytes(model):
    """On-disk size of a 16-bit save: 2 bytes per LOGICAL parameter."""
    return model_logical_numel(model) * 2


# Effective GGUF bits per weight, incl. embedding and output tensors kept at higher precision; llama.cpp's nominal bpw
# rounded up where mixed types make it model dependent, since under-estimating costs the whole export.
GGUF_BITS_PER_WEIGHT = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_1": 6.0,
    "q5_k_m": 5.7,
    "q5_k_s": 5.5,
    "q5_0": 5.5,
    "q4_1": 5.0,
    "q4_k_m": 4.9,
    "q4_k_s": 4.6,
    "q4_0": 4.5,
    "iq4_nl": 4.5,
    "iq4_xs": 4.25,
    "mxfp4": 4.25,
    "q3_k_l": 4.3,
    "q3_k_m": 3.9,
    "iq3_m": 3.7,
    "q2_k_l": 3.7,
    "q3_k_s": 3.5,
    "iq3_s": 3.5,
    "q2_k": 3.4,
    "iq3_xxs": 3.1,
    "q3_k_xs": 3.1,
    "iq2_m": 2.7,
    "iq2_s": 2.5,
    "iq2_xs": 2.31,
    "iq2_xxs": 2.06,
    "iq1_m": 1.75,
    "iq1_s": 1.56,
}

# The friendly aliases unsloth accepts, mapped onto what actually gets written.
GGUF_QUANT_ALIASES = {
    "not_quantized": "f16",
    "fast_quantized": "q8_0",
    "quantized": "q4_k_m",
    "q4_k": "q4_k_m",
    "q5_k": "q5_k_m",
    "none": "f16",
}


def gguf_bits_per_weight(quantization_type):
    """Bits per weight for a GGUF type; unknown types fall back to q8_0, since guessing small for an unknown quant is the failure this module exists to stop."""
    name = str(quantization_type or "f16").strip().lower()
    name = GGUF_QUANT_ALIASES.get(name, name)
    return GGUF_BITS_PER_WEIGHT.get(name, GGUF_BITS_PER_WEIGHT["q8_0"])


def estimate_gguf_export_bytes(
    model = None,
    quantization_methods = (),
    first_conversion = "f16",
    needs_merge = True,
    n_parameters = None,
    base_cache_copy = False,
):
    """Peak bytes a `save_pretrained_gguf` needs on one filesystem.

    Returns 0 when nothing can be measured, so callers never block on a guess.

    The peak is not the sum of the outputs, and that is the whole point. A
    LoRA export lays these on the same disk, deleting nothing in between:

      1. the 16-bit HF merge, 2 bytes per logical parameter (`needs_merge`;
         pass False for a model already on disk in HF format),
      2. the intermediate GGUF at `first_conversion` (f16 unless asked
         otherwise), which llama-quantize reads from,
      3. every requested quantized GGUF.

    So a 14GB merge plus a 4GB q4_k_m peaks near 32GB, not 18GB. Sizing it at
    "two copies of the model" is what let Gemma4 (26B A4B) Vision, Gemma4
    (31B) Vision and Qwen3 32B pass the guard, finish their merges, then die
    partway through a GGUF shard with "Not enough free space to write".

    `base_cache_copy` adds a FOURTH copy: before merging a LoRA, unsloth
    pre-warms the Hugging Face cache with the full-precision base. Gemma4
    (31B) Vision in full: 174GB free, minus a 62GB cached base, minus a 62GB
    merge, leaves 50GB - and it died at 48GB of a 65GB shard. Three copies
    would have called that export safe.

    The first conversion is counted whether or not the export keeps it: a file
    has to be written before it can be deleted, `llama-quantize` is file to
    file so it must stay readable for every quant in the loop, and
    `unsloth/save.py` unlinks it only once that loop has finished (17.7GB on
    Gemma4 31B Vision).
    """
    if n_parameters is None:
        n_parameters = model_logical_numel(model) if model is not None else 0
    n_parameters = int(n_parameters or 0)
    if n_parameters <= 0:
        return 0

    if isinstance(quantization_methods, str):
        quantization_methods = [quantization_methods]
    methods = [m for m in (quantization_methods or []) if m is not None]

    total = 0
    if needs_merge:
        total += n_parameters * 2
    if base_cache_copy:
        total += n_parameters * 2

    first = str(first_conversion or "f16").strip().lower()
    first = GGUF_QUANT_ALIASES.get(first, first)

    # A method equal to the initial conversion is already on disk and gets no
    # second pass, exactly as save_to_gguf does.
    resolved = []
    for method in methods:
        name = str(method).strip().lower()
        resolved.append(GGUF_QUANT_ALIASES.get(name, name))
    quant_only = [m for m in dict.fromkeys(resolved) if m != first]

    total += int(n_parameters * gguf_bits_per_weight(first) / 8)
    for method in quant_only:
        total += int(n_parameters * gguf_bits_per_weight(method) / 8)

    return total + DISK_SLACK_BYTES


def _format_gb(n_bytes):
    return f"{n_bytes / 1024**3:.1f}GB"


def kaggle_tmp_redirect(
    save_directory,
    need_bytes = 0,
    what = "export",
    subdirectory = "unsloth_saves",
):
    """Returns (directory, message), unchanged if nothing moves; only relative paths under /kaggle/working, in a Kaggle
    kernel, unless UNSLOTH_KAGGLE_USE_TMP is false, when /tmp can hold what /kaggle/working cannot; never move a
    caller-named dir; UNSLOTH_KAGGLE_USE_TMP=1 moves whenever /tmp is roomier."""
    if not is_kaggle_environment():
        return save_directory, None

    flag = os.environ.get("UNSLOTH_KAGGLE_USE_TMP", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return save_directory, None
    always = flag in _TRUE

    path = str(save_directory)
    if os.path.isabs(path):
        # The caller named a filesystem location. Honour it.
        return save_directory, None

    resolved = os.path.abspath(path)
    working = os.path.abspath(KAGGLE_WORKING)
    try:
        inside = os.path.commonpath([resolved, working]) == working
    except Exception:
        # Different Windows drives, or anything else commonpath dislikes.
        inside = False
    if not inside:
        return save_directory, None

    free_working = free_bytes(working)
    free_tmp = free_bytes(KAGGLE_TMP)
    if free_tmp is None or free_working is None:
        return save_directory, None
    if free_tmp <= free_working:
        # /tmp is no roomier, so moving buys nothing but a surprise.
        return save_directory, None
    if not always:
        if need_bytes <= 0:
            return save_directory, None
        if free_working >= need_bytes:
            return save_directory, None
        if free_tmp < need_bytes:
            # If /tmp cannot hold it either, keep the caller's path so its own guard raises the real error, not a move that fails too.
            return save_directory, None

    relative = os.path.relpath(resolved, working)
    target = os.path.join(KAGGLE_TMP, subdirectory, relative)
    try:
        os.makedirs(target, exist_ok = True)
    except Exception:
        return save_directory, None

    message = (
        f"Unsloth: Kaggle's working directory only has {_format_gb(free_working or 0)} free, "
        f"so the {what} goes to {target} instead ({_format_gb(free_tmp)} free). "
        f"/tmp is scratch space: it is NOT saved as kernel output, so copy or push "
        f"anything you want to keep before the session ends."
    )
    return target, message
