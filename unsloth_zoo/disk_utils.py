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

Two separate problems live here because they are the same problem seen twice:
an export needs more disk than the filesystem it was pointed at can give.

1. Kaggle's working directory is ~20GB, while the overlay mounted at /tmp on
   the very same kernel is measured in terabytes. Exports that cannot fit in
   the former fit trivially in the latter.
2. The size an export actually needs is easy to under-estimate, and an
   under-estimate is worse than no estimate at all: it reports safety it has
   not checked. See `estimate_gguf_export_bytes` for the arithmetic.

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

# Module level rather than inlined so tests can point them at a scratch tree.
# Writing to a real /tmp from a test is both slow and, on locked down CI
# machines, forbidden outright.
KAGGLE_TMP = "/tmp"
KAGGLE_WORKING = "/kaggle/working"

_FALSE = ("0", "false", "no", "off", "")
_TRUE = ("1", "true", "yes", "on")

# Slack for tokenizer files, the config, llama.cpp's own build tree and the
# filesystem's own reserve. Small next to a model, large enough that a merge
# does not fail on the last megabyte.
DISK_SLACK_BYTES = 2 * 1024**3


def is_kaggle_environment():
    """True only inside a running Kaggle kernel.

    The signal is `KAGGLE_KERNEL_RUN_TYPE` ("Interactive" or "Batch", set by
    the kernel runtime itself) AND the presence of the `/kaggle/working`
    directory that the Kaggle image creates.

    Why not the previous test, `any environment variable starting with
    KAGGLE_`: the Kaggle CLI authenticates from `KAGGLE_USERNAME`,
    `KAGGLE_KEY` and `KAGGLE_CONFIG_DIR`, which users and CI pipelines export
    on ordinary laptops and build agents. Anyone who had ever run
    `export KAGGLE_USERNAME=...` was treated as being inside a Kaggle kernel,
    and had their tokenizer cache and their pushes silently redirected to
    /tmp on their own machine.

    Requiring both halves closes both directions:
      - a stray KAGGLE_* credential export has no /kaggle/working,
      - a machine that happens to have a /kaggle/working directory has no
        KAGGLE_KERNEL_RUN_TYPE.
    Colab sets neither. Windows and Mac have no /kaggle at all, and
    os.path.isdir simply returns False there rather than raising.
    """
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
    """True inside Google Colab. Colab exports COLAB_* into every kernel."""
    for key in os.environ:
        if key.startswith("COLAB_"):
            return True
    return False


def free_bytes(path):
    """Free bytes on the filesystem holding `path`, or None if unmeasurable.

    None, not 0. Returning 0 makes an unmeasurable disk look like a full one,
    and every caller here blocks when free < needed, so a single raising
    disk_usage() would turn into a refusal to export anything at all.

    A path that does not exist yet is normal (we are sizing a directory we
    are about to create), so walk up to the nearest existing ancestor.
    """
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


def logical_numel(param):
    """Logical parameter count, which is NOT `numel()` on a quantized weight.

    bitsandbytes stores a 4-bit weight as packed uint8, two parameters per
    byte, and hangs the real shape off `quant_state`. So `numel()` reports
    exactly half the logical count per quantized tensor, and a model-wide sum
    of `numel()` under-counts a 4-bit load by nearly 2x once most of the
    model is quantized. Sizing a 16-bit merge from that halves the estimate.
    """
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
    """Total logical parameter count, or 0 if it cannot be measured."""
    try:
        return sum(logical_numel(p) for p in model.parameters())
    except Exception:
        return 0


def model_16bit_bytes(model):
    """On-disk size of a 16-bit save: 2 bytes per LOGICAL parameter."""
    return model_logical_numel(model) * 2


# Effective bits per weight of a finished GGUF, including the f16/q8 token
# embedding and output tensors llama.cpp keeps at higher precision. Values are
# the nominal bpw llama.cpp documents for each type, rounded up where the mixed
# types make the real figure model dependent. Over-estimating a quant output
# costs a little headroom; under-estimating it costs the whole export.
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
    """Bits per weight for a GGUF type, defaulting high for unknown types.

    An unknown type is far more likely to be a new large quant than a new
    tiny one, and guessing small here is the failure mode this module exists
    to stop, so unknown falls back to q8_0.
    """
    name = str(quantization_type or "f16").strip().lower()
    name = GGUF_QUANT_ALIASES.get(name, name)
    return GGUF_BITS_PER_WEIGHT.get(name, GGUF_BITS_PER_WEIGHT["q8_0"])


def estimate_gguf_export_bytes(
    model = None,
    quantization_methods = (),
    first_conversion = "f16",
    needs_merge = True,
    n_parameters = None,
    keep_intermediate_gguf = True,
    base_cache_copy = False,
):
    """Peak bytes a `save_pretrained_gguf` needs on one filesystem.

    Returns 0 when nothing can be measured, so callers never block on a
    guess.

    The peak is not the sum of the outputs, and this is the whole point.
    Exporting a LoRA model to GGUF lays down, on the same disk, in order:

      1. the 16-bit HF merge, 2 bytes per logical parameter (`needs_merge`),
      2. the intermediate GGUF at `first_conversion` (f16 unless asked
         otherwise), another 2 bytes per parameter, which llama-quantize
         reads from and which is NOT deleted before the quants are written,
      3. every requested quantized GGUF.

    So a run that produces a 14GB merge and a 4GB q4_k_m has a high-water
    mark near 32GB, not 18GB. Sizing it at "two copies of the model" is what
    let Gemma4 (26B A4B) Vision, Gemma4 (31B) Vision and Qwen3 32B pass the
    guard, complete their 16-bit merges, and then die partway through a GGUF
    shard with "Not enough free space to write".

    `base_cache_copy` adds a FOURTH copy, and it is the one that finishes the
    arithmetic on those three. Before merging a LoRA, unsloth pre-warms the
    Hugging Face cache with the full-precision base so later exports skip the
    download, which puts another 2 bytes per parameter on the same disk.
    Gemma4 (31B) Vision in full: 174GB free, minus a 62GB cached base, minus
    a 62GB merge, leaves 50GB - and it died at 48GB of a 65GB GGUF shard.
    Three copies would have called that export safe.

    Pass `needs_merge = False` for a model already on disk in HF format.

    `keep_intermediate_gguf` only reaches the arithmetic when
    `quantization_methods` is empty. That is not an oversight: `llama-quantize`
    is file to file, so the first-conversion GGUF has to stay readable for
    every quant in the loop, and `unsloth/save.py` deletes it only once the
    whole loop has finished. Sizing a quantized export as though the
    intermediate were gone before the quants run would under-count the real
    high-water mark by the size of that file, which on Gemma4 (31B) Vision is
    17.7GB, and would hand back exactly the mid-write failure this function
    exists to prevent. The flag therefore only says whether a convert-only
    export keeps its output.
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

    # Deduplicate: a method equal to the initial conversion is already on disk
    # and gets no second pass, exactly as save_to_gguf does.
    resolved = []
    for method in methods:
        name = str(method).strip().lower()
        resolved.append(GGUF_QUANT_ALIASES.get(name, name))
    quant_only = [m for m in dict.fromkeys(resolved) if m != first]

    if methods or keep_intermediate_gguf:
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
    """Move a default Kaggle save location onto the large /tmp overlay.

    Returns `(directory, message)`. `message` is None when nothing moved, and
    then `directory` is the input unchanged.

    Deliberately conservative, because silently relocating a directory the
    caller named would be a worse bug than the disk error it avoids. All of
    these must hold:

      - we are inside a real Kaggle kernel,
      - `UNSLOTH_KAGGLE_USE_TMP` has not been set to a false value,
      - the caller gave a RELATIVE path, i.e. a name resolved against
        whatever the working directory happens to be, rather than an
        absolute path they chose,
      - that path resolves to somewhere under /kaggle/working,
      - /tmp has more room than /kaggle/working, so moving gains something,
      - /kaggle/working genuinely cannot hold `need_bytes` while /tmp can.

    Set `UNSLOTH_KAGGLE_USE_TMP=1` to move whenever /tmp is roomier, without
    waiting for /kaggle/working to be measured too small.
    """
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
        # /tmp is no roomier. Moving buys nothing, so do not surprise anyone.
        return save_directory, None
    if not always:
        if need_bytes <= 0:
            return save_directory, None
        if free_working >= need_bytes:
            return save_directory, None
        if free_tmp < need_bytes:
            # /tmp cannot hold it either. Say nothing and let the caller's
            # own guard produce the real error, rather than moving the files
            # somewhere that fails just the same.
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
