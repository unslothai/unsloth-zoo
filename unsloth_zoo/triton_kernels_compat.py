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

"""OpenAI's triton_kernels (matmul_ogs) for packed MXFP4 experts: resolve once, self-check per device."""

import contextlib
import importlib
import importlib.util
import os
import sys

import torch

__all__ = [
    "get_triton_kernels",
    "triton_kernels_alias",
    "mxfp4_ogs_weight",
    "matmul_ogs_available",
    "MATMUL_OGS_RTOL",
]

_UNRESOLVED = object()
_MODULE = _UNRESOLVED
_DEVICE_OK = {}
# matmul_ogs rounds MXFP4 x bf16 in its own order: real gpt-oss-20b MoE blocks <= 6.1e-3 max rel err vs
# fp32 exact decode (bf16 dense: <= 5.4e-3); this self-check's problem 2.7e-3. A wrong layout is O(1).
MATMUL_OGS_RTOL = 1e-2



def _probe(module):
    """The pieces the packed experts call; a stripped or older build fails here, not mid-forward."""
    matmul_ogs = importlib.import_module(module.__name__ + ".matmul_ogs")
    tensor = importlib.import_module(module.__name__ + ".tensor")
    for owner, name in (
        (matmul_ogs, "matmul_ogs"), (matmul_ogs, "PrecisionConfig"), (matmul_ogs, "FnSpecs"),
        (matmul_ogs, "FusedActivation"), (tensor, "convert_layout"), (tensor, "wrap_torch_tensor"),
        (tensor, "FP4"),
    ):
        if not hasattr(owner, name):
            raise ImportError(f"triton_kernels has no {name}")
    importlib.import_module(module.__name__ + ".swiglu")
    importlib.import_module(module.__name__ + ".routing")
    importlib.import_module(module.__name__ + ".tensor_details.layout")
    return module


def _import_top_level():
    if "triton_kernels" not in sys.modules and importlib.util.find_spec("triton_kernels") is None:
        return None
    return _probe(importlib.import_module("triton_kernels"))


def _import_vllm_vendored():
    """vLLM's copy imports itself as `triton_kernels`: alias only while importing, then restore.

    A lasting alias would flip patch_gpt_oss's `import triton_kernels` on the next load in this process
    (native forward-only MXFP4 instead of the trainable path). vLLM sets its own alias when it needs one.
    """
    try:
        if importlib.util.find_spec("vllm.third_party.triton_kernels") is None:
            return None
    except (ImportError, ValueError):
        return None
    import pkgutil
    vendored = importlib.import_module("vllm.third_party.triton_kernels")
    previous = sys.modules.get("triton_kernels", _UNRESOLVED)
    before = {name for name in sys.modules if name.startswith("triton_kernels.")}
    sys.modules["triton_kernels"] = vendored
    try:
        # Every submodule now, under its own name: some calls import lazily (routing -> `.topk`), and
        # those modules import `triton_kernels.*` absolutely, which fails once the alias is gone.
        for info in pkgutil.walk_packages(vendored.__path__, vendored.__name__ + "."):
            try:
                importlib.import_module(info.name)
            except Exception:
                pass  # optional extras (proton, testing); _probe decides what is required
        return _probe(vendored)
    finally:
        # Absolute `triton_kernels.*` imports under the alias loaded second copies of those submodules;
        # left behind, a later real top-level import would mix their classes with its own.
        for name in [n for n in sys.modules if n.startswith("triton_kernels.") and n not in before]:
            sys.modules.pop(name, None)
        if previous is _UNRESOLVED:
            sys.modules.pop("triton_kernels", None)
        else:
            sys.modules["triton_kernels"] = previous


def get_triton_kernels():
    """The triton_kernels module (top-level install first, then vLLM's vendored copy) or None. Never raises."""
    global _MODULE
    if _MODULE is not _UNRESOLVED:
        return _MODULE
    module = None
    if os.environ.get("UNSLOTH_MXFP4_MATMUL_OGS", "1") != "0":
        for resolve in (_import_top_level, _import_vllm_vendored):
            try:
                module = resolve()
            except Exception:
                module = None
            if module is not None:
                break
    _MODULE = module
    return module


@contextlib.contextmanager
def triton_kernels_alias():
    """`import triton_kernels` resolves to the resolved module inside this block (load-time layout code)."""
    tk = get_triton_kernels()
    previous = sys.modules.get("triton_kernels", _UNRESOLVED)
    if tk is not None and previous is _UNRESOLVED:
        sys.modules["triton_kernels"] = tk
    try:
        yield tk
    finally:
        if tk is not None and previous is _UNRESOLVED:
            sys.modules.pop("triton_kernels", None)


def mxfp4_ogs_weight(blocks, scales):
    """GPT-OSS checkpoint (E, out, in / 32, 16) blocks + (E, out, in / 32) scales -> (weight, PrecisionConfig)
    for matmul_ogs, with the (E, in, out) logical shape its kernels expect. Copies: the layout swizzles."""
    from unsloth_zoo.temporary_patches.gpt_oss import _mxfp4_layout_arguments
    E, N, G, _ = blocks.shape
    with triton_kernels_alias() as tk:
        mo = importlib.import_module(tk.__name__ + ".matmul_ogs")
        tensor = importlib.import_module(tk.__name__ + ".tensor")
        layout = importlib.import_module(tk.__name__ + ".tensor_details.layout")
        values = blocks.reshape(E, N, G * 16).transpose(-2, -1)
        value_layout, value_opts, strided = _mxfp4_layout_arguments(layout, values)
        weight = tensor.convert_layout(tensor.wrap_torch_tensor(values, dtype = tensor.FP4), value_layout, **value_opts)
        weight.shape = torch.Size([E, G * 32, N])
        scale = tensor.convert_layout(tensor.wrap_torch_tensor(scales.transpose(-2, -1)), strided)
        return weight, mo.PrecisionConfig(weight_scale = scale, flex_ctx = mo.FlexCtx(rhs_data = mo.InFlexData()))


def _self_check(device):
    """One real-shaped routed GEMM (gpt-oss-20b gate_up rows) against exact dequant + bf16 matmul."""
    tk = get_triton_kernels()
    if tk is None or device.type != "cuda" or torch.version.hip is not None:
        return False
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
    mo = importlib.import_module(tk.__name__ + ".matmul_ogs")
    gen = torch.Generator(device = device).manual_seed(0)
    E, N, K = 4, 256, 512  # (E, out, in): K = 16 MXFP4 groups
    blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, device = device, generator = gen)
    scales = torch.randint(118, 128, (E, N, K // 32), dtype = torch.uint8, device = device, generator = gen)
    want_w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)  # (E, N, K)
    w, precision = mxfp4_ogs_weight(blocks, scales)
    routing = importlib.import_module(tk.__name__ + ".routing")
    for rows in (1, 8, 96):
        x = torch.randn(rows, K, device = device, generator = gen).to(torch.bfloat16)
        # Every row routed to expert e by a one-hot router: the call the MoE forward makes.
        for e in range(E):
            logits = torch.full((rows, E), -1e4, device = device)
            logits[:, e] = 0
            rdata, gather, scatter = routing.routing(logits, 1)
            got = mo.matmul_ogs(x, w, None, rdata, gather_indx = gather, scatter_indx = scatter,
                                precision_config = precision).float()
            want = x.float() @ want_w[e].t()
            if not torch.allclose(got, want, rtol = MATMUL_OGS_RTOL, atol = MATMUL_OGS_RTOL * want.abs().max().item()):
                return False
    return True


def matmul_ogs_available(device = None) -> bool:
    """Self-checked once per device; any import, layout, launch or accuracy failure -> False (use exact dequant)."""
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    key = str(device)
    ok = _DEVICE_OK.get(key)
    if ok is None:
        on_cuda = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
        try:
            with torch.no_grad(), torch.autocast(device.type, enabled = False), on_cuda:
                ok = bool(_self_check(device))
        except Exception:
            ok = False
        _DEVICE_OK[key] = ok
    return ok
