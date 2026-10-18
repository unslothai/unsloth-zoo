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

"""OpenAI's triton_kernels for packed MXFP4 experts, across three upstream API generations:
``matmul_ogs+routing`` (release/3.5.x, vLLM's copy), ``matmul_ogs+ragged`` (``routing.py`` removed in
30ede52, #8375) and ``matmul+ragged`` (renamed in d8119d5, #8765; no scatter reduction, ``b_mx_scale``)."""

import contextlib
import dataclasses
import importlib
import importlib.util
import os
import sys

import torch

__all__ = [
    "get_triton_kernels",
    "get_triton_kernels_backend",
    "triton_kernels_backend_for",
    "TritonKernelsBackend",
    "triton_kernels_alias",
    "mxfp4_ogs_weight",
    "matmul_ogs_available",
    "native_mxfp4_matmul_ok",
    "MATMUL_OGS_RTOL",
    "API_OGS_ROUTING",
    "API_OGS_RAGGED",
    "API_MATMUL_RAGGED",
]

_UNRESOLVED = object()
_MODULE = _UNRESOLVED
_BACKENDS = {}
_LOGGED = set()
_DEVICE_OK = {}
# matmul_ogs rounds in its own order (like bf16 dense); a wrong layout is O(1).
MATMUL_OGS_RTOL = 1e-2

API_OGS_ROUTING = "matmul_ogs+routing"
API_OGS_RAGGED = "matmul_ogs+ragged"
API_MATMUL_RAGGED = "matmul+ragged"


@dataclasses.dataclass
class GatherIndx:
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclasses.dataclass
class ScatterIndx:
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclasses.dataclass
class RoutingData:
    gate_scal: torch.Tensor
    expt_hist: torch.Tensor
    n_expts_tot: int
    n_expts_act: int
    expt_data: object = None


def _field_names(cls):
    return {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()


def _has(module, *names):
    return all(hasattr(module, name) for name in names)


def _detect_api(name):
    sub = lambda s: importlib.import_module(f"{name}.{s}")
    tensor = sub("tensor")
    if not _has(tensor, "convert_layout", "wrap_torch_tensor", "FP4"):
        raise ImportError("triton_kernels.tensor lacks convert_layout / wrap_torch_tensor / FP4")
    sub("swiglu")
    sub("tensor_details.layout")
    try:
        mo = sub("matmul_ogs")
    except ImportError:
        mo = None
    if mo is not None:
        if not _has(mo, "matmul_ogs", "PrecisionConfig", "FnSpecs", "FusedActivation"):
            raise ImportError("triton_kernels.matmul_ogs lacks matmul_ogs / PrecisionConfig / FnSpecs / FusedActivation")
        try:
            routing = sub("routing")
        except ImportError:
            routing = None
        if routing is not None and _has(routing, "routing", "RoutingData", "GatherIndx", "ScatterIndx"):
            return API_OGS_ROUTING
        if _has(mo, "RoutingData", "GatherIndx", "ScatterIndx") and _has(sub("topk"), "topk") \
                and _has(sub("tensor_details.ragged_tensor"), "make_ragged_tensor_metadata"):
            return API_OGS_RAGGED
        raise ImportError("triton_kernels.matmul_ogs without routing or ragged metadata")
    mm = sub("matmul")
    if not _has(mm, "matmul", "PrecisionConfig", "FnSpecs", "FusedActivation", "FlexCtx"):
        raise ImportError("triton_kernels.matmul lacks matmul / PrecisionConfig / FnSpecs / FusedActivation")
    if not (_has(sub("topk"), "topk") and _has(sub("tensor_details.ragged_tensor"), "make_ragged_tensor_metadata")):
        raise ImportError("triton_kernels.matmul without topk / make_ragged_tensor_metadata")
    return API_MATMUL_RAGGED


class TritonKernelsBackend:

    def __init__(self, module, api = None):
        self.module = module
        self.name = module.__name__
        self.api = api or _detect_api(self.name)
        self.source = "vllm" if self.name.startswith("vllm.") else "triton_kernels"
        sub = lambda s: importlib.import_module(f"{self.name}.{s}")
        self.tensor = sub("tensor")
        self.layout = sub("tensor_details.layout")
        self.swiglu = sub("swiglu")
        self.mm = sub("matmul" if self.api == API_MATMUL_RAGGED else "matmul_ogs")
        self.routing_module = sub("routing") if self.api == API_OGS_ROUTING else None
        if self.api == API_OGS_ROUTING:
            self.index_types = self.routing_module
        elif self.api == API_OGS_RAGGED:
            self.index_types = self.mm
        else:
            self.index_types = sys.modules[__name__]
        self._precision_fields = _field_names(self.mm.PrecisionConfig)
        self._activation_fields = _field_names(self.mm.FusedActivation)

    def __repr__(self):
        return f"TritonKernelsBackend({self.name}, {self.api})"

    def routing(self, logits, n_expts_act):
        if self.api == API_OGS_ROUTING:
            return self.routing_module.routing(logits, n_expts_act)
        topk = importlib.import_module(self.name + ".topk").topk
        sparse = topk(logits, n_expts_act, apply_softmax = True)
        meta = sparse.mask_metadata
        combine_indx, dispatch_indx = meta.col_sorted_indx, meta.row_sorted_indx
        gate_scal = sparse.vals.flatten()[combine_indx]
        return self.routing_from_sorted(
            gate_scal, meta.col_sum, combine_indx, dispatch_indx, logits.shape[-1], n_expts_act,
        )

    def ragged_metadata(self, expt_hist, n_gates):
        ragged = importlib.import_module(self.name + ".tensor_details.ragged_tensor")
        return ragged.make_ragged_tensor_metadata(expt_hist, n_gates)

    def routing_from_sorted(self, gate_scal, expt_hist, combine_indx, dispatch_indx, n_expts_tot, n_expts_act,
                            expt_data = None):
        types_ = self.index_types
        if self.api != API_OGS_ROUTING:
            expt_data = self.ragged_metadata(expt_hist, combine_indx.shape[0])
            expt_hist = expt_data.slice_sizes
        return (
            types_.RoutingData(gate_scal, expt_hist, n_expts_tot, n_expts_act, expt_data),
            types_.GatherIndx(src_indx = combine_indx, dst_indx = dispatch_indx),
            types_.ScatterIndx(src_indx = dispatch_indx, dst_indx = combine_indx),
        )

    def precision(self, weight_scale):
        mo = self.mm
        flex = mo.FlexCtx(rhs_data = mo.InFlexData())
        if "weight_scale" in self._precision_fields:
            return mo.PrecisionConfig(weight_scale = weight_scale, flex_ctx = flex)
        kwargs = dict(b_mx_scale = weight_scale, flex_ctx = flex)
        if "b_microblock_size" in self._precision_fields:
            kwargs["b_microblock_size"] = 32
        return mo.PrecisionConfig(**kwargs)

    @staticmethod
    def weight_scale(precision_config):
        scale = getattr(precision_config, "weight_scale", None)
        return scale if scale is not None else getattr(precision_config, "b_mx_scale", None)

    def swiglu_activation(self, alpha, limit):
        # `reduction_n` moved in #8483.
        mo = self.mm
        fn = self.swiglu.swiglu_fn
        if "reduction_n" in self._activation_fields:
            return mo.FusedActivation(mo.FnSpecs("swiglu", fn, ("alpha", "limit")), (alpha, limit), 2)
        return mo.FusedActivation(mo.FnSpecs("swiglu", fn, ("alpha", "limit"), reduction_n = 2), (alpha, limit))

    def unswizzle(self, tensor):
        # major_dim = -2: the checkpoint packs FP4 along dim -2; the canonical unswizzled form packs the last dim.
        layout = tensor.storage.layout
        if hasattr(layout, "unswizzle_data"):
            return layout.unswizzle_data(tensor.storage.data)
        strided = self.tensor.convert_layout(tensor, self.layout.StridedLayout(major_dim = -2))
        return strided.storage.data

    def matmul_ogs(self, x, weight, bias, routing_data, gather_indx = None, scatter_indx = None,
                   precision_config = None, gammas = None, fused_activation = None):
        kwargs = {}
        if gammas is not None:
            kwargs["gammas"] = gammas
        if fused_activation is not None:
            kwargs["fused_activation"] = fused_activation
        if self.api != API_MATMUL_RAGGED:
            return self.mm.matmul_ogs(
                x, weight, bias, routing_data, gather_indx = gather_indx, scatter_indx = scatter_indx,
                precision_config = precision_config, **kwargs,
            )
        k = routing_data.n_expts_act
        if gather_indx is not None:
            kwargs["gather_indx"] = torch.div(gather_indx.src_indx, k, rounding_mode = "trunc")
        if scatter_indx is None:
            return self.mm.matmul(
                x, weight, bias, a_ragged_metadata = routing_data.expt_data,
                precision_config = precision_config, **kwargs,
            )
        # matmul leaves the k-sum to the caller. As matmul_ogs (split_k = 1): rows in the output dtype, fp32 sum.
        # torch sums bf16 / fp16 in fp32 and rounds once, so no (tokens, hidden) fp32 temporary is needed.
        rows = self.mm.matmul(
            x, weight, bias, a_ragged_metadata = routing_data.expt_data, scatter_indx = scatter_indx.dst_indx,
            precision_config = precision_config, **kwargs,
        )
        return rows.view(-1, k, rows.shape[-1]).sum(1)

    def gate_up(self, x, weight, bias, routing_data, gather_indx, precision_config, fused_activation = None):
        return self.matmul_ogs(
            x, weight, bias, routing_data, gather_indx = gather_indx,
            precision_config = precision_config, fused_activation = fused_activation,
        )

    def down(self, h, weight, bias, routing_data, scatter_indx, precision_config, gammas = None):
        return self.matmul_ogs(
            h, weight, bias, routing_data, scatter_indx = scatter_indx,
            precision_config = precision_config, gammas = gammas,
        )


def _probe(module):
    _BACKENDS[module.__name__] = TritonKernelsBackend(module)
    return module


def _import_top_level():
    if "triton_kernels" not in sys.modules and importlib.util.find_spec("triton_kernels") is None:
        return None
    return _probe(importlib.import_module("triton_kernels"))


def _import_vllm_vendored():
    # vLLM's copy imports itself as `triton_kernels`: alias only while importing (a lasting alias flips patch_gpt_oss).
    try:
        if importlib.util.find_spec("vllm.third_party.triton_kernels") is None:
            return None
    except (ImportError, ValueError):
        return None
    import pkgutil
    vendored = importlib.import_module("vllm.third_party.triton_kernels")
    previous = sys.modules.get("triton_kernels", _UNRESOLVED)
    # Hide cached top-level children too, or the vendored tree's absolute imports reuse them.
    hidden = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith("triton_kernels.")}
    sys.modules["triton_kernels"] = vendored
    try:
        for info in pkgutil.walk_packages(vendored.__path__, vendored.__name__ + "."):
            try:
                importlib.import_module(info.name)
            except Exception:
                pass  # optional extras (proton, testing); _probe decides what is required
        return _probe(vendored)
    finally:
        # Drop the second copies the alias loaded, or a later top-level import mixes their classes with its own.
        for name in [n for n in sys.modules if n.startswith("triton_kernels.")]:
            sys.modules.pop(name, None)
        sys.modules.update(hidden)
        if previous is _UNRESOLVED:
            sys.modules.pop("triton_kernels", None)
        else:
            sys.modules["triton_kernels"] = previous


def get_triton_kernels():
    """Top-level triton_kernels, else vLLM's copy, else None; UNSLOTH_TRITON_KERNELS_SOURCE = auto | triton_kernels | vllm."""
    global _MODULE
    if _MODULE is not _UNRESOLVED:
        return _MODULE
    module = None
    if os.environ.get("UNSLOTH_MXFP4_MATMUL_OGS", "1") != "0":
        source = os.environ.get("UNSLOTH_TRITON_KERNELS_SOURCE", "auto").strip().lower()
        resolvers = {
            "triton_kernels": (_import_top_level,), "vllm": (_import_vllm_vendored,),
        }.get(source, (_import_top_level, _import_vllm_vendored))
        for resolve in resolvers:
            try:
                module = resolve()
            except Exception:
                module = None
            if module is not None:
                break
    _MODULE = module
    if module is not None:
        _log_selected(triton_kernels_backend_for(module))
    return module


def _log_selected(backend):
    if backend.name in _LOGGED:
        return
    _LOGGED.add(backend.name)
    try:
        from unsloth_zoo.log import logger
        logger.info(f"Unsloth: MXFP4 experts use triton_kernels '{backend.name}' ({backend.api} API).")
    except Exception:
        pass


def triton_kernels_backend_for(module):
    name = module if isinstance(module, str) else module.__name__
    backend = _BACKENDS.get(name)
    if backend is None:
        if isinstance(module, str):
            module = sys.modules.get(name) or importlib.import_module(name)
        backend = _BACKENDS[name] = TritonKernelsBackend(module)
    return backend


def get_triton_kernels_backend():
    tk = get_triton_kernels()
    if tk is None:
        return None
    try:
        return triton_kernels_backend_for(tk)
    except Exception:
        return None


def selected_triton_kernels():
    backend = get_triton_kernels_backend()
    return None if backend is None else (backend.name, backend.api)


@contextlib.contextmanager
def triton_kernels_alias():
    tk = get_triton_kernels()
    previous = sys.modules.get("triton_kernels", _UNRESOLVED)
    if tk is not None and previous is _UNRESOLVED:
        sys.modules["triton_kernels"] = tk
    try:
        yield tk
    finally:
        if tk is not None and previous is _UNRESOLVED:
            sys.modules.pop("triton_kernels", None)


def convert_layout_shared(tensor_module, tensor, layout, **layout_opts):
    """``convert_layout`` that keeps the storage when ``tensor`` already has ``layout``. release/3.7.x copies even
    then (3.8 returns the tensor), which duplicates every packed expert stack and, inside a CUDA graph tree's
    warmup, leaves the copy as an untracked allocation in the graph pool."""
    if not layout_opts and not isinstance(layout, type) and getattr(tensor.storage, "layout", None) == layout:
        return tensor
    return tensor_module.convert_layout(tensor, layout, **layout_opts)


def mxfp4_ogs_weight(blocks, scales):
    """GPT-OSS checkpoint (E, out, in / 32, 16) blocks + (E, out, in / 32) scales -> (weight, PrecisionConfig)
    for matmul_ogs, with the (E, in, out) logical shape its kernels expect. Copies only when the layout swizzles."""
    from unsloth_zoo.temporary_patches.gpt_oss import _mxfp4_layout_arguments
    E, N, G, _ = blocks.shape
    with triton_kernels_alias() as tk:
        backend = triton_kernels_backend_for(tk)
        tensor, layout = backend.tensor, backend.layout
        values = blocks.reshape(E, N, G * 16).transpose(-2, -1)
        value_layout, value_opts, strided = _mxfp4_layout_arguments(layout, values)
        weight = convert_layout_shared(tensor, tensor.wrap_torch_tensor(values, dtype = tensor.FP4), value_layout, **value_opts)
        weight.shape = torch.Size([E, G * 32, N])
        scale = convert_layout_shared(tensor, tensor.wrap_torch_tensor(scales.transpose(-2, -1)), strided)
        return weight, backend.precision(scale)


def _self_check(device):
    tk = get_triton_kernels()
    if tk is None or device.type != "cuda" or torch.version.hip is not None:
        return False
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
    backend = triton_kernels_backend_for(tk)
    gen = torch.Generator(device = device).manual_seed(0)
    E, N, K = 4, 256, 512  # (E, out, in): K = 16 MXFP4 groups
    blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, device = device, generator = gen)
    scales = torch.randint(118, 128, (E, N, K // 32), dtype = torch.uint8, device = device, generator = gen)
    want_w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)  # (E, N, K)
    w, precision = mxfp4_ogs_weight(blocks, scales)
    for rows in (1, 8, 96):
        x = torch.randn(rows, K, device = device, generator = gen).to(torch.bfloat16)
        for e in range(E):
            logits = torch.full((rows, E), -1e4, device = device)
            logits[:, e] = 0
            rdata, gather, scatter = backend.routing(logits, 1)
            got = backend.matmul_ogs(x, w, None, rdata, gather_indx = gather, scatter_indx = scatter,
                                     precision_config = precision).float()
            want = x.float() @ want_w[e].t()
            if not torch.allclose(got, want, rtol = MATMUL_OGS_RTOL, atol = MATMUL_OGS_RTOL * want.abs().max().item()):
                return False
    return True


def _device_capability(device):
    device = torch.device(device)
    if device.type != "cuda" or torch.version.hip is not None:
        return None
    return tuple(torch.cuda.get_device_capability(device))


def native_mxfp4_matmul_ok(device, backend) -> bool:
    """False where the native expert matmul is known to fail: sm_12x (RTX 50xx / PRO 6000, about 100 KB shared
    memory per block) with the matmul_ogs+ragged generation (release/3.6.x), whose Blackwell tiles do not fit and
    trip ``assert num_stages >= 1`` in opt_flags. release/3.5.x and 3.7+ and every other GPU keep the native path."""
    capability = _device_capability(device)
    if capability is None or backend is None:
        return True
    return not (capability[0] == 12 and backend.api == API_OGS_RAGGED)


def matmul_ogs_available(device = None) -> bool:
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
