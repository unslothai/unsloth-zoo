# SPDX-License-Identifier: AGPL-3.0-only
"""Scoped MLX quantized MoE gate and up projection fusion."""

import sys
from contextlib import contextmanager
from threading import RLock

import mlx.core as mx


_MOE_PROJECTION_FIELDS = ("weight", "scales", "biases", "bias")


_MOE_GATE_UP_CLASSES = {}
_MOE_GATE_UP_LOCK = RLock()


def _moe_switch_specs():
    specs = {}
    for path in ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers"):
        native = sys.modules.get(path)
        if native is not None and hasattr(native, "QuantizedSwitchLinear"):
            specs[native.SwitchGLU] = (
                native.QuantizedSwitchLinear, native._gather_sort, native._scatter_unsort,
            )
    return specs


def _moe_gate_up_eligible(module, projection_type):
    gate, up = module.gate_proj, module.up_proj
    if type(gate) is not projection_type or type(up) is not projection_type:
        return False
    if (module.training or gate.training or up.training
            or gate.trainable_parameters() or up.trainable_parameters()):
        return False
    if gate.bits != 8 or gate.mode != "affine" or up.mode != "affine":
        return False
    if (gate.group_size, gate.bits) != (up.group_size, up.bits):
        return False
    for name in _MOE_PROJECTION_FIELDS:
        a, b = gate.get(name), up.get(name)
        if (a is None) != (b is None):
            return False
        if a is not None and (
            not isinstance(a, mx.array)
            or not isinstance(b, mx.array)
            or a.shape != b.shape
            or a.dtype != b.dtype
        ):
            return False
    if gate.weight.ndim != 3 or gate.scales.ndim != 3 or gate.biases is None:
        return False
    return all(
        gate[name].shape[:-1] == gate.weight.shape[:-1] for name in ("scales", "biases")
    ) and (gate.get("bias") is None or gate.bias.shape == gate.weight.shape[:-1])


class _PackedMoEGateUp:
    def __init__(self, module):
        gate, up = module.gate_proj, module.up_proj
        self.original_class = type(module)
        self.active_scopes = 0
        self.gate, self.up = gate, up
        self.metadata = (gate.group_size, gate.bits, gate.mode)
        self.arrays = {
            name: mx.concatenate([gate[name], up[name]], axis = -1 if name == "bias" else -2)
            for name in _MOE_PROJECTION_FIELDS
            if gate.get(name) is not None
        }
        mx.eval(list(self.arrays.values()))
        self.views = {
            name: mx.split(array, 2, axis = -1 if name == "bias" else -2)
            for name, array in self.arrays.items()
        }
        mx.eval([view for pair in self.views.values() for view in pair])

    def attach(self):
        for name, (gate_view, up_view) in self.views.items():
            setattr(self.gate, name, gate_view)
            setattr(self.up, name, up_view)

    def matches(self, module):
        if module.gate_proj is not self.gate or module.up_proj is not self.up:
            return False
        for projection in (self.gate, self.up):
            if (
                projection.training
                or projection.trainable_parameters()
                or (projection.group_size, projection.bits, projection.mode) != self.metadata
            ):
                return False
        for name in _MOE_PROJECTION_FIELDS:
            expected = self.views.get(name, (None, None))
            if self.gate.get(name) is not expected[0] or self.up.get(name) is not expected[1]:
                return False
        return True

    def project(self, x, indices, sorted_indices):
        group_size, bits, mode = self.metadata
        result = mx.gather_qmm(
            x,
            self.arrays["weight"],
            self.arrays["scales"],
            self.arrays["biases"],
            rhs_indices = indices,
            transpose = True,
            group_size = group_size,
            bits = bits,
            mode = mode,
            sorted_indices = sorted_indices,
        )
        if "bias" in self.arrays:
            result = result + mx.expand_dims(self.arrays["bias"][indices], -2)
        return mx.split(result, 2, axis = -1)


def _fused_moe_gate_up_class(original_class, gather_sort, scatter_unsort):
    if original_class not in _MOE_GATE_UP_CLASSES:

        def fused_call(self, x, indices):
            packed = self._unsloth_moe_gate_up
            if self.training or not packed.matches(self):
                return original_class.__call__(self, x, indices)
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = gather_sort(x, indices)
            x_gate, x_up = packed.project(x, idx, do_sort)
            x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices = do_sort)
            if do_sort:
                x = scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

        _MOE_GATE_UP_CLASSES[original_class] = type(
            f"_FusedMoEGateUp{original_class.__name__}", (original_class,), {"__call__": fused_call}
        )
    return _MOE_GATE_UP_CLASSES[original_class]


@contextmanager
def fused_moe_gate_up(model):
    """Fuse quantized MoE gate and up projections while their weights stay fixed.

    Repacking at entry makes weight edits between scopes visible. Overlapping scopes
    share packing until the last exit; training or custom projections retain native calls.
    """
    patched = []
    try:
        with _MOE_GATE_UP_LOCK:
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
                specs = _moe_switch_specs()
                modules = model.named_modules() if hasattr(model, "named_modules") else ()
                for _, module in modules:
                    packed = getattr(module, "_unsloth_moe_gate_up", None)
                    if (isinstance(packed, _PackedMoEGateUp)
                            and type(module) is _MOE_GATE_UP_CLASSES.get(packed.original_class)):
                        original, fused = packed.original_class, type(module)
                    else:
                        original = type(module)
                        spec = specs.get(original)
                        if spec is None or not _moe_gate_up_eligible(module, spec[0]):
                            continue
                        packed = _PackedMoEGateUp(module)
                        fused = _fused_moe_gate_up_class(original, spec[1], spec[2])
                    packed.active_scopes += 1
                    patched.append((module, original, fused, packed))
                    if type(module) is not fused:
                        module._unsloth_moe_gate_up = packed
                        packed.attach()
                        module.__class__ = fused
        yield model
    finally:
        with _MOE_GATE_UP_LOCK:
            for module, original, fused, packed in reversed(patched):
                packed.active_scopes -= 1
                if packed.active_scopes:
                    continue
                if type(module) is fused:
                    module.__class__ = original
                if getattr(module, "_unsloth_moe_gate_up", None) is packed:
                    del module._unsloth_moe_gate_up
