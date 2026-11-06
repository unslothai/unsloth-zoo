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

"""NPU LoRA cast elimination: CPU policy tests and opt-in real NPU parity tests.

Hardware lane (torch_npu must already import successfully):
    UNSLOTH_TEST_NPU=1 python -m pytest tests/test_npu_lora_casts.py -v
The source extractor avoids importing accelerator-heavy compiler dependencies.
"""

import ast
import inspect
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch


def _compiler_symbols():
    path = Path(__file__).resolve().parents[1] / "unsloth_zoo/compiler.py"
    tree = ast.parse(path.read_text())
    names = {
        "COMPILED_LORA_FORWARD",
        "NPU_LORA_CASTS",
        "_LORA_INPUT_CASTS",
        "_patch_lora_input_cast",
        "_patch_lora_update",
        "_patch_lora_base_layer_input_cast",
        "_patch_npu_lora_casts",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    import re

    namespace = {"torch": torch, "os": os, "re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), namespace)
    return namespace


SYMBOLS = _compiler_symbols()


def _load_helper():
    # Only this policy helper needs PEFT's exact Linear identity. CPU policy tests
    # substitute a small layer so they don't require PEFT or an NPU installation.
    namespace = {"torch": torch, "torch_is_compiling": lambda: False}
    source = SYMBOLS["NPU_LORA_CASTS"].split("\n", 2)[2]
    exec(source, namespace)
    return namespace


class _Layer:
    pass


def _eligible_fixture(monkeypatch):
    namespace = _load_helper()
    namespace["_UnslothPeftLinear"] = _Layer
    layer = _Layer()
    layer.disable_adapters = layer.merged = layer.fan_in_fan_out = False
    layer.active_adapters = ["default"]
    layer.lora_variant = {}
    layer.base_layer = torch.nn.Linear(4, 4, dtype = torch.bfloat16).requires_grad_(False)
    a = torch.nn.Linear(4, 2, bias = False)
    b = torch.nn.Linear(2, 4, bias = False)
    # Tensor storage remains on CPU. Only the routing decision sees the NPU device.
    # The backend name normally becomes 'npu' when torch_npu is imported.
    device = SimpleNamespace(type = "npu")
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: device))
    monkeypatch.setattr(
        torch, "is_autocast_enabled", lambda device_type = "cuda": device_type == "npu"
    )
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device_type: torch.bfloat16)
    x = torch.zeros(2, 4, dtype = torch.bfloat16)
    result = torch.zeros_like(x)
    return namespace, layer, x, result, a, b


def test_qualified_policy(monkeypatch):
    ns, layer, x, result, a, b = _eligible_fixture(monkeypatch)
    for drop in (torch.nn.Identity(), torch.nn.Dropout(0)):
        assert ns["_npu_lora_can_skip_casts"](layer, x, result, a, b, drop, "default")


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "float32_input",
        "float32_result",
        "autocast_off",
        "autocast_fp16",
        "cuda_autocast",
        "compiling",
        "disabled",
        "merged",
        "transposed",
        "multiple",
        "variant",
        "base_fp32",
        "base_trainable",
        "weight_bf16",
        "bias",
        "dropout",
        "custom_dropout",
        "subclass",
    ],
)
def test_unsupported_policy_keeps_old_path(case, monkeypatch):
    ns, layer, x, result, a, b = _eligible_fixture(monkeypatch)
    drop = torch.nn.Identity()
    if case == "cpu":
        monkeypatch.setattr(
            torch.Tensor, "device", property(lambda self: SimpleNamespace(type = "cpu"))
        )
    elif case == "float32_input":
        x = x.float()
    elif case == "float32_result":
        result = result.float()
    elif case == "autocast_off":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: False)
    elif case == "autocast_fp16":
        monkeypatch.setattr(torch, "get_autocast_dtype", lambda *args: torch.float16)
    elif case == "cuda_autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: True)
    elif case == "compiling":
        ns["torch_is_compiling"] = lambda: True
    elif case == "disabled":
        layer.disable_adapters = True
    elif case == "merged":
        layer.merged = True
    elif case == "transposed":
        layer.fan_in_fan_out = True
    elif case == "multiple":
        layer.active_adapters.append("other")
    elif case == "variant":
        layer.lora_variant["default"] = object()
    elif case == "base_fp32":
        layer.base_layer.float()
    elif case == "base_trainable":
        layer.base_layer.requires_grad_(True)
    elif case == "weight_bf16":
        a.bfloat16()
    elif case == "bias":
        b = torch.nn.Linear(2, 4)
    elif case == "dropout":
        drop = torch.nn.Dropout(0.1)
    elif case == "custom_dropout":
        drop = lambda x: x
    elif case == "subclass":

        class Subclass(_Layer):
            pass

        layer.__class__ = Subclass
    assert not ns["_npu_lora_can_skip_casts"](layer, x, result, a, b, drop, "default")


def _forward_sources():
    from peft.tuners.lora.layer import Linear

    source = textwrap.dedent(inspect.getsource(Linear.forward))
    source = SYMBOLS["_patch_lora_update"](source)
    source = SYMBOLS["_patch_lora_input_cast"](source, force_float32 = False)
    source = SYMBOLS["_patch_lora_base_layer_input_cast"](source)
    rewritten = SYMBOLS["_patch_npu_lora_casts"](source)
    assert rewritten != source, "PEFT forward spelling changed; review the NPU rewrite"
    return source, rewritten


def _forwards():
    from peft.tuners.lora.layer import Linear

    baseline, candidate = _forward_sources()
    functions = []
    for source in (baseline, candidate):
        namespace = dict(inspect.getmodule(Linear).__dict__)
        exec(
            SYMBOLS["COMPILED_LORA_FORWARD"] + SYMBOLS["NPU_LORA_CASTS"] + source,
            namespace,
        )
        functions.append(namespace["forward"])
    return functions


def test_source_rewrite_is_conservative():
    source, rewritten = _forward_sources()
    compile(rewritten, "<npu lora>", "exec")
    for unknown in (
        source.replace("lora_forward(", "different_update("),
        source + source,
        "def forward(self, x): return x",
    ):
        assert SYMBOLS["_patch_npu_lora_casts"](unknown) == unknown
    assert SYMBOLS["_patch_npu_lora_casts"](rewritten) == rewritten


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_cpu_forward_and_gradients_unchanged(dtype, bias, autocast):
    from peft.tuners.lora.layer import Linear

    torch.manual_seed(3407)
    layer = Linear(
        torch.nn.Linear(16, 8, dtype = dtype).requires_grad_(False),
        "default",
        r = 4,
        lora_alpha = 2,
    )
    layer.lora_A.default.float()
    layer.lora_B.default.float()
    if bias:
        layer.lora_B.default.bias = torch.nn.Parameter(torch.randn(8))
    torch.nn.init.normal_(layer.lora_B.default.weight)
    x = torch.randn(2, 7, 16, dtype = dtype, requires_grad = True)
    values = []
    for forward in _forwards():
        with torch.autocast("cpu", dtype = torch.bfloat16, enabled = autocast):
            y = forward(layer, x)
        grads = torch.autograd.grad(
            y.float().sum(),
            (x, *layer.lora_A.default.parameters(), *layer.lora_B.default.parameters()),
        )
        values.append((y, *grads))
    for a, b in zip(*values):
        assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_NPU") != "1", reason = "opt-in actual NPU lane"
)
@pytest.mark.parametrize("length", [1, 71, 80, 128, 223, 457, 512])
@pytest.mark.parametrize("width", [128, 896])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_real_npu_output_and_all_gradients(length, width, scale):
    import torch_npu  # noqa: F401
    from peft.tuners.lora.layer import Linear

    torch.manual_seed(3407)
    base = torch.nn.Linear(
        896, width, bias = width == 128, device = "npu", dtype = torch.bfloat16
    ).requires_grad_(False)
    layer = Linear(base, "default", r = 8, lora_alpha = int(8 * scale))
    layer.lora_A.default.float()
    layer.lora_B.default.float()
    with torch.no_grad():
        layer.lora_A.default.weight.normal_(0, 0.02)
        layer.lora_B.default.weight.normal_(0, 0.02)
    x = torch.randn(
        1, length, 896, device = "npu", dtype = torch.bfloat16, requires_grad = True
    )
    dy = torch.randn(1, length, width, device = "npu", dtype = torch.bfloat16)
    values = []
    baseline, candidate = _forwards()
    eligible = candidate.__globals__["_npu_lora_can_skip_casts"]
    routes = []

    def record_route(*args):
        result = eligible(*args)
        routes.append(result)
        return result

    candidate.__globals__["_npu_lora_can_skip_casts"] = record_route
    for forward in (baseline, candidate):
        with torch.autocast("npu", dtype = torch.bfloat16, cache_enabled = False):
            y = forward(layer, x)
        grads = torch.autograd.grad(
            y, (x, layer.lora_A.default.weight, layer.lora_B.default.weight), dy
        )
        values.append([t.detach().cpu().contiguous() for t in (y, *grads)])
    for a, b in zip(*values):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        )

    assert routes == [True], "parity must exercise the optimized path"


def test_registration_handles_peft_linear_import_aliases(monkeypatch):
    # PEFT's bnb module can re-export layer.Linear and be discovered first.
    # Use the real registration function, but capture its generated module.
    import peft.tuners.lora.layer
    from peft.tuners.lora.layer import Linear

    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "unsloth_zoo/compiler.py").read_text()
    )
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "patch_lora_forwards"
    )
    captured = []
    namespace = dict(SYMBOLS)
    namespace.update(
        inspect = inspect,
        get_lora_layer_modules = lambda: [(Linear, "peft.tuners.lora", "Linear")],
    )

    def capture(name, source, *args, **kwargs):
        captured.append(source)

        def unsloth_forward(*args, **kwargs):
            pass

        return SimpleNamespace(unsloth_forward = unsloth_forward)

    namespace["create_new_function"] = capture
    exec(
        compile(ast.Module(body = [function], type_ignores = []), "<registration>", "exec"),
        namespace,
    )
    original = Linear.forward
    monkeypatch.setattr(Linear, "forward", original)
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    # Register through the alias, then restore the real class even on failure.
    try:
        namespace["patch_lora_forwards"]({})
    finally:
        Linear.forward = original
    assert len(captured) == 1
    assert "def _npu_lora_can_skip_casts" in captured[0]
    assert "skip_npu_casts = unsloth_skip_npu_casts" in captured[0]


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_NPU") != "1", reason = "opt-in actual NPU lane"
)
def test_real_npu_avoids_float32_activation_cast_requests(monkeypatch):
    import torch_npu  # noqa: F401
    from peft.tuners.lora.layer import Linear

    base = torch.nn.Linear(896, 128, device = "npu", dtype = torch.bfloat16).requires_grad_(
        False
    )
    layer = Linear(base, "default", r = 8, lora_alpha = 8)
    layer.lora_A.default.float()
    layer.lora_B.default.float()
    x = torch.randn(1, 80, 896, device = "npu", dtype = torch.bfloat16)
    counts = []
    real_to = torch.Tensor.to
    for forward in _forwards():
        requests = []

        def record_to(value, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else None)
            if (
                dtype == torch.float32
                and value.dtype == torch.bfloat16
                and value.device.type == "npu"
            ):
                requests.append(value.shape)
            return real_to(value, *args, **kwargs)

        with monkeypatch.context() as patch, torch.autocast(
            "npu", dtype = torch.bfloat16, cache_enabled = False
        ):
            patch.setattr(torch.Tensor, "to", record_to)
            forward(layer, x)
        counts.append(len(requests))
    assert counts == [2, 0], counts
