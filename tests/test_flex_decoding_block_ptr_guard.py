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

# Triton 3.9 removed block pointers; torch 2.6 - 2.12's flex_decoding template still builds them. Its compile then fails
# on the first decode, Dynamo skips flex_attention for the rest of the process and every later GPT-OSS prefill builds the
# full score matrix (AMD Windows, triton-windows 3.9: 4k prefill 105 MiB -> 6.7 GB). One-token decode must take the
# main flex kernel there instead.
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import pytest
import torch

pytest.importorskip("torch.nn.attention.flex_attention")

import unsloth_zoo.flex_attention.utils as U


def _fake_torch_root(tmp_path, template, files = None):
    root = tmp_path / "torch"
    files = files or {"_inductor/kernel/flex/templates/flex_decode.py.jinja": template}
    for rel, text in files.items():
        path = root.joinpath(*rel.split("/"))
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text)
    spec = importlib.machinery.ModuleSpec("torch", None, is_package = True)
    spec.submodule_search_locations = [str(root)]
    return spec


@pytest.mark.parametrize("uses", [True, False])
def test_template_probe_reads_the_installed_decode_template(tmp_path, monkeypatch, uses):
    spec = _fake_torch_root(tmp_path, "K_block_ptr = tl.make_block_ptr(" if uses else "desc = tl.make_tensor_descriptor(")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: spec if name == "torch" else real(name, *a))
    assert U._flex_decode_template_uses_block_ptr() is uses


_BLOCK_PTR, _DESCRIPTOR = "K_block_ptr = tl.make_block_ptr(", "desc = tl.make_tensor_descriptor("

# Layouts of the real torch releases (pytorch tags): 2.6 - 2.8 keep the main kernel in kernel/flex_attention.py with
# block pointers; 2.9 - 2.12 move it to templates with tensor descriptors and only flex_decode keeps block pointers.
_TORCH_LAYOUTS = {
    "2.7": ({"_inductor/kernel/flex_attention.py": _BLOCK_PTR, "_inductor/kernel/flex_decoding.py": _BLOCK_PTR},
            True, True),
    "2.12": ({"_inductor/kernel/flex/templates/flex_attention.py.jinja": _DESCRIPTOR,
              "_inductor/kernel/flex/templates/common.py.jinja": _DESCRIPTOR,
              "_inductor/kernel/flex/templates/flex_decode.py.jinja": _BLOCK_PTR}, True, False),
    "2.13": ({"_inductor/kernel/flex/templates/flex_attention.py.jinja": _DESCRIPTOR,
              "_inductor/kernel/flex/templates/common.py.jinja": _DESCRIPTOR,
              "_inductor/kernel/flex/templates/flex_decode.py.jinja": _DESCRIPTOR}, False, False),
}


@pytest.mark.parametrize("version", sorted(_TORCH_LAYOUTS))
def test_template_probes_tell_main_and_decode_kernels_apart(tmp_path, monkeypatch, version):
    files, decode_uses, main_uses = _TORCH_LAYOUTS[version]
    spec = _fake_torch_root(tmp_path, None, files)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: spec if name == "torch" else real(name, *a))
    assert U._flex_decode_template_uses_block_ptr() is decode_uses
    assert U._flex_main_template_uses_block_ptr() is main_uses


@pytest.mark.parametrize("removed", [True, False])
def test_triton_probe_sees_removed_block_pointers(monkeypatch, removed):
    core = pytest.importorskip("triton.language.core")
    if removed:
        def make_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic = None):
            raise NotImplementedError("Block pointers have been removed in favor of the tensor descriptor API")
    else:
        def make_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic = None):
            return base
    monkeypatch.setattr(core, "make_block_ptr", make_block_ptr)
    assert U._triton_block_ptr_removed() is removed


def test_installed_triton_and_torch_agree_with_the_flag():
    # Forcing the main kernel only helps when the main kernel itself has no block pointers (torch 2.9+).
    assert U.FLEX_DECODING_UNAVAILABLE == (
        bool(U.HAS_FLEX_ATTENTION) and U._triton_block_ptr_removed() and U._flex_decode_template_uses_block_ptr()
        and not U._flex_main_template_uses_block_ptr()
    )


@pytest.fixture
def S(monkeypatch):
    # attention_sink on any host: CPU-only hosts leave the CUDA-sized flex utils undefined, so stub the missing names.
    stub = types.ModuleType(U.__name__)
    stub.__dict__.update(U.__dict__)
    stub.__getattr__ = lambda name: None
    monkeypatch.setitem(sys.modules, U.__name__, stub)
    monkeypatch.delitem(sys.modules, "unsloth_zoo.flex_attention.attention_sink", raising = False)
    module = importlib.import_module("unsloth_zoo.flex_attention.attention_sink")
    monkeypatch.delitem(sys.modules, "unsloth_zoo.flex_attention.attention_sink")
    return module, stub


class _Attn(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.num_key_value_groups = 2
        self.scaling = 0.125
        self.sinks = torch.nn.Parameter(torch.zeros(heads))


def _block_mask(q_len, kv_len):
    # The routing only reads seq_lengths; a stub keeps these tests off BlockMask constructor differences across torch.
    return types.SimpleNamespace(seq_lengths = (q_len, kv_len))


def _recording_flex(seen):
    def fake_flex(query, key, value, block_mask = None, score_mod = None, enable_gqa = False, scale = None, **kwargs):
        seen.append(kwargs.get("kernel_options", "inherited"))
        return query * 1, query.float().sum(-1)
    return fake_flex


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("q_len", [1, 5])
def test_decode_takes_the_main_kernel_only_when_flex_decoding_cannot_compile(S, monkeypatch, unavailable, q_len):
    S, stub = S
    # decode_kernel_options reads its own module, the real utils.
    forced_options = {"FORCE_USE_FLEX_ATTENTION": True}
    monkeypatch.setattr(U, "PREFILL_KERNEL_OPTIONS", forced_options)
    monkeypatch.setattr(S, "PREFILL_KERNEL_OPTIONS", forced_options)
    monkeypatch.setattr(U, "FLEX_DECODING_UNAVAILABLE", unavailable)
    seen = []
    monkeypatch.setattr(S, "flex_attention", _recording_flex(seen))
    monkeypatch.setattr(S, "flex_output_and_lse", lambda output: output)
    q, k = torch.randn(2, 8, q_len, 16), torch.randn(2, 4, 128, 16)
    S.flex_attention_with_sink_inference(_Attn(8), q, k, k.clone(), _block_mask(q_len, 128))
    forced = q_len > 1 or unavailable
    assert seen == [forced_options if forced else {}], seen


@pytest.mark.parametrize("unavailable", [False, True])
def test_sink_decoding_keeps_inherited_tile_options_unless_flex_decoding_cannot_compile(S, monkeypatch, unavailable):
    S, stub = S
    forced_options = {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    monkeypatch.setattr(U, "PREFILL_KERNEL_OPTIONS", forced_options)
    monkeypatch.setattr(U, "FLEX_DECODING_UNAVAILABLE", unavailable)
    seen = []
    monkeypatch.setattr(S, "flex_attention", _recording_flex(seen))
    monkeypatch.setattr(S, "flex_output_and_lse", lambda output: output)
    attn = _Attn(8)
    attn._flex_attention_cache = lambda key: _block_mask(1, key.shape[-2])
    q, k = torch.randn(2, 8, 1, 16), torch.randn(2, 4, 128, 16)
    S.flex_attention_with_sink_decoding(attn, q, k, k.clone())
    # "inherited": no kernel_options kwarg, so the compiled flex_attention keeps its memory-sized tile options.
    assert seen == [forced_options if unavailable else "inherited"], seen


@pytest.mark.parametrize("unavailable", [False, True])
def test_sparse_flex_takes_the_main_kernel_only_when_flex_decoding_cannot_compile(monkeypatch, unavailable):
    import unsloth_zoo.flex_attention.sparse_blocks as SB
    forced_options = {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    monkeypatch.setattr(U, "PREFILL_KERNEL_OPTIONS", forced_options)
    monkeypatch.setattr(U, "FLEX_DECODING_UNAVAILABLE", unavailable)
    seen = []
    monkeypatch.setattr(U, "flex_attention", _recording_flex(seen))
    q, k = torch.randn(1, 8, 32, 16), torch.randn(1, 8, 128, 16)
    SB._flex(q, k, k.clone(), _block_mask(32, 128), 0.125)
    assert seen == [forced_options if unavailable else "inherited"], seen
