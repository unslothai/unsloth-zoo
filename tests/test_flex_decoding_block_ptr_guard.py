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

_T = "_inductor/kernel/flex/templates/"


def _templates(decode, **overrides):
    files = {_T + name + ".py.jinja": _DESCRIPTOR for name in ("flex_attention", "common", "flex_backwards", "utilities")}
    files[_T + "flex_decode.py.jinja"] = decode
    for name, text in overrides.items():
        if text is None:
            files.pop(_T + name + ".py.jinja")
        else:
            files[_T + name + ".py.jinja"] = text
    return files


# Layouts of the real torch releases (pytorch tags): 2.6 - 2.8 keep the main kernel in kernel/flex_attention.py with
# block pointers; 2.9 - 2.12 move it to templates with tensor descriptors and only flex_decode keeps block pointers.
# The unknown rows: a 2.12 tree with a main template member missing reads as None, never as free of block pointers.
_TORCH_LAYOUTS = {
    "2.7": ({"_inductor/kernel/flex_attention.py": _BLOCK_PTR, "_inductor/kernel/flex_decoding.py": _BLOCK_PTR},
            True, True),
    "2.12": (_templates(_BLOCK_PTR), True, False),
    "2.13": (_templates(_DESCRIPTOR), False, False),
    "2.12-no-common": (_templates(_BLOCK_PTR, common = None), True, None),
    "2.12-no-backwards": (_templates(_BLOCK_PTR, flex_backwards = None), True, None),
    "2.12-backwards-block-ptr": (_templates(_BLOCK_PTR, flex_backwards = _BLOCK_PTR), True, True),
    "empty": ({"_inductor/kernel/other.py": ""}, None, None),
}


@pytest.mark.parametrize("version", sorted(_TORCH_LAYOUTS))
def test_template_probes_tell_main_and_decode_kernels_apart(tmp_path, monkeypatch, version):
    files, decode_uses, main_uses = _TORCH_LAYOUTS[version]
    spec = _fake_torch_root(tmp_path, None, files)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: spec if name == "torch" else real(name, *a))
    assert U._flex_decode_template_uses_block_ptr() is decode_uses
    assert U._flex_main_template_uses_block_ptr() is main_uses


def test_unreadable_main_template_is_unknown(tmp_path, monkeypatch):
    files = _templates(_BLOCK_PTR)
    spec = _fake_torch_root(tmp_path, None, files)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: spec if name == "torch" else real(name, *a))
    import builtins
    real_open = builtins.open

    def failing_open(path, *a, **k):
        if str(path).endswith("common.py.jinja"):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", failing_open)
    assert U._flex_decode_template_uses_block_ptr() is True
    assert U._flex_main_template_uses_block_ptr() is None


@pytest.mark.parametrize("decode, main, expected", [
    (True, False, True), (True, None, False), (True, True, False), (None, False, False), (False, False, False),
])
def test_unknown_template_never_turns_the_workaround_on(decode, main, expected):
    assert U._flex_decoding_unavailable(True, decode, main) is expected
    assert U._flex_decoding_unavailable(False, decode, main) is False


def _removed(base, shape, strides, offsets, block_shape, order, _semantic = None):
    raise NotImplementedError("Block pointers have been removed in favor of the tensor descriptor API")


def _working(base, shape, strides, offsets, block_shape, order, _semantic = None):
    return base


@pytest.mark.parametrize("public, core_fn, removed", [
    (_removed, _removed, True),
    (_working, _working, False),
    # A shim (SGLang) restores only the public name, which is what generated kernels call.
    (_working, _removed, False),
    (_removed, _working, True),
])
def test_triton_probe_reads_the_public_make_block_ptr(monkeypatch, public, core_fn, removed):
    tl = pytest.importorskip("triton.language")
    core = pytest.importorskip("triton.language.core")
    monkeypatch.setattr(tl, "make_block_ptr", public)
    monkeypatch.setattr(core, "make_block_ptr", core_fn)
    assert U._triton_block_ptr_removed() is removed


def test_triton_without_make_block_ptr_counts_as_removed(monkeypatch):
    tl = pytest.importorskip("triton.language")
    monkeypatch.delattr(tl, "make_block_ptr", raising = False)
    assert U._triton_block_ptr_removed() is True


def test_installed_triton_and_torch_agree_with_the_flag():
    # Forcing the main kernel only helps when the main kernel is known to have no block pointers (torch 2.9+).
    removed = bool(U.HAS_FLEX_ATTENTION) and U._triton_block_ptr_removed()
    assert U.FLEX_DECODING_UNAVAILABLE == U._flex_decoding_unavailable(
        removed, U._flex_decode_template_uses_block_ptr(), U._flex_main_template_uses_block_ptr()
    )
    if U.FLEX_DECODING_UNAVAILABLE:
        assert U.flex_attention.__name__ == "flex_attention" and U.flex_attention.__qualname__.startswith(
            "_main_kernel_for_short_queries"
        )


def _recording_call(seen):
    def fake_flex(*args, kernel_options = None, **kwargs):
        seen.append((args, kernel_options, kwargs))
        return "out"
    return fake_flex


@pytest.mark.parametrize("requested, expected", [
    (None, {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}),
    ({}, {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}),
    ({"BLOCK_M": 16}, {"BLOCK_M": 16, "FORCE_USE_FLEX_ATTENTION": True}),
    ({"FORCE_USE_FLEX_ATTENTION": False}, {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": False}),
    ({"BACKEND": "TRITON_DECODE"}, {"BACKEND": "TRITON_DECODE"}),
])
def test_generic_flex_attention_takes_the_main_kernel_when_flex_decoding_cannot_compile(requested, expected):
    seen = []
    forced = {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    guarded = U._main_kernel_for_short_queries(_recording_call(seen), forced)
    q = torch.randn(1, 2, 1, 8)
    kwargs = {} if requested is None else {"kernel_options": requested}
    assert guarded(q, q, q, scale = 0.5, **kwargs) == "out"
    (args, options, rest), = seen
    assert options == expected and rest == {"scale": 0.5} and args[0] is q
    assert forced == {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}


def _strict_flex(seen):
    # torch's own signature, so a kernel_options given twice fails as it would there.
    def fake_flex(query, key, value, score_mod = None, block_mask = None, scale = None, enable_gqa = False,
                  return_lse = False, kernel_options = None, *, return_aux = None):
        seen.append((scale, kernel_options))
        return "out"
    return fake_flex


def test_generic_flex_attention_accepts_kernel_options_positionally():
    seen = []
    forced = {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    guarded = U._main_kernel_for_short_queries(_strict_flex(seen), forced)
    q = torch.randn(1, 2, 1, 8)
    assert guarded(q, q, q, None, None, 0.5, False, False, {"BLOCK_M": 16}) == "out"
    assert guarded(q, q, q, None, None, 0.5, False, False, None) == "out"
    assert guarded(q, q, q, None, None, 0.5, kernel_options = {"BLOCK_N": 8}) == "out"
    assert guarded(q, q, q) == "out"
    assert seen == [
        (0.5, {"BLOCK_M": 16, "FORCE_USE_FLEX_ATTENTION": True}),
        (0.5, {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}),
        (0.5, {"BLOCK_M": 32, "BLOCK_N": 8, "FORCE_USE_FLEX_ATTENTION": True}),
        (None, {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}),
    ]
    with pytest.raises(TypeError):
        guarded(q, q, q, None, None, 0.5, False, False, {"BLOCK_M": 16}, kernel_options = {"BLOCK_M": 8})


def test_kernel_options_position_matches_installed_torch():
    import inspect
    from torch.nn.attention.flex_attention import flex_attention as torch_flex
    assert list(inspect.signature(torch_flex).parameters)[U._kernel_options_position()] == "kernel_options"


@pytest.mark.parametrize("requested, expected", [
    ({"BACKEND": "AUTO"}, {"BLOCK_M": 32, "BACKEND": "TRITON"}),
    ({"BACKEND": "AUTO", "BLOCK_M": 16}, {"BLOCK_M": 16, "BACKEND": "TRITON"}),
    ({"BACKEND": "TRITON"}, {"BACKEND": "TRITON"}),
    ({"BACKEND": "FLASH"}, {"BACKEND": "FLASH"}),
])
def test_explicit_auto_backend_takes_the_main_kernel_too(requested, expected):
    # torch rejects BACKEND together with FORCE_USE_FLEX_ATTENTION, so AUTO becomes TRITON instead.
    seen = []
    guarded = U._main_kernel_for_short_queries(_strict_flex(seen), {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True})
    q = torch.randn(1, 2, 1, 8)
    guarded(q, q, q, kernel_options = dict(requested))
    assert seen == [(None, expected)]


def test_decode_option_helpers_return_copies(monkeypatch):
    forced = {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    monkeypatch.setattr(U, "PREFILL_KERNEL_OPTIONS", forced)
    monkeypatch.setattr(U, "FLEX_DECODING_UNAVAILABLE", True)
    U.decode_kernel_options()["BLOCK_M"] = 1
    U.decode_kernel_kwargs()["kernel_options"]["BLOCK_N"] = 1
    assert forced == {"BLOCK_M": 32, "FORCE_USE_FLEX_ATTENTION": True}
    assert U.decode_kernel_options() == forced and U.decode_kernel_kwargs() == {"kernel_options": forced}


def test_triton_probe_does_not_follow_a_working_shim_to_the_removed_stub(monkeypatch):
    import functools
    tl = pytest.importorskip("triton.language")
    core = pytest.importorskip("triton.language.core")
    @functools.wraps(_removed)
    def shim(*args, **kwargs):
        return _working(*args, **kwargs)
    monkeypatch.setattr(tl, "make_block_ptr", shim)
    assert U._triton_block_ptr_removed() is False
    if hasattr(core, "builtin"):
        # Triton's own @builtin wrapper around the removed stub still counts as removed.
        monkeypatch.setattr(tl, "make_block_ptr", core.builtin(_removed))
        assert U._triton_block_ptr_removed() is True
        monkeypatch.setattr(tl, "make_block_ptr", functools.wraps(core.builtin(_removed))(shim))
        assert U._triton_block_ptr_removed() is False


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
