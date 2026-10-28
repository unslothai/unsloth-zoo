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

"""An OOM in the GRPO packed (grad) forward caps the packed length rather than disabling packing for the run; the real grpo_accumulated_loss block runs on CPU."""

import contextlib
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

import unsloth_zoo.rl_replacements as rr


VOCAB, HIDDEN = 17, 8
PAD_ID, KEEP = 0, 4


class _Model(torch.nn.Module):
    def __init__(self, error = None):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.error = error
        self.packed_calls = 0

    def forward(self, input_ids = None, position_ids = None,
                packed_seq_lengths = None, use_cache = None, **kwargs):
        if packed_seq_lengths is not None:
            self.packed_calls += 1
            if self.error is not None:
                error, self.error = self.error, None
                raise error
        return SimpleNamespace(logits = self.head(torch.tanh(self.emb(input_ids))))


class _OOMInVerifyModel(_Model):
    """The packed forward succeeds; the first per-row verifier forward runs out of memory."""

    def __init__(self):
        super().__init__()
        self.armed = True

    def forward(self, input_ids = None, packed_seq_lengths = None, **kwargs):
        if packed_seq_lengths is None and self.armed:
            self.armed = False
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return super().forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)


def _block_source():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.rindex("\n", 0, src.index("    new_logprobs = None")) + 1
    end = src.rindex("\n", 0, src.index("    # ---- PrefixGrouper resolution")) + 1
    return textwrap.dedent(src[start:end])


LONG = torch.tensor([
    [PAD_ID, PAD_ID, 3, 5, 7, 9, 11, 13],
    [PAD_ID, 2,      4, 6, 8, 10, 12, 14],
])
SHORT = torch.tensor([
    [PAD_ID, PAD_ID, PAD_ID, 5, 7, 9, 11, 13],
    [PAD_ID, PAD_ID, 4,      6, 8, 10, 12, 14],
])


def _run(model, input_ids, **overrides):
    left_pad = rr.calculate_pad_tokens_in_prompt(input_ids, KEEP, PAD_ID)
    max_left_pad = int(left_pad.max())
    ns = {
        "os": rr.os, "torch": torch,
        "chunked_hidden_states_selective_log_softmax":
            rr.chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": rr.chunked_selective_log_softmax,
        "_refresh_lm_head": lambda: None, "_fp32_bind": lambda entropy = False: None, "lm_head_bias": None,
        "create_completion_attention_mask": rr.create_completion_attention_mask,
        "device_synchronize": lambda *a, **k: None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "trainer": SimpleNamespace(processing_class = SimpleNamespace(pad_token_id = PAD_ID)),
        "unwrapped_model": model,
        "lm_head": model.head.weight,
        "input_ids": input_ids,
        "left_pad_tokens_per_prompt": left_pad,
        "max_left_pad": max_left_pad,
        "completion_mask": rr.create_completion_attention_mask(
            input_ids[:, -(KEEP + max_left_pad):], left_pad, max_left_pad, PAD_ID,
        ),
        "logits_to_keep": KEEP,
        "total_rows": input_ids.shape[0],
        "multiplier": 1,
        "autocaster": contextlib.nullcontext(),
        "pixel_values": None, "token_type_ids": None, "mm_token_type_ids": None,
        "_pg_skip_pack": False,
        "logit_scale_multiply": 0.0, "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0, "temperature": 1.0,
        "_need_entropy": False,
    }
    ns.update(overrides)
    exec(_block_source(), ns)
    return ns


def test_oom_keeps_packing_for_batches_that_fit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _Model(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    ns = _run(model, LONG)
    assert ns["_pack_use"] is False
    assert getattr(model, "_unsloth_seq_packing_grad_ok", None) is not False
    oom_T = ns["_pack_T"]
    assert model._unsloth_seq_packing_grad_unsafe_T == oom_T

    calls = model.packed_calls
    ns = _run(model, LONG)
    assert ns["_pack_use"] is False
    assert model.packed_calls == calls

    ns = _run(model, SHORT)
    assert ns["_pack_T"] < oom_T
    assert ns["_pack_use"] is True
    assert model.packed_calls == calls + 1


def test_non_oom_failure_still_disables_packing():
    model = _Model(RuntimeError("kernel rejected the packed layout"))
    ns = _run(model, LONG)
    assert ns["_pack_use"] is False
    assert model._unsloth_seq_packing_grad_ok is False


def test_oom_in_the_per_row_verifier_disables_packing(monkeypatch):
    # The packed forward completed, so the OOM says nothing about the packed length.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _OOMInVerifyModel()
    ns = _run(model, LONG)
    assert model.packed_calls == 1
    assert ns["_pack_use"] is False
    assert model._unsloth_seq_packing_grad_ok is False
    assert getattr(model, "_unsloth_seq_packing_grad_unsafe_T", None) is None


def test_oom_log_reports_the_cap_not_disabled(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _Model(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    ns = _run(model, LONG, UNSLOTH_ENABLE_LOGGING = True)
    out = capsys.readouterr().out
    assert f"capped below T={ns['_pack_T']}" in out
    assert "disabled" not in out

    model = _Model(RuntimeError("kernel rejected the packed layout"))
    _run(model, LONG, UNSLOTH_ENABLE_LOGGING = True)
    assert "disabled (fell back to padded)" in capsys.readouterr().out


def test_capped_batch_skips_packing_prep(monkeypatch):
    # A batch at or above the OOM cap decides from host lengths alone: no flatten of the ids and no active-row read,
    # since both cost a GPU sync and keep buffers alive.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = _Model(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    ns = _run(model, LONG)
    assert model._unsloth_seq_packing_grad_unsafe_T == ns["_pack_T"]
    ns = _run(model, LONG)
    assert ns["_pack_use"] is False
    assert ns["_pack_T"] == int((LONG != PAD_ID).sum())
    assert "_pack_flat_ids" not in ns
    assert "_pack_active" not in ns
    # below the cap the prep runs as before
    ns = _run(model, SHORT)
    assert ns["_pack_flat_ids"].shape[1] == ns["_pack_T"]
    assert "_pack_active" in ns


def test_padded_fallback_drops_packing_index_buffers():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.index("# packing rejected/unused")
    branch = src[start:src.index("def to_device", start)]
    for name in ("_pack_keep", "_pack_lengths", "_pack_flat_ids", "_pack_nz_idx", "_pack_ctgt"):
        assert f"{name} = " in branch or f"{name} =" in branch, name
