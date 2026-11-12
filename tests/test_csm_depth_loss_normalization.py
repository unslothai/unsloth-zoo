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

"""CsmProcessor(depth_decoder_labels_ratio < 1) marks backbone-only frames -101, so the depth
decoder's divisor is its own target count over the GA window, not backbone count * 31. CPU only."""
import os
import sys

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
modeling_csm = pytest.importorskip("transformers.models.csm.modeling_csm")
from test_csm_empty_depth_decoder_batch import _patched, _tiny_csm  # noqa: E402
from test_loss_normalization_contract import _fake_trainer, _loss_utils  # noqa: E402

NC = 4
KEY = "_unsloth_csm_depth_num_items_in_batch"


def _windows():
    # Processor layout: text -100, audio frames, -101 = backbone-only frame, eos, padding -100.
    g = torch.Generator().manual_seed(0)
    out = []
    for skip in ((3, 4), (2,)):
        labels = torch.full((1, 9), -100)
        labels[0, 1:7] = 100
        labels[0, 7] = 101
        labels[0, list(skip)] = -101
        codes = torch.randint(0, 60, (1, 9, NC), generator = g)
        codes3d = codes.clone()
        codes3d[labels == -100] = -100
        codes3d[..., 1:][labels == -101] = -100
        out.append({"labels": labels, "input_ids": torch.ones_like(labels), "codes": codes, "labels3d": codes3d})
    return out


def test_batch_samples_counts_depth_targets_separately():
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _patched()
    windows = _windows()
    samples = [{"labels": w["labels"], "input_ids": w["input_ids"]} for w in windows]
    trainer = _fake_trainer(_tiny_csm(), True)
    batch, num_items = mod._unsloth_get_batch_samples(trainer, iter(samples), len(samples))[:2]

    assert int(num_items) == 7 + 7
    depth = (7 - 2) + (7 - 1)
    assert all(int(s[KEY]) == depth * (NC - 1) for s in batch)


def _depth_losses(model, windows, **per_window):
    losses = []
    for w in windows:
        kw = {k: v(w) for k, v in per_window.items()}
        losses.append(model(input_ids = w["codes"], labels = w["labels3d"], **kw).depth_decoder_loss)
    return torch.stack(losses)


def test_forward_uses_window_depth_count():
    _patched()
    model = _tiny_csm()
    windows = _windows()
    frames = [int((w["labels3d"][..., 1:] != -100).all(-1).sum()) for w in windows]
    with torch.no_grad():
        means = _depth_losses(model, windows).double()
        depth = sum(frames) * (NC - 1)
        ref = sum(m * f * (NC - 1) for m, f in zip(means, frames)) / depth
        got = _depth_losses(
            model, windows,
            num_items_in_batch = lambda w: torch.tensor(14),
            **{KEY: lambda w: torch.tensor(depth)},
        ).sum()
    torch.testing.assert_close(got.double(), ref, rtol = 1e-5, atol = 0)


def test_direct_call_scales_by_num_codebooks():
    _patched()
    model = _tiny_csm()
    w = _windows()[1]
    frames = int((w["labels3d"][..., 1:] != -100).all(-1).sum())
    shared = torch.tensor(frames)
    with torch.no_grad():
        mean = _depth_losses(model, [w])[0]
        got = _depth_losses(model, [w], num_items_in_batch = lambda w: shared)[0]
    torch.testing.assert_close(got, mean)
    # Trainer hands every microbatch the same tensor: the backbone divisor must survive.
    assert int(shared) == frames


def test_distributed_depth_count_rides_the_single_gather():
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _patched()
    trainer = _fake_trainer(_tiny_csm(), True)
    trainer.args.average_tokens_across_devices = True
    trainer.args.world_size = 2
    calls = []

    def gather(x):
        calls.append(tuple(x.shape))
        return torch.cat([x, torch.tensor([10, 0, 0, 30], dtype = x.dtype)])
    trainer.accelerator.gather = gather

    w = _windows()[1]
    batch, num_items = mod._unsloth_get_batch_samples(trainer, iter([{"labels": w["labels"]}]), 1)[:2]
    assert calls == [(4,)]
    assert int(num_items) == 7 + 10
    assert int(batch[0][KEY]) == 6 * (NC - 1) + 30
