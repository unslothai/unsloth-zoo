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

"""Vision GRPO through grpo_accumulated_loss: the fused forward-gradient route (fused_linear_grpo_loss) against the
log-prob route on a Qwen2-VL shaped batch (flattened patch rows, image_grid_thw, num_images with 0 / 1 / 2 images
per row), and both against an unchunked pass and a per-row float64 reference: per-token ratios, loss and every
gradient, the vision projector's included."""
import os
import types

import pytest
import torch

import unsloth_zoo.fused_losses.fused_grpo as fg
import unsloth_zoo.fused_losses.fused_linear as fl
import unsloth_zoo.rl_replacements as rr

H, V, P, LTK, IMG = 8, 13, 6, 4, 1
DEVICE = torch.device(os.environ.get("GRPO_FUSED_ROUTE_DEVICE", "cpu"))
# Prompt 4 (IMG = 1 marks an image slot) + completion 4, pad 0; rows 1 and 3 left padded, rows 0 and 2 end early.
INPUT_IDS = torch.tensor([
    [4, IMG, 6, 7, 8, 9, 3, 0],
    [0, IMG, IMG, 2, 10, 11, 12, 7],
    [4, 9, 6, 7, 2, 5, 0, 0],
    [0, 0, IMG, 11, 12, 3, 5, 9],
    [IMG, 5, 6, 2, 3, 5, 8, 6],
])
ROWS = INPUT_IDS.shape[0]
NUM_IMAGES = (INPUT_IDS == IMG).sum(1).tolist()  # [1, 2, 0, 1, 1]
GRID = torch.tensor([[1, 2, 2], [1, 1, 2], [1, 2, 3], [1, 1, 1], [1, 2, 1]])  # rows per image 4, 2, 6, 1, 2
PIXELS = torch.randn(int(GRID.prod(-1).sum()), P, generator = torch.Generator().manual_seed(3))


class _VLM(torch.nn.Module):
    """Images enter at their IMG slots through a trainable projector; a causal running mean carries them into the
    completion, so every completion log-prob depends on its own row's images and on no other row's."""

    def __init__(self, dtype = torch.float32):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.visual = torch.nn.Linear(P, H)
        self.mix = torch.nn.Linear(H, H)
        self.head = torch.nn.Linear(H, V, bias = False)
        self.head.weight.requires_grad_(False)  # LoRA: frozen head
        self.to(dtype)
        self.device = torch.device("cpu")
        self.forwards = []

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, pixel_values = None,
                image_grid_thw = None, **kwargs):
        x = self.emb(input_ids)
        slots = (input_ids == IMG).nonzero()
        n_img = 0 if image_grid_thw is None else image_grid_thw.shape[0]
        # A chunk must carry exactly its own rows' images, in order
        assert n_img == slots.shape[0], (n_img, slots.shape[0])
        self.forwards.append(n_img)
        if n_img:
            rows = image_grid_thw.prod(-1).tolist()
            assert sum(rows) == pixel_values.shape[0]
            feats = torch.stack([self.visual(p).mean(0) for p in pixel_values.split(rows)])
            x = x.index_put((slots[:, 0], slots[:, 1]), feats.to(x.dtype), accumulate = True)
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        count = torch.arange(1, x.shape[1] + 1, device = x.device, dtype = x.dtype)[None, :, None]
        hidden = torch.tanh(self.mix(x.cumsum(1) / count)) * 2.0
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _trainer(model, mini_batch, beta, entropy_coef):
    args = types.SimpleNamespace(
        unsloth_grpo_mini_batch = mini_batch, unsloth_logit_chunk_multiplier = 1, steps_per_generation = 2,
        use_bias_correction_kl = False, top_entropy_quantile = 1.0, entropy_coef = entropy_coef,
        use_adaptive_entropy = False, vllm_importance_sampling_mode = "token_truncate",
        vllm_importance_sampling_cap = 3.0, sapo_temperature_pos = 1.0, sapo_temperature_neg = 1.05,
        off_policy_mask_threshold = None,
    )
    return types.SimpleNamespace(
        args = args, processing_class = types.SimpleNamespace(pad_token_id = 0), model = model,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False, vllm_importance_sampling_correction = False, _autocast_dtype = None,
        beta = beta, entropy_coef = entropy_coef,
    )


def _reference_logps(dtype = torch.float64):
    """Each row on its own, in float64: [rows, LTK] completion log-probs."""
    model = _VLM(dtype)
    out = torch.zeros(ROWS, LTK, dtype = dtype)
    offsets = torch.tensor([0] + NUM_IMAGES).cumsum(0)
    row_off = torch.cat([torch.zeros(1, dtype = torch.long), GRID.prod(-1).cumsum(0)])
    with torch.no_grad():
        for r in range(ROWS):
            a, b = int(offsets[r]), int(offsets[r + 1])
            ids = INPUT_IDS[r : r + 1]
            kw = {}
            if b > a:
                kw = dict(pixel_values = PIXELS[int(row_off[a]) : int(row_off[b])].to(dtype), image_grid_thw = GRID[a:b])
            h = model(ids, attention_mask = (ids != 0).long(), logits_to_keep = LTK + 1, **kw).logits[:, :-1]
            out[r] = torch.log_softmax(model.head(h), -1).gather(-1, ids[:, -LTK:, None])[0, :, 0]
    return out


REF_LOGPS = _reference_logps()
_g = torch.Generator().manual_seed(1)
OLD = (REF_LOGPS + 0.15 * torch.randn(REF_LOGPS.shape, generator = _g, dtype = torch.float64)).float()
REF = (REF_LOGPS + 0.3 * torch.randn(REF_LOGPS.shape, generator = _g, dtype = torch.float64)).float()
ADV = torch.linspace(-1.0, 1.2, ROWS)


def _run(fused, monkeypatch, *, mini_batch = 2, loss_type = "grpo", beta = 0.0, entropy_coef = 0.0):
    monkeypatch.setattr(fg, "GRPO_FWD_GRAD", fused)
    monkeypatch.setattr(rr, "GRPO_FUSED_WITHOUT_ENTROPY", True)
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    model = _VLM().to(DEVICE)
    model.device = DEVICE
    trainer = _trainer(model, mini_batch, beta, entropy_coef)
    completion_mask = (INPUT_IDS[:, -LTK:] != 0).long().to(DEVICE)
    kwargs = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28, max_completion_length = LTK,
        num_items_in_batch = completion_mask.sum().float(), current_gradient_accumulation_steps = 2,
        num_processes = 1, importance_sampling_level = "token", sampling_per_token_logps = None,
        pixel_values = PIXELS.to(DEVICE), image_grid_thw = GRID.to(DEVICE), num_images = NUM_IMAGES,
    )
    before = fl.CALLS.get("grpo_fwd_grad", 0)
    out = rr.grpo_accumulated_loss(
        trainer, INPUT_IDS.to(DEVICE), (INPUT_IDS != 0).long().to(DEVICE), LTK, completion_mask,
        ADV.to(DEVICE), OLD.to(DEVICE), REF.to(DEVICE), **kwargs,
    )
    engaged = fl.CALLS.get("grpo_fwd_grad", 0) - before
    out[0].backward()
    grads = {n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None}
    return out, grads, engaged, model.forwards


def _close(a, b, rtol = 1e-5, atol = 1e-6):
    torch.testing.assert_close(a.detach().float().cpu(), b.detach().float().cpu(), rtol = rtol, atol = atol)


def test_route_decision_takes_vision_inputs():
    ok = dict(loss_type = "grpo")
    assert rr.grpo_fused_route_ok(ok, pixel_values = torch.ones(1))


@pytest.mark.parametrize("loss_type,beta,entropy_coef", [
    ("grpo", 0.0, 0.0), ("grpo", 0.04, 0.0), ("dr_grpo", 0.04, 0.01), ("dapo", 0.0, 0.0), ("bnpo", 0.1, 0.0),
])
def test_vision_batch_takes_the_fused_route_and_matches_the_log_prob_route(monkeypatch, loss_type, beta, entropy_coef):
    a, ga, n_fused, fwd_a = _run(True, monkeypatch, loss_type = loss_type, beta = beta, entropy_coef = entropy_coef)
    b, gb, n_base, fwd_b = _run(False, monkeypatch, loss_type = loss_type, beta = beta, entropy_coef = entropy_coef)
    assert n_fused == 1 and n_base == 0
    # mini_batch 2 over 5 rows: chunks [0:3] and [3:5], an uneven tail, images 3 then 2
    assert fwd_a == fwd_b == [3, 2]
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        elif x.numel():
            _close(x, y)
    assert ga.keys() == gb.keys() and {"visual.weight", "visual.bias", "mix.weight", "emb.weight"} <= ga.keys()
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], rtol = 1e-4, atol = 1e-6, msg = name)
    assert ga["visual.weight"].abs().sum() > 0


@pytest.mark.parametrize("fused", [True, False], ids = ["fused", "log_prob"])
def test_uneven_chunks_match_one_chunk_and_the_per_row_reference(monkeypatch, fused):
    # mini_batch 1 = one chunk of 5 rows; 3 = chunks of 2, 2, 1; 2 = chunks of 3, 2
    runs = {mb: _run(fused, monkeypatch, mini_batch = mb, beta = 0.04) for mb in (1, 2, 3)}
    assert runs[1][3] == [5] and runs[2][3] == [3, 2] and runs[3][3] == [3, 1, 1]
    one, g1 = runs[1][0], runs[1][1]
    for mb in (2, 3):
        out, g, _, _ = runs[mb]
        _close(out[0], one[0])
        _close(out[5], one[5])  # coef_1 = exp(new - old) per token
        for name in g1:
            torch.testing.assert_close(g[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = f"{mb}:{name}")
    # The per-token ratios are those of each row run on its own images, in float64
    mask = INPUT_IDS[:, -LTK:] != 0
    new = torch.log(one[5].detach().double().cpu()) + OLD.double()
    torch.testing.assert_close(new[mask], REF_LOGPS[mask], rtol = 1e-4, atol = 1e-5)
