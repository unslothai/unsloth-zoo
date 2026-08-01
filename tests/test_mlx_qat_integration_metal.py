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

"""QAT against the real product paths: merged_4bit save/reload, MLXTrainer, CPT.

tests/test_mlx_qat.py proves the QAT forward matches ``LoRALinear.fuse()``.
That is necessary but not sufficient: the shipped path is
``save_pretrained_merged(save_method='merged_4bit')`` -> ``_fuse_mlx_module``
-> ``save_model`` + config metadata -> reload. If anything in that chain
differed, the fuse-level tests would still pass while the feature did nothing.
These tests run the chain end to end on real Metal.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from unsloth_zoo.mlx.loader import FastMLXModel
from unsloth_zoo.mlx.qat import mlx_qat_module_count

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
# Unquantized source, for the rejection cases that need one.
FP_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def _require_real_metal():
    import mlx.core as _mx
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


def _ids():
    return mx.array([[3, 11, 19, 27, 35, 43, 51, 59] * 4])


def _loss(model, ids):
    logits = model(ids).astype(mx.float32)
    return nn.losses.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    ).mean()


def _train_then_save_and_reload(qat, steps=15, save_method="merged_4bit", seed=0):
    """Train briefly, save through the product path, reload, report losses.

    Seeds before loading so the QAT and non-QAT arms start from identical LoRA
    initialisations. Without this the two calls run sequentially in one
    process, inheriting whatever RNG state earlier tests left behind, and
    "QAT's saved model is better" would be a claim about random init.
    """
    ids = _ids()
    mx.random.seed(seed)
    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=64)
    kwargs = {"qat_scheme": "auto"} if qat else {}
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing=False, **kwargs,
    )
    if qat:
        assert mlx_qat_module_count(model) > 0, "qat_scheme did not patch anything"

    optimizer = optim.Adam(learning_rate=1e-4)
    step = nn.value_and_grad(model, lambda m: _loss(m, ids))
    for _ in range(steps):
        _, grads = step(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    before = float(_loss(model, ids))
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method=save_method)
        reloaded, _ = FastMLXModel.from_pretrained(directory, max_seq_length=64)
        after = float(_loss(reloaded, ids))
    return before, after


def test_qat_survives_the_merged_4bit_save_round_trip():
    """The QAT forward must predict the reloaded model, not just fuse()."""
    before, after = _train_then_save_and_reload(qat=True)
    assert after == pytest.approx(before, abs=5e-3), (
        f"QAT loss {before:.6f} did not survive merged_4bit save/reload "
        f"({after:.6f}); QAT is simulating a different quantizer than the "
        "one save_pretrained_merged writes"
    )


def test_qat_saved_model_beats_the_non_qat_saved_model():
    """The comparison that ships: both saved and reloaded, QAT wins.

    Both arms are seeded identically so the only difference is QAT itself.
    """
    base_before, base_after = _train_then_save_and_reload(qat=False, seed=0)
    qat_before, qat_after = _train_then_save_and_reload(qat=True, seed=0)

    # The non-QAT run must actually lose something, else there is nothing to fix.
    assert base_after - base_before > 10 * (qat_after - qat_before), (
        f"expected merged_4bit to degrade the non-QAT run "
        f"({base_before:.6f} -> {base_after:.6f}) far more than the QAT run "
        f"({qat_before:.6f} -> {qat_after:.6f})"
    )
    assert qat_after < base_after, (
        f"QAT saved model ({qat_after:.6f}) should beat the non-QAT saved "
        f"model ({base_after:.6f})"
    )


def test_save_method_lora_still_exports_adapters_under_qat():
    """Adapter-only export must not be confused by the QAT subclass.

    ``collect_mlx_lora_adapter_tensors`` keys on ``type(module).__name__``, so
    the stand-in class has to keep the original name.
    """
    from unsloth_zoo.mlx.utils import collect_mlx_lora_adapter_tensors

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=64)
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0,
        qat_scheme="auto", use_gradient_checkpointing=False,
    )
    tensors = collect_mlx_lora_adapter_tensors(model)
    assert tensors, "no adapter tensors collected while QAT was active"
    assert any("lora_a" in k for k in tensors)
    assert any("lora_b" in k for k in tensors)

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method="lora")
        import os
        assert any(f.endswith(".safetensors") for f in os.listdir(directory))


def test_qat_runs_through_mlx_trainer():
    """The trainer adds mx.compile and the shape guard on top of the forward."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0, qat_scheme="auto",
    )
    patched = mlx_qat_module_count(model)
    assert patched > 0

    with tempfile.TemporaryDirectory() as directory:
        trainer = MLXTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=[
                {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."}
                for i in range(16)
            ],
            args=MLXTrainingConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=6,
                learning_rate=5e-4,
                logging_steps=1,
                output_dir=directory,
                seed=3407,
                report_to="none",
            ),
        )
        trainer.train()

    history = trainer._train_loss_history
    assert len(history) >= 4, f"only {len(history)} logged losses"
    assert all(loss == loss and abs(loss) != float("inf") for loss in history), (
        f"non-finite losses under QAT: {history}"
    )
    # QAT must still be installed after training: a compile path that rebuilt
    # the modules would silently drop the fake-quant.
    assert mlx_qat_module_count(model) == patched


def test_full_finetuning_request_is_not_silently_dropped():
    """Every guard must be reachable through the public entry point.

    get_peft_model returns at its full_finetuning branch before the QAT hook,
    so a guard that only lived inside apply_mlx_qat was unreachable here and
    the request was silently ignored -- the model trained with no QAT at all
    and nothing said so.
    """
    from unsloth_zoo.mlx.qat import mlx_qat_module_count

    model, _ = FastMLXModel.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", max_seq_length=64, full_finetuning=True)
    with pytest.raises(NotImplementedError, match="full_finetuning"):
        FastMLXModel.get_peft_model(
            model, r=8, lora_alpha=16, qat_scheme="auto")
    assert mlx_qat_module_count(model) == 0


def _lora_layers_installed(model):
    from mlx_lm.tuner.lora import LoRALinear
    return sum(
        1 for _, module in model.named_modules()
        if isinstance(module, LoRALinear)
    )


# Every way a QAT request can be refused. The property under test is
# structural: validation of the would-be LoRA targets happens before any of
# them are replaced, so no rejection can leave adapters attached or
# trainability changed. Parametrized rather than written out per case so a new
# rejection added later without a preflight fails here.
_REFUSAL_CASES = [
    ("bit-width mismatch", MODEL, {}, {"lora_dropout": 0, "qat_scheme": "int8"}, ValueError),
    ("lora_dropout > 0", MODEL, {}, {"lora_dropout": 0.1, "qat_scheme": "auto"}, NotImplementedError),
    ("unquantized base", FP_MODEL, {"load_in_16bit": True, "load_in_4bit": False},
     {"lora_dropout": 0, "qat_scheme": "auto"}, ValueError),
    ("full_finetuning", FP_MODEL, {"full_finetuning": True}, {"qat_scheme": "auto"}, NotImplementedError),
    ("torchao-only scheme", MODEL, {}, {"lora_dropout": 0, "qat_scheme": "fp8-fp8"}, NotImplementedError),
    ("unknown scheme", MODEL, {}, {"lora_dropout": 0, "qat_scheme": "int3"}, ValueError),
    ("non-string scheme", MODEL, {}, {"lora_dropout": 0, "qat_scheme": 4}, TypeError),
]


@pytest.mark.parametrize(
    "label,model_name,load_kwargs,peft_kwargs,expected",
    _REFUSAL_CASES,
    ids=[case[0] for case in _REFUSAL_CASES],
)
def test_refused_requests_leave_the_model_unmutated(
    label, model_name, load_kwargs, peft_kwargs, expected,
):
    from mlx.utils import tree_flatten

    model, _ = FastMLXModel.from_pretrained(
        model_name, max_seq_length=64, **load_kwargs)

    def trainable_size():
        return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    # A freshly loaded model has everything trainable until get_peft_model
    # freezes, so "unmutated" is unchanged-from-before, not zero.
    before = trainable_size()
    with pytest.raises(expected):
        FastMLXModel.get_peft_model(
            model, r=8, lora_alpha=16,
            use_gradient_checkpointing=False, **peft_kwargs,
        )
    installed = _lora_layers_installed(model)
    assert installed == 0, f"{label}: {installed} LoRA layers left behind"
    assert trainable_size() == before, f"{label}: trainability changed"


def test_vlm_is_gated_off():
    """VLMs are refused for now: the merge path is unvalidated, not broken.

    All VLM LoRA layers (196 language + 64 vision tower with train_vision=True
    on Qwen2-VL-2B) are the same LoRALinear-over-QuantizedLinear that QAT
    handles, so this is a coverage gate. It exists so nobody gets an unverified
    numeric claim; lifting it needs a VLM train -> merged_4bit -> reload check.
    """
    VLM = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, _ = FastMLXModel.from_pretrained(VLM, max_seq_length=64)
    with pytest.raises(NotImplementedError, match="VLM"):
        FastMLXModel.get_peft_model(
            model, r=8, lora_alpha=16, lora_dropout=0,
            qat_scheme="auto", use_gradient_checkpointing=False,
        )


def test_qat_and_cpt_full_modules_are_mutually_exclusive():
    """QAT + continued-pretraining full modules cannot co-occur, by construction.

    CPT trains embed_tokens / lm_head as whole modules, which unsloth already
    rejects on a quantized base (the CCE backward zeroes the grad of a
    quantized weight). QAT in turn requires a quantized base -- there is
    nothing to fake-quantize otherwise. So the combination is impossible from
    both directions, and the user must get a clear error rather than a model
    that silently trains nothing.
    """
    # Direction 1: quantized base -> CPT rejects (pre-existing guard, fires
    # before QAT is reached).
    model, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=64)
    with pytest.raises(ValueError, match="quantized"):
        FastMLXModel.get_peft_model(
            model,
            target_modules=["all-linear", "embed_tokens", "lm_head"],
            r=8, lora_alpha=16, lora_dropout=0,
            modules_to_save=["embed_tokens", "lm_head"],
            qat_scheme="auto", use_gradient_checkpointing=False,
        )

    # Direction 2: an unquantized base is what CPT wants, and there QAT itself
    # refuses, because merged_4bit would not requantize anything.
    from unsloth_zoo.mlx.qat import apply_mlx_qat
    plain, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=64)
    plain = FastMLXModel.get_peft_model(
        plain, r=8, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing=False,
    )
    # Strip the quantization from the LoRA bases, then confirm QAT declines
    # rather than silently doing nothing.
    import mlx.nn as _nn
    from mlx_lm.tuner.lora import LoRALinear
    for _, module in plain.named_modules():
        if isinstance(module, LoRALinear) and isinstance(
                module.linear, _nn.QuantizedLinear):
            dq = _nn.Linear(
                module.linear.weight.shape[1] * 32 // module.linear.bits,
                module.linear.weight.shape[0],
                bias="bias" in module.linear,
            )
            dq.weight = mx.dequantize(
                module.linear.weight, module.linear.scales, module.linear.biases,
                group_size=module.linear.group_size, bits=module.linear.bits,
                mode=module.linear.mode,
            )
            module.linear = dq
    # The model still holds quantized modules the LoRA targets do not cover
    # (a quantized embedding), so the model-level backstop passes and the
    # target-level check is what refuses -- naming the unquantized targets.
    with pytest.raises(ValueError, match="quantized LoRA targets"):
        apply_mlx_qat(plain, "auto")
