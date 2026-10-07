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

"""Fine-tuning for the MLX Laya decision model: trainable loading, LoRA adapters and the training loop."""

import contextlib
import copy
import math
import random
import re
import time
import types

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .decision import load_decision_model
from .utils import _mlx_rng_key, _restore_mlx_rng_key

__all__ = [
    "HEAD_LEARNING_RATE",
    "MLXDecisionTrainer",
    "add_lora_adapters",
    "clef_logits",
    "clef_training_network",
    "collate_decisions",
    "decision_logits",
    "load_trainable_decision_model",
]

HEAD_LEARNING_RATE = 1e-4
_ENCODER_LINEARS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")


def load_trainable_decision_model(folder, full_finetuning = False, gradient_checkpointing = True):
    """Load a Laya checkpoint for training.

    A full fine-tune trains every weight in float32. Otherwise the encoder is frozen with float16 matmuls,
    ready for `add_lora_adapters`, and only the float32 decision head trains.
    """
    # The trainer works on the network; the loaded pipeline around it only serves requests.
    model = load_decision_model(folder, compute_dtype = mx.float32 if full_finetuning else mx.float16).model
    model.freeze()
    if full_finetuning:
        model.unfreeze()
    else:
        for part in (model.head, model.type_emb, *model.scorer):
            part.set_dtype(mx.float32)
            part.unfreeze()
    model.gradient_checkpointing = bool(gradient_checkpointing)
    model.train()
    return model


def _targeted(name, target_modules):
    if target_modules == "all-linear":
        return True
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == target or name.endswith("." + target) for target in target_modules)


def add_lora_adapters(
    model,
    r = 64,
    lora_alpha = 64,
    lora_dropout = 0.0,
    use_rslora = False,
    target_modules = "all-linear",
    random_state = 3407,
):
    """Add trainable float32 LoRA adapters to the encoder's linear layers.

    `target_modules` selects them as PEFT does: `"all-linear"`, a list of names or dotted suffixes
    (`"Wqkv"`, `"mlp.Wo"`), or a regular expression matching the whole name (`"layers.0.attn.Wqkv"`).
    """
    from mlx_lm.tuner.lora import LoRALinear

    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")
    mx.random.seed(random_state)
    scale = lora_alpha / (math.sqrt(r) if use_rslora else r)
    matched = 0
    for index, layer in enumerate(model.encoder.layers):
        for path in _ENCODER_LINEARS:
            if not _targeted(f"layers.{index}.{path}", target_modules):
                continue
            parent, name = path.split(".")
            parent = getattr(layer, parent)
            setattr(parent, name, LoRALinear.from_base(getattr(parent, name), r = r, dropout = lora_dropout, scale = scale))
            matched += 1
    if not matched:
        raise ValueError(f"Unsloth: target_modules = {target_modules!r} matches no encoder linear layer.")
    model.train(model.training)
    return model


def collate_decisions(items, pad_token_id):
    """Pad tokenized decisions (`input_ids`, `markers`, `qtype`, `target`) into one batch of MLX arrays."""
    rows = len(items)
    length = max(len(item["input_ids"]) for item in items)
    options = max(len(item["markers"]) for item in items)
    batch = {
        "input_ids": np.full((rows, length), pad_token_id, np.int64),
        "attention_mask": np.zeros((rows, length), np.int64),
        "marker_pos": np.zeros((rows, options), np.int64),
        "marker_mask": np.zeros((rows, options), np.bool_),
        "qtype": np.array([item["qtype"] for item in items], np.int64),
        "target": np.zeros((rows, options), np.float32),
    }
    for i, item in enumerate(items):
        ids, markers = item["input_ids"], item["markers"]
        batch["input_ids"][i, : len(ids)] = ids
        batch["attention_mask"][i, : len(ids)] = 1
        batch["marker_pos"][i, : len(markers)] = markers
        batch["marker_mask"][i, : len(markers)] = True
        batch["target"][i, : len(item["target"])] = item["target"]
    return {name: mx.array(value) for name, value in batch.items()}


def _soft_cross_entropy(model, batch):
    logits = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
    return -(batch["target"] * nn.log_softmax(logits, axis = -1)).sum(-1).mean()


class _LayerwiseStep:
    """Soft cross-entropy and its gradients, with the encoder run and differentiated one layer at a time.

    One evaluation of the whole graph keeps every layer's temporaries alive until it ends, and MLX's buffer cache then
    holds that working set once per batch shape. Evaluating at each layer bounds it to a single layer's worth: only
    layer inputs are kept, and each layer is recomputed for its backward pass, as gradient checkpointing does.
    """

    def __init__(self, model, compile = False):
        self.model = model
        encoder = model.encoder
        modules = [encoder.embeddings, *encoder.layers]
        # A stage needs its input's gradient only if something trainable lies below it.
        trainable = [bool(tree_flatten(module.trainable_parameters())) for module in modules]
        self.first = trainable.index(True) if True in trainable else len(modules)
        state = [model.state, mx.random.state]
        wrap = (lambda fn: mx.compile(fn, inputs = state, outputs = state)) if compile else (lambda fn: fn)

        def stage(index, module):
            run = (lambda x, mask: module(x, mask)) if index else (lambda ids, mask: module(ids))
            argnums = [0, 1] if trainable[index] and index > self.first else 0 if trainable[index] else 1

            def backward(x, mask, cotangent):
                # Gradients of sum(run(x) * cotangent) are the vector-Jacobian products, for parameter trees too.
                def paired(params, x):
                    module.update(params)
                    return (run(x, mask) * cotangent).sum()

                return mx.grad(paired, argnums = argnums)(module.trainable_parameters(), x)

            return wrap(run), wrap(backward), argnums

        self.stages = [stage(index, module) for index, module in enumerate(modules)]

        def top(x, batch):
            def loss(params, x):
                model.update(params)
                keys = batch["attention_mask"].astype(mx.bool_)[:, None, None, :]
                logits = model.decide(encoder.final_norm(x), keys, batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                return -(batch["target"] * nn.log_softmax(logits, axis = -1)).sum(-1).mean()

            params = {name: value for name, value in model.trainable_parameters().items() if name != "encoder"}
            params["encoder"] = {"final_norm": encoder.final_norm.trainable_parameters()}
            return mx.value_and_grad(loss, argnums = [0, 1] if self.first < len(modules) else 0)(params, x)

        self.top = wrap(top)

    def encode(self, batch, seeds = None):
        """The last encoder layer's output and each stage's input."""
        encoder = self.model.encoder
        masks = encoder.masks(batch["attention_mask"].astype(mx.bool_)[:, None, None, :])
        masks = [None] + [masks[layer.layer_type] for layer in encoder.layers]
        x, inputs = batch["input_ids"], []
        for (run, _, _), mask in zip(self.stages, masks):
            inputs.append(x)
            if seeds is not None:
                # The backward pass recomputes the stage, which must then draw the same dropout masks.
                seeds.append(_mlx_rng_key())
            x = run(x, mask)
            mx.eval(x)
        return x, inputs, masks

    def __call__(self, batch):
        seeds = []
        x, inputs, masks = self.encode(batch, seeds)
        loss, grads = self.top(x, batch)
        cotangent = None
        if self.first < len(self.stages):
            grads, cotangent = grads
        mx.eval(loss, grads, cotangent)
        resume = _mlx_rng_key()
        layer_grads = [{} for _ in self.stages[1:]]
        for index in range(len(self.stages) - 1, self.first - 1, -1):
            _, backward, argnums = self.stages[index]
            _restore_mlx_rng_key(seeds[index])
            result = backward(inputs[index], masks[index], cotangent)
            stage_grads, cotangent = result if argnums == [0, 1] else (result, None) if argnums == 0 else ({}, result)
            mx.eval(stage_grads, cotangent)
            if index:
                layer_grads[index - 1] = stage_grads
            else:
                grads["encoder"]["embeddings"] = stage_grads
        _restore_mlx_rng_key(resume)
        if self.first < len(self.stages):
            grads["encoder"]["layers"] = layer_grads
        return loss, grads


def _staged_logits(step, batch):
    model = step.model
    keys = batch["attention_mask"].astype(mx.bool_)[:, None, None, :]
    hidden = model.encoder.final_norm(step.encode(batch)[0])
    return model.decide(hidden, keys, batch["marker_pos"], batch["marker_mask"], batch["qtype"])


class _MarkerStep:
    """How the trainer runs a Laya network: padded batches of one decision per row."""

    def __init__(self, model, pad_token_id, compiled = False):
        self.model, self.pad_token_id = model, pad_token_id
        # Scoring stays uncompiled: a compiled stage would replay the training-mode trace it was built with.
        self.staged = _LayerwiseStep(model)
        if getattr(model, "gradient_checkpointing", False):
            self.loss_and_grad = _LayerwiseStep(model, compiled)
        else:
            value_and_grad = nn.value_and_grad(model, _soft_cross_entropy)
            self.loss_and_grad = lambda batch: value_and_grad(model, batch)
            if compiled:
                state = [model.state, mx.random.state]
                self.loss_and_grad = mx.compile(self.loss_and_grad, inputs = state, outputs = state)

    def collate(self, items):
        return collate_decisions(items, self.pad_token_id)

    def __call__(self, batch):
        return self.loss_and_grad(batch)

    def losses(self, batch):
        """Summed soft cross-entropy of a batch and the number of decisions in it."""
        logits = _staged_logits(self.staged, batch)
        return -(batch["target"] * nn.log_softmax(logits, axis = -1)).sum(), batch["target"].shape[0]


_CLEF_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


@contextlib.contextmanager
def _decoder_training(model, gradient_checkpointing = True):
    """What `MLXTrainer` sets up around a run for a decoder: differentiable kernels in place of the fused inference ones."""
    from ..gated_delta_vjp import patch_gated_delta, patch_gated_delta_vlm, patch_gated_delta_vlm_shared
    from .compile import model_has_gated_delta_layers, model_has_qwen35_attention_layers
    from .loader import _disable_fused_input_projections, _disable_fused_mrope, _fix_qwen35_attention_cache
    from .utils import acquire_mlx_training_patches, apply_gradient_checkpointing, release_mlx_training_patches, remove_gradient_checkpointing

    unfused = {"fused_apply": [], "fuse_in": []}
    acquire_mlx_training_patches()
    try:
        if gradient_checkpointing:
            apply_gradient_checkpointing(model)
        if model_has_gated_delta_layers(model):
            # mlx-vlm's copies first, or patch_gated_delta's sweep warns about them.
            patch_gated_delta_vlm()
            patch_gated_delta_vlm_shared()
            patch_gated_delta()
        if model_has_qwen35_attention_layers(model):
            _fix_qwen35_attention_cache(model)
            unfused["fused_apply"] = _disable_fused_mrope(model)
        unfused["fuse_in"] = _disable_fused_input_projections(model)
        yield
    finally:
        for flag, modules in unfused.items():
            for module in modules:
                setattr(module, flag, True)
        if gradient_checkpointing:
            remove_gradient_checkpointing(model)
        release_mlx_training_patches()


class ClefNetwork(nn.Module):
    """A loaded Clef pipeline's decoder (`encoder`) and joint head as the one parameter tree the trainer optimizes."""

    def __init__(self, pipeline, gradient_checkpointing = True):
        super().__init__()
        self.encoder, self.head = pipeline.model, pipeline.head
        self._pipeline, self._gradient_checkpointing = pipeline, bool(gradient_checkpointing)

    def training_run(self):
        return _decoder_training(self.encoder, self._gradient_checkpointing)

    def decision_step(self, compiled = False):
        return _ClefStep(self)


def clef_training_network(
    pipeline, full_finetuning = False, r = 64, lora_alpha = 64, target_modules = "all-linear", gradient_checkpointing = True, **lora
):
    """Prepare a loaded Clef pipeline for training and return its `ClefNetwork`.

    The float32 joint head always trains. The decoder trains whole under `full_finetuning`, otherwise through LoRA
    adapters (`lora` is passed to `FastMLXModel.get_peft_model`); `"all-linear"` means its language layers' projections.
    """
    from .loader import FastMLXModel
    from .utils import _get_text_model, describe_output_head

    if full_finetuning:
        # Only what a text prompt reaches: a trainable weight without a gradient would still decay.
        pipeline.model.freeze()
        _get_text_model(pipeline.model).unfreeze()
        output = describe_output_head(pipeline.model)
        if output.status != "tied":
            # The head reads the output embedding without training it.
            output.module.freeze()
    else:
        if target_modules in (None, "all-linear"):
            target_modules = list(_CLEF_LORA_TARGETS)
        # Checkpointing is applied around each run instead, so it is gone again when the model serves.
        FastMLXModel.get_peft_model(
            pipeline.model, r = r, lora_alpha = lora_alpha, target_modules = target_modules, use_gradient_checkpointing = False, **lora,
        )
    pipeline.head.unfreeze()
    network = ClefNetwork(pipeline, gradient_checkpointing)
    network.train()
    return network


def _clef_record_loss(network, item):
    """Soft cross-entropy summed over the questions of one record."""
    spans = item["option_spans"]
    logits = network._pipeline.logits(item["input_ids"], item["question_spans"], spans, item["types"])
    target = np.zeros((len(spans), logits.shape[0]), np.float32)
    start = 0
    for row, values in enumerate(item["targets"]):
        target[row, start : start + len(values)] = values
        start += len(values)
    owner = mx.array([row for row, options in enumerate(spans) for _ in options])
    own = owner[None, :] == mx.arange(len(spans))[:, None]
    return -(mx.array(target) * nn.log_softmax(mx.where(own, logits[None, :], -1e4), axis = -1)).sum()


class _ClefStep:
    """How the trainer runs a Clef: one record (a prompt holding all its questions) at a time, averaged over questions."""

    def __init__(self, network):
        self.network = network
        self.value_and_grad = nn.value_and_grad(network, _clef_record_loss)

    def collate(self, items):
        return items

    def __call__(self, items):
        total, grads = 0.0, None
        for item in items:
            loss, record = self.value_and_grad(self.network, item)
            grads = record if grads is None else tree_map(mx.add, grads, record)
            total = total + loss
            # A record is evaluated on its own, so memory is bounded by the longest prompt, not the batch.
            mx.eval(total, grads)
        count = sum(len(item["targets"]) for item in items)
        return total / count, tree_map(lambda g: g / count, grads)

    def losses(self, items):
        total = sum(_clef_record_loss(self.network, item) for item in items)
        return total, sum(len(item["targets"]) for item in items)


def clef_logits(network, items):
    """Eval-mode option logits of Clef training items: for each item, one float32 numpy array per question."""
    from .generate import generation_mode

    pipeline, out = network._pipeline, []
    was_training = network.training
    network.eval()
    try:
        # Scored as requests are served, so temperatures fitted on these logits hold there.
        with generation_mode(pipeline.model):
            for item in items:
                spans = item["option_spans"]
                logits = np.array(pipeline.logits(item["input_ids"], item["question_spans"], spans, item["types"]))
                out.append(np.split(logits, np.cumsum([len(options) for options in spans])[:-1]))
    finally:
        network.train(was_training)
    return out


def decision_logits(model, items, pad_token_id, batch_size = 16):
    """Eval-mode logits of tokenized decisions: one float32 numpy row per item, as long as its options."""
    step, out = _LayerwiseStep(model), [None] * len(items)
    order = sorted(range(len(items)), key = lambda i: -len(items[i]["input_ids"]))
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            logits = np.array(_staged_logits(step, collate_decisions([items[i] for i in chunk], pad_token_id)))
            for row, i in enumerate(chunk):
                out[i] = logits[row, : len(items[i]["markers"])]
    finally:
        model.train(was_training)
    return out


def _length_grouped_batches(lengths, batch_size, rng):
    # Similar-length micro-batches, shuffled so a step mixes lengths; the longest first so an out-of-memory shows early.
    order = list(range(len(lengths)))
    rng.shuffle(order)
    mega = batch_size * max(1, min(len(order) // (batch_size * 4), 50))
    order = [i for start in range(0, len(order), mega) for i in sorted(order[start : start + mega], key = lambda i: -lengths[i])]
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    short = [batches.pop()] if len(batches) > 1 and len(batches[-1]) < batch_size else []
    longest = max(range(len(batches)), key = lambda b: max(lengths[i] for i in batches[b]))
    first = batches.pop(longest)
    rng.shuffle(batches)
    return [first] + batches + short


def _no_decay_names(model):
    names = set()
    for path, module in model.named_modules():
        for name, _ in tree_flatten(module.trainable_parameters()):
            if isinstance(module, nn.LayerNorm) or name == "bias":
                names.add(f"{path}.{name}" if path else name)
    return names


class MLXDecisionTrainer:
    """Trains a decision model on soft cross-entropy over each decision's option logits.

    `args` is an `MLXTrainingConfig`. Parameters under `encoder.` train at `args.learning_rate` and the rest
    at `head_learning_rate`, both under `args.lr_scheduler_type`. `callbacks` are transformers `TrainerCallback`s.
    """

    def __init__(
        self,
        model,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        pad_token_id = 0,
        head_learning_rate = None,
        callbacks = None,
        processing_class = None,
    ):
        from .trainer import MLXTrainingConfig, _MLXCallbackHandler, _MLXTrainerControl, _MLXTrainerState

        self.model = model
        self.args = args or MLXTrainingConfig()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pad_token_id = pad_token_id
        self.head_learning_rate = HEAD_LEARNING_RATE if head_learning_rate is None else head_learning_rate
        self.state = _MLXTrainerState()
        self.control = _MLXTrainerControl()
        self.callback_handler = _MLXCallbackHandler(callbacks or [], model, processing_class, None, None)

    def _resolve_warmup_steps(self, total_steps):
        from .trainer import MLXTrainer

        return MLXTrainer._resolve_warmup_steps(self, total_steps)

    def _schedule_multiplier(self, total_steps):
        # As torch schedulers do, one multiplier scales every group's learning rate, so an absolute floor follows the encoder's.
        from .trainer import MLXTrainer

        base = self.args.learning_rate or self.head_learning_rate
        if not base:
            return lambda step: 0.0
        host = copy.copy(self)
        host.args = copy.copy(self.args)
        host.args.learning_rate = base
        schedule = MLXTrainer._build_schedule(host, total_steps)
        return lambda step: float(MLXTrainer._schedule_value(schedule, step)) / base

    def _optimizers(self):
        from .trainer import _normalize_mlx_optimizer_name, _resolve_adam_epsilon

        args = self.args
        name = _normalize_mlx_optimizer_name(self.args.optim)
        if name == "adamw":
            cls = optim.AdamW
        elif name == "adamw_8bit":
            from .optimizers_quantized import QuantizedMomentAdamW as cls
        else:
            raise NotImplementedError(
                f"Unsloth: decision models train with AdamW on MLX (adamw or adamw_8bit), not {self.args.optim!r}."
            )
        # MLX defaults bias_correction to False, unlike torch.
        kwargs = {"bias_correction": True}
        if args.adam_beta1 is not None or args.adam_beta2 is not None:
            kwargs["betas"] = (
                float(0.9 if args.adam_beta1 is None else args.adam_beta1),
                float(0.999 if args.adam_beta2 is None else args.adam_beta2),
            )
        if args.adam_epsilon is not None:
            kwargs["eps"] = _resolve_adam_epsilon(args.adam_epsilon)
        no_decay = _no_decay_names(self.model)
        groups = {}
        for name, _ in tree_flatten(self.model.trainable_parameters()):
            groups.setdefault((name.startswith("encoder."), name not in no_decay), []).append(name)
        return {
            key: (set(names), cls(learning_rate = 0.0, weight_decay = float(args.weight_decay or 0.0) if key[1] else 0.0, **kwargs))
            for key, names in groups.items()
        }

    def _event(self, name, **kwargs):
        self.control = self.callback_handler.call_event(name, self.args, self.state, self.control, **kwargs)

    def _log(self, logs):
        self.state.log_history.append({**logs, "step": self.state.global_step})
        self._event("on_log", logs = logs)

    def _step(self, compiled = False):
        build = getattr(self.model, "decision_step", None)
        return build(compiled) if build else _MarkerStep(self.model, self.pad_token_id, compiled)

    def _eval_batches(self):
        size = self.args.per_device_eval_batch_size or self.args.per_device_train_batch_size
        order = sorted(range(len(self.eval_dataset)), key = lambda i: -len(self.eval_dataset[i]["input_ids"]))
        return [order[i : i + size] for i in range(0, len(order), size)]

    def evaluate(self):
        """Mean soft cross-entropy over the decisions of `eval_dataset`, as `{"eval_loss": ...}`."""
        model, items = self.model, self.eval_dataset
        step = self._step()
        was_training = model.training
        model.eval()
        total, decisions = 0.0, 0
        try:
            for batch in self._eval_batches():
                loss, count = step.losses(step.collate([items[i] for i in batch]))
                total, decisions = total + loss.item(), decisions + count
        finally:
            model.train(was_training)
        metrics = {"eval_loss": total / decisions, "epoch": self.state.epoch}
        self._log(metrics)
        self._event("on_evaluate", metrics = metrics)
        return metrics

    def train(self):
        # MLX caches freed buffers by size, so every distinct batch shape leaves its working set in the cache: tens of
        # gigabytes over a run. With args.cache_limit_gb unset the cache may hold what a step has needed so far,
        # which is all the next step can reuse; a value <= 0 or args.disable_memory_limits leaves the limit alone.
        from .trainer import MLXTrainer

        args = self.args
        # The memory, wired and explicit cache caps are MLXTrainer's, under the same switches.
        limits = types.SimpleNamespace(args = args, _bytes_to_gb = MLXTrainer._bytes_to_gb)
        self._cache_follows_peak = getattr(args, "cache_limit_gb", None) is None and not getattr(args, "disable_memory_limits", False)
        prior = None
        try:
            self._memory_limits_applied = MLXTrainer._configure_memory_limits(limits)
            if self._cache_follows_peak:
                prior = mx.set_cache_limit(mx.get_peak_memory())
            with getattr(self.model, "training_run", contextlib.nullcontext)():
                return self._train()
        finally:
            if prior is not None:
                mx.set_cache_limit(prior)
            MLXTrainer._restore_memory_limits(limits)

    def _epoch_batches(self, epoch):
        args, lengths = self.args, [len(item["input_ids"]) for item in self.train_dataset]
        rng = random.Random(args.seed + epoch)
        if max(1, args.gradient_accumulation_steps) > 1:
            return _length_grouped_batches(lengths, args.per_device_train_batch_size, rng)
        # With one micro-batch per step, length grouping would make each step one question type.
        order = list(range(len(lengths)))
        rng.shuffle(order)
        return [order[i : i + args.per_device_train_batch_size] for i in range(0, len(order), args.per_device_train_batch_size)]

    def _train(self):
        from .trainer import MLXTrainOutput, _clip_grad_norm_fp32, _resolve_interval_steps

        args, model, items = self.args, self.model, self.train_dataset
        batch_size, accumulation = args.per_device_train_batch_size, max(1, args.gradient_accumulation_steps)
        steps_per_epoch = math.ceil(math.ceil(len(items) / batch_size) / accumulation)
        max_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.num_train_epochs * steps_per_epoch)
        multiplier = self._schedule_multiplier(max_steps)
        optimizers = self._optimizers()
        logging_steps, eval_steps = (_resolve_interval_steps(value, max_steps) for value in (args.logging_steps, args.eval_steps))
        # MLXTrainingConfig has no eval_strategy: a positive eval_steps evaluates on steps, otherwise once per epoch.
        eval_strategy = "no" if not self.eval_dataset else "steps" if eval_steps else "epoch"

        compiled = getattr(args, "compile", True) and getattr(args, "compile_mode", None) != "eager"
        step = self._step(compiled)

        state = self.state
        state.max_steps, state.logging_steps, state.train_batch_size = max_steps, logging_steps, batch_size
        state.num_train_epochs, state.epoch = math.ceil(max_steps / steps_per_epoch), 0.0
        model.train()
        started, logged_loss, logged_steps, total_loss = time.time(), 0.0, 0, 0.0
        self._event("on_train_begin")
        epoch = 0
        while state.global_step < max_steps and not self.control.should_training_stop:
            batches = self._epoch_batches(epoch)
            self._event("on_epoch_begin")
            accumulated, losses = None, []
            for index, batch in enumerate(batches):
                loss, grads = step(step.collate([items[i] for i in batch]))
                if accumulated is not None:
                    grads = tree_map(mx.add, accumulated, grads)
                mx.eval(loss, grads)
                if self._cache_follows_peak:
                    mx.set_cache_limit(mx.get_peak_memory())
                accumulated = grads
                losses.append(loss.item())
                if (index + 1) % accumulation and index + 1 < len(batches):
                    continue

                # The epoch's last step may hold fewer micro-batches and averages over those it has.
                step_loss, window = sum(losses) / len(losses), len(losses)
                grads = tree_map(lambda g: g / window, accumulated)
                grad_norm = None
                if args.max_grad_norm and args.max_grad_norm > 0:
                    grads, grad_norm = _clip_grad_norm_fp32(grads, args.max_grad_norm)
                scale = multiplier(state.global_step)
                learning_rate = args.learning_rate * scale
                flat = tree_flatten(grads)
                for (encoder, _), (names, optimizer) in optimizers.items():
                    optimizer.learning_rate = learning_rate if encoder else self.head_learning_rate * scale
                    optimizer.update(model, tree_unflatten([(name, g) for name, g in flat if name in names]))
                mx.eval(model.parameters(), [optimizer.state for _, optimizer in optimizers.values()])
                accumulated, losses = None, []

                state.global_step += 1
                state.epoch = epoch + (index + 1) / len(batches)
                logged_loss, logged_steps, total_loss = logged_loss + step_loss, logged_steps + 1, total_loss + step_loss
                self._event("on_step_end")
                if logging_steps and state.global_step % logging_steps == 0:
                    logs = {"loss": logged_loss / logged_steps, "learning_rate": learning_rate, "epoch": state.epoch, "peak_memory_gb": mx.get_peak_memory() / 1e9}
                    if grad_norm is not None:
                        logs["grad_norm"] = grad_norm.item()
                    logged_loss, logged_steps = 0.0, 0
                    self._log(logs)
                if eval_strategy == "steps" and state.global_step % eval_steps == 0:
                    self.evaluate()
                if state.global_step >= max_steps or self.control.should_training_stop:
                    break
            epoch += 1
            self._event("on_epoch_end")
            if eval_strategy == "epoch" and not self.control.should_training_stop:
                self.evaluate()

        runtime = time.time() - started
        metrics = {
            "train_runtime": runtime,
            "train_steps": state.global_step,
            "train_steps_per_second": state.global_step / runtime if runtime else 0.0,
            "train_loss": total_loss / max(1, state.global_step),
            "epoch": state.epoch,
        }
        self._log(metrics)
        self._event("on_train_end")
        return MLXTrainOutput(metrics)
