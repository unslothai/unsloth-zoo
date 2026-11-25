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

"""CPU defaults for transformers / TRL trainers. Installed only by the CPU backend."""

__all__ = ["install_cpu_training_defaults", "normalize_cpu_training_arguments", "cpu_packing_supported"]

import functools
import json
import os

import torch
from unsloth_zoo.utils import is_activation_checkpointing as _is_activation_checkpointing

from .runtime import available_ram, cpu_bf16_supported, cpu_fp16_supported

# bitsandbytes 8-bit / paged optimizers have no CPU kernels; AdamW states for LoRA are tiny anyway.
_GPU_ONLY_OPTIMIZERS = (
    "adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "paged_adamw",
    "adamw_apex_fused", "adafactor_8bit", "lion_8bit", "paged_lion_8bit",
    "paged_lion_32bit", "ademamix_8bit", "paged_ademamix_8bit", "paged_ademamix_32bit",
)


@functools.cache
def _cpu_adamw():
    """Fused AdamW has CPU kernels from torch 2.4; foreach is far slower over LoRA's many small tensors."""
    from packaging.version import Version
    return "adamw_torch_fused" if Version(torch.__version__.split("+")[0]) >= Version("2.4.0") else "adamw_torch"


def _name(value):
    return getattr(value, "value", value)


# TRL's SFTTrainer makes its collator padding-free for packing with these strategies (0.22: bfd; 1.x: bfd, bfd_split).
_PADDING_FREE_STRATEGIES = ("bfd", "bfd_split")


def _packing_strategy(args):
    return str(_name(getattr(args, "packing_strategy", "bfd")) or "bfd").lower()


def normalize_cpu_training_arguments(args):
    args.use_cpu = True
    if getattr(args, "fp16", False):
        # bf16 needs no loss scaling, so it wins whenever it is native; fp16 autocast (with a CPU
        # GradScaler, see _patch_cpu_fp16_accelerator) only where fp16 is native and bf16 is not.
        if cpu_bf16_supported():
            args.fp16, args.bf16 = False, True
        elif not cpu_fp16_supported():
            args.fp16, args.bf16 = False, False
    elif getattr(args, "bf16", None) and not cpu_bf16_supported():
        # bf16 autocast without AMX / AVX512-BF16 / Arm BF16 is emulated and several times slower.
        args.bf16 = False
    optim = str(_name(getattr(args, "optim", "")) or "").lower()
    if optim in _GPU_ONLY_OPTIMIZERS or any(tag in optim for tag in ("8bit", "paged", "bnb", "npu", "xla")):
        print(f"Unsloth (CPU): optim = {optim!r} needs a GPU, using {_cpu_adamw()!r}.")
        args.optim = _cpu_adamw()
    elif optim in ("adamw_torch", "adamw_torch_fused"):
        args.optim = _cpu_adamw()
    if getattr(args, "tf32", None):
        args.tf32 = None
    if hasattr(args, "dataloader_pin_memory"):
        args.dataloader_pin_memory = False
    # Packing / padding-free flatten a batch into one row with position_ids. transformers >= 4.53 on torch >= 2.6
    # turns those resets into a block-diagonal SDPA / eager mask; older stacks, and TRL's "wrapped" strategy (no
    # resets), attend across examples, so those keep TRL's padded batches.
    supported = cpu_packing_supported()
    wrapped = _packing_strategy(args) == "wrapped"
    if getattr(args, "padding_free", False) and not supported:
        print("Unsloth (CPU): padding_free needs transformers >= 4.53 and torch >= 2.6 to keep examples apart, "
              "turning it off.")
        args.padding_free = False
    for key in ("packing", "eval_packing"):
        if getattr(args, key, False) and (not supported or wrapped):
            reason = "the 'wrapped' packing strategy attends across examples without FlashAttention" if supported \
                else "keeping packed examples apart needs transformers >= 4.53 and torch >= 2.6 (or FlashAttention)"
            print(f"Unsloth (CPU): {key}: {reason}, turning it off.")
            setattr(args, key, False)
    # TRL's collator is padding-free (position_ids reset, no attention_mask) only for padding_free or bfd packing of
    # the TRAIN set, and eval batches share it: eval_packing alone packs eval rows under a plain attention mask.
    padding_free_collator = getattr(args, "padding_free", False) or (
        getattr(args, "packing", False) and _packing_strategy(args) in _PADDING_FREE_STRATEGIES)
    if getattr(args, "eval_packing", False) and not padding_free_collator:
        print("Unsloth (CPU): eval_packing without packing / padding_free gives eval batches no position_ids "
              "resets, so packed examples would attend across each other; turning it off.")
        args.eval_packing = False
    if getattr(args, "gradient_checkpointing", False):
        kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
        if isinstance(kwargs, str):
            # The CLI / JSON form, which TrainingArguments decodes only after this runs.
            kwargs = json.loads(kwargs)
        kwargs = dict(kwargs)
        kwargs.setdefault("use_reentrant", False)
        args.gradient_checkpointing_kwargs = kwargs
    return args


@functools.cache
def cpu_packing_supported():
    """transformers builds a block-diagonal mask from packed position_ids (masking_utils, 4.53+, needs torch 2.6), and
    TRL's padding-free collator leaves attention_mask out, which that detection requires. UNSLOTH_CPU_PACKING=0 opts out."""
    if os.environ.get("UNSLOTH_CPU_PACKING", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from transformers import masking_utils
        if not hasattr(masking_utils, "find_packed_sequence_indices") or \
                not getattr(masking_utils, "_is_torch_greater_or_equal_than_2_6", False):
            return False
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        batch = DataCollatorForLanguageModeling(pad_token_id = 0, padding_free = True)(
            [{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}]
        )
        return "attention_mask" not in batch and torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 0, 1]))
    except Exception:
        return False


def _packed_examples_isolated(model):
    """A packed forward (position_ids reset, no attention_mask) gives the same logits as the examples alone: True
    when this model's attention keeps packed examples apart on CPU."""
    try:
        from .loader import _disable_bnb_cpu_weight_packing
        # bitsandbytes repacks 4-bit weights on a no-grad eval forward, which breaks their backward.
        _disable_bnb_cpu_weight_packing(model)
        config = getattr(model, "config", None)
        vocab = getattr(getattr(config, "get_text_config", lambda: config)(), "vocab_size", None) or 32
        device = next(model.parameters()).device
        generator = torch.Generator().manual_seed(0)
        a = torch.randint(1, max(2, min(vocab, 1000)), (1, 3), generator = generator).to(device)
        b = torch.randint(1, max(2, min(vocab, 1000)), (1, 4), generator = generator).to(device)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device = device)
        was_training = model.training
        model.eval()
        import contextlib
        set_stance = getattr(torch.compiler, "set_stance", None)
        # Compiled training layers would build an extra no-grad graph for these 3 calls.
        eager = set_stance("force_eager") if set_stance is not None else contextlib.nullcontext()
        try:
            with torch.no_grad(), eager:
                run = lambda **kw: model(use_cache = False, **kw).logits.float()  # noqa: E731
                separate = torch.cat([run(input_ids = a), run(input_ids = b)], dim = 1)
                packed = run(input_ids = torch.cat([a, b], dim = 1), position_ids = position_ids)
        finally:
            model.train(was_training)
        # Relative to the logits' own scale (no floor: small-logit models must not get an absolute pass) and to the
        # compute dtype: fp32 rounding between the packed and separate shapes stays near 1e-6, attending across
        # examples moves the second example's logits by a sizeable fraction of their scale.
        compute = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
        tolerance = 1e-3 if compute in (torch.float32, torch.float64) else 2e-2
        scale = separate.abs().amax().clamp(min = torch.finfo(torch.float32).tiny)
        return bool(torch.isfinite(packed).all()) and float((packed - separate).abs().amax() / scale) < tolerance
    except Exception:
        return False


def _check_cpu_packing(model, training_args):
    """Per model: packing / padding_free stay on only when this model's attention keeps packed examples apart."""
    keys = [key for key in ("packing", "eval_packing", "padding_free") if getattr(training_args, key, False)]
    if not keys or model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        # A model id / path is loaded inside TRL after the dataset is packed, too late to probe its attention.
        print(f"Unsloth (CPU): cannot check a model passed as {type(model).__name__} for packed-example isolation, "
              f"turning {', '.join(keys)} off. Pass the loaded model to keep them.")
        for key in keys:
            setattr(training_args, key, False)
        return False
    if _packed_examples_isolated(model):
        # Training forwards must not build a KV cache: a cache disables transformers' packed-sequence detection.
        config = getattr(model, "config", None)
        if config is not None:
            config.use_cache = False
        return True
    print(f"Unsloth (CPU): {type(model).__name__} attends across packed examples on CPU, turning {', '.join(keys)} off.")
    for key in keys:
        setattr(training_args, key, False)
    return False


# Auto mode times eager steps, then compiles only if steps left repay a conservative cold compile at
# a conservative saving; UNSLOTH_CPU_COMPILE_COST_S / UNSLOTH_CPU_COMPILE_SAVING override.
_COMPILE_COLD_SECONDS = 150.0


_COMPILE_SAVING = 0.2


_COMPILE_DECIDE_STEP = 2  # step 1 also pays for dataloader / allocator warm-up


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _compile_worth_it(eager_step_seconds, steps_left):
    """(compile?, seconds saved, cold compile estimate)."""
    cost = _env_float("UNSLOTH_CPU_COMPILE_COST_S", _COMPILE_COLD_SECONDS)
    saved = max(0, steps_left) * eager_step_seconds * _env_float("UNSLOTH_CPU_COMPILE_SAVING", _COMPILE_SAVING)
    return saved > cost, saved, cost


def _compile_target(model):
    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and vars(candidate).get("_unsloth_cpu_compile_pending", False):
            return candidate
    return None


def _install_compile_gate(trainer):
    """Auto CPU compile: time an eager step, then compile from the next step only if the rest of the run repays it."""
    target = _compile_target(getattr(trainer, "model", None))
    if target is None:
        return None
    import time
    from transformers import TrainerCallback

    class _UnslothCPUCompileGate(TrainerCallback):
        def __init__(self):
            self.started = None

        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if self.started is None or state.global_step < _COMPILE_DECIDE_STEP or \
                    not vars(target).get("_unsloth_cpu_compile_pending", False):
                return
            seconds = time.perf_counter() - self.started
            target._unsloth_cpu_compile_pending = False
            steps_left = (state.max_steps or 0) - state.global_step
            worth, saved, cost = _compile_worth_it(seconds, steps_left)
            if worth:
                from .loader import compile_cpu_layers
                compile_cpu_layers(target)
            else:
                print(f"Unsloth (CPU): staying eager: {steps_left} steps left at {seconds:.1f} s would save about "
                      f"{saved:.0f} s with torch.compile, under its ~{cost:.0f} s cold compile. "
                      "UNSLOTH_CPU_COMPILE=1 compiles anyway.")

    trainer.add_callback(_UnslothCPUCompileGate())
    return True


def _activation_bytes_without_checkpointing(model, batch, length):
    """Upper estimate of activations autograd keeps per batch x length micro-batch without
    checkpointing: 9 hidden + 3 kv + 4 intermediate per token per layer, doubled for fp32 upcasts
    and LoRA; eager or non-SDPA/flash attention adds heads x length^2 fp32 softmax, also doubled."""
    config = getattr(model, "config", None)
    text = getattr(config, "get_text_config", lambda: config)()
    hidden = getattr(text, "hidden_size", None)
    layers = getattr(text, "num_hidden_layers", None)
    if not hidden or not layers:
        return None
    intermediate = getattr(text, "intermediate_size", None) or 4 * hidden
    heads = getattr(text, "num_attention_heads", None) or 1
    kv = (getattr(text, "num_key_value_heads", None) or heads) * (getattr(text, "head_dim", None) or hidden // heads)
    try:
        element = next(p for p in model.parameters() if p.is_floating_point()).element_size()
    except StopIteration:
        element = 4
    element = max(element, 2)
    needed = 2 * (9 * hidden + 3 * kv + 4 * intermediate) * element * layers * batch * length
    attention = getattr(text, "_attn_implementation", None) or getattr(config, "_attn_implementation", None)
    dropout = 0.0
    for name in ("attention_dropout", "attn_pdrop", "attention_probs_dropout_prob", "attn_dropout"):
        try:
            dropout = max(dropout, float(getattr(text, name, 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    # Eager attention and SDPA with dropout (CPU math path) keep B x H x L x L for backward.
    quadratic = isinstance(attention, str) and not any(fast in attention for fast in ("sdpa", "flash", "flex"))
    if quadratic or dropout > 0:
        needed += 2 * (4 + element) * heads * layers * batch * length * length
    return needed


def _args_checkpointing(training_args):
    # transformers >= 5.24 reads activation_checkpointing; gradient_checkpointing is its deprecated alias.
    return bool(getattr(training_args, "gradient_checkpointing", False) or getattr(training_args, "activation_checkpointing", False))


def _args_checkpointing_off(training_args):
    training_args.gradient_checkpointing = False
    if hasattr(training_args, "activation_checkpointing"):
        training_args.activation_checkpointing = False


def _auto_cpu_gradient_checkpointing(model, training_args):
    """use_gradient_checkpointing = "unsloth" on CPU means offloading to host RAM, which is where activations already
    live, so it is plain recomputation (one more forward per step). When the activations of a full micro-batch fit in
    a quarter of the free RAM, train without it. True / False from the user, or UNSLOTH_CPU_AUTO_GC=0, keep theirs."""
    if not isinstance(model, torch.nn.Module) or not _args_checkpointing(training_args):
        return None
    choice = getattr(model, "_unsloth_cpu_gradient_checkpointing", None)
    if choice is False:
        # TRL's SFTConfig turns checkpointing on by default, which would override use_gradient_checkpointing = False.
        _args_checkpointing_off(training_args)
        return True
    if choice != "unsloth" or os.environ.get("UNSLOTH_CPU_AUTO_GC", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    length = getattr(training_args, "max_length", None) or getattr(model, "max_seq_length", None)
    batch = getattr(training_args, "per_device_train_batch_size", None) or 1
    if not isinstance(length, int) or length <= 0:
        return None
    needed = _activation_bytes_without_checkpointing(model, batch, length)
    available = available_ram()
    if needed is None or available is None:
        return None
    ranks = 1
    try:
        ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    except ValueError:
        pass
    if needed * ranks > available // 4:
        return False
    _args_checkpointing_off(training_args)
    if _is_activation_checkpointing(model) and hasattr(model, "gradient_checkpointing_disable"):
        # PEFT forwards this to the transformers model; input require-grads hooks stay (a no-op without checkpointing).
        model.gradient_checkpointing_disable()
    print(f"Unsloth (CPU): activations for {batch} x {length} tokens (~{needed / 1024 ** 3:.2f} GB) fit in RAM "
          f"({available / 1024 ** 3:.0f} GB free), so gradient checkpointing is off (one less forward per step). "
          "Pass use_gradient_checkpointing = True or set UNSLOTH_CPU_AUTO_GC=0 to keep it.")
    return True


def _patch_post_init(cls):
    original = cls.__post_init__
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def __post_init__(self, *args, **kwargs):
        # TRL resolves bf16=None to not-fp16 first, so normalize fp16/optim before, bf16 after.
        normalize_cpu_training_arguments(self)
        result = original(self, *args, **kwargs)
        if getattr(self, "bf16", False) and not cpu_bf16_supported():
            self.bf16 = False
        return result

    __post_init__._unsloth_cpu = True
    cls.__post_init__ = __post_init__


def _patch_cpu_fp16_accelerator(trainer_cls):
    """accelerate turns fp16 into a silent fp32 CPU run (no autocast, no scaler); when fp16 is kept
    (no native bf16), enable CPU autocast and CPU GradScaler before prepare()."""
    original = trainer_cls.create_accelerator_and_postprocess
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def create_accelerator_and_postprocess(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        accelerator = getattr(self, "accelerator", None)
        if (
            accelerator is not None
            and getattr(self.args, "fp16", False)
            and accelerator.device.type == "cpu"
            and accelerator.state.mixed_precision == "fp16"
            and not accelerator.native_amp
        ):
            import torch
            accelerator.native_amp = True
            accelerator.scaler = torch.amp.GradScaler("cpu")
        if accelerator is not None and accelerator.device.type == "cpu" and accelerator.native_amp:
            _disable_cpu_autocast_cache(accelerator)
        return result

    create_accelerator_and_postprocess._unsloth_cpu = True
    trainer_cls.create_accelerator_and_postprocess = create_accelerator_and_postprocess


def _disable_cpu_autocast_cache(accelerator):
    """autocast caches a low-precision copy of fp32 LoRA adapters per outermost region, even under
    no_grad; KTO's KL pass hands the policy pass a detached copy: no adapter gradient, and
    checkpoint recompute fails on metadata; disabling the cache only recasts the small adapters."""
    from accelerate.utils import AutocastKwargs
    handler = accelerator.autocast_handler
    accelerator.autocast_handler = AutocastKwargs(enabled = handler.enabled if handler is not None else True,
                                                  cache_enabled = False)


def _patch_cpu_autocast_cache(trainer_cls):
    """4.x: own CPU autocast in compute_loss (cache on); 5.x: accelerate's, covered by handler."""
    original = getattr(trainer_cls, "autocast_smart_context_manager", None)
    if original is None or getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def autocast_smart_context_manager(self, cache_enabled = True):
        return original(self, cache_enabled = False)

    autocast_smart_context_manager._unsloth_cpu = True
    trainer_cls.autocast_smart_context_manager = autocast_smart_context_manager


def _chunked_ce_model(trainer, model):
    """The CPU model whose forward Unsloth patched (patch_chunked_causal_lm_loss), behind accelerate / PEFT wrappers."""
    unwrapped = trainer.accelerator.unwrap_model(model) if getattr(trainer, "accelerator", None) is not None else model
    for candidate in (unwrapped, getattr(unwrapped, "base_model", None), getattr(getattr(unwrapped, "base_model", None), "model", None)):
        # vars(), not getattr: PEFT forwards attribute reads to the inner model, but writes stay on the wrapper.
        if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False):
            return candidate
    return None


def _chunked_head_target(trainer, model):
    """The patched CPU model behind `model` whose plain lm_head can score from the last hidden state, or None."""
    from .ops import _plain_head_weight
    target = _chunked_ce_model(trainer, model)
    head = _plain_head_weight(target.get_output_embeddings()) if target is not None else None
    # head.device: a CPU-patched model moved to an accelerator keeps TRL's forward (the chunked scorer is CPU-only).
    if head is None or head.device.type != "cpu" or getattr(trainer, "is_encoder_decoder", False) \
            or getattr(trainer, "aux_loss_enabled", False):
        return None
    return target


def _chunked_token_logps(target, model, input_ids, attention_mask, targets, log_sum_sq = False):
    """Per-token log p(targets[:, t]) and vocabulary-mean logits at position t (dict), or None if a wrapper replaced
    the patched forward. The head runs inside the patched forward (ops.patch_chunked_causal_lm_loss), so a trainable
    lm_head under DDP is used inside the wrapped forward, and accelerate's autocast covers it as it covers TRL's."""
    target._unsloth_cpu_logps_targets = targets
    if log_sum_sq:
        target._unsloth_cpu_logps_sq = True
    try:
        outputs = model(input_ids, attention_mask = attention_mask, use_cache = False)
    finally:
        target.__dict__.pop("_unsloth_cpu_logps_targets", None)
        target.__dict__.pop("_unsloth_cpu_logps_sq", None)
    if not isinstance(outputs, dict) or "log_probs" not in outputs:
        return None
    return outputs


def _orpo_chunked_forward(trainer, original):
    """TRL's ORPO concatenated_forward without the [2B, L, V] logits: the policy's next-token log-probs come from the
    last hidden state in row chunks (cpu_chunked_log_probs, the CPU twin of TRL's fused LM head, which TRL's DPO / KTO
    already use). Its nll loss, length-averaged log-probs and logits metric are those same per-token values. Only
    the plain decoder case; anything else, or UNSLOTH_CPU_CHUNKED_PREFERENCE=0, runs TRL's own forward."""

    def concatenated_forward(model, batch):
        target = _chunked_head_target(trainer, model)
        if target is None or getattr(trainer, "label_pad_token_id", -100) != -100 \
                or any(key in batch for key in ("pixel_values", "pixel_attention_mask", "image_sizes")):
            return original(model, batch)
        concatenated = trainer.concatenated_inputs(
            batch, is_encoder_decoder = False, padding_value = trainer.padding_value, device = trainer.accelerator.device,
        )
        input_ids = concatenated["concatenated_input_ids"]
        attention_mask = concatenated["concatenated_attention_mask"]
        labels = concatenated["concatenated_labels"][:, 1:]
        targets = input_ids[:, 1:]
        label_mask = labels != -100
        if labels.shape != targets.shape or not torch.equal(labels[label_mask], targets[label_mask]):
            return original(model, batch)  # labels that are not the next input token: TRL's two targets differ
        len_chosen = batch["chosen_labels"].shape[0]
        outputs = _chunked_token_logps(target, model, input_ids, attention_mask, targets)
        if outputs is None:
            return original(model, batch)  # a wrapper replaced the patched forward
        log_probs, mean_logits = outputs["log_probs"], outputs["mean_logits"]
        # nn.CrossEntropyLoss over the chosen rows' attended next tokens (TRL's ORPO nll over prompt + response).
        nll_mask = attention_mask[:len_chosen, 1:] == 1
        chosen_nll_loss = -log_probs[:len_chosen][nll_mask].sum() / nll_mask.sum()
        # get_batch_logps(average_log_prob = True)
        all_logps = (log_probs * label_mask).sum(-1) / label_mask.sum(-1)
        # TRL only ever takes .mean() of the logits it returns: per-token vocabulary means average to the same value.
        return (all_logps[:len_chosen], all_logps[len_chosen:], mean_logits[:len_chosen], mean_logits[len_chosen:],
                chosen_nll_loss)

    concatenated_forward._unsloth_cpu = True
    return concatenated_forward


def _dpo_flush_and_truncate(trainer, attention_mask, input_ids, loss_mask):
    """TRL 0.2x DPOTrainer.concatenated_forward's flush / truncation of the decoder inputs (no token_type_ids)."""
    from trl.trainer.utils import flush_left, flush_right
    max_length = trainer.max_length
    if max_length is not None and max_length < attention_mask.size(1):
        if trainer.truncation_mode == "keep_start":
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            return attention_mask[:, :max_length], input_ids[:, :max_length], loss_mask[:, :max_length]
        if trainer.truncation_mode == "keep_end":
            attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
            attention_mask, input_ids, loss_mask = (
                attention_mask[:, -max_length:], input_ids[:, -max_length:], loss_mask[:, -max_length:])
            return flush_left(attention_mask, input_ids, loss_mask)
        return None  # TRL raises on an unknown mode: let it
    return flush_left(attention_mask, input_ids, loss_mask)


def _dpo_chunked_forward(trainer, original):
    """TRL 0.2x DPOTrainer.concatenated_forward without the [2B, L, V] logits (TRL 1.x DPO already scores through
    its fused LM head, which Unsloth routes to cpu_chunked_log_probs). Same flush / truncation, then the per-token
    log-probs, WPO weights (logsumexp(2 log p)), RPO / SFT nll, IPO / LD-DPO arithmetic and mean-logits metrics from
    the chunked per-token values. Padding-free, vision, token_type_ids, MoE aux loss and encoder-decoder models, or
    UNSLOTH_CPU_CHUNKED_PREFERENCE=0, run TRL's own forward."""

    def concatenated_forward(model, batch, is_ref_model = False):
        target = _chunked_head_target(trainer, model)
        if target is None or getattr(trainer, "padding_free", False):
            return original(model, batch, is_ref_model = is_ref_model)
        num_examples = batch["prompt_input_ids"].shape[0]
        padding_value = getattr(trainer, "pad_token_id", None)
        if padding_value is None:
            padding_value = trainer.padding_value
        concatenated_batch = trainer.concatenated_inputs(batch, padding_value = padding_value)
        if any(key in concatenated_batch for key in ("pixel_values", "pixel_attention_mask", "image_sizes", "token_type_ids")):
            return original(model, batch, is_ref_model = is_ref_model)
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        input_ids = torch.cat((concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]), dim = 1)
        attention_mask = torch.cat((prompt_attention_mask, concatenated_batch["completion_attention_mask"]), dim = 1)
        loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), concatenated_batch["completion_attention_mask"]), dim = 1)
        flushed = _dpo_flush_and_truncate(trainer, attention_mask, input_ids, loss_mask)
        if flushed is None:
            return original(model, batch, is_ref_model = is_ref_model)
        attention_mask, input_ids, loss_mask = flushed
        logits_to_keep = None
        if getattr(trainer, "use_logits_to_keep", False):
            # TRL keeps the logits from the first completion position on (+1 for the first label).
            logits_to_keep = (loss_mask.shape[1] - loss_mask.nonzero(as_tuple = True)[1].min()).item() + 1

        labels = torch.roll(input_ids, shifts = -1, dims = 1)
        loss_mask = torch.roll(loss_mask, shifts = -1, dims = 1).bool()
        labels = labels.masked_fill(~loss_mask, 0)  # TRL's dummy token on ignored positions
        use_weighting = bool(getattr(trainer, "use_weighting", False))
        outputs = _chunked_token_logps(target, model, input_ids, attention_mask, labels, log_sum_sq = use_weighting)
        if outputs is None:
            return original(model, batch, is_ref_model = is_ref_model)
        token_logps, mean_logits, log_sum_sq = outputs["log_probs"], outputs["mean_logits"], outputs.get("log_sum_sq_probs")
        if logits_to_keep is not None:
            # TRL's logits for the last logits_to_keep positions: the same per-token values, sliced.
            labels, loss_mask = labels[:, -logits_to_keep:], loss_mask[:, -logits_to_keep:]
            token_logps, mean_logits = token_logps[:, -logits_to_keep:], mean_logits[:, -logits_to_keep:]
            log_sum_sq = log_sum_sq[:, -logits_to_keep:] if log_sum_sq is not None else None

        per_token_logps = token_logps.masked_fill(~loss_mask, 0)
        per_token_logps = torch.roll(per_token_logps, shifts = 1, dims = 1)
        all_logps = per_token_logps[:, 1:].sum(-1)
        output = {}
        if use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper, as TRL computes it from the full logits.
                per_token_logps_adjusted = per_token_logps - log_sum_sq
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                output["policy_weights"] = torch.clamp(
                    torch.exp(all_weights[:num_examples] + all_weights[num_examples:]), max = 1)
        loss_type = trainer.loss_type
        if trainer.args.rpo_alpha is not None or "sft" in loss_type:
            # F.cross_entropy(chosen logits[:, :-1], chosen labels[:, :-1], ignore_index = 0)
            chosen_labels = labels[:num_examples, :-1]
            keep = chosen_labels != 0
            output["nll_loss"] = -token_logps[:num_examples, :-1][keep].sum() / keep.sum()
        if "ipo" in loss_type:
            all_logps = all_logps / loss_mask.sum(-1)
        ld_alpha = getattr(trainer.args, "ld_alpha", None)
        if ld_alpha is not None and not is_ref_model:
            completion_lengths = loss_mask.sum(dim = 1)
            public_lengths = torch.min(completion_lengths[:num_examples], completion_lengths[num_examples:])
            public_lengths = torch.cat([public_lengths, public_lengths], dim = 0)
            position_ids = torch.arange(per_token_logps.size(1), device = per_token_logps.device).expand_as(per_token_logps)
            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)
            front_logps = (per_token_logps * (ld_mask & mask).float()).sum(dim = 1)
            rear_logps = (per_token_logps * (~ld_mask & mask).float()).sum(dim = 1)
            all_logps = front_logps + ld_alpha * rear_logps
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        # logits[rows][loss_mask].mean(): every position has V logits, so the mean of per-position means is equal.
        output["mean_chosen_logits"] = mean_logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = mean_logits[num_examples:][loss_mask[num_examples:]].mean()
        return output

    concatenated_forward._unsloth_cpu = True
    return concatenated_forward


def _kto_chunked_forward(trainer, original):
    """TRL 0.2x KTOTrainer.forward without the [B, L, V] completion / KL logits: summed log-probs from the last
    hidden state in row chunks. TRL only ever takes .nansum() of the logits it returns, so the returned stand-ins are
    the per-position vocabulary sums ([B, L], same total). Encoder-decoder and MoE aux-loss models, or
    UNSLOTH_CPU_CHUNKED_PREFERENCE=0, run TRL's own forward."""
    from .ops import _plain_head_weight

    def scores(target, model, input_ids, attention_mask, labels):
        pad = getattr(trainer, "label_pad_token_id", -100)  # TRL 0.27+ (trl.experimental.kto) hardcodes -100
        shifted = labels[:, 1:]
        loss_mask = shifted != pad
        # One target per position (the last one a dummy) so every position's logits sum is scored, as TRL sums them.
        targets = torch.cat((shifted.masked_fill(~loss_mask, 0), shifted.new_zeros(shifted.shape[0], 1)), dim = 1)
        outputs = _chunked_token_logps(target, model, input_ids, attention_mask, targets)
        if outputs is None:
            return None
        logps = (outputs["log_probs"][:, :-1] * loss_mask).sum(-1)
        vocab = _plain_head_weight(target.get_output_embeddings()).shape[0]
        return logps, outputs["mean_logits"] * float(vocab)

    def forward(model, batch):
        target = _chunked_head_target(trainer, model)
        if target is None or batch["completion_input_ids"].shape != batch["completion_labels"].shape:
            return original(model, batch)
        KL_logps = None
        if trainer.calculate_KL:
            with torch.no_grad():
                kl = scores(target, model, batch["KL_completion_input_ids"], batch["KL_completion_attention_mask"],
                            batch["KL_completion_labels"])
            if kl is None:
                return original(model, batch)
            KL_logps = kl[0]
        completion = scores(target, model, batch["completion_input_ids"], batch["completion_attention_mask"],
                            batch["completion_labels"])
        if completion is None:
            return original(model, batch)
        completion_logps, completion_logit_sums = completion
        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )
        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]
        return (completion_logps[chosen_idx, ...], completion_logps[rejected_idx, ...],
                completion_logit_sums[chosen_idx, ...], completion_logit_sums[rejected_idx, ...], KL_logps)

    forward._unsloth_cpu = True
    return forward


# trainer module suffix -> (the method replaced per instance, the methods it relies on / feeds, factory)
_CHUNKED_PREFERENCE_FORWARDS = {
    "orpo_trainer": ("concatenated_forward", ("concatenated_inputs", "get_batch_logps", "get_batch_loss_metrics"),
                     _orpo_chunked_forward),
    "dpo_trainer": ("concatenated_forward", ("concatenated_inputs", "get_batch_loss_metrics", "compute_ref_log_probs"),
                    _dpo_chunked_forward),
    "kto_trainer": ("forward", ("get_batch_logps", "get_batch_loss_metrics", "_compute_kl_logps"),
                    _kto_chunked_forward),
}


def _install_chunked_preference_forward(trainer):
    """Per trainer instance: TRL's own ORPOTrainer (trl.trainer.orpo_trainer up to 0.2x, trl.experimental.orpo in 1.x)
    and TRL 0.2x's DPOTrainer / KTOTrainer (1.x has neither method: its fused LM head is already chunked on CPU), not
    a subclass that overrides the replaced method or what it relies on / feeds (its scoring and logits consumer)."""
    if os.environ.get("UNSLOTH_CPU_CHUNKED_PREFERENCE", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    module = getattr(getattr(type(trainer), "get_batch_loss_metrics", None), "__module__", "") or ""
    if not module.startswith("trl."):
        return False
    spec = _CHUNKED_PREFERENCE_FORWARDS.get(module.rsplit(".", 1)[-1])
    if spec is None:
        return False
    method, relies_on, factory = spec
    for name in (method, *relies_on):
        if (getattr(getattr(type(trainer), name, None), "__module__", "") or "") != module:
            return False
    bound = getattr(trainer, method)
    if getattr(bound, "_unsloth_cpu", False):
        return True
    setattr(trainer, method, factory(trainer, bound))
    return True


def _patch_cpu_preference_trainers(trainer_cls):
    """Trainer.__init__ builds the accelerator once per trainer: the hook that sees every TRL trainer instance without
    importing trl.experimental (which warns on import)."""
    original = trainer_cls.create_accelerator_and_postprocess
    if getattr(original, "_unsloth_cpu_preference", False):
        return

    @functools.wraps(original)
    def create_accelerator_and_postprocess(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _install_chunked_preference_forward(self)
        except Exception:
            pass
        return result

    create_accelerator_and_postprocess._unsloth_cpu = getattr(original, "_unsloth_cpu", False)
    create_accelerator_and_postprocess._unsloth_cpu_preference = True
    trainer_cls.create_accelerator_and_postprocess = create_accelerator_and_postprocess


def _count_sft_tokens(trainer, model, inputs):
    """TRL's `num_tokens` bookkeeping from SFTTrainer.compute_loss, which the chunked path bypasses."""
    metrics = getattr(trainer, "_metrics", None)
    if metrics is None or not hasattr(trainer, "_total_train_tokens"):
        return
    mode = "train" if model.training else "eval"
    if mode == "train":
        mask = inputs.get("attention_mask", None)
        if mask is not None:
            trainer._total_train_tokens += trainer.accelerator.gather_for_metrics(mask.sum()).sum().item()
        elif inputs.get("position_ids", None) is not None:
            trainer._total_train_tokens += inputs["position_ids"].size(1)
    metrics[mode]["num_tokens"] = [trainer._total_train_tokens]


def _patch_sft_compute_loss(sft_trainer_cls):
    """Chunked CE returns no logits; TRL's entropy / token-accuracy metrics need them. Skip just those."""
    original = sft_trainer_cls.compute_loss
    if getattr(original, "_unsloth_cpu", False):
        return
    from transformers import Trainer

    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None, **kwargs):
        chunked = None
        unwrapped = self.accelerator.unwrap_model(model) if getattr(self, "accelerator", None) is not None else model
        for candidate in (unwrapped, getattr(unwrapped, "base_model", None), getattr(getattr(unwrapped, "base_model", None), "model", None)):
            # vars(), not getattr: PEFT forwards attribute reads to the inner model, but writes stay on the wrapper.
            if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False):
                chunked = candidate
                break
        if chunked is None:
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        if not torch.is_grad_enabled():
            # Evaluation / prediction: the stock forward returns logits, so keep TRL's entropy and accuracy metrics.
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        if str(_name(getattr(self.args, "loss_type", None) or "nll")) != "nll":
            # TRL's other SFT objectives (dft, ...) weight per-token logits: stock forward.
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        # Only this SFT step turns the chunked loss on; DPO / KTO / ORPO and plain forwards keep logits.
        inputs["use_cache"] = False
        chunked._unsloth_cpu_chunked_ce_active = True
        try:
            result = Trainer.compute_loss(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch)
        finally:
            chunked._unsloth_cpu_chunked_ce_active = False
        _count_sft_tokens(self, model, inputs)
        return result

    compute_loss._unsloth_cpu = True
    sft_trainer_cls.compute_loss = compute_loss


# Notebooks still pass these to SFTTrainer; stock TRL (CPU backend) moved them to SFTConfig.
_LEGACY_SFT_KWARGS = {
    "dataset_text_field": "dataset_text_field",
    "max_seq_length": "max_length",
    "packing": "packing",
    "dataset_num_proc": "dataset_num_proc",
}


def _patch_max_seq_length_alias(config_cls):
    """TRL >= 0.20 dropped SFTConfig(max_seq_length), which notebooks and Studio pass: map it to max_length."""
    fields = getattr(config_cls, "__dataclass_fields__", {})
    if "max_seq_length" in fields or "max_length" not in fields:
        return
    original = config_cls.__init__
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if "max_seq_length" in kwargs:
            max_seq_length = kwargs.pop("max_seq_length")
            if max_seq_length is not None:
                kwargs["max_length"] = max_seq_length
        return original(self, *args, **kwargs)

    __init__._unsloth_cpu = True
    config_cls.__init__ = __init__


def _sft_loss_takes_num_items(trainer):
    """The CPU SFT loss (chunked CE or TRL's fused LM head) divides by num_items_in_batch itself, whatever the
    model's accepts_loss_kwargs says."""
    if getattr(trainer, "compute_loss_func", None) is not None or getattr(trainer, "label_smoother", None) is not None:
        return False
    loss_type = str(_name(getattr(trainer.args, "loss_type", None) or "nll"))
    model = trainer.model
    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False) and loss_type == "nll":
            return True
    try:
        import trl.trainer.utils as trl_utils
    except Exception:
        return False
    return hasattr(trl_utils, "add_fused_lm_head") and loss_type in ("nll", "dft")


def _count_items_for_sft_loss(trainer):
    """Without the count each micro-batch loss is its own mean, so accumulation / DDP weight rows wrongly."""
    if getattr(trainer, "model_accepts_loss_kwargs", True) or not _sft_loss_takes_num_items(trainer):
        return False
    trainer.model_accepts_loss_kwargs = True
    return True


def _make_cpu_sft_trainer(base):
    if getattr(base, "_unsloth_cpu", False):
        return base

    class SFTTrainer(base):
        _unsloth_cpu = True

        def __init__(self, *args, **kwargs):
            tokenizer = kwargs.pop("tokenizer", None)
            if tokenizer is not None and kwargs.get("processing_class") is None:
                kwargs["processing_class"] = tokenizer
            legacy = {k: kwargs.pop(k) for k in list(kwargs) if k in _LEGACY_SFT_KWARGS}
            if legacy:
                from trl import SFTConfig
                args = list(args)
                training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
                if training_args is None:
                    training_args = SFTConfig(output_dir = "outputs")
                elif not isinstance(training_args, SFTConfig):
                    # TRL rebuilds a plain TrainingArguments from to_dict(), which drops attributes set here.
                    values = training_args.to_dict()
                    values["hub_token"] = getattr(training_args, "hub_token", None)
                    values.pop("push_to_hub_token", None)
                    training_args = SFTConfig(**values)
                if "args" in kwargs or len(args) <= 1:
                    kwargs["args"] = training_args
                else:
                    args[1] = training_args
                for key, value in legacy.items():
                    setattr(training_args, _LEGACY_SFT_KWARGS[key], value)
                normalize_cpu_training_arguments(training_args)
            # Configs write in-process as dataset_num_proc = 1, but datasets >= 4.1 builds a Pool(1) for it
            # (under spawn its child re-imports an unguarded __main__): stock TRL must see None while it maps.
            training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
            if training_args is not None:
                _check_cpu_packing(kwargs.get("model", args[0] if args else None), training_args)
                _auto_cpu_gradient_checkpointing(kwargs.get("model", args[0] if args else None), training_args)
            num_proc = getattr(training_args, "dataset_num_proc", None)
            serial = isinstance(num_proc, int) and not isinstance(num_proc, bool) and 0 <= num_proc <= 1
            if serial:
                training_args.dataset_num_proc = None
            try:
                super().__init__(*args, **kwargs)
            finally:
                if serial:
                    training_args.dataset_num_proc = num_proc
            _count_items_for_sft_loss(self)
            _install_compile_gate(self)

    # Pickled by reference: by value (TRL's tokenize map, dataset_num_proc > 1) fails on torch config modules.
    SFTTrainer.__module__ = __name__
    SFTTrainer.__qualname__ = "_CPUSFTTrainer"
    return SFTTrainer


def __getattr__(name):
    # Spawn workers unpickling the CPU class get it, not stock TRL's SFTTrainer.
    if name == "_CPUSFTTrainer":
        install_cpu_training_defaults()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def install_cpu_training_defaults():
    """Idempotent. Only ever called from a CPU-backend process."""
    import sys
    from .runtime import fit_cpu_threads
    # Before any kernel runs: the pool size is read on first use by every op.
    fit_cpu_threads()
    if os.environ.get("UNSLOTH_CPU_SDPA_GQA", "1") != "0":
        from .ops import patch_cpu_sdpa_gqa
        patch_cpu_sdpa_gqa()
    if os.environ.get("UNSLOTH_CPU_LORA_INPUT_CAST", "0") != "1":
        from .ops import patch_peft_lora_autocast_input
        patch_peft_lora_autocast_input()
    from transformers import Trainer, TrainingArguments
    _patch_post_init(TrainingArguments)
    if hasattr(Trainer, "create_accelerator_and_postprocess"):
        _patch_cpu_fp16_accelerator(Trainer)
        _patch_cpu_preference_trainers(Trainer)
    _patch_cpu_autocast_cache(Trainer)
    try:
        import trl
        from trl.trainer import sft_trainer
    except Exception:
        return
    for name in ("SFTConfig", "DPOConfig", "GRPOConfig", "ORPOConfig", "KTOConfig", "RewardConfig"):
        cls = getattr(trl, name, None)
        if cls is not None and "__post_init__" in vars(cls):
            _patch_post_init(cls)
    if getattr(trl, "SFTConfig", None) is not None:
        _patch_max_seq_length_alias(trl.SFTConfig)
    _patch_sft_compute_loss(sft_trainer.SFTTrainer)
    from .ops import patch_trl_fused_lm_head
    patch_trl_fused_lm_head()
    trainer = _make_cpu_sft_trainer(sft_trainer.SFTTrainer)
    globals()["_CPUSFTTrainer"] = trainer
    for module in (sft_trainer, trl, sys.modules.get("trl.trainer")):
        if module is not None:
            setattr(module, "SFTTrainer", trainer)
