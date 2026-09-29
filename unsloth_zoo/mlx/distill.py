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

"""Off-policy knowledge distillation (GKD) for the MLX trainer: a frozen teacher
scores the student's batch and the student trains on TRL's generalized JSD."""

# Bound at import: a deferred `import mlx.core` would resolve to the
# tests/mlx_simulation stub once that has replaced sys.modules["mlx.core"].
import mlx.core as mx
import mlx.nn as nn

from .utils import _model_logits, _normalize_cce_label_dtype

__all__ = [
    "generalized_jsd_loss",
    "assert_tokenizers_compatible",
    "estimate_distillation_peak_bytes",
    "largest_tokens_that_fit",
    "preflight_memory",
    "validate_gkd_config",
    "TOKENIZER_PROBES",
    "DEFAULT_CHUNK_SIZE",
    "NAIVE_ACTIVATION_MULTIPLIER",
    "CHUNKED_ACTIVATION_MULTIPLIER",
    "MEMORY_SAFETY_FRACTION",
]

DEFAULT_CHUNK_SIZE = 128

# Peak as a multiple of one batch*seq*vocab float32 logit buffer, fitted on a
# Qwen2.5-0.5B student + Qwen2.5-3B teacher and rounded up: an oversized step
# does not raise, it aborts with "[METAL] Command buffer execution failed" and
# wedges the GPU context.
NAIVE_ACTIVATION_MULTIPLIER = 16.0
CHUNKED_ACTIVATION_MULTIPLIER = 13.0
MEMORY_SAFETY_FRACTION = 0.85

TOKENIZER_PROBES = (
    "The capital of France is Paris.",
    "user\nhi\nassistant\n",
    "def f(x):\n    return x**2  # comment",
    "emoji and accents: cafe, naive, jalapeno",
    "1234567890 !@#$%^&*()",
)


def _kl_divergence_log_target(input_log_probs, target_log_probs):
    """Elementwise ``F.kl_div(input, target, log_target=True)``."""
    return mx.exp(target_log_probs) * (target_log_probs - input_log_probs)


def _jsd_per_position(student_logits, teacher_logits, beta, temperature):
    # float32: the mixture is a logaddexp of two near-equal terms that the outer
    # combination then subtracts, which bf16 cannot resolve.
    student_log_probs = nn.log_softmax(student_logits.astype(mx.float32) / temperature, axis=-1)
    teacher_log_probs = nn.log_softmax(teacher_logits.astype(mx.float32) / temperature, axis=-1)

    if beta == 0.0:
        per_token = _kl_divergence_log_target(student_log_probs, teacher_log_probs)
    elif beta == 1.0:
        per_token = _kl_divergence_log_target(teacher_log_probs, student_log_probs)
    else:
        beta_array = mx.array(beta, dtype=mx.float32)
        mixture_log_probs = mx.logaddexp(
            student_log_probs + mx.log(1 - beta_array),
            teacher_log_probs + mx.log(beta_array),
        )
        per_token = (
            beta * _kl_divergence_log_target(mixture_log_probs, teacher_log_probs)
            + (1 - beta) * _kl_divergence_log_target(mixture_log_probs, student_log_probs)
        )
    return per_token.sum(axis=-1)


def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels = None,
    beta = 0.5,
    temperature = 1.0,
    chunk_size = DEFAULT_CHUNK_SIZE,
):
    """TRL's ``GKDTrainer.generalized_jsd_loss`` in MLX: beta 0 is forward KL,
    1 reverse KL; mean over non -100 positions (TRL's labels=None batchmean
    divides by batch size instead); ``chunk_size`` positions at a time."""
    if labels is None:
        mask = mx.ones(student_logits.shape[:2])
    else:
        mask = (labels != -100)
    denominator = mx.maximum(mask.sum(), 1)

    sequence_length = student_logits.shape[1]
    if not _is_chunked(chunk_size, sequence_length):
        per_position = _jsd_per_position(student_logits, teacher_logits, beta, temperature)
        return (per_position * mask).sum() / denominator

    total = mx.zeros(())
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        per_position = _jsd_per_position(
            student_logits[:, start:end, :],
            teacher_logits[:, start:end, :],
            beta,
            temperature,
        )
        total = total + (per_position * mask[:, start:end]).sum()
    return total / denominator


def _is_chunked(chunk_size, sequence_length):
    return chunk_size is not None and 0 < chunk_size < sequence_length


def assert_tokenizers_compatible(student_tokenizer, teacher_tokenizer,
                                 student_vocab_size = None, teacher_vocab_size = None):
    """Raise unless the vocab widths match and both tokenizers encode the probes
    identically: equal widths alone still permit misaligned targets."""
    if (student_vocab_size is not None and teacher_vocab_size is not None
            and student_vocab_size != teacher_vocab_size):
        raise ValueError(
            "Unsloth: GKD needs the teacher and student to share a tokenizer, but "
            f"their logit widths differ (student {student_vocab_size}, teacher "
            f"{teacher_vocab_size}). Cross-family distillation would need token "
            "remapping, which is not implemented. Pick a teacher from the "
            "student's family (e.g. Qwen2.5-3B for a Qwen2.5/Qwen3 student)."
        )
    if student_tokenizer is None or teacher_tokenizer is None:
        import warnings
        warnings.warn(
            "Unsloth: GKD could not compare the student and teacher tokenizers "
            "(no tokenizer given); only the logit widths were checked. Token ids "
            "must mean the same thing to both models.",
            UserWarning,
            stacklevel = 2,
        )
        return True

    _assert_vocab_mappings_match(student_tokenizer, teacher_tokenizer)
    for probe in TOKENIZER_PROBES:
        student_ids = student_tokenizer.encode(probe)
        teacher_ids = teacher_tokenizer.encode(probe)
        if list(student_ids) != list(teacher_ids):
            raise ValueError(
                "Unsloth: GKD needs the teacher and student to share a tokenizer. "
                "Their logit widths match but they encode text differently, so the "
                "teacher's distribution would not line up with the student's "
                f"tokens. First disagreement on {probe!r}: student produced "
                f"{len(student_ids)} ids, teacher produced {len(teacher_ids)}. "
                "Training would silently optimise against misaligned targets."
            )
    return True


def _assert_vocab_mappings_match(student_tokenizer, teacher_tokenizer):
    """Every token must map to the same id. Differences confined to added
    tokens (Qwen3's <think> vs a Qwen2.5 teacher) warn: those ids only matter
    when the data contains them. Tokenizers without get_vocab skip this."""
    try:
        student_vocab = student_tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()
    except Exception:
        return
    differing = sorted(
        token for token in student_vocab.keys() | teacher_vocab.keys()
        if student_vocab.get(token) != teacher_vocab.get(token)
    )
    if not differing:
        return
    added = set()
    for tokenizer in (student_tokenizer, teacher_tokenizer):
        try:
            added.update(tokenizer.get_added_vocab())
        except Exception:
            pass
    base = [token for token in differing if token not in added]
    if base:
        raise ValueError(
            "Unsloth: GKD needs the teacher and student to share a tokenizer, but "
            f"{len(base)} vocabulary tokens map to different ids (e.g. "
            f"{base[:5]!r}), so the teacher's distribution would not line up "
            "with the student's tokens."
        )
    import warnings
    warnings.warn(
        f"Unsloth: GKD teacher and student disagree on {len(differing)} added "
        f"tokens ({differing[:5]!r}); the teacher's targets at those ids are "
        "not meaningful if your data contains them.",
        UserWarning,
        stacklevel = 3,
    )


def estimate_distillation_peak_bytes(batch_size, sequence_length, vocab_size,
                                     resident_bytes, chunked = True,
                                     bytes_per_element = 4):
    logit_buffer = batch_size * sequence_length * vocab_size * bytes_per_element
    multiplier = CHUNKED_ACTIVATION_MULTIPLIER if chunked else NAIVE_ACTIVATION_MULTIPLIER
    return int(resident_bytes + multiplier * logit_buffer)


def largest_tokens_that_fit(vocab_size, resident_bytes, budget_bytes,
                            chunked = True, bytes_per_element = 4):
    multiplier = CHUNKED_ACTIVATION_MULTIPLIER if chunked else NAIVE_ACTIVATION_MULTIPLIER
    headroom = budget_bytes - resident_bytes
    if headroom <= 0:
        return 0
    return int(headroom / (multiplier * vocab_size * bytes_per_element))


def preflight_memory(batch_size, sequence_length, vocab_size, resident_bytes,
                     system_bytes, chunked = True, bytes_per_element = 4,
                     skip = False):
    """Raise before the first step if this configuration cannot fit; ``skip``
    downgrades the refusal to a warning since the estimate is conservative."""
    budget = int(system_bytes * MEMORY_SAFETY_FRACTION)
    estimate = estimate_distillation_peak_bytes(
        batch_size, sequence_length, vocab_size, resident_bytes,
        chunked = chunked, bytes_per_element = bytes_per_element,
    )
    if estimate <= budget:
        return estimate

    gb = 1024 ** 3
    if skip:
        import warnings
        warnings.warn(
            "Unsloth: gkd_skip_memory_preflight=True, proceeding with a GKD "
            f"configuration estimated at {estimate / gb:.2f} GB against a "
            f"{budget / gb:.2f} GB usable budget "
            f"({system_bytes / gb:.2f} GB system). If this does not fit, the "
            "process will NOT raise a recoverable error -- it aborts with "
            "'[METAL] Command buffer execution failed: Impacting Interactivity' "
            "and leaves the GPU context wedged for subsequent runs.",
            UserWarning,
            stacklevel = 2,
        )
        return estimate

    max_tokens = largest_tokens_that_fit(
        vocab_size, resident_bytes, budget,
        chunked = chunked, bytes_per_element = bytes_per_element,
    )
    raise ValueError(
        "Unsloth: this GKD configuration does not fit in memory.\n"
        f"  requested      : batch_size={batch_size} x seq_len={sequence_length} "
        f"= {batch_size * sequence_length} tokens\n"
        f"  estimated peak : {estimate / gb:.2f} GB"
        f"{'' if chunked else ' (unchunked loss)'}\n"
        f"  system memory  : {system_bytes / gb:.2f} GB "
        f"(usable budget {budget / gb:.2f} GB at "
        f"{int(MEMORY_SAFETY_FRACTION * 100)}%)\n"
        f"  largest that fits at vocab {vocab_size}: "
        f"{max_tokens} tokens (e.g. batch_size=1 x seq_len={max_tokens})\n"
        "Reduce per_device_train_batch_size or max_seq_length, or use a smaller "
        "teacher, or set gkd_skip_memory_preflight=True. Proceeding would abort "
        "the process with a Metal command-buffer failure rather than a "
        "recoverable error."
    )


def validate_gkd_config(gkd_beta, gkd_temperature, gkd_lmbda):
    if not (0.0 <= gkd_beta <= 1.0):
        raise ValueError(
            f"Unsloth: gkd_beta must be in [0, 1], got {gkd_beta}. 0 is forward KL "
            "(classic KD), 1 is reverse KL, 0.5 is symmetric JSD."
        )
    if gkd_temperature <= 0.0:
        raise ValueError(
            f"Unsloth: gkd_temperature must be > 0, got {gkd_temperature}."
        )
    if gkd_lmbda:
        raise ValueError(
            f"Unsloth: gkd_lmbda={gkd_lmbda} selects on-policy distillation, where "
            "the student generates and the teacher scores its samples. That needs "
            "generation inside the training loop, which the MLX trainer does not "
            "have. Only off-policy KD (gkd_lmbda=0, teacher scores your dataset) "
            "is supported."
        )
    return True


def load_teacher(model_name_or_path):
    """Load a teacher (mlx_lm.load returns it in eval mode) and freeze it."""
    from mlx.utils import tree_flatten
    from mlx_lm import load

    teacher, teacher_tokenizer = load(model_name_or_path)
    teacher.freeze()
    trainable = tree_flatten(teacher.trainable_parameters())
    if trainable:
        raise RuntimeError(
            f"Unsloth: teacher {model_name_or_path} still reports "
            f"{len(trainable)} trainable tensors after freeze(); it would be "
            "pulled into the optimizer state."
        )
    return teacher, teacher_tokenizer


def build_gkd_loss_fn(teacher_model, args, vocab_size, batch_size,
                      resident_bytes, system_bytes):
    """Loss fn with the trainer's ``(model, batch, lengths, labels=None) ->
    (loss, ntoks)`` contract; the teacher runs under ``stop_gradient``."""
    beta = float(getattr(args, "gkd_beta", 0.5))
    temperature = float(getattr(args, "gkd_temperature", 1.0))
    chunk_size = int(getattr(args, "gkd_chunk_size", DEFAULT_CHUNK_SIZE))
    validate_gkd_config(beta, temperature, getattr(args, "gkd_lmbda", 0.0))

    # The loss sees max_seq_length - 1 positions at most. A shorter batch that
    # fits in one chunk runs unchunked, so budget whichever branch costs more.
    sequence_length = max(1, int(getattr(args, "max_seq_length", 512)) - 1)
    chunked = _is_chunked(chunk_size, sequence_length)
    if chunked and NAIVE_ACTIVATION_MULTIPLIER * chunk_size > CHUNKED_ACTIVATION_MULTIPLIER * sequence_length:
        sequence_length, chunked = chunk_size, False
    preflight_memory(
        batch_size = batch_size,
        sequence_length = sequence_length,
        vocab_size = vocab_size,
        resident_bytes = resident_bytes,
        system_bytes = system_bytes,
        chunked = chunked,
        skip = bool(getattr(args, "gkd_skip_memory_preflight", False)),
    )

    def loss_fn(model, batch, lengths, labels=None):
        # Same positions as make_baseline_loss_fn: the lengths window, further
        # restricted by -100 labels when labels are given.
        inputs = batch[:, :-1]
        steps = mx.arange(1, inputs.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        if labels is not None:
            mask = mx.logical_and(mask, _normalize_cce_label_dtype(labels[:, 1:]) != -100)
        shifted_labels = mx.where(mask, mx.zeros_like(inputs), mx.full(inputs.shape, -100))
        student_logits = _model_logits(model(inputs))
        teacher_logits = mx.stop_gradient(_model_logits(teacher_model(inputs)))
        loss = generalized_jsd_loss(
            student_logits, teacher_logits, shifted_labels,
            beta = beta, temperature = temperature, chunk_size = chunk_size,
        )
        return loss, mask.sum()

    loss_fn._unsloth_gkd = True
    return loss_fn
