# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Off-policy knowledge distillation (GKD) for the MLX trainer.

A frozen teacher scores the student's own batch; the student is trained against
the teacher's distribution with TRL's generalized Jensen-Shannon divergence.

Three things here are load-bearing and were established by measurement, not
assumption. Read before changing any of them.

1. TEACHER AND STUDENT MUST SHARE A TOKENIZER. Matching ``vocab_size`` alone is
   NOT sufficient: two checkpoints can report the same width while encoding text
   differently, which trains the student against misaligned targets and produces
   no error at all. ``assert_tokenizers_compatible`` therefore compares encoded
   ids on probe strings as well as the logit width. Verified working pairs
   include Qwen2.5-0.5B <- Qwen2.5-3B, Llama-3.2-1B <- Llama-3.2-3B, and
   Qwen3-0.6B <- Qwen2.5-3B (tokenizers agree across generations). Llama <-> Qwen
   in either direction fails the width check.

2. THE LOSS IS CHUNKED BY DEFAULT. The divergence materializes several
   [batch, seq, vocab] float32 tensors, and vocab is ~152k for Qwen. Chunking
   over sequence positions took a measured batch=2/seq=1024 step from 17.39 GB
   (over a 16 GB machine, crashes) to 15.29 GB (fits). The unchunked path is kept
   only so a test can assert the two agree numerically.

3. OVERSIZED BATCHES DO NOT RAISE A CATCHABLE OOM. On Apple Silicon they abort
   the process with

       [METAL] Command buffer execution failed: Impacting Interactivity

   and leave the GPU context wedged for subsequent runs. ``preflight_memory``
   therefore estimates peak up front and raises a normal Python error instead.
   Its estimator is a pure function so it is testable without MLX installed.
"""

# Bind MLX at import, NOT inside each function. Deferred `import mlx.core` calls
# resolve against sys.modules at call time, so anything that swaps in a stub
# afterwards (tests/mlx_simulation does exactly this) would silently hand this
# module the stub instead of real MLX. Binding here captures whichever MLX was
# present when this module was first imported and keeps it.
import mlx.core as mx
import mlx.nn as nn

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

# Sequence positions per chunk. 128 was measured; larger trades memory for fewer
# kernel launches.
DEFAULT_CHUNK_SIZE = 128

# Peak activation cost expressed as a multiple of one teacher-logit buffer
# (batch * seq * vocab * 4 bytes). Fitted from measured training steps on a
# Qwen2.5-0.5B LoRA student with a Qwen2.5-3B teacher (vocab 151936):
#   unchunked  batch/seq  1/512 -> 7.47 GB, 2/512 -> 12.05 GB, 2/1024 -> 17.39 GB
#   chunked    batch/seq  2/512 -> 10.46 GB, 2/1024 -> 15.29 GB
# Rounded UP: a preflight that under-estimates lets the process wedge Metal.
NAIVE_ACTIVATION_MULTIPLIER = 16.0
CHUNKED_ACTIVATION_MULTIPLIER = 13.0

# Never plan to fill RAM. Unified memory is shared with the OS and display.
MEMORY_SAFETY_FRACTION = 0.85

# Short, deliberately varied strings: plain text, chat control tokens, code with
# runs of whitespace, and non-ASCII. Two tokenizers that agree on all of these
# and on logit width are treated as interchangeable for distillation.
TOKENIZER_PROBES = (
    "The capital of France is Paris.",
    "user\nhi\nassistant\n",
    "def f(x):\n    return x**2  # comment",
    "emoji and accents: cafe, naive, jalapeno",
    "1234567890 !@#$%^&*()",
)


def _kl_divergence_log_target(input_log_probs, target_log_probs):
    """Elementwise ``F.kl_div(input, target, log_target=True)``.

    That is ``exp(target) * (target - input)``, matching PyTorch so the port of
    TRL's loss stays comparable term by term.
    """
    return mx.exp(target_log_probs) * (target_log_probs - input_log_probs)


def _jsd_per_position(student_logits, teacher_logits, beta, temperature):
    """Divergence summed over the vocab axis, one value per (batch, position)."""
    student_log_probs = nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_log_probs = nn.log_softmax(teacher_logits / temperature, axis=-1)

    if beta == 0.0:
        # Forward KL: classic KD, teacher as the target distribution.
        per_token = _kl_divergence_log_target(student_log_probs, teacher_log_probs)
    elif beta == 1.0:
        # Reverse KL: mode-seeking. Note it does NOT minimise forward KL.
        per_token = _kl_divergence_log_target(teacher_log_probs, student_log_probs)
    else:
        beta_array = mx.array(beta, dtype=student_log_probs.dtype)
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
    """TRL's ``GKDTrainer.generalized_jsd_loss`` ported to MLX.

    ``beta=0`` is forward KL (classic KD), ``beta=1`` reverse KL, and values in
    between interpolate against the log-space mixture. Positions where ``labels``
    is -100 are excluded; the mean is over supervised positions only.

    ``chunk_size`` splits the sequence axis so only ``[batch, chunk, vocab]``
    intermediates are alive at once. Pass ``chunk_size=0`` for the unchunked
    path, which exists so tests can assert the two agree numerically -- it is not
    the default because it costs ~2 GB more at batch=2/seq=1024.
    """
    if labels is None:
        mask = mx.ones(student_logits.shape[:2])
    else:
        mask = (labels != -100)
    denominator = mx.maximum(mask.sum(), 1)

    sequence_length = student_logits.shape[1]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= sequence_length:
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


def assert_tokenizers_compatible(student_tokenizer, teacher_tokenizer,
                                 student_vocab_size = None, teacher_vocab_size = None):
    """Raise unless the two tokenizers are interchangeable for distillation.

    Width equality is checked first because a mismatch is what MLX would report
    as an opaque ``broadcast_shapes`` failure deep in the loss. Probe-id equality
    is checked second because width equality ALONE is not sufficient: two
    checkpoints can share a vocab size and still encode text differently, which
    would silently train the student against misaligned targets.
    """
    if (student_vocab_size is not None and teacher_vocab_size is not None
            and student_vocab_size != teacher_vocab_size):
        raise ValueError(
            "Unsloth: GKD needs the teacher and student to share a tokenizer, but "
            f"their logit widths differ (student {student_vocab_size}, teacher "
            f"{teacher_vocab_size}). Cross-family distillation would need token "
            "remapping, which is not implemented. Pick a teacher from the "
            "student's family (e.g. Qwen2.5-3B for a Qwen2.5/Qwen3 student)."
        )

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


def estimate_distillation_peak_bytes(batch_size, sequence_length, vocab_size,
                                     resident_bytes, chunked = True,
                                     bytes_per_element = 4):
    """Estimated peak bytes for one distillation training step.

    Pure arithmetic with no MLX import, so it is testable anywhere. Peak is
    modelled as the already-resident weights plus a multiple of one teacher-logit
    buffer; the multiplier was fitted from measured steps and rounded up, because
    under-estimating here lets the process wedge the GPU rather than raise.
    """
    logit_buffer = batch_size * sequence_length * vocab_size * bytes_per_element
    multiplier = CHUNKED_ACTIVATION_MULTIPLIER if chunked else NAIVE_ACTIVATION_MULTIPLIER
    return int(resident_bytes + multiplier * logit_buffer)


def largest_tokens_that_fit(vocab_size, resident_bytes, budget_bytes,
                            chunked = True, bytes_per_element = 4):
    """Largest batch*sequence token count that fits in ``budget_bytes``."""
    multiplier = CHUNKED_ACTIVATION_MULTIPLIER if chunked else NAIVE_ACTIVATION_MULTIPLIER
    headroom = budget_bytes - resident_bytes
    if headroom <= 0:
        return 0
    return int(headroom / (multiplier * vocab_size * bytes_per_element))


def preflight_memory(batch_size, sequence_length, vocab_size, resident_bytes,
                     system_bytes, chunked = True, bytes_per_element = 4,
                     skip = False):
    """Raise before the first step if this configuration cannot fit.

    An oversized distillation step does not raise a catchable OOM on Apple
    Silicon; it aborts with ``[METAL] Command buffer execution failed: Impacting
    Interactivity`` and leaves the GPU context wedged for later runs. Failing
    here with a normal Python error is the whole point.

    ``skip=True`` downgrades the refusal to a warning and proceeds. The estimator
    is deliberately conservative and reads total system memory, so it can reject a
    configuration that would actually have fit; refusing by default is right, but
    refusing with no recourse is not.
    """
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
        "teacher. Proceeding would abort the process with a Metal command-buffer "
        "failure rather than a recoverable error."
    )


def validate_gkd_config(gkd_beta, gkd_temperature, gkd_lmbda):
    """Config gate for the GKD knobs. Pure, so the Linux gate can execute it."""
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
    """Load a teacher, freeze it, and assert it contributes no trainable state.

    Frozen once here rather than per step: the teacher must stay out of
    ``trainable_parameters()``, out of the optimizer state, and out of the state
    tuple handed to ``mx.compile``.
    """
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


def build_gkd_loss_fn(student_model, teacher_model, args, is_vlm,
                      vocab_size, resident_bytes, system_bytes):
    """Return a loss function matching the trainer's ``(model, batch, lengths,
    labels=None) -> (loss, ntoks)`` contract.

    The teacher forward runs under ``stop_gradient``: it is evaluated inside the
    traced step (the contract gives us no way to hand logits in) but contributes
    no gradient and no optimizer state.
    """
    beta = float(getattr(args, "gkd_beta", 0.5))
    temperature = float(getattr(args, "gkd_temperature", 1.0))
    chunk_size = int(getattr(args, "gkd_chunk_size", DEFAULT_CHUNK_SIZE))
    validate_gkd_config(beta, temperature, getattr(args, "gkd_lmbda", 0.0))

    if is_vlm:
        raise ValueError(
            "Unsloth: GKD distillation is text-only on MLX. The teacher would "
            "need the student's exact image preprocessing for its logits to line "
            "up, which is not implemented."
        )

    preflight_memory(
        batch_size = int(getattr(args, "per_device_train_batch_size", 1)),
        sequence_length = int(getattr(args, "max_seq_length", 512)),
        vocab_size = vocab_size,
        resident_bytes = resident_bytes,
        system_bytes = system_bytes,
        chunked = chunk_size > 0,
        skip = bool(getattr(args, "gkd_skip_memory_preflight", False)),
    )

    def loss_fn(model, batch, lengths, labels=None):
        inputs = batch[:, :-1]
        if labels is None:
            steps = mx.arange(1, inputs.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            shifted_labels = mx.where(mask, mx.zeros_like(inputs), mx.full(inputs.shape, -100))
        else:
            shifted_labels = labels[:, 1:]
            mask = (shifted_labels != -100)
        student_logits = model(inputs)
        teacher_logits = mx.stop_gradient(teacher_model(inputs))
        loss = generalized_jsd_loss(
            student_logits, teacher_logits, shifted_labels,
            beta = beta, temperature = temperature, chunk_size = chunk_size,
        )
        return loss, mask.sum()

    loss_fn._unsloth_gkd = True
    return loss_fn

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
