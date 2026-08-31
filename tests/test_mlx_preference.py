"""Focused behavioral coverage for MLX ORPO and DPO."""

from __future__ import annotations

import math
import sys
import types
import warnings

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    prefixes = ("mlx", "mlx_lm", "mlx_vlm")

    def _owned(name):
        return (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        )

    # The unsloth_zoo.mlx.* entries are saved as well as the mlx* ones, because
    # this fixture drops them and the next importer therefore builds a NEW module
    # object. Anything that ran earlier in this process and did
    # `from unsloth_zoo.mlx.utils import ...` at import time keeps the OLD one, so
    # it and the code under test end up on two copies of the same module globals:
    # the training window one opens is invisible to the other, and a gated-delta
    # op quietly takes the cached path instead of the VJP. Serially that never
    # happens (alphabetical order puts this file last of the pair); under
    # `-n N --dist loadfile` it depends on which worker draws which file.
    from mlx_simulation import (
        restore_modules,
        simulate_mlx_on_torch,
        snapshot_modules,
    )
    from mlx_simulation.mlx_stub import _MLXFinder

    real_modules = snapshot_modules(_owned)
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, _MLXFinder)
    ]
    restore_modules(real_modules, _owned)


class Tokenizer:
    bos_token = None
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(character) % 43) for character in text]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
        continue_final_message=False, **kwargs,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class MappingTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=True):
        return self.mapping[text]


class CapturingTokenizer(Tokenizer):
    """Records every rendering, so which messages were rendered is observable."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return super().apply_chat_template(messages, **kwargs)


class SpecialTokenizer(Tokenizer):
    """Character-level, but its add_special_tokens actually adds a special.

    The tokenizers above accept the flag and ignore it, so nothing they encode
    can tell an explicit override from the default.
    """

    bos_token = "<bos>"
    bos_token_id = 1
    trailing_special = False

    def encode(self, text, add_special_tokens=True):
        values = super().encode(text, add_special_tokens)
        if add_special_tokens:
            values.insert(0, self.bos_token_id)
            if self.trailing_special:
                values.append(self.eos_token_id)
        return values


class DeclaredOnlyBosTokenizer(Tokenizer):
    """Declares a bos_token_id its add_special_tokens never emits.

    Phi-4-mini is the real one: bos_token_id is set, but encoding with
    add_special_tokens=True produces no BOS, so the model's own convention is
    to have none. Asking the id alone would put one in for DPO only.
    """

    bos_token = None
    bos_token_id = 7


class TrailingSpecialTokenizer(SpecialTokenizer):
    """Also closes with its EOS, as tokenizers with add_eos_token set do."""

    trailing_special = True


class TemplateScriptTokenizer(Tokenizer):
    """Renders a row's three texts from a script and merges the given pairs.

    A template can re-render the prompt's final turn once a completion follows
    it, leaving the prompt no longer a text prefix of one branch or of either,
    and a merge stops a character prefix from being a token prefix. Scripting
    the renderings is the smallest fixture that reaches those branches.
    """

    def __init__(self, renders, merges=()):
        self.renders = list(renders)
        self.merges = tuple(merges)

    def encode(self, text, add_special_tokens=True):
        values = []
        position = 0
        while position < len(text):
            pair = text[position:position + 2]
            if pair in self.merges:
                values.append(46 + self.merges.index(pair))
                position += 2
            else:
                values.append(3 + (ord(text[position]) % 43))
                position += 1
        return values

    def apply_chat_template(self, messages, **kwargs):
        return self.renders.pop(0)


def rows(count=6):
    return [
        {"prompt": f"question {index}: ", "chosen": "yes", "rejected": "no"}
        for index in range(count)
    ]


def policy(kind="dpo", **kwargs):
    from unsloth_zoo.mlx.preference import PreferenceLengthPolicy
    options = dict(
        max_length=64, max_prompt_length=None, max_completion_length=None,
        truncation_mode="keep_end",
    )
    options.update(kwargs)
    options.setdefault("max_seq_length", options["max_length"])
    return PreferenceLengthPolicy(kind=kind, **options)


def build_plan(**kwargs):
    from unsloth_zoo.mlx.preference import create_preference_batch_plan
    options = dict(
        batch_size=2, length_policy=policy(), dataset_order="sequential",
        grad_accum=2,
    )
    options.update(kwargs)
    return create_preference_batch_plan(rows(), Tokenizer(), **options)


def test_configs_are_objective_specific_and_keyword_only():
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXORPOConfig, MLXTrainingConfig

    assert not hasattr(MLXTrainingConfig(), "beta")
    assert MLXORPOConfig(beta=0.2).beta == 0.2
    assert MLXDPOConfig(beta=0.3, label_smoothing=0.1).reference_free is False
    with pytest.raises(TypeError):
        MLXORPOConfig(*([None] * 100))


@pytest.mark.parametrize(
    "epochs,expected_batches,expected_steps",
    [(0.5, 2, 1), (1.0, 3, 2), (1.5, 5, 3), (2.0, 6, 4)],
)
def test_fractional_epoch_budget_matches_finite_sft_contract(
    epochs, expected_batches, expected_steps,
):
    from unsloth_zoo.mlx.trainer import _resolve_training_steps

    plan = build_plan(num_epochs=epochs, dataset_order="torch_randperm", seed=7)
    args = types.SimpleNamespace(
        max_steps=0, num_train_epochs=epochs, gradient_accumulation_steps=2,
    )
    assert len(plan) == expected_batches
    assert plan.cycle_length == 3
    assert _resolve_training_steps(
        args, plan, None, includes_epochs=True,
    ) == expected_steps


def test_plan_is_lazy_and_concatenates_chosen_then_rejected():
    import mlx.core as mx

    plan = build_plan(num_batches=2)
    assert all(not isinstance(value, mx.array) for value in plan.rows)
    batch, lengths, normalizers = plan[0]
    assert batch.shape[0] == 4
    assert lengths.tolist()[0][0] == lengths.tolist()[2][0]
    assert normalizers.tolist()[1:] == [4, 2]


def test_default_order_randomizes_length_batches_from_first_visit():
    from unsloth_zoo.mlx.preference import create_preference_batch_plan

    dataset = [
        {"prompt": "p", "chosen": "x" * size, "rejected": "y" * size}
        for size in range(1, 8)
    ]
    plan = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=2, length_policy=policy(),
        num_batches=4, grad_accum=1, dataset_order="default", seed=1,
    )
    assert plan.schedule == ((6,), (4, 5), (0, 1), (2, 3))


def test_a_refused_row_names_its_index():
    from unsloth_zoo.mlx.preference import create_preference_batch_plan

    dataset = rows(3)
    dataset[1] = {"prompt": "", "chosen": "y", "rejected": "n"}
    with pytest.raises(ValueError, match="dataset row 1") as error:
        create_preference_batch_plan(
            dataset, Tokenizer(), batch_size=1, length_policy=policy(max_length=8),
            num_batches=1, grad_accum=1, dataset_order="sequential", append_eos=False,
        )
    assert "anything before it to predict from" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


def test_prompt_only_eos_does_not_consume_first_response_token():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    class PromptEOSTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=True):
            values = super().encode(text, add_special_tokens)
            if text.endswith(": "):
                values.append(self.eos_token_id)
            return values

    tokenized = tokenize_preference_row(
        PromptEOSTokenizer(), rows(1)[0], length_policy=policy(),
    )
    expected_first = Tokenizer().encode("question 0: yes")[len(
        Tokenizer().encode("question 0: ")
    )]
    assert tokenized.chosen_ids[0] == expected_first


@pytest.mark.parametrize(
    "mapping,row,expected",
    [
        (
            {"ab": [1, 2], "abc": [1, 3, 4], "abd": [1, 5, 6]},
            {"prompt": "ab", "chosen": "c", "rejected": "d"},
            ((1, 3, 4), (1, 5, 6), (1, 1)),
        ),
        (
            {"a": [1], "ac": [2, 3], "ad": [4, 5]},
            {"prompt": "a", "chosen": "c", "rejected": "d"},
            ((2, 3), (4, 5), (0, 0)),
        ),
        (
            {"ab": [1, 2], "abc": [1, 2, 3], "abd": [1, 4, 5]},
            {"prompt": "ab", "chosen": "c", "rejected": "d"},
            ((1, 2, 3), (1, 4, 5), (2, 1)),
        ),
    ],
)
def test_orpo_boundary_merges_preserve_each_branch(mapping, row, expected):
    from unsloth_zoo.mlx.preference import (
        create_preference_batch_plan,
        tokenize_preference_row,
    )

    tokenizer = MappingTokenizer(mapping)
    tokenized = tokenize_preference_row(
        tokenizer, row, length_policy=policy("orpo", max_length=8),
    )
    chosen, rejected, prompt_lengths = expected
    assert tokenized.chosen == chosen
    assert tokenized.rejected == rejected

    _, lengths, _ = create_preference_batch_plan(
        [row], tokenizer, batch_size=1, length_policy=policy("orpo", max_length=8),
        num_batches=1, grad_accum=1, dataset_order="sequential",
    )[0]
    assert lengths.tolist() == [
        [prompt_lengths[0], len(chosen)],
        [prompt_lengths[1], len(rejected)],
    ]


def test_sft_and_preference_share_boundary_merge_tolerance():
    from unsloth_zoo.mlx.utils import _tokenize_mlx_prompt_completion

    input_ids, labels = _tokenize_mlx_prompt_completion(
        MappingTokenizer({"ab": [1, 2], "abc": [1, 3, 4]}),
        "ab", "c", completion_only_loss=True,
    )
    assert input_ids == [1, 3, 4]
    assert labels == [-100, 3, 4]





def test_sft_rejects_a_prompt_mismatch_before_the_boundary():
    from unsloth_zoo.mlx.utils import _tokenize_mlx_prompt_completion

    # SFT re-tokenizes the same text, so an early mismatch is a real error.
    with pytest.raises(ValueError, match="differ before the final prompt token"):
        _tokenize_mlx_prompt_completion(
            MappingTokenizer({"abc": [1, 2, 3], "abcd": [9, 2, 4, 5]}),
            "abc", "d", completion_only_loss=True,
        )


def test_orpo_steps_back_one_token_the_way_trl_does():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    # build_tokenized_answer steps back one token and checks no further.
    tokenizer = MappingTokenizer(
        {
                "abc": [1, 2, 3],
                "abcd": [9, 2, 4, 5],
                "abce": [8, 2, 6, 7],
        }
    )

    tokenized = tokenize_preference_row(
        tokenizer,
        {"prompt": "abc", "chosen": "d", "rejected": "e"},
        length_policy=policy("orpo", max_length=8),
    )
    assert (tokenized.chosen_prompt_ids, tokenized.chosen_ids) == ((9, 2), (4, 5))
    assert (tokenized.rejected_prompt_ids, tokenized.rejected_ids) == ((8, 2), (6, 7))


def test_orpo_refuses_two_prompts_that_disagree_beyond_a_merge():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    # Two differing prompt tokens is not a merge, and ORPOTrainer refuses it.
    tokenizer = MappingTokenizer(
        {
            "abc": [1, 2, 3],
            "abcd": [9, 7, 4, 5],
            "abce": [8, 6, 6, 7],
        }
    )

    with pytest.raises(ValueError, match="only the last prompt token may move"):
        tokenize_preference_row(
            tokenizer,
            {"prompt": "abc", "chosen": "d", "rejected": "e"},
            length_policy=policy("orpo", max_length=8),
        )


# Distinct characters, so a head slice can never pass for a tail slice.
PROMPT, CHOSEN, REJECTED = "abcdefghijklmnopqrst", "uvwxyzABCD", "EFGH"
LONG_ROW = {"prompt": PROMPT, "chosen": CHOSEN, "rejected": REJECTED}
SWAPPED = {"prompt": PROMPT, "chosen": REJECTED, "rejected": CHOSEN}
TURN = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
ANSWER, OTHER = {"role": "assistant", "content": "A2"}, {"role": "assistant", "content": "A3"}


def encoded(text, *, eos=False):
    ids = tuple(Tokenizer().encode(text))
    return ids + (Tokenizer.eos_token_id,) if eos else ids


def tokenize(row=None, kind="dpo", tokenizer=None, append_eos=True, **kwargs):
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    return tokenize_preference_row(
        tokenizer or Tokenizer(), LONG_ROW if row is None else row,
        length_policy=policy(kind, **kwargs), append_eos=append_eos)


def resolved(kind="dpo", **kwargs):
    from unsloth_zoo.mlx.preference import resolve_preference_length_policy

    options = dict(max_length=1024, max_prompt_length=512,
                   max_completion_length=None, truncation_mode="keep_end")
    options.update(kwargs)
    return resolve_preference_length_policy(
        kind, types.SimpleNamespace(**options),
        max_seq_length=options.pop("max_seq_length", 2048))


@pytest.mark.parametrize(
    "kind,options,chosen,rejected,row",
    [
        ("dpo", dict(max_prompt_length=5, max_completion_length=3),
         ((-5, None), (None, 3)), ((-5, None), (None, 3)), LONG_ROW),
        ("dpo", dict(max_length=8),
         ((0, 0), (-8, None)), ((-3, None), (None, None)), LONG_ROW),
        ("orpo", dict(max_length=14, max_prompt_length=8),
         ((-8, None), (None, 6)), ((-8, None), (None, 6)), LONG_ROW),
        ("orpo", dict(max_length=14, max_prompt_length=8),
         ((-8, None), (None, 6)), ((-8, None), (None, 6)), SWAPPED),
        ("orpo", dict(max_length=25, max_prompt_length=8, truncation_mode="keep_start"),
         ((None, 8), (None, None)), ((None, 8), (None, None)), LONG_ROW),
    ],
)
def test_truncation_follows_each_trl_trainer(kind, options, chosen, rejected, row):
    tokenized = tokenize(row, kind, **options)
    for side, (prompt, response) in (("chosen", chosen), ("rejected", rejected)):
        assert getattr(tokenized, side) == encoded(PROMPT)[slice(*prompt)] + \
            encoded(row[side], eos=True)[slice(*response)]


@pytest.mark.parametrize(
    "kind,row,options,message",
    [
        ("dpo", {"prompt": PROMPT, "chosen": CHOSEN, "rejected": CHOSEN[::-1]},
         dict(max_length=15, truncation_mode="keep_start"), "anything before it to predict from"),
        ("dpo", {"prompt": "", "chosen": "y", "rejected": "n"},
         dict(max_length=8, append_eos=False), "anything before it to predict from"),
        ("dpo", {"prompt": "", "chosen": "y", "rejected": "no"},
         dict(max_length=8, append_eos=False), None),
        ("orpo", {"prompt": "", "chosen": "y", "rejected": "no"},
         dict(max_length=8, append_eos=False), "anything before it to predict from"),
        ("orpo", {"prompt": "", "chosen": "no", "rejected": "y"},
         dict(max_length=8, append_eos=False), "anything before it to predict from"),
        ("orpo", {"prompt": "", "chosen": CHOSEN, "rejected": CHOSEN[::-1]},
         dict(max_length=1, max_prompt_length=1), "anything before it to predict from"),
        ("orpo", {"prompt": "p" * 6, "chosen": "c" * 20, "rejected": "d" * 4},
         dict(max_length=8, max_prompt_length=12), "more than the 8 a batch can hold"),
        ("orpo", {"prompt": "p" * 6, "chosen": "d" * 4, "rejected": "c" * 20},
         dict(max_length=8, max_prompt_length=12), "more than the 8 a batch can hold"),
    ],
)
def test_the_length_guards_refuse_only_untrainable_rows(kind, row, options, message):
    if message is None:
        flipped = {**row, "chosen": row["rejected"], "rejected": row["chosen"]}
        assert tokenize(row, kind, **options) and tokenize(flipped, kind, **options)
        return
    with pytest.raises(ValueError, match=message):
        tokenize(row, kind, **options)


@pytest.mark.parametrize(
    "kind,overrides,expected,warning",
    [
        ("orpo", dict(max_length=None), (512, 512), "is not set"),
        ("orpo", dict(max_prompt_length=None), (1024, 128), "is not set"),
        ("dpo", dict(max_length=None, max_prompt_length=None, max_seq_length=300),
         (300, None), None),
        ("orpo", dict(max_seq_length=512), (512, 256), "no longer leaves room"),
        ("orpo", dict(max_seq_length=400), (400, 200), "no longer leaves room"),
        ("orpo", dict(max_length=8, max_prompt_length=12, max_seq_length=4), (4, 12), None),
        ("dpo", dict(max_seq_length=512), (512, 512), None),
    ],
)
def test_the_resolved_budget_matches_each_objectives_arithmetic(
    kind, overrides, expected, warning,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        length_policy = resolved(kind, **overrides)
    said = " ".join(str(record.message) for record in caught)
    assert (length_policy.max_length, length_policy.max_prompt_length) == expected
    assert length_policy.max_seq_length == overrides.get("max_seq_length", 2048)
    assert length_policy.max_completion_length is None
    assert warning in said if warning else "no longer leaves room" not in said


def test_an_inherited_max_length_says_it_narrowed_the_budget():
    from unsloth_zoo.mlx.preference import resolve_preference_length_policy

    def resolve(max_seq_length, explicit, kind="dpo"):
        args = types.SimpleNamespace(
            max_length=1024, max_prompt_length=512,
            max_completion_length=None, truncation_mode="keep_end",
            _unsloth_mlx_max_length_explicit=explicit,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_preference_length_policy(
                kind, args, max_seq_length=max_seq_length,
            )
        return [str(item.message) for item in caught]

    said = resolve(2048, explicit=False)
    assert len(said) == 1 and "budgeted more tightly" in said[0]
    assert "max_prompt_length=None" in said[0]
    # ORPO already made an open bound 128, so advising None would cap, not widen.
    orpo = resolve(2048, explicit=False, kind="orpo")
    assert len(orpo) == 1 and "max_prompt_length=None" not in orpo[0]
    assert "resolves to 128" in orpo[0]
    assert resolve(2048, explicit=True) == []
    assert resolve(1024, explicit=False) == []


def test_orpo_says_what_an_open_prompt_bound_costs_only_when_it_matters():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    # A row that fits never consults the bound, so leaving it open still works.
    open_bound = policy("orpo", max_length=64, max_prompt_length=None)
    assert open_bound.max_prompt_length is None
    assert tokenize_preference_row(
        Tokenizer(), {"prompt": "ab", "chosen": "cd", "rejected": "ef"},
        length_policy=open_bound,
    )
    with pytest.raises(ValueError, match="cannot be left open here"):
        tokenize_preference_row(
            Tokenizer(),
            {"prompt": PROMPT, "chosen": CHOSEN, "rejected": CHOSEN[::-1]},
            length_policy=policy("orpo", max_length=8, max_prompt_length=None),
        )


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"chosen": "The sky is blue.", "rejected": "The sky is green."},
         ("The sky is", " blue.", " green.")),
        ({"prompt": "Q1", "chosen": TURN + [ANSWER], "rejected": TURN + [OTHER]},
         (TURN, [ANSWER], [OTHER])),
    ],
)
def test_an_implicit_prompt_is_recovered_the_way_trl_recovers_it(row, expected):
    from unsloth_zoo.mlx.preference import _maybe_extract_prompt

    recovered, was_recovered = _maybe_extract_prompt(dict(row))
    keys = ("prompt", "chosen", "rejected")
    assert (*(recovered[key] for key in keys), was_recovered) == (*expected, True)


def test_dpo_tokenizes_the_prompt_and_each_completion_the_way_trl_does():
    # DPO never tokenizes the pair, so a mid-token recovery splits, not refuses.
    tokenized = tokenize(
        {"chosen": "foobar", "rejected": "foobaz"}, max_length=16,
        tokenizer=MappingTokenizer({"fooba": [1, 2], "foobar": [3], "foobaz": [4],
                                    "r": [5], "z": [6]}),
    )
    assert (tokenized.chosen, tokenized.rejected) == ((1, 2, 5), (1, 2, 6))
    assert tokenized.chosen_prompt_ids == tokenized.rejected_prompt_ids == (1, 2)


def test_encode_honours_an_explicit_special_token_override():
    from unsloth_zoo.mlx.utils import encode_mlx_text

    tokenizer = SpecialTokenizer()
    plain = [3 + (ord(character) % 43) for character in "ab"]
    assert encode_mlx_text(tokenizer, "ab") == [tokenizer.bos_token_id] + plain
    assert encode_mlx_text(tokenizer, "ab", add_special_tokens=False) == plain


# A user-ended prompt, so the template renders prompt, chosen, rejected in order.
SCRIPTED_ROW = {
    "prompt": [{"role": "user", "content": "p"}],
    "chosen": [{"role": "assistant", "content": "c"}],
    "rejected": [{"role": "assistant", "content": "r"}],
}


def test_orpo_scans_for_the_boundary_only_once_the_prompt_stops_being_a_prefix():
    # Neither branch still starts with the rendered prompt, so a strict check
    # would refuse a row that only has to split earlier.
    tokenized = tokenize(
        SCRIPTED_ROW, "orpo", append_eos=False, max_length=8, max_prompt_length=4,
        tokenizer=TemplateScriptTokenizer(["abc", "aXY", "aXZ"]),
    )
    assert tokenized.chosen_prompt_ids == encoded("a")
    assert tokenized.chosen_ids == encoded("XY")
    assert tokenized.rejected_ids == encoded("XZ")


def test_orpo_keeps_only_the_boundary_both_branches_prove_shared():
    # The boundaries disagree, so both prompts stop at the lower.
    tokenized = tokenize(
        SCRIPTED_ROW, "orpo", append_eos=False, max_length=8, max_prompt_length=4,
        tokenizer=TemplateScriptTokenizer(["ab", "abc", "aXc"]),
    )
    assert tokenized.chosen_prompt_ids == tokenized.rejected_prompt_ids == encoded("a")
    assert tokenized.chosen_ids == encoded("bc")
    assert tokenized.rejected_ids == encoded("Xc")


def test_dpo_clips_the_prompt_to_what_both_branches_still_agree_with():
    # Mid-token clip: what DPO tolerates, and what the whole prompt would empty.
    tokenizer = TemplateScriptTokenizer(
        ["abXc", "abXd", "abXe"], merges=("Xc", "Xd", "Xe"),
    )
    tokenized = tokenize(
        SCRIPTED_ROW, "dpo", tokenizer=tokenizer, append_eos=False, max_length=16,
    )
    assert tokenized.chosen_prompt_ids == tuple(tokenizer.encode("abX"))
    assert tokenized.chosen_ids == tuple(tokenizer.encode("d"))
    assert tokenized.rejected_ids == tuple(tokenizer.encode("e"))


def test_dpo_tokenizes_each_completion_without_special_tokens():
    tokenized = tokenize(
        {"prompt": "ab", "chosen": "cd", "rejected": "ef"}, "dpo",
        tokenizer=SpecialTokenizer(), append_eos=False, max_length=16,
    )
    assert tokenized.chosen_ids == encoded("cd")
    assert tokenized.rejected_ids == encoded("ef")


def test_the_dpo_prompt_keeps_its_bos_and_gains_no_trailing_special():
    tokenizer = TrailingSpecialTokenizer()
    tokenized = tokenize(
        {"prompt": "ab", "chosen": "cd", "rejected": "ef"}, "dpo",
        tokenizer=tokenizer, append_eos=False, max_length=16,
    )
    assert tokenized.chosen_prompt_ids == (tokenizer.bos_token_id,) + encoded("ab")
    assert tokenized.chosen_ids == encoded("cd")


def test_dpo_adds_no_bos_the_tokenizer_itself_would_not_emit():
    """A declared bos_token_id is not a promise the tokenizer emits one.

    DPO encodes the prompt with add_special_tokens=False and re-adds the BOS,
    so reading bos_token_id directly would give DPO a leading token that this
    tokenizer's SFT and ORPO paths, and TRL's own DPOTrainer, all leave out --
    the two objectives would then disagree on the same model.
    """
    tokenizer = DeclaredOnlyBosTokenizer()
    rows = {"prompt": "ab", "chosen": "cd", "rejected": "ef"}
    dpo = tokenize(rows, "dpo", tokenizer=tokenizer, append_eos=False,
                   max_length=16)
    orpo = tokenize(rows, "orpo", tokenizer=tokenizer, append_eos=False,
                    max_length=16)
    assert dpo.chosen_prompt_ids[0] != tokenizer.bos_token_id
    assert dpo.chosen_prompt_ids == orpo.chosen_prompt_ids


def test_dpo_still_adds_a_bos_the_tokenizer_does_emit():
    """The converse: where add_special_tokens emits a BOS, DPO keeps it."""
    tokenizer = SpecialTokenizer()
    tokenized = tokenize(
        {"prompt": "ab", "chosen": "cd", "rejected": "ef"}, "dpo",
        tokenizer=tokenizer, append_eos=False, max_length=16,
    )
    assert tokenized.chosen_prompt_ids[0] == tokenizer.bos_token_id


def test_an_explicit_partial_assistant_turn_is_finished_by_its_completions():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    tokenizer = CapturingTokenizer()
    tokenize_preference_row(
        tokenizer,
        {
            "prompt": [{"role": "assistant", "content": "prefix"}],
            "chosen": [{"role": "assistant", "content": " chosen"}],
            "rejected": [{"role": "assistant", "content": " rejected"}],
        },
        length_policy=policy(),
    )
    assert [message["content"] for message in tokenizer.calls[1][0]] == ["prefix chosen"]
    assert [message["content"] for message in tokenizer.calls[2][0]] == ["prefix rejected"]


def test_a_recovered_prompt_never_absorbs_its_completions():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    # A recovered prompt's last assistant message is a whole turn.
    tokenizer = CapturingTokenizer()
    tokenize_preference_row(
        tokenizer,
        {"prompt": "Q1", "chosen": TURN + [ANSWER], "rejected": TURN + [OTHER]},
        length_policy=policy(),
    )
    assert [message["content"] for message in tokenizer.calls[1][0]] == ["Q1", "A1", "A2"]
    assert [message["content"] for message in tokenizer.calls[2][0]] == ["Q1", "A1", "A3"]


def test_content_parts_and_chat_template_options_are_preserved():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    tokenizer = CapturingTokenizer()
    row = {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
        "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
        "rejected": [{"role": "assistant", "content": "B"}],
        "tools": [],
        "chat_template_kwargs": {"flag": 1},
    }
    result = tokenize_preference_row(tokenizer, row, length_policy=policy())
    assert result.chosen != result.rejected
    assert all(call[1]["tools"] == [] and call[1]["flag"] == 1 for call in tokenizer.calls)
    assert tokenizer.calls[0][0][0]["content"] == "Q"


def test_assistant_ended_prompt_continues_the_same_message():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    tokenizer = Tokenizer()
    row = {
        "prompt": [{"role": "assistant", "content": "prefix"}],
        "chosen": [{"role": "assistant", "content": " chosen"}],
        "rejected": [{"role": "assistant", "content": " rejected"}],
    }
    result = tokenize_preference_row(tokenizer, row, length_policy=policy())
    assert (
        result.chosen_prompt_ids
        and result.rejected_prompt_ids
        and result.chosen_ids
        and result.rejected_ids
    )


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "p", "chosen": [{"role": "assistant", "content": "a"}], "rejected": "b"},
        {"prompt": [{"role": "system", "content": "p"}], "chosen": [], "rejected": []},
        {
            "prompt": [{"role": "system", "content": "p"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        },
    ],
)
def test_ambiguous_rows_are_rejected(row):
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    with pytest.raises(ValueError):
        tokenize_preference_row(Tokenizer(), row, length_policy=policy())


class TinyModel:
    def __init__(self):
        import mlx.nn as nn
        self.embedding = nn.Embedding(64, 8)
        self.output = nn.Linear(8, 64, bias=False)

    def __call__(self, tokens):
        return self.output(self.embedding(tokens))


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_microbatch_loss_matches_one_logical_batch(objective):
    from unsloth_zoo.mlx.preference import make_dpo_loss_fn, make_orpo_loss_fn

    dataset = rows(3)
    from unsloth_zoo.mlx.preference import create_preference_batch_plan
    micro = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=2, length_policy=policy(),
        num_batches=2, grad_accum=2, dataset_order="sequential",
    )
    whole = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=3, length_policy=policy(),
        num_batches=1, grad_accum=1, dataset_order="sequential",
    )
    model = TinyModel()
    loss_fn = (
        make_orpo_loss_fn(beta=0.1)
        if objective == "orpo"
        else make_dpo_loss_fn(beta=0.1, reference_free=True)
    )
    micro_value = sum(float(loss_fn(model, *micro[index])[0]) for index in range(2)) / 2
    whole_value = float(loss_fn(model, *whole[0])[0])
    assert math.isclose(micro_value, whole_value, rel_tol=2e-5, abs_tol=2e-5)


def test_orpo_odds_term_is_finite_for_perfect_float16_logp():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import _orpo_terms

    value = _orpo_terms(
        mx.array([0.0], dtype=mx.float16),
        mx.array([-1.0], dtype=mx.float16),
    )
    assert math.isfinite(float(value[0]))
    assert value.dtype == mx.float32


def test_dpo_label_smoothing_matches_conservative_formula():
    from unsloth_zoo.mlx import preference

    plan = build_plan(num_batches=1, grad_accum=1)
    batch, lengths, normalizers = plan[0]
    model = TinyModel()
    epsilon = 0.2
    beta = 0.3
    policy = preference._response_logps(model, batch, lengths)
    pairs = batch.shape[0] // 2
    logits = beta * (policy[:pairs] - policy[pairs:])
    expected = -(
        (1 - epsilon) * preference._log_sigmoid(logits)
        + epsilon * preference._log_sigmoid(-logits)
    ).mean()
    actual, weight = preference.make_dpo_loss_fn(
        beta=beta, label_smoothing=epsilon, reference_free=True,
    )(model, batch, lengths, normalizers)
    assert float(actual) == pytest.approx(float(expected), rel=1e-6)
    assert int(weight) == 1


def test_orpo_nll_covers_the_full_chosen_sequence():
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx import preference

    plan = build_plan(num_batches=1, grad_accum=1)
    batch, lengths, normalizers = plan[0]
    mx.random.seed(0)
    model = TinyModel()
    beta = 0.15
    targets = batch[:, 1:]
    ce = nn.losses.cross_entropy(
        model(batch[:, :-1]), targets, reduction="none",
    ).reshape(targets.shape)
    pairs = batch.shape[0] // 2
    steps = mx.arange(1, targets.shape[1] + 1)
    response_mask = preference._response_mask(targets, lengths)
    response_logp = -(ce * response_mask).sum(axis=1) / mx.maximum(
        response_mask.sum(axis=1), mx.array(1.0),
    )
    odds = preference._orpo_terms(
        response_logp[:pairs], response_logp[pairs:],
    ).mean()
    full_mask = (steps < lengths[:pairs, 1:]).astype(mx.float32)
    full_nll = (ce[:pairs] * full_mask).sum() / full_mask.sum()
    response_nll = (
        ce[:pairs] * response_mask[:pairs]
    ).sum() / response_mask[:pairs].sum()
    actual, _ = preference.make_orpo_loss_fn(beta)(
        model, batch, lengths, normalizers,
    )
    assert float(actual) == pytest.approx(float(full_nll + beta * odds), rel=1e-6)
    assert not math.isclose(
        float(actual), float(response_nll + beta * odds), rel_tol=1e-4,
    )


def test_dropout_context_restores_exact_state():
    from unsloth_zoo.mlx.preference import PreferenceRunContext
    from unsloth_zoo.mlx.utils import _read_mlx_lora_dropout

    class Dropout:
        def __init__(self):
            self._p_1 = 0.75
            self.p = 0.25

    class Model:
        def __init__(self):
            self.dropout = Dropout()
            self.probability = types.SimpleNamespace(p=0.6)

        def named_modules(self):
            return [
                ("", self), ("dropout", self.dropout),
                ("probability", self.probability),
            ]

    model = Model()
    adapter = types.SimpleNamespace(dropout=model.dropout)
    before = (model.dropout._p_1, model.dropout.p)
    context = PreferenceRunContext(model)
    assert model.dropout._p_1 == 1.0
    assert model.probability.p == 0.6
    assert _read_mlx_lora_dropout(adapter) == 0.25
    context.restore()
    assert (model.dropout._p_1, model.dropout.p) == before


def test_adapter_config_keeps_dropout_while_preference_context_is_active():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import PreferenceRunContext
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    class Dropout:
        def __init__(self):
            self._p_1 = 0.7
            self.p = 0.3

    class Adapter:
        def __init__(self):
            self.lora_a = mx.zeros((8, 4))
            self.lora_b = mx.zeros((4, 16))
            self.scale = 2.0
            self.dropout = Dropout()

    class Model:
        def __init__(self):
            self.adapter = Adapter()

        def named_modules(self):
            return [
                ("", self), ("q_proj", self.adapter),
                ("q_proj.dropout", self.adapter.dropout),
            ]

    model = Model()
    context = PreferenceRunContext(model)
    try:
        config = _enrich_mlx_adapter_config(model, {})
        assert config["lora_parameters"]["dropout"] == pytest.approx(0.3)
        assert config["dropout"] == pytest.approx(0.3)
    finally:
        context.restore()


def test_referenced_dpo_rejects_non_lora_and_reference_free_accepts_it():
    from unsloth_zoo.mlx.preference import build_reference_policy

    class Model:
        _hf_repo = "base"

        def named_modules(self):
            return [("", self)]

        def parameters(self):
            return {}

        def trainable_parameters(self):
            return {}

    with pytest.raises(ValueError, match="requires plain LoRA"):
        build_reference_policy(
            Model(), reference_free=False, resume_provenance=None,
        )
    policy, provenance = build_reference_policy(
        Model(), reference_free=True, resume_provenance=None,
    )
    assert policy is None and provenance == {"kind": "reference_free"}


def test_reference_forward_restores_scale_and_neftune_after_failure(monkeypatch):
    from unsloth_zoo.mlx import preference

    adapter = types.SimpleNamespace(scale=0.75)
    neftune = types.SimpleNamespace(_neftune_noise_enabled=True)
    policy = preference.LoRAReferencePolicy([adapter], [neftune])
    monkeypatch.setattr(
        preference, "_response_logps",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("forward failed")),
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        policy.forward(object(), object(), object())
    assert adapter.scale == 0.75
    assert neftune._neftune_noise_enabled is True


def test_referenced_dpo_accepts_only_fresh_or_same_run_plain_lora():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import build_reference_policy

    class PlainLoRA:
        def __init__(self, second_nonzero=False):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[1.0 if second_nonzero else 0.0]])
            self.scale = 1.0

    class Model:
        _hf_repo = "base/repo"
        _unsloth_base_revision = "main"
        _unsloth_base_commit_hash = "abc123"

        def __init__(self, adapter, extra=False):
            self.adapter = adapter
            self.extra = extra

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter)]

        def parameters(self):
            values = {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                }
            }
            if self.extra:
                values["norm"] = mx.array([1.0])
            return values

        def trainable_parameters(self):
            return self.parameters()

    fresh = Model(PlainLoRA())
    policy, provenance = build_reference_policy(
        fresh, reference_free=False, resume_provenance=None,
    )
    assert policy.modules == (fresh.adapter,)
    assert provenance["base_commit"] == "abc123"

    resumed = Model(PlainLoRA(second_nonzero=True))
    resumed_policy, resumed_provenance = build_reference_policy(
        resumed, reference_free=False, resume_provenance=provenance,
    )
    assert resumed_policy.modules == (resumed.adapter,)
    assert resumed_provenance == provenance

    with pytest.raises(ValueError, match="fresh zero-delta"):
        build_reference_policy(
            Model(PlainLoRA(second_nonzero=True)),
            reference_free=False,
            resume_provenance=None,
        )
    with pytest.raises(ValueError, match="LoRA-only trainable"):
        build_reference_policy(
            Model(PlainLoRA(), extra=True),
            reference_free=False,
            resume_provenance=None,
        )


def test_referenced_dpo_rejects_dora():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import build_reference_policy

    DoRA = type("DoRALinear", (), {})
    adapter = DoRA()
    adapter.lora_a = mx.array([[1.0]])
    adapter.lora_b = mx.array([[0.0]])
    adapter.scale = 1.0

    class Model:
        def named_modules(self):
            return [("", self), ("q_proj", adapter)]

        def parameters(self):
            return {"q_proj": {"lora_a": adapter.lora_a, "lora_b": adapter.lora_b}}

        trainable_parameters = parameters

    with pytest.raises(ValueError, match="does not support DoRA"):
        build_reference_policy(
            Model(), reference_free=False, resume_provenance=None,
        )


def test_preference_trainers_forward_shared_constructor_state():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._config = {"model_type": "tiny"}

    def formatter(row):
        return row
    processor = object()
    for trainer_class, config in (
        (MLXORPOTrainer, MLXORPOConfig()),
        (MLXDPOTrainer, MLXDPOConfig(reference_free=True)),
    ):
        original = config.to_dict()
        trainer = trainer_class(
            Model(), Tokenizer(), rows(1), args=config,
            formatting_func=formatter, processor=processor,
            max_seq_length=96,
        )
        assert trainer.args is not config
        assert config.to_dict() == original
        assert trainer.args.beta == config.beta
        assert trainer.args.max_seq_length == 96
        assert config.max_seq_length == original["max_seq_length"] == 2048
        assert trainer.formatting_func is formatter
        assert trainer.processor is processor


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_preference_trainer_runs_through_shared_training_loop(
    objective, monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    common = dict(
        max_steps=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        compile=True,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        disable_memory_limits=True,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        logging_steps=1,
        output_dir=str(tmp_path),
    )
    if objective == "orpo":
        trainer = MLXORPOTrainer(
            _tiny_model(), Tokenizer(), rows(3), args=MLXORPOConfig(**common),
        )
    else:
        trainer = MLXDPOTrainer(
            _tiny_model(), Tokenizer(), rows(3),
            args=MLXDPOConfig(reference_free=True, **common),
        )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    result = trainer.train()
    assert result["train_steps"] == 1
    assert result["trained_tokens"] > 0


def test_referenced_dpo_freezes_accidental_norm_before_reference_validation(
    monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    class Adapter:
        def __init__(self):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.0]])
            self.scale = 1.0

    class Norm:
        def __init__(self):
            self.weight = mx.array([1.0])
            self.frozen = False

        def freeze(self, keys=None, recurse=False):
            assert keys == ["weight"] and recurse is False
            self.frozen = True

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self.adapter = Adapter()
            self.norm = Norm()
            self._config = {"model_type": "tiny"}

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

        def train(self, mode=True):
            return self

        @property
        def state(self):
            return []

        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                },
                "norm": {"weight": self.norm.weight},
            }

        def trainable_parameters(self):
            parameters = self.parameters()
            if self.norm.frozen:
                parameters.pop("norm")
            return parameters

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter), ("norm", self.norm)]

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    model = Model()
    trainer = MLXDPOTrainer(
        model,
        Tokenizer(),
        rows(2),
        args=MLXDPOConfig(
            max_steps=1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            compile=True,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            disable_memory_limits=True,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            logging_steps=1,
            output_dir=str(tmp_path),
        ),
    )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None

    result = trainer.train()

    assert model.norm.frozen is True
    assert result["train_steps"] == 1


def test_final_loss_averages_logical_updates_with_ragged_epoch_tail(
    monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.preference import make_orpo_loss_fn
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    config = MLXORPOConfig(
        max_steps=2, gradient_accumulation_steps=2,
        per_device_train_batch_size=2, dataset_order="sequential",
        compile=False, gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False, disable_memory_limits=True,
        max_grad_norm=0.0, max_grad_leaf_norm=0.0, logging_steps=10,
        output_dir=str(tmp_path),
    )
    model = _tiny_model()
    trainer = MLXORPOTrainer(model, Tokenizer(), rows(5), args=config)
    plan, _ = trainer._prepare_data(False)
    objective = make_orpo_loss_fn(config.beta)
    values = [float(objective(model, *plan[index])[0]) for index in range(3)]
    expected = ((values[0] + values[1]) / 2 + values[2]) / 2
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    result = trainer.train()
    assert result["train_steps"] == 2
    assert result["train_loss"] == pytest.approx(expected, rel=1e-6)


def test_trainer_applies_preference_formatter_once_per_row():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._config = {"model_type": "tiny"}

    calls = []

    def formatter(row):
        calls.append(row["id"])
        return {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }

    dataset = [dict(row, id=index) for index, row in enumerate(rows(3))]
    trainer = MLXORPOTrainer(
        _tiny_model(), Tokenizer(), dataset, formatting_func=formatter,
        args=MLXORPOConfig(max_steps=1, gradient_accumulation_steps=1),
    )
    trainer._prepare_data(False)
    assert calls == [0, 1, 2]


def test_mismatched_referenced_resume_rejects_before_adapter_hydration(tmp_path):
    import json
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.0]])
            self.scale = 1.0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = Adapter()
            self._config = {"model_type": "tiny"}
            self._hf_repo = "base/repo"
            self._unsloth_base_revision = "main"
            self._unsloth_base_commit_hash = "abc123"
            self.loaded = False

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter)]

        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                }
            }

        trainable_parameters = parameters

        def load_weights(self, *_args, **_kwargs):
            self.loaded = True
            self.adapter.lora_b = mx.array([[9.0]])

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "adapters.safetensors").touch()
    (checkpoint / "optimizer_state.safetensors").touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 1,
        "preference_reference": {"kind": "reference_free"},
    }))
    model = Model()
    trainer = MLXDPOTrainer(
        model, Tokenizer(), rows(1),
        args=MLXDPOConfig(
            max_steps=2, compile=False, gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False, disable_memory_limits=True,
            output_dir=str(tmp_path / "output"),
        ),
    )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda *_args: None,
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        trainer.train(resume_from_checkpoint=str(checkpoint))
    assert model.loaded is False
    assert model.adapter.lora_b.tolist() == [[0.0]]


def _tiny_model(lora=False):
    """lora=True adds one adapter at zero delta, as referenced DPO requires."""
    import mlx.core as mx
    import mlx.nn as nn

    class Adapter:
        def __init__(self):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.0]])
            self.scale = 1.0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self._config = {"model_type": "tiny"}
            if lora:
                self.q_proj = Adapter()

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

        def train(self, mode=True):
            return self

        @property
        def state(self):
            return []

        if lora:
            def named_modules(self):
                return [("", self), ("q_proj", self.q_proj)]

            def parameters(self):
                return {"q_proj": {
                    "lora_a": self.q_proj.lora_a, "lora_b": self.q_proj.lora_b,
                }}

            def trainable_parameters(self):
                return self.parameters()

    return Model()


def _generation_common(tmp_path, **overrides):
    common = dict(
        max_steps=1, gradient_accumulation_steps=2, per_device_train_batch_size=2,
        compile=True, gradient_checkpointing=False, disable_memory_limits=True,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0, max_grad_leaf_norm=0.0, logging_steps=1,
        output_dir=str(tmp_path), max_seq_length=64, generate_during_eval=True,
        num_generation_prompts=2, generation_max_tokens=8, eval_steps=1,
        generation_temperature=0.25,
    )
    common.update(overrides)
    return common


def _run_generation_trainer(trainer, monkeypatch, calls, generate_batch=None):
    """Drive one training step whose evaluation samples, recording engine calls.

    ``generate_batch`` replaces the recording stub, for tests needing the engine
    to fail or to return text of their own choosing.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx import generate as generate_module
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    def fake_generate_batch(model, tokenizer, requests, *, defaults=None):
        call = len(calls) + 1
        calls.append({
            "requests": list(requests),
            "defaults": defaults,
            "scales": [module.scale for _, module in iter_mlx_lora_modules(model)],
        })
        # Distinct per call and per row, so a mis-mapped sample reads wrong.
        return [
            types.SimpleNamespace(
                token_ids=[1, 2], text=f"c{call}r{index}", logprobs=[0.0, 0.0],
                finish_reason="length", stop_match=None,
            )
            for index, _ in enumerate(requests)
        ]

    monkeypatch.setattr(
        generate_module, "generate_batch", generate_batch or fake_generate_batch,
    )

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    return trainer.train()


def test_generation_fields_stay_appended_for_a_wholesale_config_copy():
    """A config round-tripped without the generation fields is still a copy.

    An unregistered field flips the copy detection, letting a copied default
    warmup_steps override an explicit warmup_ratio.
    """
    import dataclasses
    from unsloth_zoo.mlx.trainer import MLXDPOConfig

    generation_fields = {
        "generate_during_eval", "num_generation_prompts",
        "generation_max_tokens", "generation_temperature",
    }
    baseline = MLXDPOConfig()
    provided = {
        field.name: getattr(baseline, field.name)
        for field in dataclasses.fields(MLXDPOConfig)
        if field.name not in generation_fields
    }
    provided["warmup_ratio"] = 0.25
    config = MLXDPOConfig(**provided)
    assert config._unsloth_mlx_warmup_steps_explicit is False


def test_callback_requested_eval_samples_without_a_step_cadence(
    tmp_path, monkeypatch,
):
    """A callback raising should_evaluate reaches the sampling pass."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(
            **_generation_common(tmp_path, reference_free=True, eval_steps=0)
        ),
    )

    class RequestEval:
        def on_step_end(self, args, state, control, **kwargs):
            control.should_evaluate = True

    assert trainer.last_generation_samples == [], "readable before the first eval"
    trainer.add_callback(RequestEval())
    calls = []
    _run_generation_trainer(trainer, monkeypatch, calls)
    assert len(calls) == 1


@pytest.mark.parametrize("overrides,with_eval,error,message", [
    ({}, False, ValueError, "needs an eval_dataset"),
    ({"generation_max_tokens": 64}, True, ValueError,
     "smaller than max_seq_length"),
    ({"num_generation_prompts": 0}, True, ValueError,
     "num_generation_prompts must be at least 1"),
    # The engine validates these itself, and both of its rejections must land at
    # configuration time rather than at the first evaluation.
    ({"generation_max_tokens": 0}, True, ValueError,
     "generation_max_tokens is invalid"),
    ({"generation_max_tokens": 8.5}, True, TypeError,
     "generation_max_tokens is invalid"),
    ({"generation_temperature": -1.0}, True, ValueError,
     "generation_temperature is invalid"),
])
def test_generate_during_eval_rejects_a_configuration_it_cannot_honour(
    tmp_path, overrides, with_eval, error, message,
):
    """What a sampling pass cannot honour fails while the config is checked."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3),
        eval_dataset=rows(2) if with_eval else None,
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, **overrides)),
    )
    with pytest.raises(error, match=message):
        trainer._prepare_data(False)


def test_generation_prompt_is_the_standalone_encoding_not_the_training_boundary():
    """A merged seam is prompt when there is no completion to merge with.

    tokenize_preference_row gives that token to the response span so supervision
    starts after the merge; generation must not inherit that boundary. ORPO is
    the objective that still merges: DPO tokenizes each completion apart from
    the prompt, so a seam there simply splits and never needs folding back.
    """
    from unsloth_zoo.mlx.preference import (
        encode_generation_prompt, tokenize_preference_row,
    )

    tokenizer = MappingTokenizer({"ab": [1, 2], "abc": [1, 3, 4], "abd": [1, 5, 6]})
    row = {"prompt": "ab", "chosen": "c", "rejected": "d"}
    trained = tokenize_preference_row(
        tokenizer, row, length_policy=resolved("orpo", max_seq_length=16),
    )
    assert trained.chosen_prompt_ids == (1,)

    _text, prompt_ids = encode_generation_prompt(
        tokenizer, row, max_seq_length=16, max_new_tokens=4,
    )
    assert prompt_ids == (1, 2)


def test_generation_prompt_reserves_room_for_the_sample():
    """The prompt keeps its end and leaves max_new_tokens of context free."""
    from unsloth_zoo.mlx.preference import encode_generation_prompt

    tokenizer = MappingTokenizer({"p": [1, 2, 3, 4, 5, 6, 7, 8]})
    row = {"prompt": "p", "chosen": "", "rejected": ""}
    _text, prompt_ids = encode_generation_prompt(
        tokenizer, row, max_seq_length=10, max_new_tokens=6,
    )
    assert prompt_ids == (5, 6, 7, 8)


def test_referenced_dpo_samples_the_reference_with_scales_zeroed(
    tmp_path, monkeypatch,
):
    """The reference sample is the base policy, and the scales come back."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    model = _tiny_model(lora=True)
    trainer = MLXDPOTrainer(
        model, Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path)),
    )
    calls = []
    _run_generation_trainer(trainer, monkeypatch, calls)

    assert len(calls) == 2, "policy and reference"
    assert all(scale != 0.0 for scale in calls[0]["scales"])
    assert all(scale == 0.0 for scale in calls[1]["scales"])
    assert all(
        module.scale != 0.0 for _, module in iter_mlx_lora_modules(model)
    ), "scales restored after sampling"
    samples = trainer.last_generation_samples
    assert [row["policy"] for row in samples] == ["c1r0", "c1r1"]
    assert [row["reference"] for row in samples] == ["c2r0", "c2r1"], (
        "each row reports its own reference, from the reference call"
    )
    # The reference must decode under the budget the prompt was truncated for.
    for call in calls:
        assert call["defaults"].max_tokens == 8
        assert call["defaults"].sampling.temperature == 0.25


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_unreferenced_objectives_sample_the_policy_only(
    tmp_path, monkeypatch, objective,
):
    """ORPO has no reference and reference-free DPO has none either."""
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    common = _generation_common(tmp_path, generation_max_tokens=56)
    if objective == "orpo":
        cls, args = MLXORPOTrainer, MLXORPOConfig(**common)
    else:
        cls, args = MLXDPOTrainer, MLXDPOConfig(reference_free=True, **common)
    trainer = cls(
        _tiny_model(lora=True), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=args,
    )
    calls = []
    _run_generation_trainer(trainer, monkeypatch, calls)
    assert len(calls) == 1
    assert trainer.last_generation_samples[0]["reference"] is None

    from unsloth_zoo.mlx.preference import encode_generation_prompt
    samples = trainer.last_generation_samples
    for index, row in enumerate(rows(2)):
        request = calls[0]["requests"][index]
        text, expected = encode_generation_prompt(
            Tokenizer(), row, max_seq_length=64, max_new_tokens=56,
        )
        # Token ids, not text: the engine encodes without special tokens, so
        # text would drop what training adds. The budget bites at this length.
        assert request.prompt is None
        assert tuple(request.prompt_token_ids) == expected
        assert len(request.prompt_token_ids) == 64 - 56
        assert samples[index]["prompt"] == text
    assert calls[0]["defaults"].max_tokens == 56
    assert calls[0]["defaults"].sampling.temperature == 0.25


def test_generation_samples_every_split_of_a_dict_eval_dataset(
    tmp_path, monkeypatch,
):
    """A dict of splits is resolved, not iterated as prompts named by key.

    Each split holds more rows than num_generation_prompts, so the per-split cap
    has to bite for the count to come out right.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3),
        eval_dataset={"a": rows(3), "b": rows(3)},
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )
    calls = []
    _run_generation_trainer(trainer, monkeypatch, calls)
    assert len(calls) == 1
    assert len(calls[0]["requests"]) == 4, "two prompts from each split"
    assert [row["split"] for row in trainer.last_generation_samples] == [
        "a", "a", "b", "b",
    ]


@pytest.mark.parametrize("eval_steps,expected", [(0, True), (0.5, False), (2, False)])
def test_sampling_without_a_cadence_says_so(tmp_path, capsys, eval_steps, expected):
    """No cadence samples nothing, which is worth a word rather than silence.

    eval_steps is an HF interval, so a ratio in (0, 1) is a real cadence.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, eval_steps=eval_steps)),
    )
    trainer._prepare_data(False)
    said = "no evaluation cadence" in capsys.readouterr().out
    assert said is expected


def test_a_reused_trainer_does_not_report_the_previous_run_samples(
    tmp_path, monkeypatch,
):
    """Same contract the eval metrics get: a run reports only its own."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )
    _run_generation_trainer(trainer, monkeypatch, [])
    assert trainer.last_generation_samples

    trainer.args.eval_steps = 0
    _run_generation_trainer(trainer, monkeypatch, [])
    assert trainer.last_generation_samples == []


def test_sampling_runs_in_eval_mode_and_restores_training_mode(
    tmp_path, monkeypatch,
):
    """_evaluate restores training mode before sampling starts, so the sampler
    has to enter eval mode itself or it decodes under NEFTune noise and
    dropout -- printing text the policy does not actually produce. TRL samples
    inside its evaluation_loop, in eval mode."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )
    # The shim's Module.eval() is a no-op with no `training` attribute, so the
    # observable here is the call order, not the flag. The flag itself is
    # covered on the Metal lane.
    order = []
    _eval, _train = trainer.model.eval, trainer.model.train
    trainer.model.eval = lambda: (order.append("eval"), _eval())[1]
    trainer.model.train = lambda: (order.append("train"), _train())[1]

    calls = []
    _run_generation_trainer(trainer, monkeypatch, calls)

    assert calls, "the sampler never ran"
    assert order[-1] == "train", f"run did not end in training mode: {order}"
    # Scoring's own eval/train pair, then a second pair around the sampler.
    assert order.count("eval") >= 2, (
        f"sampling did not enter eval mode of its own: {order}"
    )


def test_a_failing_sampler_does_not_take_the_run_down(tmp_path, monkeypatch):
    """Sampling is a diagnostic. A raise here used to abort training and drop
    the evaluation that had already succeeded but was not yet logged."""
    from unsloth_zoo.mlx import generate as generate_module
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )

    class Exploding:
        # Not generate_batch: _run_generation_trainer substitutes its own, so a
        # failure injected there is overwritten before the sampler ever sees it.
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("engine out of memory")

    monkeypatch.setattr(generate_module, "GenerationRequest", Exploding)
    result = _run_generation_trainer(trainer, monkeypatch, [])

    assert result is not None, "the run was aborted by a failed sampler"
    assert trainer._last_eval_metrics, "the evaluation was discarded"
    assert any(
        "eval_loss" in entry for entry in trainer.state.log_history
    ), "the completed evaluation never reached the log"


def test_prompt_preparation_runs_inside_the_rng_guard(tmp_path, monkeypatch):
    """Building the eval batches draws from the shared RNG, so it has to run
    inside the preservation. Falsified against a run that builds none."""
    import random
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    held_out = [{"prompt": f"eval {index}: ", "chosen": "yes", "rejected": "no"}
                for index in range(4)]

    def training_draws(evaluating):
        drawn, calls = [], []
        random.seed(11)
        trainer = MLXDPOTrainer(
            _tiny_model(), Tokenizer(), rows(4),
            eval_dataset=held_out if evaluating else None,
            formatting_func=lambda row: (random.random(), row)[1],
            args=MLXDPOConfig(**_generation_common(
                tmp_path, reference_free=True, max_steps=2, eval_steps=1,
                generate_during_eval=evaluating)))
        # After each step, so the second lands past the first evaluation.
        trainer.add_step_callback(lambda *_a, **_k: drawn.append(random.random()))
        _run_generation_trainer(trainer, monkeypatch, calls)
        return drawn, calls

    evaluated, calls = training_draws(True)
    control, no_calls = training_draws(False)
    assert calls and not no_calls, "one run evaluated and the other did not"
    assert len(evaluated) == len(control) == 2
    assert evaluated == control, "building eval batches moved the training draws"


def test_generation_prompt_rejects_a_budget_with_no_room():
    """A non-positive budget would slice the wrong end, or the whole prompt."""
    from unsloth_zoo.mlx.preference import encode_generation_prompt

    with pytest.raises(ValueError, match="at least one"):
        encode_generation_prompt(
            MappingTokenizer({"p": [1, 2]}),
            {"prompt": "p", "chosen": "", "rejected": ""},
            max_seq_length=8, max_new_tokens=8,
        )


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_evaluation_reports_the_trl_metric_set(objective, tmp_path, monkeypatch):
    """No perplexity: a preference loss is not a per-token likelihood."""
    from unsloth_zoo.mlx.preference import PREFERENCE_EVAL_METRICS
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    cls, config = ((MLXORPOTrainer, MLXORPOConfig) if objective == "orpo"
                   else (MLXDPOTrainer, MLXDPOConfig))
    common = _generation_common(
        tmp_path, max_steps=1, eval_steps=1, generate_during_eval=False)
    if objective == "dpo":
        common["reference_free"] = True
    trainer = cls(_tiny_model(), Tokenizer(), rows(4), eval_dataset=rows(3),
                  args=config(**common))
    _run_generation_trainer(trainer, monkeypatch, [])
    metrics = trainer._last_eval_metrics
    assert not [name for name in PREFERENCE_EVAL_METRICS[objective]
                if f"eval_{name}" not in metrics]
    assert "eval_loss" in metrics and "eval_perplexity" not in metrics
    assert any("eval_loss" in entry for entry in trainer.state.log_history)


def test_eval_loss_weights_every_pair_once_across_a_ragged_tail():
    """Three rows at batch_size 2 leave a one-pair tail. Weighting each batch
    equally would give that tail the same say as the two pairs before it."""
    from unsloth_zoo.mlx.preference import (
        create_preference_batch_plan, make_preference_eval_fn,
    )

    options = dict(length_policy=resolved(max_seq_length=64), num_epochs=1,
                   grad_accum=1, preserve_dataset_order=True)
    split = create_preference_batch_plan(
        rows(3), Tokenizer(), batch_size=2, **options)
    whole = create_preference_batch_plan(
        rows(3), Tokenizer(), batch_size=3, **options)
    assert len(split) == 2 and len(whole) == 1
    model = TinyModel()
    eval_fn = make_preference_eval_fn("dpo", beta=0.1, reference_free=True)
    total, weight = 0.0, 0
    for index in range(len(split)):
        loss, pairs, _stats = eval_fn(model, *split[index])
        total += float(loss) * int(pairs)
        weight += int(pairs)
    assert weight == 3, "every pair is weighted once"
    assert math.isclose(total / weight, float(eval_fn(model, *whole[0])[0]),
                        rel_tol=2e-5, abs_tol=2e-5)


def test_the_orpo_logit_sum_is_accumulated_in_float32():
    """ORPO reduces raw logits, so the accumulator dtype is the model's own.

    mx.sum does not promote, and a batch holds millions of logits: in bf16 the
    running sum stops registering addends long before the end, and the cast in
    mx.stack lands after the damage. DPO is safe by construction -- its float32
    response mask promotes the multiply before the reduction.

    This shim runs in float32, so the assertion is on the dtype rather than on
    the value; only a real-mlx run reproduces the drift itself.
    """
    import mlx.core as mx
    from unsloth_zoo.mlx import preference as pref

    logits = mx.random.uniform(shape=(2, 16, 64)) * 2.0
    for dtype in (mx.bfloat16, mx.float16):
        summed, count = pref._orpo_logit_sum(logits.astype(dtype))
        assert summed.dtype == mx.float32
        assert float(count) == 2 * 16 * 64
        assert float(summed) == pytest.approx(
            float(logits.astype(dtype).astype(mx.float32).sum()), rel=1e-3)


def test_a_token_mean_is_taken_over_every_token_not_every_batch():
    """Two pairs averaging 1 over two tokens then a one-pair tail averaging 10
    over four is a mean of 7; by-pair weighting gives 4, equal weighting 5.5."""
    import mlx.core as mx
    from unsloth_zoo.mlx import preference as pref
    from unsloth_zoo.mlx.trainer import MLXTrainer

    names = pref.PREFERENCE_EVAL_METRICS["dpo"]
    at = names.index("logits/chosen")
    over = pref.PREFERENCE_EVAL_DENOMINATORS["dpo"][at]

    def scored(logit_sum, tokens, pairs):
        values = [0.0] * pref.PREFERENCE_EVAL_STATS_WIDTH["dpo"]
        values[at], values[over] = logit_sum, tokens
        return mx.array(0.0), mx.array(pairs), mx.array(values)

    batches = {"full": scored(2.0, 2.0, 2), "tail": scored(40.0, 4.0, 1)}
    loss_fn = lambda _m, name, _l, _labels=None: batches[name]
    loss_fn._unsloth_preference_metrics = names
    loss_fn._unsloth_preference_denominators = pref.PREFERENCE_EVAL_DENOMINATORS["dpo"]
    loss_fn._unsloth_preference_stats_width = pref.PREFERENCE_EVAL_STATS_WIDTH["dpo"]

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model, trainer.stop_requested = _tiny_model(), False
    trainer._evaluate(
        [("full", None, None), ("tail", None, None)], loss_fn, is_vlm=False)
    assert trainer._last_eval_metrics["eval_logits/chosen"] == pytest.approx(7.0)


@pytest.mark.parametrize("eval_dataset,overrides,message", [
    pytest.param(None, {"load_best_model_at_end": True},
                 "need an eval_dataset", id="nothing-to-select-on"),
    pytest.param({}, {"load_best_model_at_end": True},
                 "at least one eval split", id="empty-mapping"),
    pytest.param([], {}, "eval dataset is empty", id="empty-split"),
    pytest.param({"a": rows(1), "b": []}, {}, "eval dataset is empty",
                 id="one-empty-split"),
    pytest.param(iter(rows(2)), {}, "requires a finite", id="unsized"),
    pytest.param(rows(2), {"max_eval_batches": 1}, "max_eval_batches",
                 id="capped-by-batch-count"),
])
def test_an_evaluation_that_cannot_be_trusted_is_rejected_before_the_run(
    eval_dataset, overrides, message, tmp_path,
):
    """Each would otherwise report a number that reads like a real evaluation."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(4), eval_dataset=eval_dataset,
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, generate_during_eval=False,
            **overrides)))
    with pytest.raises(ValueError, match=message):
        trainer._prepare_data(False)


def test_the_best_metric_direction_reaches_callbacks_that_read_it(tmp_path):
    """HF's EarlyStoppingCallback reads args.greater_is_better itself, so an
    unresolved direction stops a run whose reward accuracy is still climbing."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(4), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, generate_during_eval=False,
            metric_for_best_model="eval_rewards/accuracies")))
    assert trainer.args.greater_is_better is True


def test_sampled_text_cannot_drive_the_terminal(tmp_path, monkeypatch, capsys):
    """Dataset prompts and model completions are untrusted text bound for a
    terminal, and str.split() leaves ESC/BEL untouched, so an OSC sequence
    arrived intact and a CSI erase could rewrite the log above it. Falsified
    against the whitespace-only version: the raw escapes appear in the output.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3),
        eval_dataset={"ev\x1b[2Kil": rows(2)},
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )

    def hostile_generate_batch(model, tokenizer, requests, *, defaults=None):
        return [
            types.SimpleNamespace(
                token_ids=[1, 2],
                text="ok\x1b]52;c;U0VDUkVU\x07forged‮desrever",
                logprobs=[0.0, 0.0], finish_reason="length", stop_match=None,
            )
            for _ in requests
        ]

    _run_generation_trainer(
        trainer, monkeypatch, [], generate_batch=hostile_generate_batch,
    )

    printed = capsys.readouterr().out
    assert "forged" in printed, "the text itself still shows"
    for raw in ("\x1b]52", "\x07", "‮", "\x1b[2K"):
        assert raw not in printed, f"{raw!r} reached the terminal"
    assert "\\x1b" in printed, "the escape is shown, not executed"


def test_a_failing_decode_joins_the_rank_consensus(tmp_path, monkeypatch):
    """A rank that unwinds never reaches the _distributed_should_stop() call
    between the two decodes, stranding a healthy peer there, so the failing rank
    must join a consensus first. Falsified against an unguarded decode: no
    consensus is joined at all.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path, reference_free=True)),
    )

    joined = []
    original = trainer._distributed_any_flag
    monkeypatch.setattr(
        trainer, "_distributed_any_flag",
        lambda flag: (joined.append(bool(flag)), original(flag))[1],
    )

    def exploding_generate_batch(model, tokenizer, requests, *, defaults=None):
        raise RuntimeError("engine out of memory")

    result = _run_generation_trainer(
        trainer, monkeypatch, [], generate_batch=exploding_generate_batch,
    )

    assert result is not None, "the run was aborted by a failed sampler"
    assert True in joined, "the failing rank never joined the consensus"
    assert trainer.last_generation_samples == []
    assert trainer._last_eval_metrics, "the evaluation was discarded"


def test_a_failed_reference_decode_publishes_no_samples(
    tmp_path, monkeypatch, capsys,
):
    """A referenced run that loses its reference half skips the whole set:
    reference=None is what ORPO and reference-free DPO publish, so the policy
    half would read as a legitimate policy-only set, after _decode already said
    it was skipping. Falsified against the unguarded publish: two rows arrive,
    each with reference None.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    model = _tiny_model(lora=True)
    trainer = MLXDPOTrainer(
        model, Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(tmp_path)),
    )

    seen = []

    def failing_reference(model, tokenizer, requests, *, defaults=None):
        seen.append(1)
        if len(seen) > 1:
            raise RuntimeError("engine out of memory")
        return [
            types.SimpleNamespace(
                token_ids=[1, 2], text=f"policy{index}", logprobs=[0.0, 0.0],
                finish_reason="length", stop_match=None,
            )
            for index, _ in enumerate(requests)
        ]

    result = _run_generation_trainer(
        trainer, monkeypatch, [], generate_batch=failing_reference,
    )

    assert result is not None, "the run was aborted by a failed sampler"
    assert len(seen) == 2, "the policy decode ran, then the reference one failed"
    assert trainer.last_generation_samples == [], (
        "a policy-only set was published for a failed reference pass"
    )
    printed = capsys.readouterr().out
    assert "reference generation failed" in printed
    assert "policy0" not in printed, "samples printed after saying they are skipped"
    assert all(
        module.scale != 0.0 for _, module in iter_mlx_lora_modules(model)
    ), "scales restored after the failure"
    assert trainer._last_eval_metrics, "the evaluation itself still stands"


def test_best_model_without_a_cadence_says_so(tmp_path, capsys):
    """Asking for best-model selection and silently getting none is the worst
    outcome: _best_step stays None and the restore is skipped. A callback owning
    the cadence is legitimate, so this is a notice, not a refusal."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    def _notice(eval_strategy=None, **overrides):
        trainer = MLXDPOTrainer(
            _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
            args=MLXDPOConfig(**_generation_common(
                tmp_path, reference_free=True, generate_during_eval=False,
                load_best_model_at_end=True, **overrides)),
        )
        # Not a config field; _ensure_callback_args_compat attaches it to args.
        if eval_strategy is not None:
            trainer.args.eval_strategy = eval_strategy
        trainer._prepare_data(False)
        return "no evaluation cadence is set" in capsys.readouterr().out

    assert _notice(eval_steps=0) is True
    assert _notice(eval_steps=1) is False
    assert _notice(eval_steps=0, eval_strategy="epoch") is False


def test_the_sampling_cadence_notice_understands_an_epoch_strategy(
    tmp_path, capsys,
):
    """eval_strategy="epoch" is a cadence without eval_steps, so reading
    eval_steps alone called a real cadence "no cadence". Falsified against the
    eval_steps-only test: the notice prints for the epoch run."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, eval_steps=0)),
    )
    trainer.args.eval_strategy = "epoch"
    trainer._prepare_data(False)
    assert "no evaluation cadence" not in capsys.readouterr().out


def test_an_eval_split_is_held_to_the_length_it_declares(tmp_path, monkeypatch):
    """len() is checked at configuration time but the plan builder iterates to
    exhaustion, so a split claiming 2 rows and yielding 3 had all 3 scored, and
    an endless sized iterable would hang. Falsified against the unbounded loop:
    3 rows are scored and nothing is raised."""
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    class LyingSplit:
        """Declares two rows, yields three."""

        def __init__(self, rows_):
            self._rows = rows_

        def __len__(self):
            return len(self._rows) - 1

        def __iter__(self):
            return iter(self._rows)

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=LyingSplit(rows(3)),
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, max_steps=1, eval_steps=1,
            generate_during_eval=False)),
    )
    # The plan is built at the first evaluation, not at _prepare_data.
    with pytest.raises(ValueError, match="more rows than the 2 it declares"):
        _run_generation_trainer(trainer, monkeypatch, [])


def test_a_step_cadence_under_eval_strategy_no_is_no_cadence(tmp_path, capsys):
    """eval_steps > 0 under eval_strategy="no" evaluates never, so reading
    eval_steps alone suppressed the notice for the one case it exists to report.
    Falsified against the eval_steps-only test: nothing prints.
    """
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    trainer = MLXDPOTrainer(
        _tiny_model(), Tokenizer(), rows(3), eval_dataset=rows(2),
        args=MLXDPOConfig(**_generation_common(
            tmp_path, reference_free=True, generate_during_eval=False,
            load_best_model_at_end=True, eval_steps=1)),
    )
    trainer.args.eval_strategy = "no"
    trainer._prepare_data(False)
    assert "no evaluation cadence is set" in capsys.readouterr().out

@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_evaluation_accepts_a_wrapped_model_output(objective):
    """A text-only VLM load takes the preference path, and mlx-vlm wrappers return
    a LanguageModelOutput where mlx_lm models return the array. Evaluation reads
    the same forward the training loss does, so it has to accept both."""
    from unsloth_zoo.mlx.preference import (
        create_preference_batch_plan, make_preference_eval_fn,
    )

    class Wrapped(TinyModel):
        def __call__(self, tokens):
            class LanguageModelOutput:
                def __init__(self, logits):
                    self.logits = logits
            return LanguageModelOutput(super().__call__(tokens))

    plan = create_preference_batch_plan(
        rows(2), Tokenizer(), batch_size=2,
        length_policy=resolved(max_seq_length=64),
        num_epochs=1, grad_accum=1, preserve_dataset_order=True)
    eval_fn = make_preference_eval_fn(objective, beta=0.1, reference_free=True)

    bare = TinyModel()
    wrapped = Wrapped()
    wrapped.embedding, wrapped.output = bare.embedding, bare.output

    from_bare = eval_fn(bare, *plan[0])
    from_wrapped = eval_fn(wrapped, *plan[0])
    assert math.isclose(float(from_bare[0]), float(from_wrapped[0]),
                        rel_tol=1e-6, abs_tol=1e-6)
    assert int(from_bare[1]) == int(from_wrapped[1])
