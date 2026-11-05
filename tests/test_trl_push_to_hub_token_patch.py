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

"""patch_trl_push_to_hub_token must never break TrainingArguments -> SFTConfig.

transformers 5 dropped push_to_hub_token from TrainingArguments. TRL 0.22-0.27
bare-pop it from to_dict() (KeyError on transformers 5), while TRL 1.x pops it
only below transformers 5 and passes everything else to SFTConfig(**d), so a
to_dict() that carries the key raises TypeError there.
"""

import copy
import json
import pickle

import pytest
import torch

transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")

from transformers import TrainingArguments  # noqa: E402

from unsloth_zoo.utils import Version  # noqa: E402
from unsloth_zoo.temporary_patches import misc  # noqa: E402

TF5 = Version(transformers.__version__) >= Version("5.0.0")


@pytest.fixture
def patched(monkeypatch):
    # Restore the real to_dict after each test, whatever the patch did.
    monkeypatch.setattr(TrainingArguments, "to_dict", TrainingArguments.to_dict)
    misc.patch_trl_push_to_hub_token()
    yield


def _args(tmp_path):
    return TrainingArguments(
        output_dir = str(tmp_path / "out"),
        report_to = "none",
        max_steps = 1,
        per_device_train_batch_size = 2,
        use_cpu = True,
    )


def _tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4}
    tk = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object = tk, pad_token = "<pad>", eos_token = "<eos>", unk_token = "<unk>"
    )
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = len(vocab),
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
        pad_token_id = 0,
        eos_token_id = 1,
    )
    return LlamaForCausalLM(cfg), tok


def test_patch_marks_to_dict_only_on_transformers_5(patched):
    marked = getattr(TrainingArguments.to_dict, "_unsloth_patched", False)
    assert marked == TF5


def test_to_dict_does_not_gain_push_to_hub_token(patched, tmp_path):
    d = _args(tmp_path).to_dict()
    if TF5:
        assert "push_to_hub_token" not in d
    else:
        assert "push_to_hub_token" in d


def test_bare_pop_is_safe(patched, tmp_path):
    # What TRL 0.22-0.27 do in SFTTrainer / IterativeSFTTrainer __init__.
    d = _args(tmp_path).to_dict()
    d.pop("push_to_hub_token")
    assert "push_to_hub_token" not in d
    with pytest.raises(KeyError):
        d.pop("definitely_not_a_field")
    assert d.pop("definitely_not_a_field", 7) == 7


def test_to_dict_result_still_serialises(patched, tmp_path):
    a = _args(tmp_path)
    d = a.to_dict()
    assert isinstance(d, dict)
    plain = {**d}
    assert type(plain) is dict
    assert json.loads(json.dumps(d)) == json.loads(json.dumps(plain))
    assert pickle.loads(pickle.dumps(d)) == d
    assert copy.deepcopy(d) == d
    json.loads(a.to_json_string())
    a.to_sanitized_dict()


def test_patch_is_idempotent(patched):
    first = TrainingArguments.to_dict
    misc.patch_trl_push_to_hub_token()
    assert TrainingArguments.to_dict is first


def test_sft_config_from_to_dict(patched, tmp_path):
    from trl import SFTConfig

    a = _args(tmp_path)
    d = a.to_dict()
    d["hub_token"] = a.hub_token
    if not TF5:
        d.pop("push_to_hub_token")
    assert isinstance(SFTConfig(**d), SFTConfig)


def test_sft_trainer_from_training_arguments(patched, tmp_path):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _tiny_model_and_tokenizer()
    ds = Dataset.from_dict({"text": ["hello world"] * 4})
    trainer = SFTTrainer(
        model = model, args = _args(tmp_path), train_dataset = ds, processing_class = tok
    )
    assert isinstance(trainer.args, SFTConfig)
    assert trainer.args.max_steps == 1
