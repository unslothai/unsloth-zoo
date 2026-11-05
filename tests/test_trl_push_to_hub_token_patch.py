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

"""patch_trl_push_to_hub_token must never break TrainingArguments -> SFTConfig."""

import copy
import importlib.metadata
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
TRL_BARE_POP = Version(trl.__version__) < Version("0.27.2")
# Importing unsloth_zoo may already have patched to_dict for the installed TRL.
UNPATCHED_TO_DICT = getattr(TrainingArguments.to_dict, "__wrapped__", TrainingArguments.to_dict)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "to_dict", UNPATCHED_TO_DICT)
    misc.patch_trl_push_to_hub_token()
    yield


@pytest.fixture(params = ["0.27.1", "0.27.2"])
def patched_as_trl(request, monkeypatch):
    # Fake the installed TRL version the patch reads from package metadata.
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: request.param if name == "trl" else real_version(name),
    )
    monkeypatch.setattr(TrainingArguments, "to_dict", UNPATCHED_TO_DICT)
    misc.patch_trl_push_to_hub_token()
    yield Version(request.param) < Version("0.27.2")


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


def test_patch_only_on_transformers_5_with_bare_pop_trl(patched_as_trl):
    marked = getattr(TrainingArguments.to_dict, "_unsloth_patched", False)
    assert marked == (TF5 and patched_as_trl)


def test_bare_pop_trl_finds_push_to_hub_token(patched_as_trl, tmp_path):
    d = _args(tmp_path).to_dict()
    if TF5 and not patched_as_trl:
        assert "push_to_hub_token" not in d
    else:
        # What TRL 0.22-0.27.1 do in SFTTrainer / IterativeSFTTrainer __init__.
        d.pop("push_to_hub_token")


def test_to_dict_result_still_serialises(patched_as_trl, tmp_path):
    a = _args(tmp_path)
    d = a.to_dict()
    assert type(d) is dict
    assert json.loads(json.dumps(d)) == d
    assert pickle.loads(pickle.dumps(d)) == d
    assert copy.deepcopy(d) == d
    path = tmp_path / "args.pt"
    torch.save(d, path)
    assert torch.load(path, weights_only = True) == d
    json.loads(a.to_json_string())
    a.to_sanitized_dict()


def test_patch_is_idempotent(patched_as_trl):
    first = TrainingArguments.to_dict
    misc.patch_trl_push_to_hub_token()
    assert TrainingArguments.to_dict is first


def test_sft_config_from_to_dict(patched, tmp_path):
    from trl import SFTConfig

    a = _args(tmp_path)
    d = a.to_dict()
    d["hub_token"] = a.hub_token
    if not TF5 or TRL_BARE_POP:
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
