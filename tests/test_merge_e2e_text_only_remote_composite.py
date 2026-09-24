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

"""merged_16bit of a text_only load of a repo-code composite (Nemotron-Omni, InternVL).

Same contract as #969 for Gemma 3: the merge writes the checkpoint's own tensors
(language_model.* plus the vision parts), so the config beside them must be the
composite's. For a repo-code composite that config only loads with the repo's code,
so it has to be read with the load's trust_remote_code and the code files copied.
Before: AutoConfig(trust_remote_code=False) raised, the text config was saved, and a
reload initialised every text weight at random without an error.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch

import _merge_e2e_helpers as H

_CONFIGURATION = '''
from transformers import PretrainedConfig, LlamaConfig


class TinyVisionConfig(PretrainedConfig):
    model_type = "tiny_vision_zoo"

    def __init__(self, hidden_size = 8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class TinyOmniConfig(PretrainedConfig):
    model_type = "tiny_omni_zoo"
    is_composition = True

    def __init__(self, llm_config = None, vision_config = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_config = LlamaConfig(**(llm_config or {}))
        self.vision_config = TinyVisionConfig(**(vision_config or {}))
'''

_MODELING = '''
from torch import nn
from transformers import PreTrainedModel, LlamaForCausalLM
from .configuration_tiny_omni import TinyOmniConfig


class TinyOmni(PreTrainedModel):
    config_class = TinyOmniConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        self.language_model = LlamaForCausalLM(config.llm_config)
        self.vision_model = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size)
        self.post_init()

    def forward(self, pixel_values, input_ids = None, **kwargs):
        return self.language_model(input_ids = input_ids, **kwargs)
'''

_LLM = dict(hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2, num_attention_heads = 2,
            num_key_value_heads = 1, vocab_size = 64, max_position_embeddings = 64, tie_word_embeddings = False)


def _write_base(tmp_path):
    import transformers as T
    from safetensors.torch import save_file
    base = os.path.join(str(tmp_path), "base")
    os.makedirs(base)
    open(os.path.join(base, "configuration_tiny_omni.py"), "w").write(_CONFIGURATION)
    open(os.path.join(base, "modeling_tiny_omni.py"), "w").write(_MODELING)
    json.dump({
        "model_type": "tiny_omni_zoo", "architectures": ["TinyOmni"],
        "auto_map": {"AutoConfig": "configuration_tiny_omni.TinyOmniConfig",
                     "AutoModel": "modeling_tiny_omni.TinyOmni",
                     "AutoModelForCausalLM": "modeling_tiny_omni.TinyOmni"},
        "llm_config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"], **_LLM},
        "vision_config": {"hidden_size": 8},
    }, open(os.path.join(base, "config.json"), "w"))
    torch.manual_seed(H.SEED)
    text = T.LlamaForCausalLM(T.LlamaConfig(**_LLM)).to(torch.float32)
    tensors = {"language_model." + k: v.contiguous() for k, v in text.state_dict().items()}
    tensors["vision_model.weight"] = torch.randn(8, 8)
    tensors["vision_model.bias"] = torch.randn(8)
    save_file(tensors, os.path.join(base, "model.safetensors"))
    return base, text.state_dict()


def _text_only_peft(base, state, trusted):
    """What Unsloth's text_only branch leaves in memory: the bare causal LM with the checkpoint's
    weights, _name_or_path pointing at the composite, and the load's trust decision recorded."""
    import transformers as T
    from peft import LoraConfig, get_peft_model
    model = T.LlamaForCausalLM(T.LlamaConfig(**_LLM)).to(torch.float32)
    model.load_state_dict(state)
    model.config._name_or_path = base
    if trusted:
        model._unsloth_trust_remote_code = True
    pm = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
                                          target_modules = ["q_proj", "v_proj"]))
    H.seed_lora(pm)
    return pm


def test_remote_composite_export_is_the_composite_with_its_code(tmp_path):
    import transformers as T
    H.set_offline_cpu_env()
    os.environ["HF_MODULES_CACHE"] = os.path.join(str(tmp_path), "modules")
    base, state = _write_base(tmp_path)
    out = os.path.join(str(tmp_path), "merged")
    pm = _text_only_peft(base, state, trusted = True)
    ids = torch.tensor([[1, 5, 9, 13, 17]])
    with torch.no_grad():
        trained = pm(input_ids = ids).logits

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        H.run_merge(pm, base, out, save_dtype = torch.float32)
    assert not any("Could not read the base config" in str(w.message) for w in caught)

    saved = json.load(open(os.path.join(out, "config.json")))
    assert saved["architectures"] == ["TinyOmni"] and saved["model_type"] == "tiny_omni_zoo"
    assert saved["auto_map"]["AutoModel"] == "modeling_tiny_omni.TinyOmni"
    for f in ("configuration_tiny_omni.py", "modeling_tiny_omni.py"):
        assert os.path.isfile(os.path.join(out, f)), f"{f} not carried into the export"
    written = H.read_safetensors_dir(out)
    assert torch.equal(written["vision_model.weight"], H.read_safetensors_dir(base)["vision_model.weight"])

    reloaded, info = T.AutoModel.from_pretrained(out, trust_remote_code = True, local_files_only = True,
                                                 output_loading_info = True, dtype = torch.float32)
    assert type(reloaded).__name__ == "TinyOmni"
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not info.get(field), f"{field}: {sorted(info.get(field))[:6]}"
    with torch.no_grad():
        got = reloaded.language_model(input_ids = ids).logits
    torch.testing.assert_close(got, trained, atol = 1e-4, rtol = 1e-4)


def test_untrusted_load_keeps_the_warned_fallback(tmp_path):
    # A load that never ran the repo's code must not start running it at export time.
    H.set_offline_cpu_env()
    os.environ["HF_MODULES_CACHE"] = os.path.join(str(tmp_path), "modules")
    base, state = _write_base(tmp_path)
    out = os.path.join(str(tmp_path), "merged")
    pm = _text_only_peft(base, state, trusted = False)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        H.run_merge(pm, base, out, save_dtype = torch.float32)
    assert any("Could not read the base config" in str(w.message) for w in caught)
    assert not any(f.endswith(".py") for f in os.listdir(out))


def test_text_configs_sees_llm_config():
    import unsloth_zoo.saving_utils as SU
    import transformers as T
    from types import SimpleNamespace as NS
    llm = T.LlamaConfig(vocab_size = 77)
    cfg = NS(llm_config = llm, get_text_config = lambda: cfg)
    assert any(h is llm for h in SU._text_configs(cfg))
    assert SU._config_vocab_size(cfg) == 77


def test_trust_marker_found_through_peft_layers():
    import unsloth_zoo.saving_utils as SU
    from types import SimpleNamespace as NS
    inner = NS(_unsloth_trust_remote_code = True)
    assert SU._loaded_with_trust_remote_code(NS(base_model = NS(model = inner)))
    assert not SU._loaded_with_trust_remote_code(NS(base_model = NS(model = NS())))
    # Only a real True counts (a MagicMock-like truthy attribute does not).
    assert not SU._loaded_with_trust_remote_code(NS(_unsloth_trust_remote_code = "yes"))


def test_trust_is_not_carried_to_a_different_repo(tmp_path):
    # The load trusted `base`; an export reading another repo (a resolved FP8 -> 16bit sibling, a
    # name-mapped repo) must not run that repo's code on the strength of the first approval.
    from unsloth_zoo import saving_utils as S
    H.set_offline_cpu_env()
    os.environ["HF_MODULES_CACHE"] = os.path.join(str(tmp_path), "modules")
    base, state = _write_base(tmp_path)
    pm = _text_only_peft(base, state, trusted = True)
    assert type(S._read_export_base_config(base, None, pm, source_is_loaded_repo = True)).__name__.startswith("TinyOmni")
    with pytest.raises(Exception):
        S._read_export_base_config(base, None, pm, source_is_loaded_repo = False)
    assert not os.path.isdir(os.path.join(str(tmp_path), "modules", "transformers_modules", "base"))


def test_trust_is_not_carried_to_a_substituted_source(tmp_path, monkeypatch):
    # The load trusted `base`, but the source resolution hands back another directory (a local
    # copy found in the working directory, a sibling): its code must not run under that approval.
    import shutil
    from unsloth_zoo import saving_utils as S
    H.set_offline_cpu_env()
    os.environ["HF_MODULES_CACHE"] = os.path.join(str(tmp_path), "modules")
    base, state = _write_base(tmp_path)
    other = os.path.join(str(tmp_path), "substituted")
    shutil.copytree(base, other)
    real = S.determine_base_model_source
    monkeypatch.setattr(S, "determine_base_model_source", lambda name, *a, **k: (other, *real(name, *a, **k)[1:]))
    out = os.path.join(str(tmp_path), "merged")
    pm = _text_only_peft(base, state, trusted = True)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        H.run_merge(pm, base, out, save_dtype = torch.float32)
    assert any("Could not read the base config" in str(w.message) and "substituted" in str(w.message) for w in caught)
    assert not any(f.endswith(".py") for f in os.listdir(out))


def test_a_relative_load_path_is_the_same_trusted_source(tmp_path, monkeypatch):
    # Loaded as `./base`, resolved to an absolute directory by the source lookup: still the repo the
    # load trusted, so the export is the composite with its code, not the text-only fallback.
    H.set_offline_cpu_env()
    os.environ["HF_MODULES_CACHE"] = os.path.join(str(tmp_path), "modules")
    base, state = _write_base(tmp_path)
    monkeypatch.chdir(os.path.dirname(base))
    pm = _text_only_peft(base, state, trusted = True)
    pm.config._name_or_path = os.path.join(".", os.path.basename(base))
    out = os.path.join(str(tmp_path), "merged")
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        H.run_merge(pm, os.path.abspath(base), out, save_dtype = torch.float32)
    assert not any("Could not read the base config" in str(w.message) for w in caught)
    assert json.load(open(os.path.join(out, "config.json")))["architectures"] == ["TinyOmni"]
    assert os.path.isfile(os.path.join(out, "modeling_tiny_omni.py"))


def test_hub_repo_code_is_pinned_to_the_loaded_commit(monkeypatch, tmp_path):
    # A Hub repo is re-read, and its code copied, at the commit the trusted load ran, never the
    # branch head; with no known commit the trusted re-read is refused.
    import transformers
    import huggingface_hub
    from unsloth_zoo import saving_utils as S
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("trust_remote_code"): raise ValueError("needs repo code")
        return "config"
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    model = torch.nn.Linear(1, 1)
    model.config = type("C", (), {"_name_or_path": "org/repo"})()
    model._unsloth_trust_remote_code = True
    assert S._is_export_source_loaded_repo("org/repo", model)
    assert not S._is_export_source_loaded_repo("org/repo-bf16", model)
    with pytest.raises(ValueError):
        S._read_export_base_config("org/repo", None, model, source_is_loaded_repo = True)
    assert len(calls) == 1

    commit = "a" * 40
    model._unsloth_trust_remote_code_commit = commit
    assert S._read_export_base_config("org/repo", None, model, source_is_loaded_repo = True) == "config"
    assert calls[-1]["revision"] == commit and calls[-1]["code_revision"] == commit

    seen = []
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files",
                        lambda self, name, token = None, revision = None: seen.append(revision) or ["modeling_x.py", "sub/y.py"])
    src = tmp_path / "modeling_x.py"; src.write_text("# code")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda name, file, token = None, revision = None: seen.append(revision) or str(src))
    assert S._copy_remote_code_files("org/repo", str(tmp_path / "out"), revision = commit) == ["modeling_x.py"]
    assert seen == [commit, commit]
