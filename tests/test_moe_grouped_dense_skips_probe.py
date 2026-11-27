"""Dense models skip the grouped_mm probe; any MoE signal still reaches it. CPU only."""
import types

import pytest
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return dict(self.__dict__)


class _Model(nn.Module):
    def __init__(self, config, body=None):
        super().__init__()
        self.config = config
        self.body = body if body is not None else nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


@pytest.fixture
def probe(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return False   # no grouped_mm: enable_grouped_moe returns right after the probe

    monkeypatch.delenv("UNSLOTH_MOE_GROUPED", raising=False)
    monkeypatch.setattr(ML, "_grouped_mm_supported", fake)
    return calls


def test_dense_config_skips_probe(probe):
    dense = _Model(_Cfg(model_type="llama", hidden_size=4, num_hidden_layers=2,
                        rope_scaling={"type": "linear", "factor": 2.0}))
    assert ML.enable_grouped_moe(dense, verbose=False) == 0
    assert ML.auto_enable_grouped_moe(dense) is None
    assert probe == []


def test_dense_vlm_text_config_skips_probe(probe):
    vlm = _Model(_Cfg(model_type="gemma3", text_config=_Cfg(hidden_size=4), vision_config=_Cfg(hidden_size=4)))
    assert ML.enable_grouped_moe(vlm, verbose=False) == 0
    assert probe == []


@pytest.mark.parametrize("cfg", [
    _Cfg(num_experts=8),                                        # Qwen2/3-MoE, OLMoE
    _Cfg(num_local_experts=8),                                  # Mixtral, gpt-oss
    _Cfg(n_routed_experts=8),                                   # DeepSeek, GLM4-MoE
    _Cfg(moe_num_experts=8),                                    # Ernie 4.5
    _Cfg(text_config=_Cfg(num_experts=8)),                      # Qwen3-VL-MoE, Llama 4
    _Cfg(ffn_config=_Cfg(moe_num_experts=8)),                   # DBRX
    _Cfg(text_config={"num_local_experts": 8}),                 # sub-config kept as a dict
])
def test_moe_config_still_probes(probe, cfg):
    assert ML.enable_grouped_moe(_Model(cfg), verbose=False) == 0
    assert probe == [1]


def test_unset_expert_count_is_dense(probe):
    ML.enable_grouped_moe(_Model(_Cfg(num_experts=None, num_local_experts=0)), verbose=False)
    assert probe == []


def test_moe_structure_without_config_still_probes(probe):
    block_cls = type(next(iter(ML._BLOCK_SPECS)), (nn.Module,), {})
    ML.enable_grouped_moe(_Model(_Cfg(), nn.Sequential(block_cls())), verbose=False)
    holder = nn.Module()
    holder.experts = nn.ModuleList([nn.Linear(4, 4)])   # fused / stacked experts module
    ML.enable_grouped_moe(_Model(None, nn.Sequential(holder)), verbose=False)
    ML.enable_grouped_moe(nn.Sequential(holder), verbose=False)   # no config attribute at all
    assert probe == [1, 1, 1]


def test_previous_grouped_patch_still_reaches_restore(probe, monkeypatch):
    """A block carrying our patch keeps the old path, so a now-ineligible block is restored after the probe."""
    lin = nn.Linear(4, 4)
    original = lin.forward
    lin._orig_moe_forward = original
    lin.forward = types.MethodType(ML.grouped_moe_forward, lin)
    model = _Model(_Cfg(), nn.Sequential(lin))
    monkeypatch.setattr(ML, "_grouped_mm_supported", lambda: probe.append(1) or True)
    assert ML.enable_grouped_moe(model, verbose=False) == 0
    assert probe == [1]
    assert not hasattr(lin, "_orig_moe_forward") and lin.forward == original


def test_grouped_off_restores_without_probe(probe, monkeypatch):
    lin = nn.Linear(4, 4)
    original = lin.forward
    lin._orig_moe_forward = original
    lin.forward = types.MethodType(ML.grouped_moe_forward, lin)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED", "0")
    assert ML.enable_grouped_moe(_Model(_Cfg(num_experts=8), nn.Sequential(lin)), verbose=False) == 0
    assert probe == []
    assert not hasattr(lin, "_orig_moe_forward") and lin.forward == original


def test_unreadable_config_probes(probe):
    class _Bad:
        @property
        def __dict__(self):
            raise RuntimeError("boom")

    ML.enable_grouped_moe(_Model(_Bad()), verbose=False)
    assert probe == [1]
