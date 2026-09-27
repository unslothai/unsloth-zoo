"""gradient_checkpointing_enable() on remote-code wrappers that leave supports_gradient_checkpointing False.

nvidia Nemotron-3-Nano-Omni's NemotronH_Nano_Omni_Reasoning_V3 (and its inner remote NemotronHForCausalLM)
keep the PreTrainedModel default False while their decoder blocks are GradientCheckpointingLayer, so
transformers raises "does not support gradient checkpointing". The zoo patch inherits the flag from a
checkpointable submodule. CPU only.
"""
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer


class _Cfg(PretrainedConfig):
    model_type = "unsloth_gc_inherit_test"

    def __init__(self, hidden_size=8, n_layers=3, **kw):
        self.hidden_size, self.n_layers = hidden_size, n_layers
        super().__init__(**kw)


class _Block(GradientCheckpointingLayer):
    calls = 0

    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(h, h)

    def forward(self, x):
        type(self).calls += 1
        return x + torch.tanh(self.lin(x))


class _Inner(PreTrainedModel):  # remote NemotronHForCausalLM stand-in: flag left False
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(16, config.hidden_size)
        self.layers = nn.ModuleList(_Block(config.hidden_size) for _ in range(config.n_layers))
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class _Wrapper(PreTrainedModel):  # NemotronH_Nano_Omni_Reasoning_V3 stand-in
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.language_model = _Inner(config)
        self.mlp1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(self, input_ids):
        return self.language_model(input_ids)


class _NoLayers(PreTrainedModel):
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.lin = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()


@pytest.fixture
def patched():
    saved = PreTrainedModel.gradient_checkpointing_enable
    try:
        from unsloth_zoo.temporary_patches.misc import patch_gradient_checkpointing_enable_inherit
    except ImportError:
        patch_gradient_checkpointing_enable_inherit = None
    if patch_gradient_checkpointing_enable_inherit is not None:
        patch_gradient_checkpointing_enable_inherit()
    yield
    PreTrainedModel.gradient_checkpointing_enable = saved


@pytest.mark.parametrize("which", ["wrapper", "inner"])
def test_wrapper_and_inner_enable_and_recompute(patched, which):
    torch.manual_seed(0)
    model = _Wrapper(_Cfg())
    target = model if which == "wrapper" else model.language_model
    assert type(target).supports_gradient_checkpointing is False
    target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(b.gradient_checkpointing for b in model.language_model.layers)
    assert target.is_gradient_checkpointing
    # The class default is untouched; only this instance inherits.
    assert type(target).supports_gradient_checkpointing is False
    model.train()
    ids = torch.randint(0, 16, (2, 5))
    _Block.calls = 0
    model(ids).sum().backward()
    # 3 blocks, forward + recompute in backward = 6 calls.
    assert _Block.calls == 6
    assert model.language_model.layers[0].lin.weight.grad is not None


def test_grads_match_no_checkpointing(patched):
    torch.manual_seed(0)
    a = _Wrapper(_Cfg())
    b = _Wrapper(_Cfg())
    b.load_state_dict(a.state_dict())
    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    a.train(); b.train()
    ids = torch.randint(0, 16, (2, 5))
    a(ids).pow(2).sum().backward()
    b(ids).pow(2).sum().backward()
    n_checked = 0
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), n
            n_checked += 1
    assert n_checked >= 7


def test_model_without_checkpointable_submodule_still_raises(patched):
    with pytest.raises(ValueError, match="does not support gradient checkpointing"):
        _NoLayers(_Cfg()).gradient_checkpointing_enable()


def test_kill_switch(monkeypatch):
    saved = PreTrainedModel.gradient_checkpointing_enable
    try:
        from unsloth_zoo.temporary_patches.misc import patch_gradient_checkpointing_enable_inherit
    except ImportError:
        pytest.skip("patch not present")
    monkeypatch.setenv("UNSLOTH_GC_INHERIT", "0")
    try:
        patch_gradient_checkpointing_enable_inherit()
        assert PreTrainedModel.gradient_checkpointing_enable is saved
    finally:
        PreTrainedModel.gradient_checkpointing_enable = saved
