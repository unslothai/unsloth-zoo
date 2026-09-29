# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# model_type set on the config instance, not the class (Ling-2.6-flash). CPU only.

import pytest

from transformers import PretrainedConfig

from unsloth_zoo.hf_utils import get_transformers_model_type


_UNRESOLVED_MESSAGE = "Cannot determine model type for config file"


class _SubConfig(PretrainedConfig):
    model_type = ""

    def __init__(self, model_type = "bailing_hybrid_text", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type


class _BailingLikeConfig(PretrainedConfig):
    model_type = ""

    def __init__(self, model_type = "bailing_hybrid", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.hidden_size = 16
        self.num_hidden_layers = 1


class _EmptyEverywhereConfig(PretrainedConfig):
    model_type = ""


def test_instance_only_model_type_resolves():
    config = _BailingLikeConfig()
    assert config.to_dict()["model_type"] == ""
    assert config.model_type == "bailing_hybrid"

    assert get_transformers_model_type(config) == ["bailing_hybrid"]


def test_unknown_remote_type_is_not_prefix_truncated():
    import transformers.models
    assert "bailing_hybrid" not in set(dir(transformers.models))

    assert get_transformers_model_type(_BailingLikeConfig()) == ["bailing_hybrid"]


def test_prefix_colliding_remote_type_is_not_rewritten():
    import transformers.models
    assert "llama" in set(dir(transformers.models))

    config = _EmptyEverywhereConfig()
    config.model_type = "llama_hybrid_not_really"

    assert get_transformers_model_type(config) == ["llama_hybrid_not_really"]


def test_nested_sub_config_instance_attribute_is_reached():
    config = _EmptyEverywhereConfig()
    config.text_config = _SubConfig()

    assert get_transformers_model_type(config) == ["bailing_hybrid_text"]


def test_sub_config_inside_a_list_is_reached():
    config = _EmptyEverywhereConfig()
    config.sub_configs = [_SubConfig(model_type = "some_remote_type")]

    assert get_transformers_model_type(config) == ["some_remote_type"]


def test_to_dict_answer_wins_over_instance_attribute():
    class _NamedOnClass(PretrainedConfig):
        model_type = "llama"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model_type = "not_llama"

    assert get_transformers_model_type(_NamedOnClass()) == ["llama"]


def test_still_unresolved_when_nothing_names_the_architecture():
    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_EmptyEverywhereConfig())


def test_plain_object_without_model_type_still_raises():
    class _NoModelType:
        def to_dict(self):
            return {"architectures": ["MyCustomModel"]}

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_NoModelType())


def test_plain_config_unchanged():
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 1,
        intermediate_size = 32, vocab_size = 32,
    )
    assert get_transformers_model_type(config) == ["llama"]


@pytest.mark.parametrize("model_type", [
    "llama'); import os; os.system('touch /tmp/pwned",
    "llama import os",
    "llama\nimport os",
    "llama)",
])
def test_injected_instance_model_type_is_rejected(model_type):
    config = _EmptyEverywhereConfig()
    config.model_type = model_type

    with pytest.raises(ValueError, match = "Invalid model_type"):
        get_transformers_model_type(config)


def test_instance_model_type_path_traversal_cannot_survive():
    import re

    config = _EmptyEverywhereConfig()
    config.model_type = "../../../etc/passwd"

    result = get_transformers_model_type(config)
    assert all(re.fullmatch(r"[a-z0-9_]+", t) for t in result), result


def test_cyclic_config_graph_terminates():
    parent = _EmptyEverywhereConfig()
    child = _EmptyEverywhereConfig()
    parent.child = child
    child.parent = parent

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(parent)


def test_non_config_attributes_are_not_walked():
    class _NotAConfig:
        model_type = "definitely_not_the_model"

    class _Holder:
        """Plain object so `str()` in the error message does not JSON-dump the helper."""
        def __init__(self):
            self.some_helper = _NotAConfig()
        def to_dict(self):
            return {"model_type": ""}

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_Holder())


def test_mock_config_raises_instead_of_hanging():
    from unittest import mock
    for config in (mock.MagicMock(), mock.Mock()):
        with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
            get_transformers_model_type(config)
