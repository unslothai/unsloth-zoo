# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`model_type` set on the config INSTANCE instead of on the config class.

`PretrainedConfig.to_dict` ends with `output["model_type"] = self.__class__.model_type`,
so it reports the class attribute and overwrites whatever `__init__` assigned to the
object. `get_transformers_model_type` walked `to_dict()` alone, so a trust_remote_code
config that leaves `model_type = ""` on the class and sets `self.model_type` in
`__init__` was unreadable and raised "Cannot determine model type for config file".

That is not hypothetical: `inclusionAI/Ling-2.6-flash` ships
`BailingMoeV2_5Config` with `model_type = ""` at class level and
`self.model_type = "bailing_hybrid"` in `__init__`, so Unsloth refused to load a model
transformers itself resolves. The classes below reproduce that exact shape offline.

The fallback is only consulted when the `to_dict` walk yields nothing usable, so every
config that already answers through `to_dict` resolves exactly as before, and the
returned name still goes through the same validation - the result is interpolated into
an import path and a compiled-cache filename.

CPU-only and network-free.
"""

import pytest

from transformers import PretrainedConfig

from unsloth_zoo.hf_utils import get_transformers_model_type


_UNRESOLVED_MESSAGE = "Cannot determine model type for config file"


class _SubConfig(PretrainedConfig):
    """A sub-config that names itself on the instance only, like its parent."""
    model_type = ""

    def __init__(self, model_type = "bailing_hybrid_text", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type


class _BailingLikeConfig(PretrainedConfig):
    """The `BailingMoeV2_5Config` shape: empty class attribute, real instance one."""
    model_type = ""

    def __init__(self, model_type = "bailing_hybrid", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.hidden_size = 16
        self.num_hidden_layers = 1


class _EmptyEverywhereConfig(PretrainedConfig):
    """Nothing names this architecture anywhere, so it stays unresolved."""
    model_type = ""


# --- the defect ---------------------------------------------------------------

def test_instance_only_model_type_resolves():
    config = _BailingLikeConfig()
    # The precondition that made this fail: the serialization disagrees with the object
    assert config.to_dict()["model_type"] == ""
    assert config.model_type == "bailing_hybrid"

    assert get_transformers_model_type(config) == ["bailing_hybrid"]


def test_unknown_remote_type_is_not_prefix_truncated():
    """`bailing_hybrid` is not a `transformers.models` module. The trimming loop only
    replaces a name when some prefix of it IS one, so an unknown remote type is
    returned whole and the caller sees the architecture the config actually named."""
    import transformers.models
    assert "bailing_hybrid" not in set(dir(transformers.models))

    assert get_transformers_model_type(_BailingLikeConfig()) == ["bailing_hybrid"]


def test_prefix_colliding_remote_type_is_not_rewritten():
    """The trimming loop exists for transformers' own mislabels (gemma3_text ->
    gemma3). A remote-code name that merely shares a prefix with a shipped module
    must not be rewritten to it, or the model is dispatched to the wrong
    architecture's optimized path."""
    import transformers.models
    assert "llama" in set(dir(transformers.models))

    config = _EmptyEverywhereConfig()
    config.model_type = "llama_hybrid_not_really"

    assert get_transformers_model_type(config) == ["llama_hybrid_not_really"]


def test_nested_sub_config_instance_attribute_is_reached():
    """Composite remote-code configs hide the only real name on a sub-config."""
    config = _EmptyEverywhereConfig()
    config.text_config = _SubConfig()

    assert get_transformers_model_type(config) == ["bailing_hybrid_text"]


def test_sub_config_inside_a_list_is_reached():
    config = _EmptyEverywhereConfig()
    config.sub_configs = [_SubConfig(model_type = "some_remote_type")]

    assert get_transformers_model_type(config) == ["some_remote_type"]


# --- nothing that resolved before may change ----------------------------------

def test_to_dict_answer_wins_over_instance_attribute():
    """The fallback is consulted only when `to_dict` yields nothing usable, so a
    config that already answers keeps answering the same way."""
    class _NamedOnClass(PretrainedConfig):
        model_type = "llama"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # An instance attribute that disagrees must not be able to take over
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


# --- the fallback is behind the same guards as the to_dict path ---------------

@pytest.mark.parametrize("model_type", [
    "llama'); import os; os.system('touch /tmp/pwned",
    "llama import os",
    "llama\nimport os",
    "llama)",
])
def test_injected_instance_model_type_is_rejected(model_type):
    """A remote-code config controls its instance attribute exactly as much as it
    controls its serialization, so the import-path guard must cover both."""
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
    """Sub-configs are walked off `__dict__`, which can be cyclic."""
    parent = _EmptyEverywhereConfig()
    child = _EmptyEverywhereConfig()
    parent.child = child
    child.parent = parent

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(parent)


def test_non_config_attributes_are_not_walked():
    """Only config-shaped objects are followed, so an unrelated attribute that happens
    to carry a `model_type` string cannot name the architecture."""
    class _NotAConfig:
        model_type = "definitely_not_the_model"

    class _Holder:
        """Plain object, so `str()` in the error message stays a plain repr -
        `PretrainedConfig.__repr__` JSON-dumps itself and would choke on the helper."""
        def __init__(self):
            self.some_helper = _NotAConfig()
        def to_dict(self):
            return {"model_type": ""}

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_Holder())


_MOCK_PROBE = r"""
import builtins, sys
from unittest import mock
from unsloth_zoo import hf_utils

if sys.argv[1] == "duck":
    # The duck-typed branch only runs when PretrainedConfig cannot be imported
    real_import = builtins.__import__
    def _no_pretrained_config(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "transformers" and fromlist and "PretrainedConfig" in fromlist:
            raise ImportError("simulated layout change")
        return real_import(name, globals, locals, fromlist, level)
    builtins.__import__ = _no_pretrained_config
    assert hf_utils._instance_attribute_model_types(mock.MagicMock()) == []
else:
    for config in (mock.MagicMock(), mock.Mock()):
        try:
            hf_utils.get_transformers_model_type(config)
        except TypeError as error:
            assert "Cannot determine model type" in str(error)
        else:
            raise AssertionError("a Mock config resolved to a model type")
print("OK")
"""


@pytest.mark.parametrize("mode", ["isinstance", "duck"])
def test_mock_config_raises_instead_of_hanging(mode):
    # Mock fabricates an attribute on every access and stores it on `__dict__`, so a
    # walk that reads `model_type` off each node grows the graph forever. Before the
    # instance-attribute fallback this raised TypeError; it must still raise, promptly.
    # A subprocess with a timeout, because a hang cannot be interrupted in-process
    # portably (no SIGALRM on Windows).
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _MOCK_PROBE, mode],
            capture_output = True, text = True, timeout = 240, env = env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("instance-attribute walk did not terminate on a Mock config")
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-2000:]
