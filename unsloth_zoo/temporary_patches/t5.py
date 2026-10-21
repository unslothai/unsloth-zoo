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

import ast
import functools
import inspect
import linecache

from .common import TEMPORARY_PATCHES
from .utils import raise_error

__all__ = [
    "patch_UMT5_scale_decoder_outputs",
    "patch_UMT5_causal_decoder_self_attention",
    "patch_config_model_type_from_architectures",
]

_PATCHED = "_unsloth_scale_decoder_outputs_patched"
_SCALE_TEST = 'getattr(self.config, "scale_decoder_outputs", self.config.tie_word_embeddings)'


def _patch_umt5_config(config_cls):
    if config_cls.__dict__.get(_PATCHED, False):
        return
    original_init = config_cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        tie = kwargs.get("tie_word_embeddings")
        scale = kwargs.get("scale_decoder_outputs")
        original_init(self, *args, **kwargs)
        # Mesh TF / t5x rescale decoder outputs only when the softmax shares the embedding.
        self.scale_decoder_outputs = (tie is not False) if scale is None else bool(scale)

    config_cls.__init__ = __init__
    setattr(config_cls, _PATCHED, True)


def _rewrite_method(modeling, cls, method_name, find):
    """Splice `find(method_node) -> (node, new_text)` into the source on the node's own line, then swap
    the method's __code__ and the module's linecache so the compiled cache reads the same text."""
    source_file = inspect.getsourcefile(modeling)
    module_source = "".join(linecache.getlines(source_file)) or inspect.getsource(modeling)

    def method_node(source):
        nodes = [
            node
            for klass in ast.parse(source).body
            if isinstance(klass, ast.ClassDef) and klass.name == cls.__name__
            for node in klass.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ]
        return nodes[0] if len(nodes) == 1 else None

    method = method_node(module_source)
    hit = method and find(method)
    if not hit or hit[0].lineno != hit[0].end_lineno:
        return False
    node, new_text = hit
    lines = module_source.splitlines(keepends = True)
    line = lines[node.lineno - 1].encode("utf-8")
    line = line[: node.col_offset] + new_text.encode("utf-8") + line[node.end_col_offset :]
    lines[node.lineno - 1] = line.decode("utf-8")
    patched_source = "".join(lines)

    method = method_node(patched_source)
    method.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body = [method], type_ignores = []), source_file, "exec"),
        modeling.__dict__,
        namespace,
    )
    original = inspect.unwrap(getattr(cls, method_name))
    patched = namespace[method_name]
    if original.__code__.co_freevars != patched.__code__.co_freevars:
        return False
    original.__code__ = patched.__code__
    linecache.cache[source_file] = (len(patched_source), None, lines, source_file)
    setattr(cls, _PATCHED, True)
    return True


def _find_scale_test(forward):
    tests = [
        node.test
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.config.tie_word_embeddings"
        and "self.model_dim" in ast.unparse(node)
    ]
    return (tests[0], _SCALE_TEST) if len(tests) == 1 else None


def _find_uncausal_self_attention(init):
    calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "UMT5Attention"
        and not any(keyword.arg == "is_causal" for keyword in node.keywords)
    ]
    if len(calls) != 1:
        return None
    call = ast.unparse(calls[0])
    return calls[0], call[:-1] + ", is_causal=config.is_decoder)"


def patch_UMT5_scale_decoder_outputs():
    """transformers 5 forces tie_word_embeddings, so untied UMT5 logits get d_model**-0.5 (T5 fix: transformers#43681)."""
    try:
        import transformers.models.umt5.modeling_umt5 as modeling
        from transformers.models.umt5.configuration_umt5 import UMT5Config
    except ImportError:
        return
    except Exception as e:
        return raise_error("UMT5ForConditionalGeneration.forward", e)
    try:
        if not UMT5Config.__dict__.get(_PATCHED, False):
            if UMT5Config(tie_word_embeddings = False).tie_word_embeddings is not True:
                return  # transformers 4.x honours the flag
        model_cls = modeling.UMT5ForConditionalGeneration
        if model_cls.__dict__.get(_PATCHED, False) or _rewrite_method(
            modeling, model_cls, "forward", _find_scale_test
        ):
            _patch_umt5_config(UMT5Config)
    except Exception as e:
        return raise_error("UMT5ForConditionalGeneration.forward", e)
TEMPORARY_PATCHES.append(patch_UMT5_scale_decoder_outputs)


def patch_UMT5_causal_decoder_self_attention():
    """transformers 5.15 - 5.17 never pass is_causal to the decoder's UMT5Attention, so under SDPA it sees future tokens (transformers#49134)."""
    try:
        import transformers.models.umt5.modeling_umt5 as modeling
    except ImportError:
        return
    except Exception as e:
        return raise_error("UMT5LayerSelfAttention.__init__", e)
    try:
        layer_cls = modeling.UMT5LayerSelfAttention
        if layer_cls.__dict__.get(_PATCHED, False):
            return
        if "is_causal" not in inspect.signature(modeling.UMT5Attention.__init__).parameters:
            return
        _rewrite_method(modeling, layer_cls, "__init__", _find_uncausal_self_attention)
    except Exception as e:
        return raise_error("UMT5LayerSelfAttention.__init__", e)
TEMPORARY_PATCHES.append(patch_UMT5_causal_decoder_self_attention)


@functools.lru_cache(maxsize = 1)
def _architecture_model_types():
    from transformers.models.auto import modeling_auto

    model_types = {}
    for name in dir(modeling_auto):
        mapping = getattr(modeling_auto, name)
        if not name.endswith("_MAPPING_NAMES") or not isinstance(mapping, dict):
            continue
        for model_type, classes in mapping.items():
            for cls in classes if isinstance(classes, (list, tuple)) else (classes,):
                model_types.setdefault(cls, set()).add(model_type)
    return model_types


def patch_config_model_type_from_architectures():
    """transformers 5 dropped AutoConfig's repo-name fallback: infer a missing model_type (google/umt5-*) from architectures."""
    try:
        import transformers
        from packaging.version import Version
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        try:
            from transformers import PreTrainedConfig
        except ImportError:
            from transformers import PretrainedConfig as PreTrainedConfig
    except Exception as e:
        return raise_error("PreTrainedConfig.get_config_dict", e)
    if Version(transformers.__version__) < Version("5.0.0"):
        return
    original = PreTrainedConfig.__dict__.get("get_config_dict")
    if not isinstance(original, classmethod) or getattr(original.__func__, _PATCHED, False):
        return
    original = original.__func__

    @functools.wraps(original)
    def get_config_dict(cls, *args, **kwargs):
        config_dict, unused_kwargs = original(cls, *args, **kwargs)
        if (
            cls is PreTrainedConfig
            and isinstance(config_dict, dict)
            and "model_type" not in config_dict
            and "auto_map" not in config_dict
            and isinstance(config_dict.get("architectures"), list)
        ):
            known = _architecture_model_types()
            model_types = set()
            for architecture in config_dict["architectures"]:
                model_types |= known.get(architecture, {None})
            if len(model_types) == 1 and next(iter(model_types)) in CONFIG_MAPPING:
                config_dict["model_type"] = next(iter(model_types))
        return config_dict, unused_kwargs

    setattr(get_config_dict, _PATCHED, True)
    PreTrainedConfig.get_config_dict = classmethod(get_config_dict)
TEMPORARY_PATCHES.append(patch_config_model_type_from_architectures)
