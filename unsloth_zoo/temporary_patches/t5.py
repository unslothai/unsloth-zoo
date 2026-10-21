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


def _patch_umt5_forward(modeling):
    model_cls = modeling.UMT5ForConditionalGeneration
    if model_cls.__dict__.get(_PATCHED, False):
        return True
    source_file = inspect.getsourcefile(modeling)
    module_source = "".join(linecache.getlines(source_file)) or inspect.getsource(modeling)
    tree = ast.parse(module_source)
    forwards = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == model_cls.__name__
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1:
        return False
    forward_node = forwards[0]
    tests = [
        node
        for node in ast.walk(forward_node)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.config.tie_word_embeddings"
        and "self.model_dim" in ast.unparse(node)
    ]
    if len(tests) != 1 or tests[0].test.lineno != tests[0].test.end_lineno:
        return False
    test = tests[0].test

    # Same line count, so tracebacks and the compiler's inspect.getsource stay aligned.
    lines = module_source.splitlines(keepends = True)
    line = lines[test.lineno - 1].encode("utf-8")
    line = line[: test.col_offset] + _SCALE_TEST.encode("utf-8") + line[test.end_col_offset :]
    lines[test.lineno - 1] = line.decode("utf-8")

    tests[0].test = ast.copy_location(ast.parse(_SCALE_TEST, mode = "eval").body, test)
    forward_node.decorator_list = []
    ast.fix_missing_locations(forward_node)
    namespace = {}
    exec(
        compile(ast.Module(body = [forward_node], type_ignores = []), source_file, "exec"),
        modeling.__dict__,
        namespace,
    )
    original_body = inspect.unwrap(model_cls.forward)
    patched_body = namespace["forward"]
    if original_body.__code__.co_freevars != patched_body.__code__.co_freevars:
        return False
    original_body.__code__ = patched_body.__code__
    linecache.cache[source_file] = (len("".join(lines)), None, lines, source_file)
    setattr(model_cls, _PATCHED, True)
    return True


def patch_UMT5_scale_decoder_outputs():
    """transformers 5 forces UMT5Config.tie_word_embeddings = True, so every untied UMT5
    checkpoint (all of google/umt5-*) rescales logits by d_model**-0.5 before its own lm_head.
    Record the checkpoint's tie flag as `scale_decoder_outputs`, as T5Config does upstream."""
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
        if _patch_umt5_forward(modeling):
            _patch_umt5_config(UMT5Config)
    except Exception as e:
        return raise_error("UMT5ForConditionalGeneration.forward", e)
TEMPORARY_PATCHES.append(patch_UMT5_scale_decoder_outputs)


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
    """transformers 5 dropped AutoConfig's repo-name fallback, so configs without `model_type`
    (google/umt5-*) raise "Unrecognized model". Infer it from an unambiguous `architectures`."""
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
