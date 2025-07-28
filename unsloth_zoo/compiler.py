# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "UNSLOTH_COMPILE_LOCATION",
    "get_transformers_model_type",
    "unsloth_compile_transformers",
    "create_new_function",
]

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import inspect
import re
import importlib
import importlib.util
import numpy as np
import os
import torch
import subprocess
import types
import time
import logging
import tempfile
import sys
import textwrap
from .utils import (
    Version,
    is_main_process,
    is_distributed,
    distributed_function,
)
import triton
import regex
from .peft_utils import get_lora_layer_modules
from importlib.metadata import version as importlib_version
from packaging.version import Version
import functools
from .compiler_replacements import compiler_replacements
from . import DEVICE_TYPE
from .temporary_patches.common import get_torch_compile_options

# Compiled cache location
global COMBINED_UNSLOTH_NAME
COMBINED_UNSLOTH_NAME = "unsloth_compiled_module"

global UNSLOTH_COMPILE_LOCATION
if 'UNSLOTH_COMPILE_LOCATION' not in globals():
    _loc = os.getenv("UNSLOTH_COMPILE_LOCATION", None)
    if _loc:
        UNSLOTH_COMPILE_LOCATION = _loc
    else:
        UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"

global UNSLOTH_COMPILE_USE_TEMP
UNSLOTH_COMPILE_USE_TEMP = False

# Disable some compilations if old versions are seen
OLD_TORCH_VERSION = Version(torch.__version__) < Version("2.5.0")

# device capability
major = None
minor = None
if DEVICE_TYPE == "cuda":
    major, minor = torch.cuda.get_device_capability()
    OLD_CUDA_ARCH_VERSION = (major <= 7) and (minor < 5)
elif DEVICE_TYPE == "xpu":
    OLD_CUDA_ARCH_VERSION = False
pass

OLD_TRITON_VERSION = Version(triton.__version__) < Version("3.0.0")

# Check if Unsloth Studio is allowed
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass

# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())
pass

DISABLED_KEYWORDS = [
    "select_best_resolution", # Llava NeXT errors out
    "original_aspect_ratio > current_aspect_ratio",  # Llava NeXT errors out
    "causal_mask[start:end, start:end] = 0", # Pixtral Dynamic slicing on data-dependent value is not supported
    "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", # Gemma3 create_masks_for_generate
    "create_causal_mask(**mask_kwargs)", # Gemma3 create_masks_for_generate
    "compute_mup_vector", # used in falcon h1 init and not needed to compile + inductor complains
    "segment_sum", # falcon h1
    "apply_mask_to_padding_states", # falcon h1
    "reshape_into_chunks", # falcon h1
    "pad_tensor_by_size", # falcon h1
]

_license_header = """
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import torch
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass
"""

_disabled_sdpa_code = f"""{_license_header}

from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_compiled_ce_loss_function,
    unsloth_compiled_fused_ce_loss_function,
)

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass

"""

# Patch Layernorm, Conv
_patch_functions = [
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "RMSNorm", "LayerNorm",
    # "CrossEntropyLoss",
]


def get_transformers_model_type(
    model_name,
    token = None,
    revision = None,
    trust_remote_code = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    from transformers import AutoConfig
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()

    config = AutoConfig.from_pretrained(
        model_name,
        token = token,
        revision = revision,
        trust_remote_code = trust_remote_code,
    )
    if not was_disabled: enable_progress_bars()

    model_types = []
    config = str(config.to_dict())
    model_types = re.findall(r"'model_type': '([^\s\']{1,})'", config)
    model_types = [x.replace("-", "_").lower() for x in model_types]
    # Add splitted modules for eg gemma3_text -> gemma3
    model_types += [x.split("_")[0] for x in model_types]
    model_types = list(dict().fromkeys(model_types))

    from transformers import models
    models = dir(models)
    all_model_types = set()
    for name in models:
        for model_type in model_types:
            if model_type in name.lower():
                all_model_types.add(model_type)
                break
    pass

    all_model_types = list(all_model_types)
    return all_model_types
pass


# Empty causal mask
def no_update_causal_mask(*args, **kwargs): return None

# Patch SDPA
def replace_with_grouped_query_attention(module, source):
    # All Unsloth Zoo code licensed under LGPLv3
    if "enable_gqa" not in torch.nn.functional.scaled_dot_product_attention.__doc__: return source

    grouped_query_attention_finder = \
        r"(key_states \= repeat_kv[^\n]{1,}\n[\s]{1,}"\
        r"value_states \= repeat_kv[^\n]{1,}\n[\s]{1,}"\
        r"(.+?)"\
        r"query_states \= query_states\.contiguous\(\)\n[\s]{1,}"\
        r"key_states \= key_states\.contiguous\(\)\n[\s]{1,}"\
        r"value_states \= value_states\.contiguous\(\))"

    found = re.findall(grouped_query_attention_finder, source, flags = re.DOTALL | re.MULTILINE,)
    if len(found) == 1:
        found = found[0]
        # Should be == 2, but Llama has key_states = self.k_norm(key_states)
        if found[0].count("key_states = ") >= 2 and found[0].count("value_states = ") >= 2:
            print(f"Unsloth: Transforming {module}.")
            all_source = source
            source = re.sub(
                grouped_query_attention_finder,
                r"\2pass\n",
                source,
                flags = re.DOTALL | re.MULTILINE,
            )
            source = source\
                .replace(
                    "dropout_p=self.dropout if self.training else 0.0,",
                    "dropout_p=self.dropout if self.training else 0.0, "\
                    "enable_gqa=self.num_key_value_groups != 1,",
                ).replace(
                    "dropout_p=self.attention_dropout if self.training else 0.0,",
                    "dropout_p=self.attention_dropout if self.training else 0.0, "\
                    "enable_gqa=self.num_key_value_groups != 1,",
                )
        pass
    pass

    source = re.sub(
        r"if output_attentions\:.+?return super\(\)\.forward.+?\)",
        "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
        source,
        flags = re.DOTALL | re.MULTILINE,
    )
    return source
pass

def _get_compile_folder(use_tempfile = False):
    global UNSLOTH_COMPILE_LOCATION
    global UNSLOTH_COMPILE_USE_TEMP
    if UNSLOTH_COMPILE_USE_TEMP or use_tempfile:
        UNSLOTH_COMPILE_USE_TEMP = True
        location = os.path.join(tempfile.gettempdir(), UNSLOTH_COMPILE_LOCATION)
        if not os.path.exists(location):
            print(
                f"Unsloth: We'll be using `{location}` for temporary Unsloth patches."
            )
            os.makedirs(location, exist_ok = True)
    else:
        location = UNSLOTH_COMPILE_LOCATION
        if os.path.exists(location): return location, UNSLOTH_COMPILE_USE_TEMP
        try:
            # Try creating the directory
            os.makedirs(location, exist_ok = True)
        except:
            # Instead use a temporary location!
            UNSLOTH_COMPILE_USE_TEMP = True
            location = os.path.join(tempfile.gettempdir(), location)
            os.makedirs(location, exist_ok = True)
            print(
                f"Unsloth: We'll be using `{location}` for temporary Unsloth patches."
            )
    return location, UNSLOTH_COMPILE_USE_TEMP
pass

def get_compile_folder(use_tempfile = False):
    location, UNSLOTH_COMPILE_USE_TEMP = distributed_function(2, _get_compile_folder, use_tempfile)
    return location, UNSLOTH_COMPILE_USE_TEMP
pass

# Mask creation functions
@functools.lru_cache(1)
def get_mask_functions():
    try:
        import transformers.masking_utils
        masking_utils = dir(transformers.masking_utils)
        return [x for x in masking_utils if x.startswith("create")]
    except:
        return []
pass

def create_new_function(
    name,
    new_source,
    model_location,
    functions,
    prepend = "",
    append = "",
    overwrite = True,
    add_torch_compile = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    old_new_source = new_source
    do_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

    if new_source[0] == " ":
        spaces = new_source.find("def")
        new_source = new_source.split("\n")
        new_source = "\n".join(x[spaces:] for x in new_source)
    pass

    if add_torch_compile:
        new_source = \
            "@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n"\
            f"{new_source}"
    pass

    # Import items to make the function executable
    items = [x for x in functions if ((x in new_source) and (x != name) and not (f"def {x}(" in new_source))]
    # Patch for SiglipEncoder and others
    if "SiglipEncoder" in new_source: items += ["SiglipEncoder"]
    # Check for create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
    mask_functions = get_mask_functions()
    for mask_function in mask_functions:
        if mask_function in new_source: items += [mask_function]
    pass
    # Full import script
    imports = "from torch import Tensor\n"
    imports += "import torch\n"
    imports += "import torch.nn as nn\n"
    imports += "from torch.nn import functional as F\n"
    imports += "from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable\n"
    imports += f"from {model_location} import (" + ", ".join(x for x in items) + ")" if len(items) != 0 else ""
    new_source = imports + "\n\n" + new_source
    new_source = prepend + new_source + append

    # Check versioning
    try: unsloth_zoo_version = importlib_version("unsloth_zoo")
    except: unsloth_zoo_version = "0"
    try: unsloth_version = importlib_version("unsloth")
    except: unsloth_version = "0"
    try: transformers_version = importlib_version("transformers")
    except: transformers_version = "0"
    try: trl_version = importlib_version("trl")
    except: trl_version = "0"

    versioning = '"""\n' + \
        f'{unsloth_zoo_version}\n'\
        f'{unsloth_version}\n'\
        f'{transformers_version}\n'\
        f'{trl_version}\n__UNSLOTH_VERSIONING__\n' + '"""\n'

    write_new_source = versioning + new_source

    # Write function
    global UNSLOTH_COMPILE_USE_TEMP
    file_source = None
    compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = False)
    function_location = os.path.join(compile_folder, f"{name}.py")

    # Check if file was already created!
    if not overwrite and os.path.isfile(function_location):

        # Check if exactly equivalent
        with open(function_location, "r", encoding="utf-8") as f:
            file_source = f.read()

        if file_source != write_new_source:
            overwrite = True
        elif not overwrite:
            if "__UNSLOTH_VERSIONING__" not in file_source:
                overwrite = True
            else:
                versions = file_source[:file_source.find('__UNSLOTH_VERSIONING__')]
                if versioning[:versioning.find('__UNSLOTH_VERSIONING__')] != versions:
                    overwrite = True
    pass
    if os.environ.get("UNSLOTH_COMPILE_OVERWRITE", "1") == "0":
        overwrite = False

    # Check location
    def write_file(function_location, write_new_source):
        with open(function_location, "wb", buffering = 0) as file:
            file.write(write_new_source.encode("utf-8"))
            file.flush()
            os.fsync(file.fileno())
        return None
    pass

    if overwrite or not os.path.isfile(function_location):
        try:
            distributed_function(1, write_file, function_location, write_new_source)
        except Exception as error:
            if UNSLOTH_COMPILE_USE_TEMP:
                raise RuntimeError(error)
            else:
                # Failed so instead use a temporary directory
                compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = True)
                function_location = os.path.join(compile_folder, f"{name}.py")
                distributed_function(1, write_file, function_location, write_new_source)
            pass
        pass
    pass

    # Now import modules! Use a tempfile if it fails on the first try!
    old_path = None
    new_module = None

    def import_module(compile_folder, name):
        # Add directory to sys.path temporarily if it's not already there
        if compile_folder not in sys.path:
            old_path = list(sys.path)
            # Fail if name already exists!
            if name in old_path:
                raise OSError(f"Unsloth: File {name} already exists")
            sys.path.insert(0, compile_folder)
        # Try standard import
        new_module = importlib.import_module(name)
        return new_module, old_path
    pass

    try:
        new_module, old_path = import_module(compile_folder, name)
    except Exception as e:
        new_module = None
        # Try using temp directory instead!
        if not UNSLOTH_COMPILE_USE_TEMP:
            compile_folder, UNSLOTH_COMPILE_USE_TEMP = get_compile_folder(use_tempfile = True)
            function_location = os.path.join(compile_folder, f"{name}.py")
            distributed_function(1, write_file, function_location, write_new_source)
            if is_main_process():
                print(f"Standard import failed for {name}: {e}. Using tempfile instead!")
            try:
                new_module, old_path = import_module(compile_folder, name)
            except Exception as e:
                new_module = None
                if is_main_process():
                    print(f"Standard import failed for {name}: {e}. Using spec.loader.exec_module instead!")
        pass
        # Fallback to direct module loading
        if new_module is None:
            try:
                module_name = f"unsloth_cache_{name}"
                file_location = os.path.join(compile_folder, name) + ".py"
                spec = importlib.util.spec_from_file_location(module_name, file_location)
                new_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = new_module
                spec.loader.exec_module(new_module)
            except Exception as e:
                raise RuntimeError(f"Direct module loading failed for {name}: {e}")
        pass
    finally:
        # Restore original sys.path if we modified it
        if old_path is not None:
            sys.path = old_path

    if new_module is None:
        raise ImportError(f'Unsloth: Cannot import {name} from {UNSLOTH_COMPILE_LOCATION}')

    return new_module
pass


def create_standalone_class(
    module,
    model_location,
    functions,
    fullgraph = False,
    forward_source = None,
    disable = False,
    add_loss_kwargs = False,
    new_init = None,
) -> str:
    # All Unsloth Zoo code licensed under LGPLv3
    # Create optimized standalone forward function
    f = eval(f"{model_location}.{module}")
    full_class = inspect.getsource(f)
    old_source = inspect.getsource(f.forward)
    old_init   = inspect.getsource(f.__init__)
    if forward_source is None: forward_source = old_source

    # We disable this for nn.Embedding modules if torch is older than 2.5 since
    if OLD_TORCH_VERSION and "nn.Embedding(" in old_init:
        disable = True

    source = re.sub(
        "def forward",
        f"def {module}_forward",
        forward_source,
    )
    spaces = re.search(r"[^\s\n]", source).span(0)[0]
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)

    # For cuda_kernels_forward, we disable
    if "cuda_kernels_forward" in source:
        disable = True

    if disable is not None:
        compile = \
            f"@torch.compile(fullgraph = {fullgraph}, dynamic = True, options = torch_compile_options)" \
            if not disable else \
            "@torch.compiler.disable(recursive = False)"
    else:
        compile = ""

    # Create new forward calling optimized function
    parameters = inspect.signature(f.forward).parameters
    # .parameters removes **kwargs and *args so we get it back!
    keys = list(parameters.keys())
    values = list(parameters.values())
    for j, value in enumerate(values):
        value = str(value)
        if   value.startswith("**"): keys[j] = "**" + keys[j]
        elif value.startswith("*"):  keys[j] = "*"  + keys[j]
    pass
    parameters = ", ".join(keys)

    # Now create the forward function!
    definition = re.findall(r"[\s\n]{1,}def[^\(]{1,}\([^\)]{1,}\)[^\:]{0,}\:", old_source, flags = re.MULTILINE)[0]
    leftover = full_class[full_class.find(definition) + len(definition):]

    # Add **loss_kwargs
    if add_loss_kwargs and "**" not in parameters:
        parameters += ", **loss_kwargs"
        definition = re.sub(r"(\,[\n][\s]{1,}\))", r",**loss_kwargs\1", definition)
        source = re.sub(r"(\,[\n]\) \-\>)", r",**loss_kwargs\1", source)
    pass

    source = f"{compile}\n{source}\n"

    left = re.match(r"[\s\n]{4,}", leftover).span()[1]
    new_forward = definition + leftover[:left] + \
        f"return {module}_forward({parameters})\n"
    full_class = full_class.replace(old_source, new_forward)

    # New init as well
    if new_init is not None:
        full_class = full_class.replace(old_init, new_init)

    # Combine all into file
    source = source + full_class

    # Remove @auto_docstring
    source = re.sub(r"@auto_docstring[\s]{0,}(\([^\)]{1,}\))?", "", source)
    # source = source.replace("@auto_docstring", "")

    # Fix Gemma 3 ignore_index being not set!
    source = source.replace("self.config.ignore_index", "-100")

    # Force embeddings with offsets to clamp_(0, max_size)
    # This fixes some weird OOBs accesses for Gemma 3N for example
    source = re.sub(
        r"self\.([A-Za-z\_]{0,}embedding)\(input_ids (\-|\+) (self\.[A-Za-z\_]{1,})\)",
        r"self.\1((input_ids \2 \3).clamp_(0))",
        source,
    )
    return source
pass


_cross_entropy_code = """
from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \\
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\\n'\\
    "```\\nimport os\\n"\\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\\n"\\
    "trainer.train()\\n```\\n"\\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass

"""

__DYNAMO__RECOMPILING__ = """

    # Set compiler stance to fail on recompiles for inference
    global INFERENCE_RUNS
    if torch_dynamo_eval_frame is not None:
        old_stance = torch_dynamo_eval_frame._stance.stance
    else:
        old_stance = None
    if old_stance is not None and INFERENCE_RUNS == 1:
        # Skip guards and return to eager -> we still need guards!
        torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
        if UNSLOTH_ENABLE_LOGGING:
            logger_compiler.info(
                f"Unsloth: Removing compiler guards after 1 inference run. "\\
                f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\\
                f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
            )
    elif old_stance == "eager_on_recompile":
        pass
    elif old_stance == "default" and INFERENCE_RUNS > 1:
        # Reset compiler stance
        torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
        if UNSLOTH_ENABLE_LOGGING:
            logger_compiler.info(
                f"Unsloth: Reseting guards. "\\
                f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\\
                f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
            )
        INFERENCE_RUNS = 0
    INFERENCE_RUNS += 1
"""

# Replace Cross Entropy cells with fused linear lm heads
cross_entropy_find_1 = """
logits = self.lm_head(hidden_states$INDEXING$
$LOGITSCALINGMULTIPLY$
$LOGITSCALINGDIVISION$
$LOGITSOFTCAPPING$
loss = None
if labels is not None:$SPACES$
$UPCASTING$
$LOGITSUPCAST$
$LABELSDEVICE$
shift_logits = logits[..., :-1, :]$CONTIGUOUS$
shift_labels = labels[..., 1:]$CONTIGUOUS$
loss_fct = $CROSSENTROPYLOSS$
shift_logits = shift_logits.view(-1, $VOCABSIZE$)
shift_labels = shift_labels.view(-1)
shift_labels = shift_labels.to(shift_logits.device)
loss = loss_fct(shift_logits, shift_labels)
"""

cross_entropy_replacement_1 = """
NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

n_items = None
all_locals = locals()
if 'loss_kwargs' in all_locals:
    __kwargs = all_locals['loss_kwargs']
    if type(__kwargs) is dict:
        n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
if n_items is None and 'kwargs' in all_locals:
    __kwargs = all_locals['kwargs']
    if type(__kwargs) is dict:
        n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
if n_items is None:
    all_locals = all_locals.values()
    for __kwargs in all_locals:
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
            break
pass

requires_grad_ = self.lm_head.weight.requires_grad
requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32

if RETURN_HIDDEN_STATES:
    logits = hidden_states\\1
elif labels is None:
    __DYNAMO__RECOMPILING__
    logits = self.lm_head(hidden_states\\1)
elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None and not requires_grad_):
    loss = fast_linear_cross_entropy(
        hidden_states        = hidden_states\\1,
        lm_head              = self.lm_head,
        labels               = labels,
        num_items_in_batch   = n_items,
        logit_softcapping    = None if (\\4) == () else (\\4),
        logit_scale_multiply = None if (\\2) == () else (\\2),
        logit_scale_divide   = None if (\\3) == () else (\\3),
    )
elif ((\\2) == () and (\\3) == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
    loss = fused_linear_cross_entropy(
        hidden_states      = hidden_states\\1,
        lm_weight          = self.lm_head.weight,
        labels             = labels.to(self.lm_head.weight.device),
        num_items_in_batch = n_items,
        logit_softcapping  = None if (\\4) == () else (\\4),
    )
else:
    lm_head_weight = self.lm_head.weight
    lm_head_bias   = getattr(self.lm_head, "bias", None)

    # ========= NEW fused =========
    _hidden_states = hidden_states\\1
    torch._dynamo.mark_dynamic(_hidden_states, 1)
    torch._dynamo.mark_dynamic(labels, 1)
    loss = unsloth_compiled_fused_ce_loss_function(
        hidden_states        = _hidden_states,
        lm_head_weight       = lm_head_weight,
        lm_head_bias         = lm_head_bias,
        output_labels        = labels,
        logit_scale_multiply = (\\2) if (\\2) != () else 0,
        logit_scale_divide   = (\\3) if (\\3) != () else 0,
        logit_softcapping    = (\\4) if (\\4) != () else 0,
        vocab_size           = (\\6),
        n_items              = n_items if n_items is not None else 0,
        requires_grad_       = requires_grad_,
    )

    # ========= OLD non fused =========
    # logits = self.lm_head(hidden_states\\1.to(lm_head_weight.device))
    # torch._dynamo.mark_dynamic(logits, 1)
    # torch._dynamo.mark_dynamic(labels, 1)
    # loss = unsloth_compiled_ce_loss_function(
    #     output_logits        = logits,
    #     output_labels        = labels,
    #     logit_scale_multiply = (\\2) if (\\2) != () else 0,
    #     logit_scale_divide   = (\\3) if (\\3) != () else 0,
    #     logit_softcapping    = (\\4) if (\\4) not in (None, (),) else 0,
    #     vocab_size           = (\\6),
    #     n_items              = n_items if n_items is not None else 0,
    #     requires_grad_       = requires_grad_,
    # )


    # if (\\2) != ():
    #     logits = logits * (\\2)
    # if (\\3) != ():
    #     logits = logits / (\\3)
    # if (\\4) != ():
    #     logits = logits / (\\4)
    #     logits = torch.tanh(logits)
    #     logits = logits * (\\4)
    # shift_logits = logits[..., :-1, :].float().contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # reduction = 'mean' if n_items is None else 'sum'
    # loss_fct = torch.nn.CrossEntropyLoss(reduction = reduction)
    # shift_logits = shift_logits.view(-1, \\6)
    # shift_labels = shift_labels.view(-1)
    # shift_labels = shift_labels.to(shift_logits.device)
    # loss = loss_fct(shift_logits, shift_labels)
    # if n_items is not None: loss = loss / n_items
""".replace("__DYNAMO__RECOMPILING__", __DYNAMO__RECOMPILING__)

cross_entropy_find_2 = """
logits = self.lm_head(hidden_states$INDEXING$
$LOGITSCALINGMULTIPLY$
$LOGITSCALINGDIVISION$
$LOGITSOFTCAPPING$
loss = None
if labels is not None:$SPACES$loss = self.loss_function($NEWLINES$$LOGITS$, $LABELS$, $VOCABSIZE$$KWARGS$$NEWLINES$)
"""

cross_entropy_replacement_2 = """
NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

n_items = None
if (\\9) != () and type(\\9) is dict:
    n_items = (\\9).get("num_items_in_batch", None) or (\\9).get("n_items", None)
if n_items is None:
    all_locals = locals()
    if 'loss_kwargs' in all_locals:
        __kwargs = all_locals['loss_kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
    if n_items is None and 'kwargs' in all_locals:
        __kwargs = all_locals['kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
    if n_items is None:
        all_locals = all_locals.values()
        for __kwargs in all_locals:
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
                break
pass

requires_grad_ = self.lm_head.weight.requires_grad
requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32

if RETURN_HIDDEN_STATES:
    logits = hidden_states\\1
elif labels is None:
    __DYNAMO__RECOMPILING__
    logits = self.lm_head(hidden_states\\1)
elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
    loss = fast_linear_cross_entropy(
        hidden_states        = hidden_states\\1,
        lm_head              = self.lm_head,
        labels               = labels,
        num_items_in_batch   = n_items,
        logit_softcapping    = None if (\\4) == () else (\\4),
        logit_scale_multiply = None if (\\2) == () else (\\2),
        logit_scale_divide   = None if (\\3) == () else (\\3),
    )
elif ((\\2) == () and (\\3) == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
    loss = fused_linear_cross_entropy(
        hidden_states      = hidden_states\\1,
        lm_weight          = self.lm_head.weight,
        labels             = labels.to(self.lm_head.weight.device),
        num_items_in_batch = n_items,
        logit_softcapping  = None if (\\4) == () else (\\4),
    )
elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
    lm_head_weight = self.lm_head.weight
    lm_head_bias   = getattr(self.lm_head, "bias", None)

    # ========= NEW fused =========
    _hidden_states = hidden_states\\1
    torch._dynamo.mark_dynamic(_hidden_states, 1)
    torch._dynamo.mark_dynamic(labels, 1)
    loss = unsloth_compiled_fused_ce_loss_function(
        hidden_states        = _hidden_states,
        lm_head_weight       = lm_head_weight,
        lm_head_bias         = lm_head_bias,
        output_labels        = labels,
        logit_scale_multiply = (\\2) if (\\2) != () else 0,
        logit_scale_divide   = (\\3) if (\\3) != () else 0,
        logit_softcapping    = (\\4) if (\\4) not in (None, (),) else 0,
        vocab_size           = (\\8),
        n_items              = n_items if n_items is not None else 0,
        requires_grad_       = requires_grad_,
    )


    # ========= OLD non fused =========
    # logits = self.lm_head(hidden_states\\1.to(lm_head_weight.device))
    # torch._dynamo.mark_dynamic(logits, 1)
    # torch._dynamo.mark_dynamic(labels, 1)
    # loss = unsloth_compiled_ce_loss_function(
    #     output_logits        = logits,
    #     output_labels        = labels,
    #     logit_scale_multiply = (\\2) if (\\2) != () else 0,
    #     logit_scale_divide   = (\\3) if (\\3) != () else 0,
    #     logit_softcapping    = (\\4) if (\\4) not in (None, (),) else 0,
    #     vocab_size           = (\\8),
    #     n_items              = n_items if n_items is not None else 0,
    #     requires_grad_       = requires_grad_,
    # )
else:
    logits = self.lm_head(hidden_states\\1)
    if (\\2) != ():
        logits = logits * (\\2)
    if (\\3) != ():
        logits = logits / (\\3)
    if (\\4) not in (None, (),):
        logits = logits / (\\4)
        logits = torch.tanh(logits)
        logits = logits * (\\4)
    loss = self.loss_function(\\6, \\7.to(self.lm_head.weight.device), vocab_size=\\8, **\\9)
""".replace("__DYNAMO__RECOMPILING__", __DYNAMO__RECOMPILING__)

cross_entropy_find_3 = """
$OUTPUTLOGITS$
$LOGITSCALINGMULTIPLY$
$LOGITSCALINGDIVISION$
$LOGITSOFTCAPPING$
loss = None
if labels is not None:$SPACES$
$UPCASTING$
$LOGITSUPCAST$
$LABELSDEVICE$
$LOGITSHIFTING$
$VLMATTENTIONMASK$
loss_fct = $CROSSENTROPYLOSS$
shift_logits = shift_logits.view(-1, $VOCABSIZE$)
shift_labels = shift_labels.view(-1)###
$LOGITSDEVICE$###
loss = loss_fct(shift_logits, shift_labels)
"""

cross_entropy_replacement_3 = """
NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

all_locals = locals()
n_items = None
if 'loss_kwargs' in all_locals:
    __kwargs = all_locals['loss_kwargs']
    if type(__kwargs) is dict:
        n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
if n_items is None and 'kwargs' in all_locals:
    __kwargs = all_locals['kwargs']
    if type(__kwargs) is dict:
        n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
if n_items is None:
    all_locals = all_locals.values()
    for __kwargs in all_locals:
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
            break
pass

requires_grad_ = self.lm_head.weight.requires_grad
requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32

if RETURN_HIDDEN_STATES:
    logits = hidden_states\\1
elif labels is None:
    __DYNAMO__RECOMPILING__
    logits = self.lm_head(hidden_states\\1)
else:
    lm_head_weight = self.lm_head.weight
    lm_head_bias   = getattr(self.lm_head, "bias", None)

    # ========= NEW fused =========
    _hidden_states = hidden_states\\1
    torch._dynamo.mark_dynamic(_hidden_states, 1)
    torch._dynamo.mark_dynamic(labels, 1)
    if attention_mask is not None:
        torch._dynamo.mark_dynamic(attention_mask, 1)
    loss = unsloth_compiled_fused_ce_loss_function(
        hidden_states        = _hidden_states,
        lm_head_weight       = lm_head_weight,
        lm_head_bias         = lm_head_bias,
        output_labels        = labels,
        logit_scale_multiply = (\\2) if (\\2) != () else 0,
        logit_scale_divide   = (\\3) if (\\3) != () else 0,
        logit_softcapping    = (\\4) if (\\4) not in (None, (),) else 0,
        vocab_size           = (\\7),
        n_items              = n_items if n_items is not None else 0,
        mask                 = \\6,
        requires_grad_       = requires_grad_,
    )

    # ========= OLD non fused =========
    # logits = self.lm_head(hidden_states\\1.to(lm_head_weight.device))
    # torch._dynamo.mark_dynamic(logits, 1)
    # torch._dynamo.mark_dynamic(labels, 1)
    # if attention_mask is not None:
    #     torch._dynamo.mark_dynamic(attention_mask, 1)
    # loss = unsloth_compiled_ce_loss_function(
    #     output_logits        = logits,
    #     output_labels        = labels,
    #     logit_scale_multiply = (\\2) if (\\2) != () else 0,
    #     logit_scale_divide   = (\\3) if (\\3) != () else 0,
    #     logit_softcapping    = (\\4) if (\\4) not in (None, (),) else 0,
    #     vocab_size           = (\\7),
    #     n_items              = n_items if n_items is not None else 0,
    #     mask                 = \\6,
    #     requires_grad_       = requires_grad_,
    # )
""".replace("__DYNAMO__RECOMPILING__", __DYNAMO__RECOMPILING__)

ce_finders = [
    (cross_entropy_find_1, cross_entropy_replacement_1,),
    (cross_entropy_find_2, cross_entropy_replacement_2,),
    (cross_entropy_find_3, cross_entropy_replacement_3,),
]


def apply_fused_lm_head(forward, module = None):
    # All Unsloth Zoo code licensed under LGPLv3
    UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
    for jj, (cross_entropy_find, cross_entropy_replacement) in enumerate(ce_finders):
        cross_entropy_find = cross_entropy_find.strip()\
            .replace("*", r"\*").replace("^", r"\^")\
            .replace("-", r"\-").replace("_", r"\_")\
            .replace(":", r"\:").replace("+", r"\+")\
            .replace(".", r"\.").replace(",", r"\,")\
            .replace("(", r"\(").replace(")", r"\)")\
            .replace("[", r"\[").replace("]", r"\]")\
            .replace(
                "\n",
                r"(?:[\s\n]{0,}(?:\#[^\n]{1,}[\n][\s\n]{1,})?){0,}"
            )

        # Replace $ with anything and % with num_logits_to_keep or .float()
        cross_entropy_find = cross_entropy_find\
            .replace("$INDEXING$",     r"([^\n^\)]{0,})\)(?:\.float\(\))?[\n][\s]{0,}")\
            .replace("$UPCASTING$",    r"(?:\.float\(\))?")\
            .replace("$SPACES$",       r"[\n]([\s]{1,})(?:\#[^\n]{1,}[\n][\s\n]{1,})?")\
            .replace("$LOGITS$",       r"(logits=logits|logits)")\
            .replace("$LABELS$",       r"(labels=labels|labels)")\
            .replace("$VOCABSIZE$",
                     r"(?:vocab_size\=)?"\
                     r"("\
                     r"self\.config\.vocab_size|"\
                     r"self\.vocab_size|"\
                     r"self\.config\.vocab_size|"\
                     r"self\.config\.text_config\.vocab_size"\
                     ")")\
            .replace("$KWARGS$",       r"(?:, \*\*(loss_kwargs|kwargs))?")\
            .replace("$LOGITSUPCAST$", r"(?:logits = logits\.float\(\))?")\
            .replace("$LABELSDEVICE$", r"(?:labels = labels\.to\([^\)]{1,}\))?")\
            .replace("$LOGITSCALINGMULTIPLY$",
                     r"(?:[\n\s]{0,}logits = logits \* (self\.[^ \n]{1,})[^\n]{0,})?###")\
            .replace("$LOGITSCALINGDIVISION$",
                     r"(?:[\n\s]{0,}logits = logits \/ (self\.[^ \n]{1,})[^\n]{0,})?###")\
            .replace("$LOGITSOFTCAPPING$",
                     r"(?:[\n\s]{0,}(?:if self\.[^\n\s]{1,} is not None:\n)?"\
                     r"[\s\n]{0,}logits = logits \/ (self\.[^ \n]{1,})\n"\
                     r"[\s\n]{0,}logits = torch\.tanh\(logits\)\n"\
                     r"[\s\n]{0,}logits = logits \* self\.[^ \n]{1,}\n)?")\
            .replace("$CROSSENTROPYLOSS$",
                     r"(?:CrossEntropyLoss\(\)|"\
                     r"nn\.CrossEntropyLoss\(\)|"\
                     r"torch\.nn\.CrossEntropyLoss\(\)"\
                     r")")\
            .replace(r"$VLMATTENTIONMASK$",
                     r"(?:"\
                     r"(?:"\
                     r"shift_logits = logits\[\.\.\.\, :-1, :\]$CONTIGUOUS$"\
                     r"shift_labels = labels\[\.\.\.\, 1:\]$CONTIGUOUS$"\
                     r")?"
                     r"if ([a-zA-Z\_]{1,}_mask) is not None:###"\
                     r"shift_attention_mask = @@@###"\
                     r"shift_logits = @@@###"\
                     r"shift_labels = @@@###"\
                     r"else:###"\
                     r"shift_logits = [^\n]{1,}###"\
                     r"shift_labels = [^\n]{1,}###"\
                     r")?")\
            .replace(r"$LOGITSHIFTING$",
                     r"(?:"\
                     r"shift_logits = logits\[\.\.\.\, :-1, :\]$CONTIGUOUS$###"\
                     r"shift_labels = labels\[\.\.\.\, 1:\]$CONTIGUOUS$###"\
                     r")?")\
            .replace(r"$LOGITSDEVICE$",
                     r"(?:"\
                     r"\.to\([^\)]{1,}\)|shift_labels = shift_labels\.to\([^\)]{1,}\)"
                     r")")\
            .replace(r"$OUTPUTLOGITS$",
                     r"(?:"\
                     r"logits = outputs\.logits|"\
                     r"logits = self\.lm_head\(hidden_states\)|"\
                     r"logits = self\.lm_head\(hidden_states$INDEXING$"
                     r")")\
            .replace("$INDEXING$",
                     r"([^\n^\)]{0,})\)(?:\.float\(\))?[\n][\s]{0,}")\
            .replace(r"shift_", r"(?:shift_|flat_)")\
            .replace("$CONTIGUOUS$",   r"(?:\.contiguous\(\))?")\
            .replace(r"shift\_", r"(?:shift\_|flat\_)")\
            .replace(r"###", r"(?:[\s\n]{0,}(?:\#[^\n]{1,}[\n][\s\n]{1,})?){0,}")\
            .replace(r"@@@", r"[^\[]{1,}\[[^\]]{1,}\][^\n]{0,}\n")\
            .replace(r"$EMPTY$", r"()")\
            .replace(r"$NEWLINES$", r"[\s\n]{0,}")

        # print(cross_entropy_find)
        cross_entropy_replacement = cross_entropy_replacement\
            .replace(
                "$KWARGS$",
                "locals().get('loss_kwargs', {}) or locals().get('kwargs', {})"
            )

        # Fix Idefics and Idefics3
        forward = forward.replace(
            "loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))",

            "shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)\n"\
            "shift_labels = shift_labels.view(-1)\n"\
            "shift_labels = shift_labels.to(shift_logits.device)\n"\
            "loss = loss_fct(shift_logits, shift_labels)"
        )

        # Find matches
        if r"loss\_function" in cross_entropy_find and "loss_function" not in forward:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"(1) Unsloth skipping patching fast linear cross entropy for {module}")
            continue
        elif r"loss\_function" not in cross_entropy_find and "loss_function" in forward:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"(2) Unsloth skipping patching fast linear cross entropy for {module}")
            continue
        elif "CrossEntropyLoss" not in cross_entropy_find and "CrossEntropyLoss" in forward:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"(3) Unsloth skipping patching fast linear cross entropy for {module}")
            continue
        elif "CrossEntropyLoss" in cross_entropy_find and "CrossEntropyLoss" not in forward:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"(4) Unsloth skipping patching fast linear cross entropy for {module}")
            continue
        try:
            finder = regex.findall(
                cross_entropy_find,
                forward,
                flags = regex.DOTALL | regex.MULTILINE,
                timeout = 1
            )
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"Unsloth failed patching fast linear cross entropy with error: {str(e)}")
            continue
        if len(finder) == 0: continue
        if UNSLOTH_ENABLE_LOGGING:
            print(f"[{jj+1}/3 pattern] Successfully patched fast linear cross entropy for {module}")
        pass
        # print(forward)

        spaces = finder[0][4]
        if spaces.count(" ") != len(spaces):
            spaces = finder[0][3]
        replacement = cross_entropy_replacement.strip().split("\n")
        replacement = "\n".join((len(spaces)-4)*" " + x for x in replacement)
        replacement = \
            "logits = EMPTY_LOGITS\n" + \
            (len(spaces)-4)*" " + "loss = None\n" + \
            replacement + "\n"
        try:
            forward = regex.sub(
                cross_entropy_find,
                replacement,
                forward,
                flags = regex.DOTALL | regex.MULTILINE,
            )
        except:
            continue
        # Return logits back
        if "logits = outputs.logits" in cross_entropy_find:
            forward = forward.replace(
                "logits = EMPTY_LOGITS",
                "logits = outputs.logits",
            )
        # Fix vocab_size = (vocab_size=
        forward = regex.sub(
            r"vocab_size[ ]{0,}=[ ]{0,}\(vocab_size[ ]{0,}=",
            "vocab_size = (",
            forward,
        )
        # Fix , **
        forward = forward.replace(", **)", ")")
        forward = forward.replace(",**)", ")")
        forward = forward.replace(",** )", ")")
        # print(forward)
        return forward
    pass
    return forward
pass


def test_apply_fused_lm_head():
    import inspect
    forwards = []
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    forwards.append(Qwen2VLForConditionalGeneration)
    from transformers.models.granite.modeling_granite import GraniteForCausalLM
    forwards.append(GraniteForCausalLM)
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    forwards.append(Gemma2ForCausalLM)
    from transformers.models.cohere.modeling_cohere import CohereForCausalLM
    forwards.append(CohereForCausalLM)
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    forwards.append(GemmaForCausalLM)
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    forwards.append(LlamaForCausalLM)
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    forwards.append(MistralForCausalLM)
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
    forwards.append(PaliGemmaForConditionalGeneration)
    from transformers.models.idefics.modeling_idefics import IdeficsForVisionText2Text
    forwards.append(IdeficsForVisionText2Text)
    from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration
    forwards.append(Idefics3ForConditionalGeneration)
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
    forwards.append(Mistral3ForConditionalGeneration)
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
    forwards.append(MllamaForConditionalGeneration)
    from transformers.models.mllama.modeling_mllama import MllamaForCausalLM
    forwards.append(MllamaForCausalLM)
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
    forwards.append(Llama4ForCausalLM)
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    forwards.append(Llama4ForConditionalGeneration)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    forwards.append(Qwen3ForCausalLM)
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    forwards.append(Qwen2_5_VLForConditionalGeneration)
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
    forwards.append(Gemma3ForConditionalGeneration)
    forwards = [(f.__name__, inspect.getsource(f.forward),) for f in forwards]
    for name, forward in forwards:
        # print("=" * 30)
        # print(name)
        forward = apply_fused_lm_head(forward, name)
        if "NOT_RETURN_LOGITS" not in forward:
            print(f"Failed patching fast CE forward for {name}")
        if "loss = outputs.loss" in forward:
            print(f"Failed patching fast CE forward for {name} since `loss = outputs.loss` exists")
        # return apply_fused_lm_head(forward, name)
        # print(apply_fused_lm_head(forward, name))
        # print("=" * 30)
    pass
pass

# Fix attention_mask not masking out labels for VLMs
def apply_mask_attention_mask_out(source):
    if not len(re.findall(r"attention_mask[\s]{0,}\=attention_mask[\s]{0,}\,\n", source)): return source
    if not len(re.findall(r"labels[\s]{0,}\=labels[\s]{0,}\,\n", source)): return source
    if "ForConditionalGeneration" in source:
        source = re.sub(
            r"labels[\s]{0,}\=labels[\s]{0,}\,\n",
            "labels=mask_attention_mask_out(labels = labels, attention_mask = attention_mask),\n",
            source,
        )
    return source
pass


# Patch remaining functions
def convert_attention_masks_to_bool(module, old_source):
    # All Unsloth Zoo code licensed under LGPLv3
    # Convert attention mask creation functions to boolean
    source = re.sub(r"\([\s]{0,}", "(", old_source)
    source = re.sub(r"[\s]{0,}\)", ")", source)
    all_splits = source.strip().split("\n")
    splits = all_splits[-1].strip()
    if "return" not in splits: return old_source
    vars = re.findall(r"return[\s]{1,}(?:([^\,]{1,})\,[\s]{0,}){0,}([^\s]{1,})", splits)
    if len(vars) != 1: return old_source
    vars = vars[0]

    good_vars = []
    for var in vars:
        for split in all_splits:
            if re.search(re.escape(var) + ".+?" + r"torch\.finfo\(.+?\)\.min", split):
                good_vars.append(var)
    pass
    if len(good_vars) == 0: return old_source
    good_vars = set(good_vars)
    final = all_splits[-1]
    for var in good_vars:
        if len(var) == 0: continue
        final = final.replace(var, var + f"!=torch.finfo({var}.dtype).min")
    pass
    all_splits[-1] = final
    new_source = "\n".join(all_splits)
    print(f"Unsloth: Boolean mask for {module}")
    return new_source
pass


# We need to manually replace some items
# For example HF 4.53.1 breaks Qwen2VL since None wasn't provided
custom_gradient_checkpointing_replacements = [
    (
        """hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )""",
        """hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )""",
    ),
    (
        """hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )""",
        """hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )""",
    )
]
replace_gradient_checkpointing = """
for LAYER in MODULELIST_ITEM:
$if self.gradient_checkpointing and self.training:
$    hidden_states = self._gradient_checkpointing_func(
$        LAYER.__call__, ARGS
$    )
$else:
$    hidden_states = LAYER(ARGS)
"""
def patch_gradient_checkpointing(module, source):
    # All Unsloth Zoo code licensed under LGPLv3
    try: init = inspect.getsource(source.__init__)
    except: return None
    if "nn.ModuleList" not in init: return None
    try: forward = inspect.getsource(source.forward)
    except: return None
    if "_gradient_checkpointing_func" in forward: return None

    # Fix Qwen2 missing None for gradient checkpointing
    for custom_find, custom_replace in custom_gradient_checkpointing_replacements:
        forward = forward.replace(custom_find, custom_replace)
    pass

    # No gradient checkpointing?
    modulelist_items = re.findall(r"(self\.[^\s]{1,}) = .*?nn\.ModuleList\(", init)
    if len(modulelist_items) != 1: return None
    modulelist_item = modulelist_items[0]

    # Check in forward source
    finder = \
        r"for ([^\s]{1,}) in " + modulelist_item + r"\:[\n]" + \
        r"([\s]{4,})hidden_states = \1\(([^\)]{1,})\)"
    find = re.findall(finder, forward)
    if len(find) == 0:
        print(f"Unsloth: Failed patching {module} with gradient checkpointing")
        return None
    pass

    layer, spaces, args = find[0]
    span = re.search(finder, forward).span(0)
    replacer = replace_gradient_checkpointing.strip()

    # Gradient checkpointing calling must remove arg=arg convention
    args = re.sub(r"([^\s]{1,})[\s]?\=[\s]?\1", r"\1", args)

    replacer = replacer\
        .replace("LAYER", layer).replace("MODULELIST_ITEM", modulelist_item)\
        .replace("ARGS", args).replace("$", spaces)
    forward = forward.replace(forward[span[0] : span[1]], replacer)

    # Also fix init
    spaces = init.find("def")
    init = init + "\n" + (spaces + 4) * " " + "self.gradient_checkpointing = False\n\n"
    return init, forward
pass

DTYPE_MISMATCH_FIND = """
attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
attention_mask_tensor = (1.0 - attention_mask_tensor).int()
"""

DTYPE_MISMATCH_REPLACE = """
if attention_mask_tensor.dtype == torch.bool:
    attention_mask_tensor = attention_mask_tensor.int()
elif torch.is_floating_point(attention_mask_tensor):
    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
"""

def patch_finfo_attention_mask_dtype_mismatch(module, source):
    try:
        old_block = textwrap.dedent(DTYPE_MISMATCH_FIND).strip()
        new_block = textwrap.dedent(DTYPE_MISMATCH_REPLACE).strip()
        if not old_block or not new_block:
            return source

        first_line = old_block.split('\n')[0]
        if not first_line:
            return source

        for line in source.split('\n'):
            if first_line in line:
                indent = line[:len(line) - len(line.lstrip())]
                break
        else:
            return source

        indented_old = textwrap.indent(old_block, indent)
        indented_new = textwrap.indent(new_block, indent)

        return source.replace(indented_old, indented_new)
    except:
        return source
pass

# Torch.compiling makes things slower - rather just leave it as addmm
COMPILED_LORA_FORWARD = """
torch_addmm = torch.addmm
torch_add   = torch.add
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x) @ lora_A.weight.t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
        output,
        bias,
        alpha = scaling,
    )
    return output
pass

"""

COMPILED_LORA_FORWARD_forced_float32 = """
torch_addmm = torch.addmm
torch_add   = torch.add
torch_float16 = torch.float16
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x.to(torch_float16)) @ lora_A.weight.to(torch_float16).t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]).to(torch_float16),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.to(torch_float16).t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
        output,
        bias.to(torch_float16),
        alpha = scaling,
    )
    return output
pass

"""

def patch_lora_forwards(torch_compile_options):
    # All Unsloth Zoo code licensed under LGPLv3
    Linear_LoRA_Layers = get_lora_layer_modules()
    success = 0
    could_not_replace_modules = []
    for function, parent, child in Linear_LoRA_Layers:
        if not hasattr(function, "forward"): continue
        if function.forward.__name__ == "unsloth_forward": continue

        exec(f"import {parent}", locals(), globals())
        source = inspect.getsource(function.forward)

        spaces = source.find("def")
        source = source.split("\n")
        source = "\n".join(x[spaces:] for x in source)
        old_hash = hash(source)

        # Remove cloning
        source = source.replace("result = result.clone()", "")

        # Use addmm
        old1 = "output = lora_B(lora_A(dropout(x))) * scaling"
        old2 = "result = result + lora_B(lora_A(dropout(x))) * scaling"
        add = "result = result + output"

        if (old1 not in source and add not in source) and \
            (old2 not in source):
            pass
        else:
            replace = "return lora_forward(result, lora_A, lora_B, dropout, x, scaling)"
            source = source.replace(old1, replace)
            source = source.replace(old2, replace)
        pass

        # Update function name
        source = source.replace(
            "def forward",
            "def unsloth_forward",
            1,
        )

        # Check failed upcasting
        replacements = [
            "x = x.to(lora_A.weight.dtype)",
            "x = self._cast_input_dtype(x, lora_A.weight.dtype)",
        ]
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0":
            if "torch.is_autocast_enabled()" not in source:
                new = "if not torch.is_autocast_enabled(): "\
                    "result, x = "\
                        "result.to(lora_A.weight.dtype), "\
                        "x.to(lora_A.weight.dtype)"
                for replace in replacements:
                    source = source.replace(replace, new)
        else:
            for replace in replacements:
                source = source.replace(replace, "")
        pass
        source = source.replace(
            "self._check_forward_args(x, *args, **kwargs)",
            "",
        )

        if hash(source) != old_hash:
            success += 1
            compiled_lora_forward = \
                COMPILED_LORA_FORWARD \
                if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0" \
                else COMPILED_LORA_FORWARD_forced_float32

            forward = create_new_function(
                f"{child}_peft_forward",
                compiled_lora_forward + source,
                parent,
                dir(eval(parent)),
                prepend = \
                    f"\ntorch_compile_options = {torch_compile_options}\n"
            ).unsloth_forward
            exec(f"{parent}.{child}.forward = forward", globals(), locals())
        else:
            could_not_replace_modules.append(parent)
    pass
    if success <= 5:
        print("Unsloth: Not an error, but could not optimize some PEFT modules.")
    
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print("Unsloth: Not an error, but could not optimize some PEFT modules.")
        print(could_not_replace_modules)
    return
pass


def patch_residual_stream(source):
    # All Unsloth Zoo code licensed under LGPLv3

    # if self.is_gated: hidden_state = self.gate_ffn.tanh() * hidden_state
    # if self.is_gated: hidden_state = self.gate_attn.tanh() * hidden_state
    source = re.sub(
        r"if self\.([^\(]{2,})\:\n"\
        r"[\s]{4,}"\
        r"(hidden\_state(?:s)?) \= ([^\s]{4,}) \* \2\n"\
        r"[\s]{4,}"\
        r"\2 \= residual \+ \2",

        r"\2 = residual + \2 * (\3 if self.\1 else 1.0)",

        source,
    )

    # hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
    # hidden_states = residual + hidden_states * self.residual_multiplier
    matches = re.findall(
        r"[\s]{4,}"\
        r"((hidden\_state(?:s)?) \= residual \+ "\
        r"(?:"\
        r"(?:\2 \* ([^\n]{3,}))"\
        r"|"\
        r"(?:([^\n]{3,}) \* \2)"\
        r"))\n",

        source,
    )
    if len(matches) == 0: return source

    for (full_match, h, left, right,) in matches:
        s = left or right
        replace = \
            f"s = {s}; {h} = "\
            f"torch.add(residual, {h}, alpha = s) "\
            f"if type(s) is float else "\
            f"torch.addcmul(residual, {h}, s)\n"
        source = source.replace(full_match, replace)
    pass
    return source
pass


def patch_gradient_accumulation(modeling_file, module):
    # All Unsloth Zoo code licensed under LGPLv3

    functions = dir(modeling_file)
    module = eval(f"modeling_file.{module}")
    try:
        forward = module.forward
        source = inspect.getsource(forward)
    except:
        return None
    has_kwargs = tuple(inspect.signature(forward).parameters.values())[-1].kind == inspect._VAR_KEYWORD
    if has_kwargs: return None

    __init__ = inspect.getsource(module.__init__)

    # Only get ._from_config type objects
    inner_classes = re.findall(r"(self\.[^ ]{1,}) \= ([^\.]{1,})\._from_config", __init__)
    if len(inner_classes) == 0: return None

    total_has_kwargs = False
    for (call_class, inner_class) in inner_classes:
        inner_class = eval(f"modeling_file.{inner_class}")
        has_kwargs = tuple(inspect.signature(inner_class.forward).parameters.values())[-1].kind == inspect._VAR_KEYWORD
        if not has_kwargs: continue

        total_has_kwargs = True
        print(f"Unsloth: Patching {inner_class.__name__} within {module.__name__} to fix gradient accumulation.")
        regex_find = rf"{call_class}\(([^\)]{{1,}})\)"
        source = re.sub(regex_find, rf"{call_class}(\1, **kwargs)", source, flags = re.DOTALL | re.MULTILINE)
    pass

    if total_has_kwargs:
        # Fix **kwargs for function def
        regex_find = r"def forward\(([^\)]{1,})\)"
        source = re.sub(regex_find, r"def forward(\1, **kwargs)", source, flags = re.DOTALL | re.MULTILINE)

        # Remove double commas
        source = re.sub(r"\,[\s]{0,}\,", ",", source)
    else:
        return None

    # Now replace old forward with new one
    source = inspect.getsource(module).replace(inspect.getsource(forward), source)
    return source
pass


# Pre fix up some modules like Gemma3n
def fixup_fused_lm_head(source):
    # Gemma 3N
    source = source.replace(
        "if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:",
        "if self.config.get_text_config().final_logit_softcapping is not None:",
    )
    source = source.replace(
        "logits = logits / final_logit_softcapping",
        "logits = logits / self.config.get_text_config().final_logit_softcapping",
    )
    source = source.replace(
        "logits = logits * final_logit_softcapping",
        "logits = logits * self.config.get_text_config().final_logit_softcapping",
    )
    # END Gemma 3N fixes
    return source
pass


# =====================================
# Image models inside timm
def rms_norm2d(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    assert len(normalized_shape) == 1
    original_dtype = x.dtype
    v = x.to(torch.float32).pow(2)
    v = torch.mean(v, dim=1, keepdim=True)
    x = x.to(torch.float32) * torch.rsqrt(v + eps)
    if weight is not None:
        x = x.to(torch.float32) * weight.to(torch.float32).reshape(1, -1, 1, 1)
    return x.to(original_dtype)
pass


def compile_timm_models(UNSLOTH_ENABLE_LOGGING, torch_compile_options):
    try:
        import timm
    except:
        return
    try:
        import timm.layers.fast_norm
        timm.layers.fast_norm.is_fast_norm = lambda *args, **kwargs: False
        timm.layers.fast_norm.rms_norm2d = rms_norm2d
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Compiled timm.layers.fast_norm")
    except:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed compiling timm.layers.fast_norm")
    pass
    # Try compiling norms and activation combinations
    try:
        import timm.layers.norm_act
        norms = dir(timm.layers.norm_act)
        norms = [x for x in norms if "Act" in x]
        for norm in norms:
            try:
                exec(f"from timm.layers.norm_act import {norm}")
            except:
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"Unsloth: Failed compiling from timm.layers.norm_act import {norm}")
                continue
            pass
            forward = eval(norm).forward
            if hasattr(forward, "get_compiler_config"): continue
            forward = torch.compile(forward, fullgraph = True, dynamic = None, options = torch_compile_options)
            exec(f"timm.layers.norm_act.{norm}.forward = forward")
            if UNSLOTH_ENABLE_LOGGING:
                print(f"Unsloth: Compiled timm.layers.norm_act.{norm}")
        pass
    except:
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Failed compiling timm.layers.norm_act")
    pass
    # Compile EfficientNet blocks
    try:
        import timm.models._efficientnet_blocks
        efficientnet_blocks = inspect.getsource(timm.models._efficientnet_blocks)

        blocks = re.findall(r"class ([^ ]{1,})\(.*?nn\.Module\)\:", efficientnet_blocks)
        for block in blocks:
            try:
                exec(f"from timm.models._efficientnet_blocks import {block}")
            except:
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"Unsloth: Failed compiling from timm.models._efficientnet_blocks import {block}")
                continue
            pass
            forward = eval(block).forward
            if hasattr(forward, "get_compiler_config"): continue
            forward = torch.compile(forward, fullgraph = True, dynamic = None, options = torch_compile_options)
            exec(f"timm.models._efficientnet_blocks.{block}.forward = forward")
            if UNSLOTH_ENABLE_LOGGING:
                print(f"Unsloth: Compiled timm.models._efficientnet_blocks.{block}")
    except:
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Failed compiling timm.models._efficientnet_blocks")
    pass
pass

def compile_causal_conv1d(UNSLOTH_ENABLE_LOGGING=False):
    # For Liquid, Falcon and other Mamba type models
    # We disable compiling on them!
    try:
        import causal_conv1d
        causal_conv1d.causal_conv1d_fn     = \
            torch.compiler.disable(causal_conv1d.causal_conv1d_fn,     recursive = True)
        causal_conv1d.causal_conv1d_update = \
            torch.compiler.disable(causal_conv1d.causal_conv1d_update, recursive = True)
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Disabled compiling causal_conv1d")
        return True
    except Exception as e:
        print(e, str(e))
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Failed compiling causal_conv1d")
        return False
pass

def compile_mamba_ssm(UNSLOTH_ENABLE_LOGGING=False):
    # For Liquid, Falcon and other Mamba type models
    # We disable compiling on them!
    try:
        import mamba_ssm
        mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined        = \
            torch.compiler.disable(
                mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined,
                recursive = True
            )
        mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined = \
            torch.compiler.disable(
            mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined,
            recursive = True
            )
        mamba_ssm.ops.triton.selective_state_update.selective_state_update = \
            torch.compiler.disable(
                mamba_ssm.ops.triton.selective_state_update.selective_state_update,
                recursive = True
            )
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Disabled compiling mamba_ssm")
        return True
    except:
        if UNSLOTH_ENABLE_LOGGING:
            print(f"Unsloth: Failed compiling mamba_ssm")
        return False
pass


# if module ends with any of these, disable compile
DISABLE_COMPILE_MODULES = [
    "ParallelExperts",
    "GraniteMoeHybridMoE",
    "GraniteMoeHybridMambaLayer",
]


def unsloth_compile_transformers(
    model_type             : str = "llama",
    sdpa_dynamic_mask      : bool = True,
    sdpa_bool_masks        : bool = True,
    sdpa_gqa_replace       : bool = True,
    sdpa_dynamic_compile   : bool = True,
    compile_attention      : bool = True,
    disable_causal_masks   : bool = True,
    compile_torch_modules  : bool = True,
    compile_custom_modules : bool = True,
    compile_function_calls : bool = True,
    fuse_lm_head           : bool = True,
    gradient_checkpointing : bool = True,
    manual_replacements    : bool = True,
    fast_lora_forwards     : bool = True,
    fast_residual_stream   : bool = False,
    accurate_accumulation  : bool = True,
    epilogue_fusion        : bool = True,
    max_autotune           : bool = False,
    shape_padding          : bool = True,
    cudagraphs             : bool = False,
    debug                  : bool = False,
    fullgraph              : bool = True,
    import_from_cache      : bool = False,
    disable                : bool = False,
    return_logits          : bool = False,
    supports_sdpa          : list = None,
):
    # import transformers logging module and instantiate model_type logging instance.
    from transformers import logging as transformers_logging
    model_logger = transformers_logging.get_logger(f"modeling_{model_type}")

    # All Unsloth Zoo code licensed under LGPLv3
    disable = disable or (os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1")
    if fast_residual_stream:
        raise NotImplementedError("Unsloth: Fast residual stream optimization makes things slower!")
    pass

    model_location = f"transformers.models.{model_type}.modeling_{model_type}"
    exec(f"import {model_location}", globals())
    modeling_file = eval(model_location)
    if hasattr(modeling_file, "__UNSLOTH_PATCHED__"): return

    # Use transformers model_type logger to suppress message: Remove `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`
    exec("model_logger.addFilter(HideLoggingMessage('use_cache'))", globals(), locals())
    # Use transformers model_type logger to suppress message: You have set `compile_config`, but we are unable to meet the criteria for compilation.
    exec("model_logger.addFilter(HideLoggingMessage('compile_config'))", globals(), locals())

    # Instead of Inductor Compilation:
    try:
        import torch._inductor.async_compile
        from torch.hub import tqdm
        def replaced_tqdm(*args, **kwargs):
            kwargs["desc"] = "Unsloth: Compiling kernels"
            return tqdm(*args, **kwargs)
        torch._inductor.async_compile.tqdm = replaced_tqdm
    except:
        print("Unsloth: Failed editing tqdm to replace Inductor Compilation:")
    pass

    # torch_compile_options
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    UNSLOTH_ENABLE_LOGGING        = os.environ.get("UNSLOTH_ENABLE_LOGGING",        "0") == "1"
    torch_compile_options = get_torch_compile_options(
        epilogue_fusion = epilogue_fusion,
        max_autotune = max_autotune,
        shape_padding = shape_padding,
        debug = UNSLOTH_COMPILE_DEBUG,
        cudagraphs = cudagraphs,
        coordinate_descent_tuning = UNSLOTH_COMPILE_MAXIMUM,
        logging = UNSLOTH_ENABLE_LOGGING,
        combo_kernels = False, # Causes incompatible gradient sizes on 2.6
        group_fusion = True,
        memory_planning = True,
        multi_kernel = False, # Sometimes fails
        use_block_ptr = False, # Sometimes fails
    )

    # Compile timm models
    compile_timm_models(UNSLOTH_ENABLE_LOGGING, torch_compile_options)

    # Disable compiling mamba type models
    has_causal_conv1d = compile_causal_conv1d(UNSLOTH_ENABLE_LOGGING)
    has_mamba_ssm = compile_mamba_ssm(UNSLOTH_ENABLE_LOGGING)

    # Return logits
    UNSLOTH_RETURN_LOGITS = "0" if not return_logits else "1"
    if "UNSLOTH_RETURN_LOGITS" not in os.environ:
        os.environ["UNSLOTH_RETURN_LOGITS"] = UNSLOTH_RETURN_LOGITS
    else:
        UNSLOTH_RETURN_LOGITS = os.environ["UNSLOTH_RETURN_LOGITS"] == "1"
    pass

    # Fullgraph
    UNSLOTH_FULLGRAPH = "1" if fullgraph else "0"
    if "UNSLOTH_FULLGRAPH" not in os.environ:
        os.environ["UNSLOTH_FULLGRAPH"] = UNSLOTH_FULLGRAPH
    else:
        UNSLOTH_FULLGRAPH = os.environ["UNSLOTH_FULLGRAPH"]
    pass
    UNSLOTH_FULLGRAPH = UNSLOTH_FULLGRAPH == "1"

    # Patch PEFT lora forwards
    if (not disable) and fast_lora_forwards:
        print("Unsloth: Patching LoRA to make it faster")
        patch_lora_forwards(torch_compile_options)
    pass

    modeling_file.__UNSLOTH_PATCHED__ = True
    functions = dir(modeling_file)
    full_source = inspect.getsource(modeling_file)
    # Order functions by ascending order
    functions = list(np.array(functions)[np.argsort([full_source.find(x) for x in functions])])
    ordered_functions = functions.copy()

    # If mamba type, but no fast causal functions, warn!
    if not has_causal_conv1d and \
        ("causal_conv1d_fn" in full_source or "causal_conv1d_update" in full_source):

        print(
            "**********\n"\
            "Unsloth: Please install `causal_conv1d` to speed up Mamba training via `pip install causal_conv1d`\n"\
            "If you don't, training will still work, just might be slower for Mamba type models.\n"\
            "**********\n"
        )
    pass

    # If mamba type, but no fast causal functions, warn!
    if not has_mamba_ssm and \
        ("mamba_chunk_scan_combined" in full_source or "mamba_split_conv1d_scan_combined" in full_source or "selective_state_update" in full_source):
        print(
            "**********\n"\
            "Unsloth: Please install `mamba_ssm` to speed up Mamba training via `pip install mamba_ssm`\n"\
            "If you don't, training will still work, just might be slower for Mamba type models.\n"\
            "**********\n"
        )
    pass

    # Get class LlamaAttention(nn.Module)
    torch_modules = re.findall(r"class ([^\s]{1,})\(.+?\.Module\)", full_source)
    # Also get class LlamaSdpaAttention(LlamaAttention)
    inherited_class = "(?:" + "|".join(re.findall(r"class ([^\s]{1,})\(.+?\.Module\)", full_source)) + ")"
    inherited_modules = re.findall(r"class ([^\s]{1,})\(" + inherited_class + r"\)", full_source)
    # OrderedSet
    torch_modules = list(dict.fromkeys(torch_modules + inherited_modules))
    # Get all functions as well
    functions = [x for x in functions if x not in torch_modules or not compile_torch_modules or not compile_custom_modules]

    # Remove if no forward function
    final_torch_modules = []
    for module in torch_modules:
        source = eval(f"modeling_file.{module}")
        if hasattr(source, "forward"): final_torch_modules.append(module)
    pass
    torch_modules = final_torch_modules

    # Remove functions which have gradient checkpointing in them
    # Also check if it's an attention module
    gradient_checkpointed_modules = []
    scaled_dot_product_attention_modules = []
    full_attention_modules = []
    for module in torch_modules:
        source = eval(f"modeling_file.{module}")
        try: source = inspect.getsource(source)
        except: continue
        if "_gradient_checkpointing_func" in source:
            gradient_checkpointed_modules.append(module)
        elif "scaled_dot_product_attention" in source or "ALL_ATTENTION_FUNCTIONS" in source:
            scaled_dot_product_attention_modules.append(module)
        elif "nn.functional.softmax" in source or "flash_attn_varlen_func" in source or "_flash_attention_forward" in source:
            full_attention_modules.append(module)
    pass
    removal = set(
        scaled_dot_product_attention_modules + \
        full_attention_modules + \
        gradient_checkpointed_modules
    )
    torch_modules = [x for x in torch_modules if x not in removal]

    # Check SDPA to load as eager or SDPA (Pixtral / Mistral 3 for eg doesn't have SDPA)
    if supports_sdpa is not None:
        assert(type(supports_sdpa) is list and len(supports_sdpa) == 1)
        if len(scaled_dot_product_attention_modules) != 0:
            if supports_sdpa[0] != False: supports_sdpa[0] = True
        elif "_supports_sdpa = True" in full_source:
            if supports_sdpa[0] != False: supports_sdpa[0] = True
        else:
            supports_sdpa[0] = False
    pass

    # Get functions which are called
    called_functions = []
    for function in functions:
        # Start of text
        defined = re.findall(r"\bdef[\s]{1,}" + re.escape(function),full_source, flags = re.DOTALL)
        # Disable self.
        called = re.findall(r"[\s]{1,}" + re.escape(function) + r"\(.+?\)", full_source, flags = re.DOTALL)
        if len(defined) != 0 and len(called) != 0:
            called_functions.append(function)
    pass

    # Check if fullgraph can be used
    torch_modules = {x : True for x in torch_modules}
    for module in torch_modules.keys():
        source = eval(f"modeling_file.{module}")
        try: source = inspect.getsource(source.__init__)
        except: continue
        fullgraph = not ("nn.Linear" in source or "nn.ModuleList" in source)

        # Eg SiglipVisionEmbeddings and CLIPVisionEmbeddings
        if str(module).endswith("VisionEmbeddings"):
            # sometimes we attach a post forward call to make sure requires grad is set
            # this breaks full graph mode and fails so instead we relax the full graph check
            # We attach via post forward call, since the forward call only passes keyword
            # arguments in transformers and pre_forward hook doesn't pass kwargs.
            fullgraph = False

        # Check if other modules is used as well
        for another_module in torch_modules:
            if another_module in source:
                fullgraph = fullgraph and torch_modules[another_module]
        pass
        torch_modules[module] = fullgraph if UNSLOTH_FULLGRAPH else False
    pass

    # Get other classes
    other_classes = re.findall(r"class ([^\s]{1,})\(.+?\)", full_source)
    other_classes = [x for x in other_classes if x not in torch_modules and x not in removal]

    # Fix scaled dot product attention up if possible
    scaled_dot_product_attention_modules = {x:None for x in scaled_dot_product_attention_modules}
    disabled_scaled_dot_product_attention_modules = []

    for module in scaled_dot_product_attention_modules.keys():
        source = eval(f"{model_location}.{module}")
        try: source = inspect.getsource(source.forward)
        except: continue

        causal_mask_find = \
            r"(is_causal \= True if (.+?\_mask) is None and q_len \> 1 else False[\n\s]{1,})"\
            r"([A-Za-z0-9\_]{1,}[\s]{1,}\=[\s]{1,}[A-Za-z\.]{1,}scaled\_dot\_product\_attention)"\
            r"(.+?attn\_mask[\s]{0,}\=[\s]{0,})\2"\
            r"(.+?is\_causal[\s]{0,}\=[\s]{0,})is\_causal"

        scaled_dot_product_attention_find = \
            r"(\=[\s]{1,}[A-Za-z\.]{1,}scaled\_dot\_product\_attention)"

        new_source = source
        if sdpa_dynamic_mask:
            new_source = re.sub(
                r"if output_attentions\:.+?return super\(\)\.forward.+?\)",
                "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
                new_source,
                flags = re.DOTALL | re.MULTILINE,
            )
        else:
            if len(re.findall(causal_mask_find, source, flags = re.DOTALL)) == 1:
                new_source = re.sub(
                    causal_mask_find,
                    r"\1\3\4None\5True",
                    source,
                    flags = re.DOTALL,
                )
                new_source = source
            else:
                new_source = re.sub(
                    scaled_dot_product_attention_find,
                    "= disable_compile_scaled_dot_product_attention",
                    source,
                    flags = re.DOTALL,
                )
                disabled_scaled_dot_product_attention_modules.append(module)
            pass
        pass
        scaled_dot_product_attention_modules[module] = new_source
    pass

    all_standalone_classes = {}

    # Fix modules with _update_causal_mask if SDPA can be used with causal masks
    remove_causal_masks = []
    if disable_causal_masks:
        for module in other_classes:
            source = eval(f"{model_location}.{module}")
            if not hasattr(source, "_update_causal_mask"): continue

            try: source = inspect.getsource(source.__init__)
            except: continue

            can_remove = True
            for x in disabled_scaled_dot_product_attention_modules:
                if x in source:
                    can_remove = False
                    break
            pass
            if can_remove: remove_causal_masks.append(module)
        pass
    pass

    # Remove modules which have attention mechanisms
    # since torch.compile will compile too many kernels
    bad_torch_modules = set()
    # actively disable certain modules
    disable_modules = set()
    for module, fullgraph in torch_modules.items():
        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "forward"): continue
        try:
            init   = inspect.getsource(source.__init__)
            source = inspect.getsource(source.forward)
        except: continue

        if "attn_weights" in source or "self.self_attn" in source or "_ATTENTION_CLASSES" in init:

            print(f"Unsloth: Will not compile {module} since it looks like it calls attention modules!")
            bad_torch_modules.add(module)
        pass

        if "self.encoder" in source or "BaseModelOutput" in source:

            print(f"Unsloth: Will not compile {module} since it looks like a vision encoder!")
            bad_torch_modules.add(module)
        pass

        # Check if creating arrays in inside the function
        # Error: DataDependentOutputException: aten._local_scalar_dense.default
        if "torch.arange(" in source or "torch.zeros(" in source or "torch.ones(" in source:
            print(f"Unsloth: Failed compiling function {module} since array creations are done.")
            bad_torch_modules.add(module)
        pass

        # Remove decoder layers
        if "for layer in self." in source:
            print(f"Unsloth: Failed compiling function {module} since it looks like a decoder!")
            bad_torch_modules.add(module)
        pass

        # Remove padding
        if "nn.functional.pad" in source or "padding" in source:
            print(f"Unsloth: Failed compiling function {module} since there is padding done.")
            bad_torch_modules.add(module)
        pass

        # if more modules need to be disabled consider adding to a global list
        if any([module.endswith(x) for x in DISABLE_COMPILE_MODULES]):
            print(f"Unsloth: Disabling compile for {module} since it's marked for disabling.")
            bad_torch_modules.add(module)
            disable_modules.add(module)
        pass

        # Check for residual streams optimizations
        if fast_residual_stream and "residual" in source:
            new_source = patch_residual_stream(source)
            if new_source != source:
                try:
                    new_module = create_standalone_class(
                        module,
                        model_location,
                        functions,
                        fullgraph = False,
                        disable = None,
                        forward_source = new_source,
                    )
                    print(f"Unsloth: Faster residual stream for {module}")
                    all_standalone_classes[module] = new_module
                except:
                    continue
            pass
        pass
    pass
    # Add back to functions since failed compiling
    functions += list(bad_torch_modules)

    if len(disable_modules) > 0:
        for module in disable_modules:
            try:
                new_module = create_standalone_class(
                    module,
                    model_location,
                    functions,
                    fullgraph = False,
                    disable = True,
                )
                all_standalone_classes[module] = new_module
            except:
                print(f"Unsloth: Failed to disable {module}.")
                continue
        pass
    pass

    # Now patch modules ie LlamaRMSNorm
    if compile_custom_modules:
        for module, fullgraph in torch_modules.items():
            if module in bad_torch_modules: continue
            try:
                new_module = create_standalone_class(
                    module,
                    model_location,
                    functions,
                    fullgraph = fullgraph,
                )
                print(f"Unsloth: Compiled module {module}.")
                all_standalone_classes[module] = new_module
            except:
                continue
        pass
    pass

    # SDPA
    if compile_attention:
        for module, forward_source in scaled_dot_product_attention_modules.items():
            if sdpa_gqa_replace:
                forward_source = replace_with_grouped_query_attention(
                    module,
                    forward_source,
                )
            pass
            try:
                new_module = create_standalone_class(
                    module,
                    model_location,
                    functions,
                    fullgraph = fullgraph,
                    disable = sdpa_dynamic_compile,
                    forward_source = forward_source,
                )
                print(f"Unsloth: Fast Attention patch for {module}.")
                all_standalone_classes[module] = new_module
            except:
                continue
        pass

        # Patch full attention modules
        for module in full_attention_modules:
            try:
                new_module = create_standalone_class(
                    module,
                    model_location,
                    functions,
                    fullgraph = False,
                    disable = True,
                )
                print(f"Unsloth: Slow Attention patch for {module}.")
                all_standalone_classes[module] = new_module
            except:
                continue
        pass
    pass

    # Remove causal masks
    do_not_remove = False
    for module in remove_causal_masks:
        if module.endswith(("ForConditionalGeneration", "Gemma3Model")):
            do_not_remove = True
            print(f"Unsloth: Will not remove causal mask for {model_location} since it's a VLM!")
            break
    pass
    for module in remove_causal_masks:
        if do_not_remove: continue

        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "_update_causal_mask"): continue

        # Don't remove for VLMs!
        if module.endswith(("ForConditionalGeneration")):
            print(f"Unsloth: Will not remove causal mask for {module} since it's a VLM!")
            continue

        exec(f"{model_location}.{module}._update_causal_mask = no_update_causal_mask", globals())
        print(f"Unsloth: Removed causal mask for {module} to reduce memory usage.")
    pass

    # Patch LM Head
    if fuse_lm_head:
        from transformers.generation import GenerationMixin
        modules = dir(modeling_file)

        for module in modules:
            # Disable if torch < 2.5 or V100s 7.0 (Tesla T4 7.5 works) or old Triton < 3
            if OLD_CUDA_ARCH_VERSION or OLD_TORCH_VERSION or OLD_TRITON_VERSION:
                continue

            module_class = eval(f"modeling_file.{module}")
            if hasattr(module_class, "forward") and issubclass(module_class, GenerationMixin):
                try:
                    source = inspect.getsource(module_class.forward)
                except:
                    continue
                # Fix some arguments up like for Gemma 3N
                new_source = fixup_fused_lm_head(source)
                # Apply fused LM transforms
                new_source = apply_fused_lm_head(new_source, module)
                # print(new_source)
                new_source = apply_mask_attention_mask_out(new_source)
                if new_source != source:
                    new_module = create_standalone_class(
                        module,
                        model_location,
                        functions,
                        fullgraph = False,
                        disable = True,
                        forward_source = new_source,
                        add_loss_kwargs = True,
                    )
                    print(f"Unsloth: Fast fused linear cross entropy patch for {module}.")
                    all_standalone_classes[module] = new_module
                pass
            pass
        pass
    pass

    # Allow gradient checkpointing if not enabled
    if gradient_checkpointing:
        for module in other_classes:
            source = eval(f"{model_location}.{module}")
            output = patch_gradient_checkpointing(module, source)
            if output is None: continue

            init, forward = output
            new_module = create_standalone_class(
                module,
                model_location,
                functions,
                fullgraph = False,
                disable = True,
                forward_source = forward,
                add_loss_kwargs = False,
                new_init = init,
            )
            all_standalone_classes[module] = new_module
            print(f"Unsloth: Patched {module} by adding gradient checkpointing")
        pass
    pass

    # torch.finfo fix for transformers > 4.52.4 affect qwen2vl, qwen25vl, and glm4vl
    for module in other_classes:
        if module in all_standalone_classes:
            source = all_standalone_classes[module]
        else:
            module_cls = eval(f"{model_location}.{module}")
            if hasattr(module_cls, "forward"):
                source = inspect.getsource(module_cls.forward)
            else:
                continue
            new_source = patch_finfo_attention_mask_dtype_mismatch(module, source)
            if new_source != source:
                new_module = create_standalone_class(
                    module,
                    model_location,
                    functions,
                    fullgraph = False,
                    disable = True,
                    forward_source = new_source,
                )
                all_standalone_classes[module] = new_module
                print(f"Unsloth: Patched {module} by fixing finfo dtype mismatch in attention mask")
            pass
        pass
    pass

    # Manually replace hand written parts
    if manual_replacements:
        for module in compiler_replacements:
            if module in all_standalone_classes or \
                module in bad_torch_modules or \
                module in remove_causal_masks:

                print(f"Unsloth: Manual replacement for {module}")
                all_standalone_classes[module] = compiler_replacements[module]
        pass
    pass

    # Patch Trainer
    from transformers.trainer import Trainer
    try:
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
            Trainer._original_training_loop = inner_training_loop
        else:
            inner_training_loop = Trainer._original_training_loop
    except:
        raise RuntimeError('Unsloth: Unsuccessfully patched inner_training_loop')
    pass

    import transformers.trainer
    items_in_trainer = dir(transformers.trainer)
    good_items = []
    for item in items_in_trainer:
        if item in inner_training_loop: good_items.append(item)
    pass
    exec("from transformers.trainer import (" + ", ".join(x for x in good_items) + ")", globals())

    start = re.search(r'logger\.info\([\"\'].+?Running training', inner_training_loop).span(0)[0]
    end = inner_training_loop.find("\n\n", start)
    original_debug = inner_training_loop[start:end]
    spaces = re.search(r'\n([\s\t]{1,})', original_debug).group(0)[1:]
    front_spaces = re.match(r'([\s\t]{1,})', inner_training_loop).group(0)

    debug_info = """debug_info = \\
        f"==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = {len(set(p.device for p in model.parameters()))}\\n"\\
        f"   {chr(92)}{chr(92)}   /|    Num examples = {num_examples:,} | Num Epochs = {num_train_epochs:,} | Total steps = {max_steps:,}\\n"\\
        f"O^O/ {chr(92)}_/ {chr(92)}    Batch size per device = {self._train_batch_size:,} | Gradient accumulation steps = {args.gradient_accumulation_steps}\\n"\\
        f"{chr(92)}        /    Data Parallel GPUs = {args.world_size} | Total batch size ({self._train_batch_size} x {args.gradient_accumulation_steps} x {args.world_size}) = {total_train_batch_size:,}\\n"\\
        f' "-____-"     Trainable parameters = {get_model_param_count(model, trainable_only=True):,} of {get_model_param_count(model):,} ({get_model_param_count(model, trainable_only=True)/get_model_param_count(model)*100:.2f}% trained)'
        f"🦥 Unsloth needs about 1-3 minutes to load everything - please wait!"
        logger.warning(debug_info)
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()"""

    debug_info = debug_info.split('\n')
    debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
    inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

    debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once('Unsloth is running with multi GPUs - the effective batch size is multiplied by ' + str(n_total_devices))
        debug_info ="""
    debug_info = debug_info.split('\n')
    debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
    inner_training_loop = inner_training_loop.replace("debug_info =", debug_info, 1)

    front_spaces = re.match(r"[\t\s]{1,}", inner_training_loop).group(0)
    inner_training_loop = re.sub(r"^" + front_spaces, "", inner_training_loop, flags = re.MULTILINE)
    inner_training_loop = inner_training_loop.replace(
        "train_dataloader = tpu_spmd_dataloader(train_dataloader)",
        "raise RuntimeError('Unsloth: TPUs are not yet supported!')"
    )
    inner_training_loop = inner_training_loop.replace(
        "_inner_training_loop",
        "_fast_inner_training_loop", 1,
    )
    inner_training_loop = inner_training_loop.replace(
        "is_torch_tpu_available()",
        "False",
    )
    exec(inner_training_loop, globals())
    Trainer._inner_training_loop = _fast_inner_training_loop

    # All other functions
    if compile_function_calls:
        mask_functions = get_mask_functions()
        # Fix up function signatures
        for module in called_functions:
            function = eval(f"{model_location}.{module}")

            parameters = inspect.signature(function)
            params = list(parameters.parameters.keys())
            source = inspect.getsource(function)

            where = source.find(str(parameters))
            if where == -1: where = source.find("\n") + 1
            else: where = where + len(str(parameters))
            code_section = source[where:]
            cleaned_code_section = re.sub(r'\"\"\".+?\"\"\"', "", code_section, flags = re.DOTALL)

            bad_params = []
            for param in params:
                if not param in cleaned_code_section:
                    bad_params.append(param)
            pass
            if len(bad_params) == 0: continue

            for bad_param in bad_params:
                parameters = re.sub(
                    re.escape(bad_param) + r"[\s]{0,}\=[\s]{0,}None[\s]{0,}\,",
                    "", # Remove them entirely
                    str(parameters),
                    flags = re.DOTALL,
                )
            pass
            parameters = f"def {module}" + parameters + code_section
            print(f"Unsloth: Fixed up function {module}.")

            parameters = \
                f"@torch.compile(fullgraph = {UNSLOTH_FULLGRAPH}, dynamic = True, options = torch_compile_options)\n{parameters}"
            all_standalone_classes[module] = parameters
        pass

        for module in called_functions:
            if module in all_standalone_classes: continue
            function = eval(f"{model_location}.{module}")
            source = inspect.getsource(function)

            if sdpa_bool_masks:
                source = convert_attention_masks_to_bool(module, source)

            # Check erroring out
            bad = False
            for keyword in DISABLED_KEYWORDS:
                if keyword in source:
                    bad = True
                    break
            pass
            if not bad:
                source = f"@torch.compile(fullgraph = {UNSLOTH_FULLGRAPH}, dynamic = True, options = torch_compile_options)\n{source}"
                print(f"Unsloth: Compiled function {module}.")
            else:
                print(f"Unsloth: Cannot compile function {module} since disabled keyword is in it.")
            # Skip mask creation functions
            bad = False
            for mask_function in mask_functions:
                if mask_function == module:
                    bad = True
                    print(f"Unsloth: Will skip copying source of {module}.")
                    break
            pass
            if not bad:
                all_standalone_classes[module] = source
        pass
    pass

    # Fix gradient accumulation issues if there's no **kwargs
    if accurate_accumulation:
        for module in other_classes:
            new_source = patch_gradient_accumulation(modeling_file, module)
            if new_source is None: continue
            if module in all_standalone_classes:
                print(f"Unsloth: Will override already patched {module} with gradient accumulation fix.")
            all_standalone_classes[module] = new_source
        pass
    pass

    # Order all components
    final_all_standalone_classes = []
    for module in ordered_functions:
        if module in all_standalone_classes:
            final_all_standalone_classes.append(all_standalone_classes[module])
        pass
    pass

    all_code = "\n\n".join(final_all_standalone_classes)

    try:
        combined_module = create_new_function(
            f"{COMBINED_UNSLOTH_NAME}_{model_type}",
            all_code,
            model_location,
            functions,
            prepend = \
                _disabled_sdpa_code + \
                f"\ntorch_compile_options = {torch_compile_options}\n" + \
                _cross_entropy_code + "\n"
        )
    except Exception as exception:
        if not disable:
            raise RuntimeError(exception)
        combined_module = None

    if compile_torch_modules and not disable:

        from .patch_torch_functions import patch_torch_functions
        patch_torch_functions()

        for module in _patch_functions:
            try: source = eval(f"{model_location}.torch")
            except: continue
            if not hasattr(source, "nn"): continue
            if not hasattr(source.nn, module): continue
            function = eval(f"source.nn.{module}")
            if not hasattr(function, "forward"): continue
            if hasattr(function.forward, "get_compiler_config"): continue

            source = inspect.getsource(function.forward).rstrip()
            forward = create_new_function(
                module, source, model_location, functions,
                prepend = \
                    _license_header + \
                    f"\ntorch_compile_options = {torch_compile_options}\n",
                append = ".to(input.dtype)\n",
                overwrite = False,
                add_torch_compile = False,
            ).forward

            exec(f"{model_location}.torch.nn.{module}.forward = forward", globals(), locals())
            try: exec( f"{model_location}.nn.{module}.forward = forward", globals(), locals())
            except: pass
            if combined_module is not None:
                exec( f"combined_module.torch.nn.{module}.forward = forward", globals(), locals())
                try: exec(  f"combined_module.nn.{module}.forward = forward", globals(), locals())
                except: pass
            pass
        pass
    pass
    # Quick exit
    if combined_module is None or disable: return

    # Import and replace with new module
    for module in all_standalone_classes.keys():
        exec(f"{model_location}.{module} = combined_module.{module}", globals(), locals())
    pass

    # Finally edit dictionary items inside the target file
    replaced_classes = all_standalone_classes.keys()
    check_dicts = dir(eval(f"{model_location}"))
    for check in check_dicts:
        item = eval(f"{model_location}.{check}")
        if type(item) is not dict: continue

        for key, value in item.items():
            value = str(value)
            found = False
            for replaced_class in replaced_classes:
                if replaced_class in value:
                    exec(f"{model_location}.{check}['{key}'] = combined_module.{replaced_class}", globals(), locals())
                    # print(f"Unsloth: Replacing {check} with {replaced_class}")
                    break
                pass
            pass
        pass
    pass
    return
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
