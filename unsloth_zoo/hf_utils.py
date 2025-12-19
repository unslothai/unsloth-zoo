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
import tempfile
import shutil
import torch
import json
import re
from .log import logger
try:
    from transformers import PreTrainedConfig
    PretrainedConfig = PreTrainedConfig
except:
    from transformers import PretrainedConfig

HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__

__all__ = [
    "HAS_TORCH_DTYPE",
    "dtype_from_config",
    "add_dtype_kwargs",
    "set_dtype_in_config",
    "get_transformers_model_type",
    "fix_lora_auto_mapping",
    "get_auto_processor",
]

def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    if HAS_TORCH_DTYPE:
        check_order = ['torch_dtype', 'dtype']
    dtype = None
    for dtype_name in check_order:
        if dtype is None:
            dtype = getattr(config, dtype_name, None)
    return dtype

def set_dtype_in_config(config, dtype):
    try:
        # if dtype is not a string, convert it to a string
        string_dtype = str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else dtype
        if HAS_TORCH_DTYPE:
            setattr(config, "torch_dtype", string_dtype)
        else:
            setattr(config, "dtype", string_dtype)
    except:
        set_dtype_in_config_fallback(config, string_dtype)

def set_dtype_in_config_fallback(config, dtype):
    try:
        string_dtype = str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else dtype
        if HAS_TORCH_DTYPE:
            config.__dict__["torch_dtype"] = string_dtype
        else:
            config.__dict__["dtype"] = string_dtype
    except:
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            print("Unsloth: Failed to set dtype in config, fallback failed too")

def add_dtype_kwargs(dtype, kwargs_dict=None):
    if kwargs_dict is None:
        kwargs_dict = {}
    if HAS_TORCH_DTYPE:
        kwargs_dict["torch_dtype"] = dtype
    else:
        kwargs_dict["dtype"] = dtype
    return kwargs_dict

def _dtype_stringify(x):
    # Convert *values* (not the config) into JSON-safe strings when they are dtypes
    try:
        if isinstance(x, torch.dtype):
            # str(torch.float16) -> "torch.float16" -> "float16"
            return str(x).split(".", 1)[-1]
        if isinstance(x, str) and x.startswith("torch."):
            tail = x.split(".", 1)[-1]
            # Only strip "torch." if the tail is actually a dtype on torch
            if hasattr(torch, tail) and isinstance(getattr(torch, tail), torch.dtype):
                return tail
    except Exception:
        pass

    return x

def _normalize_dict_dtypes(obj):
    if isinstance(obj, dict):
        return {k: _normalize_dict_dtypes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_dict_dtypes(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_dict_dtypes(v) for v in obj)
    return _dtype_stringify(obj)


def get_transformers_model_type(config, trust_remote_code=False):
    """ Gets model_type from config file - can be PEFT or normal HF """
    if config is None:
        raise RuntimeError(
            f"Unsloth: No config file found - are you sure the `model_name` is correct?\n"\
            f"If you're using a model on your local device, confirm if the folder location exists.\n"\
            f"If you're using a HuggingFace online model, check if it exists."
        )
    model_types = None

    from peft import PeftConfig
    # Handle model.peft_config["default"]
    if type(config) is dict and "default" in config:
        config = config["default"]
    
    retry_config = False
    if issubclass(type(config), PeftConfig):
        model_type_list = re.finditer(r"transformers\.models\.([^\.]{2,})\.modeling_\1", str(config))
        model_type_list = list(model_type_list)
        if len(model_type_list) == 0:
            logger.info("*** `model_type_list` in `get_transformers_model_type` is None!")
        if len(model_type_list) != 0:
            # Use transformers.models.gpt_oss.modeling_gpt_oss
            model_type = model_type_list[0].group(1)
            model_types = [model_type]
        elif getattr(config, "auto_mapping", None) is not None:
            # Use GptOssForCausalLM
            model_type = config.auto_mapping.get("base_model_class", None)
            if model_type is not None:
                model_type = str(model_type)
                model_type = model_type.rsplit("For", 1)[0].lower()
                # Find exact name of modeling path
                import transformers.models
                supported_model_types = dir(transformers.models)
                for modeling_file in supported_model_types:
                    if model_type == modeling_file.lower().replace("_", "").replace(".", "_").replace("-", "_"):
                        model_types = [modeling_file]
                        break
            pass
        pass

        # Get original base model
        base_model_name_or_path = getattr(config, "base_model_name_or_path", None)
        if base_model_name_or_path is None:
            raise TypeError("Unsloth: adapter_config.json's `base_model_name_or_path` is None?")
        base_model_name_or_path = str(base_model_name_or_path)
        # Set model name for patching purposes
        os.environ["UNSLOTH_MODEL_NAME"] = base_model_name_or_path.lower()

        # Last resort use model name unsloth/gpt-oss-20b-unsloth-bnb-4bit
        if model_types is None:
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(
                    base_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                )
                retry_config = True
            except ImportError as error:
                config = None
                raise error
            except Exception as error:
                from transformers import __version__ as transformers_version
                autoconfig_error = str(error)
                if "architecture" in autoconfig_error:
                    raise ValueError(
                        f"`{base_model_name_or_path}` is not supported yet in `transformers=={transformers_version}`.\n"
                        f"Please update transformers via `pip install --upgrade transformers` and try again."
                    )
                config = None
        pass
    else:
        retry_config = True
    pass

    # Check since we might have tried AutoConfig fallback last resort for LoRA
    if retry_config:
        from collections.abc import Mapping, Sequence
        def find(data, target_key):
            stack = [data]
            while stack:
                obj = stack.pop()
                if isinstance(obj, Mapping):
                    # Emit values for matches
                    if target_key in obj:
                        yield obj[target_key]
                    # Keep walking into nested values
                    stack.extend(obj.values())
                elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                    # Walk sequences (lists/tuples/sets), but not strings/bytes
                    stack.extend(obj)
        model_types = list(find(getattr(config, "to_dict", lambda *args, **kwargs: {})(), "model_type"))
    pass
    if model_types is None:
        raise TypeError(f"Unsloth: Cannot determine model type for config file: {str(config)}")
    # Standardize model_type
    final_model_types = []
    for model_type in model_types:
        model_type = model_type.lower()
        model_type = model_type.replace("-", "_")
        model_type = model_type.replace("/", "_")
        model_type = model_type.replace(".", "_")
        final_model_types.append(model_type)
    final_model_types = sorted(final_model_types)

    # Check if model type is correct
    # Gemma-3 270M has `gemma3_text` which is wrong
    import transformers.models
    all_model_types = dir(transformers.models)
    found_type = False
    for j, model_type in enumerate(final_model_types):
        if model_type not in all_model_types:
            # Try splitting on _ gemma3_text -> gemma3
            model_types = list(model_type)
            model_types = ["".join(model_types[:i]) for i in range(len(model_types), 0, -1)]
            for current_model_type in model_types:
                if current_model_type in all_model_types:
                    final_model_types[j] = current_model_type
                    found_type = True
                    break
        else:
            found_type = True
    pass
    if not found_type:
        logger.info(f"*** Could not find model_type for config = {str(config)} ***")
    final_model_types = sorted(final_model_types)
    return final_model_types
pass


def fix_lora_auto_mapping(model):
    # Fix LoraConfig's auto_mapping_dict
    if getattr(model, "peft_config", None) is None: return

    peft_config = model.peft_config
    values = peft_config.values() if type(peft_config) is dict else [peft_config]
    for config in values:
        # See https://github.com/huggingface/peft/blob/20a9829f76419149f5e447b856bc0abe865c28a7/src/peft/peft_model.py#L347
        if getattr(model, "_get_base_model_class", None) is not None:
            base_model_class = model._get_base_model_class(
                is_prompt_tuning = getattr(config, "is_prompt_learning", False),
            )
        elif getattr(model, "base_model", None) is not None:
            base_model_class = model.base_model.__class__
        else:
            base_model_class = model.__class__
        pass
        parent_library = base_model_class.__module__
        auto_mapping_dict = {
            "base_model_class": base_model_class.__name__,
            "parent_library": parent_library,
            "unsloth_fixed" : True,
        }
        if getattr(config, "auto_mapping", None) is None:
            config.auto_mapping = auto_mapping_dict
    pass
pass


def get_auto_processor(name, **kwargs):
    # Allow AutoProcessor to work if config.json does not exist
    if not os.path.exists(name):
        return None
    try:
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
    except:
        return None

    reversal_map = { v : k for k, v in PROCESSOR_MAPPING_NAMES.items() }
    processor_class = None
    model_type = None

    # Find "processor_class" : "Gemma3Processor"
    for filename in [
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ]:
        processor_config = os.path.join(name, filename)
        if os.path.exists(processor_config):
            try:
                with open(processor_config, "r") as f: f = f.read()
                config = json.loads(f)
                processor_class = config["processor_class"]
                model_type = reversal_map[processor_class]
                break
            except:
                pass
    pass
    # model_module = __import__(f"transformers.models.{model_type}")
    # processor = getattr(model_module, processor_class)

    if model_type is None:
        # Try loading adapter_config.json
        adapter_config = os.path.join(name, "adapter_config.json")
        if os.path.exists(adapter_config):
            try:
                from peft import PeftConfig
                peft_config = PeftConfig.from_pretrained(name)
                model_type = get_transformers_model_type(peft_config)[0]
            except:
                pass
    pass
    if model_type is None:
        # Try doing AutoTokenizer instead
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(name, **kwargs)
        except:
            raise TypeError(f"Unsloth: Failed loading a AutoProcessor from `{name}`")
    pass

    # Make a temporary directory to copy all files
    temp_directory = tempfile.TemporaryDirectory()
    temp_name = temp_directory.name

    # Make a fake config.json file with just the model_type
    config_file = {"model_type" : model_type}
    with open(os.path.join(temp_name, "config.json"), "w") as f:
        f.write(json.dumps(config_file))

    # Copy other files
    filenames = os.listdir(name)
    for filename in filenames:
        if "model" not in filename and "safetensors" not in filename and "bin" not in filename:
            try:
                shutil.copy(os.path.join(name, filename), os.path.join(temp_name, filename))
            except:
                pass
    pass

    # Try importing again!
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(temp_name, **kwargs)
    except:
        processor = None
    temp_directory.cleanup()

    # Try doing AutoTokenizer instead
    if processor is None:
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(name, **kwargs)
        except:
            raise TypeError(f"Unsloth: Failed loading a AutoProcessor from `{name}`")
    return processor
pass
