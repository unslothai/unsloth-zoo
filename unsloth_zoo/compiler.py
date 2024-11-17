# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
    "compile_transformers_model_type",
]

import inspect
import re
import importlib
import numpy as np
import os
import torch

global COMBINED_UNSLOTH_NAME
global UNSLOTH_COMPILE_LOCATION
global UNSLOTH_CREATED_FUNCTIONS
COMBINED_UNSLOTH_NAME = "unsloth_compiled_module"
UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"
UNSLOTH_CREATED_FUNCTIONS = []


_disabled_sdpa_code = """
import torch
scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass

"""

# Patch Layernorm, Conv
_patch_functions = [
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "LayerNorm", "RMSNorm",
]


def get_transformers_model_type(
    model_name,
    token = None,
    revision = None,
    trust_remote_code = False
):
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

    model_type = config.model_type.replace("_", "").replace("-", "").lower()
    from transformers import models
    models = dir(models)
    all_model_types = set()
    for name in models:
        if model_type in name.lower():
            all_model_types.add(model_type)
    pass
    all_model_types = list(all_model_types)
    return all_model_types
pass


# Empty causal mask
def no_update_causal_mask(*args, **kwargs): return None

# Patch SDPA
def replace_with_grouped_query_attention(module, source):
    # Code licensed under LGPL
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
            source = source.replace(
                "dropout_p=self.dropout if self.training else 0.0,",
                "dropout_p=self.dropout if self.training else 0.0, enable_gqa=self.num_key_value_groups != 1,",
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


def create_new_function(name, new_source, model_location, functions, prepend = "", append = ""):
    # Code licensed under LGPL
    global UNSLOTH_CREATED_FUNCTIONS
    global UNSLOTH_COMPILE_LOCATION
    if new_source[0] == " ":
        spaces = new_source.find("def")
        new_source = new_source.split("\n")
        new_source = "\n".join(x[spaces:] for x in new_source)
    pass

    # Import items to make the function executable
    items = [x for x in functions if ((x in new_source) and (x != name) and not (f"def {x}" in new_source))]
    imports = "from torch import Tensor\n"
    imports += f"from {model_location} import (" + ", ".join(x for x in items) + ")" if len(items) != 0 else ""
    new_source = imports + "\n\n" + new_source
    new_source = prepend + new_source + append

    # Fix super() Not necessary anymore!
    # new_source = new_source.replace("super()", "super(type(self), self)")

    # Check location
    if not os.path.exists(UNSLOTH_COMPILE_LOCATION): os.makedirs(UNSLOTH_COMPILE_LOCATION)

    location = os.path.join(UNSLOTH_COMPILE_LOCATION, f"{name}.py")
    with open(location, "w") as file:
        file.write(new_source)
        file.flush()
        os.fsync(file)
    pass

    new_module = importlib.import_module(UNSLOTH_COMPILE_LOCATION + "." + name)
    # Must save to global state or else temp file closes
    UNSLOTH_CREATED_FUNCTIONS.append(location)
    return new_module
pass


def create_standalone_class(
    module,
    model_location,
    functions,
    fullgraph = False,
    forward_source = None,
    disable = False,
) -> str:
    # Code licensed under LGPL
    # Create optimized standalone forward function
    f = eval(f"{model_location}.{module}")
    full_class = inspect.getsource(f)
    old_source = inspect.getsource(f.forward)
    if forward_source is None: forward_source = old_source

    source = re.sub(
        "def forward",
        f"def {module}_forward",
        forward_source,
    )
    spaces = re.search(r"[^\s\n]", source).span(0)[0]
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)

    compile = \
        f"torch.compile(fullgraph = {fullgraph}, dynamic = True, options = torch_compile_options)" \
        if not disable else \
        "torch.compiler.disable(recursive = False)"

    source = f"@{compile}\n{source}\n"

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
    left = re.match("[\s\n]{4,}", leftover).span()[1]
    new_forward = definition + leftover[:left] + \
        f"return {module}_forward({parameters})\n"
    full_class = full_class.replace(old_source, new_forward)

    # Combine all into file
    source = source + full_class
    return source
pass


# Patch remaining functions
def convert_attention_masks_to_bool(module, old_source):
    # Code licensed under LGPL
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


def unsloth_compile_transformers(
    model_type           : str = "llama",
    sdpa_causal_only     : bool = False,
    sdap_bool_masks      : bool = True,
    sdpa_gqa_replace     : bool = True,
    sdpa_disable_compile : bool = True,
    remove_causal_masks  : bool = True,
    import_from_cache    : bool = False,
    compile_functions    : bool = True,
):
    # Code licensed under LGPL
    model_location = f"transformers.models.{model_type}.modeling_{model_type}"
    exec(f"import {model_location}", globals())
    modeling_file = eval(model_location)
    functions = dir(modeling_file)
    full_source = inspect.getsource(modeling_file)
    # Order functions by ascending order
    functions = list(np.array(functions)[np.argsort([full_source.find(x) for x in functions])])
    ordered_functions = functions.copy()

    # Get class LlamaAttention(nn.Module)
    torch_modules = re.findall(r"class (.+?)\(.+?\.Module\)", full_source)
    # Also get class LlamaSdpaAttention(LlamaAttention)
    inherited_class = "(?:" + "|".join(re.findall(r"class (.+?)\(.+?\.Module\)", full_source)) + ")"
    inherited_modules = re.findall(r"class (.+?)\(" + inherited_class + "\)", full_source)
    # OrderedSet
    torch_modules = list(dict.fromkeys(torch_modules + inherited_modules))
    # Get all functions as well
    functions = [x for x in functions if x not in torch_modules]

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
        elif "scaled_dot_product_attention" in source:
            scaled_dot_product_attention_modules.append(module)
        elif "nn.functional.softmax" in source:
            full_attention_modules.append(module)
    pass
    removal = set(
        scaled_dot_product_attention_modules + \
        full_attention_modules + \
        gradient_checkpointed_modules
    )
    torch_modules = [x for x in torch_modules if x not in removal]

    # Get functions which are called
    called_functions = []
    for function in functions:
        # Start of text
        defined = re.findall(r"\bdef[\s]{1,}" + re.escape(function),full_source, flags = re.DOTALL)
        # Disable self.
        called = re.findall(r"[\s]{1,}" + re.escape(function) + "\(.+?\)", full_source, flags = re.DOTALL)
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

        # Check if other modules is used as well
        for another_module in torch_modules:
            if another_module in source:
                fullgraph = fullgraph and torch_modules[another_module]
        pass
        torch_modules[module] = fullgraph
    pass

    # Get other classes
    other_classes = re.findall(r"class (.+?)\(.+?\)", full_source)
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
        if sdpa_causal_only:
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
        else:
            new_source = re.sub(
                r"if output_attentions\:.+?return super\(\)\.forward.+?\)",
                "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
                new_source,
                flags = re.DOTALL | re.MULTILINE,
            )
        pass
        scaled_dot_product_attention_modules[module] = new_source
    pass

    # Fix modules with _update_causal_mask if SDPA can be used with causal masks
    remove_causal_masks = []
    if remove_causal_masks:
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
    for module, fullgraph in torch_modules.items():
        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "forward"): continue
        try: source = inspect.getsource(source.forward)
        except: continue
        if "attn_weights" in source or "self.self_attn" in source:
            print(f"Unsloth: Will not compile {module}.")
            bad_torch_modules.add(module)
    pass

    # Now patch modules
    all_standalone_classes = {}
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

    # SDPA
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
                disable = sdpa_disable_compile,
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

    # Remove causal masks
    for module in remove_causal_masks:
        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "_update_causal_mask"): continue

        exec(f"{model_location}.{module}._update_causal_mask = no_update_causal_mask")
        print(f"Unsloth: Removed causal mask for {module} to reduce memory usage.")
    pass

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
            f"@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n{parameters}"
        all_standalone_classes[module] = parameters
    pass

    # All other functions
    for module in called_functions:
        if module in all_standalone_classes: continue
        function = eval(f"{model_location}.{module}")
        source = inspect.getsource(function)
        if sdap_bool_masks:
            source = convert_attention_masks_to_bool(module, source)
        source = f"@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n{source}"
        all_standalone_classes[module] = source
        print(f"Unsloth: Compiled function {module}.")
    pass

    # Order all components
    final_all_standalone_classes = []
    for module in ordered_functions:
        if module in all_standalone_classes:
            final_all_standalone_classes.append(all_standalone_classes[module])
        pass
    pass

    all_code = "\n\n".join(final_all_standalone_classes)

    if import_from_cache:
        try:
            combined_module = importlib.import_module(f"{UNSLOTH_COMPILE_LOCATION}.{COMBINED_UNSLOTH_NAME}")
            import_from_cache = True
        except:
            import_from_cache = False
    else:
        import_from_cache = False
    if not import_from_cache:
        combined_module = create_new_function(
            COMBINED_UNSLOTH_NAME,
            all_code,
            model_location,
            functions,
            prepend = \
                _disabled_sdpa_code + \
                f"\ntorch_compile_options = {torch_compile_options}\n"
        )
    pass

    if compile_functions:
        for module in _patch_functions:
            try: source = eval(f"{model_location}.torch")
            except: continue
            if not hasattr(source, "nn"): continue
            if not hasattr(source.nn, module): continue
            function = eval(f"source.nn.{module}")
            if not hasattr(function, "forward"): continue
            if hasattr(function.forward, "get_compiler_config"): continue

            source = inspect.getsource(function.forward).rstrip()
            forward = create_new_function(module, source, model_location, functions, append = ".to(input.dtype)\n").forward
            exec(f"{model_location}.torch.nn.{module}.forward = forward", globals())
            try:  exec(f"{model_location}.nn.{module}.forward = forward", globals())
            except: pass
            exec( f"combined_module.torch.nn.{module}.forward = forward", globals())
            try:  exec( f"combined_module.nn.{module}.forward = forward", globals())
            except: pass
        pass
    pass

    # Import and replace with new module
    for module in all_standalone_classes.keys():
        exec(f"{model_location}.{module} = combined_module.{module}", globals())
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
                    exec(f"{model_location}.{check}['{key}'] = combined_module.{replaced_class}", globals())
                    # print(f"Unsloth: Replacing {check} with {replaced_class}")
                    break
                pass
            pass
        pass
    pass
    return
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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