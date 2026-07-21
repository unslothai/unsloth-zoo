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
#
# ============================================================================
# Gemma-4 targeted float32 patches for fp16 (UNSLOTH_FORCE_FLOAT32) training.
#
# Direct analog of the Gemma3 float32 patches in gemma.py. When gemma4 is in
# unsloth_zoo.model_lists.FORCE_FLOAT32, a fp16 request loads bf16 weights,
# down-casts them to fp16, and relies on these per-module patches to keep the
# residual stream in float32 while matmuls run in fp16. Without them fp16 NaNs
# the grad_norm in backward (forward peaks ~350, far below fp16's 65504, so it
# is a backward gradient overflow), and an fp32 RoPE upcast of q/k while v stays
# fp16 makes SDPA raise a dtype mismatch. All three gate on
# UNSLOTH_FORCE_FLOAT32 == "1", so bf16 / fp32 (and fp16 on other archs) are untouched.
#
# Only the precision-sensitive ops upcast (targeted, not a whole-model wrap):
#   * ScaledWordEmbedding -> returns float32, keeping the residual highway fp32.
#   * RMSNorm             -> computes in float32, returns fp16 (clamped); residual
#     adds (fp32 + fp16) stay fp32.
#   * TextAttention       -> q/k/v + scores all float32 (no SDPA mismatch), output
#     down-cast to fp16 for o_proj.
# MLP / MoE experts are already handled by gemma4.py's fp16 overflow clamp plus
# the grouped-GEMM expert forward, so no extra MLP patch is needed here.
# ============================================================================

import os
import ast
import inspect
import linecache
import sys
import torch
from .common import TEMPORARY_PATCHES, logger
from .utils import patch_function, raise_error

# Mirrors gemma.py: flex dispatch can be turned off globally.
_UNSLOTH_FLEX_ATTENTION_DISABLED = os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0"


def _unsloth_gemma4_ple_cast_input(module, x):
    """Cast at a learned PLE Linear boundary without touching the fp32 carrier."""
    get_base_layer = getattr(module, "get_base_layer", None)
    base_layer = get_base_layer() if callable(get_base_layer) else \
        getattr(module, "base_layer", module)
    weight = getattr(module, "weight", None)
    if weight is None:
        weight = getattr(base_layer, "weight", None)
    if weight is None:
        return x
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        return x
    dtype = getattr(weight, "dtype", None)
    if dtype is None or not getattr(dtype, "is_floating_point", False):
        return x
    # Skip sub-2-byte floats (e.g. float8_e4m3fn / float8_e5m2): casting the fp32
    # carrier down to fp8 destroys it, and fp8 matmul kernels expect a higher
    # precision input they scale internally. getattr keeps this a no-op on older
    # torch builds whose dtypes lack `itemsize`.
    if getattr(dtype, "itemsize", 2) < 2:
        return x
    return x if x.dtype == dtype else x.to(dtype)
pass


def _is_gemma4_ple_linear_call(node, attr, argument):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == argument
    )


def _is_gemma4_ple_fixed_call(node, attr, argument):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "_unsloth_gemma4_ple_cast_input"
        and len(node.args[0].args) == 2
        and isinstance(node.args[0].args[0], ast.Attribute)
        and node.args[0].args[0].attr == attr
        and isinstance(node.args[0].args[0].value, ast.Name)
        and node.args[0].args[0].value.id == "self"
        and isinstance(node.args[0].args[1], ast.Name)
        and node.args[0].args[1].id == argument
    )


def _gemma4_ple_validate(model_cls, method_name, targets):
    """Classify a PLE method WITHOUT mutating anything.

    Returns (status, payload) where status is one of:
      "ALREADY"     - marker set, or every boundary is already cast (safe no-op).
      "PATCH"       - payload carries what is needed to rewrite the method.
      "DRIFT"       - the method exists but its source shape is unexpected.
      "UNAVAILABLE" - the class / source cannot be introspected here.
    payload is None unless status == "PATCH".
    """
    marker = f"_unsloth_ple_dtype_{method_name}_patched"
    if getattr(model_cls, marker, False):
        return "ALREADY", None
    module = sys.modules.get(model_cls.__module__)
    source_file = inspect.getsourcefile(model_cls)
    if module is None or source_file is None:
        return "UNAVAILABLE", None
    module_source = inspect.getsource(module)
    tree = ast.parse(module_source, filename = source_file)
    class_nodes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == model_cls.__name__]
    if len(class_nodes) != 1:
        return "DRIFT", None
    method_nodes = [x for x in class_nodes[0].body if isinstance(x, ast.FunctionDef) and x.name == method_name]
    if len(method_nodes) != 1:
        return "DRIFT", None
    method_node = method_nodes[0]

    raw_calls = {}
    for attr, argument in targets:
        raw = [x for x in ast.walk(method_node) if _is_gemma4_ple_linear_call(x, attr, argument)]
        fixed = [x for x in ast.walk(method_node) if _is_gemma4_ple_fixed_call(x, attr, argument)]
        if not raw and len(fixed) == 1:
            raw_calls[attr] = []
            continue
        if len(raw) != 1 or fixed:
            return "DRIFT", None
        raw_calls[attr] = raw
    if all(not raw_calls[attr] for attr, _ in targets):
        return "ALREADY", None
    return "PATCH", (module, source_file, module_source, method_node, raw_calls)


def _patch_gemma4_ple_dtype_on_method(model_cls, method_name, targets, dry_run = False):
    status, payload = _gemma4_ple_validate(model_cls, method_name, targets)
    if dry_run:
        return status

    marker = f"_unsloth_ple_dtype_{method_name}_patched"
    if status == "ALREADY":
        # Covers both the marker-preset case and "every boundary already cast".
        setattr(model_cls, marker, True)
        return False
    if status in ("DRIFT", "UNAVAILABLE"):
        if status == "DRIFT":
            logger.warning(
                f"Unsloth: Gemma4 PLE dtype patch skipped for "
                f"`{model_cls.__name__}.{method_name}` - the expected source shape was "
                f"not found (transformers source drift). Forced-float32 runs may hit a "
                f"dtype mismatch here until the patch is updated."
            )
        return False

    module, source_file, module_source, method_node, raw_calls = payload
    patched_module_source = module_source
    for attr, argument in targets:
        if not raw_calls[attr]:
            continue
        old = ast.get_source_segment(module_source, raw_calls[attr][0])
        new = f"self.{attr}(_unsloth_gemma4_ple_cast_input(self.{attr}, {argument}))"
        if old is None or patched_module_source.count(old) != 1:
            return False
        patched_module_source = patched_module_source.replace(old, new, 1)

    class PLEInputCast(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            for attr, argument in targets:
                if _is_gemma4_ple_linear_call(node, attr, argument):
                    node.args[0] = ast.copy_location(
                        ast.Call(
                            func = ast.Name(id = "_unsloth_gemma4_ple_cast_input", ctx = ast.Load()),
                            args = [node.func, node.args[0]],
                            keywords = [],
                        ),
                        node.args[0],
                    )
            return node

    method_node.decorator_list = []
    method_node = PLEInputCast().visit(method_node)
    ast.fix_missing_locations(method_node)
    module.__dict__["_unsloth_gemma4_ple_cast_input"] = _unsloth_gemma4_ple_cast_input
    namespace = {}
    exec(compile(ast.Module(body = [method_node], type_ignores = []), source_file, "exec"), module.__dict__, namespace)
    original_body = inspect.unwrap(getattr(model_cls, method_name))
    patched_body = namespace[method_name]
    if original_body.__code__.co_freevars != patched_body.__code__.co_freevars:
        return False
    original_body.__code__ = patched_body.__code__
    linecache.cache[source_file] = (
        len(patched_module_source), None, patched_module_source.splitlines(keepends = True), source_file,
    )
    setattr(model_cls, marker, True)
    return True


def patch_Gemma4PLEInputDtype():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") != "1": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        Gemma4TextModel = transformers.models.gemma4.modeling_gemma4.Gemma4TextModel
        Gemma4TextDecoderLayer = transformers.models.gemma4.modeling_gemma4.Gemma4TextDecoderLayer
    except Exception as e:
        return raise_error("Gemma4 PLE Linear input dtype", e)
    specs = (
        (Gemma4TextModel, "project_per_layer_inputs",
            (("per_layer_model_projection", "inputs_embeds"),)),
        (Gemma4TextDecoderLayer, "forward",
            (("per_layer_input_gate", "hidden_states"), ("per_layer_projection", "hidden_states"))),
    )
    try:
        # Validate every boundary before mutating any, so a drift in one method can
        # never leave a half-patched model that still crashes at the other boundary.
        statuses = [
            ((cls, method), _patch_gemma4_ple_dtype_on_method(cls, method, targets, dry_run = True))
            for cls, method, targets in specs
        ]
        # Abort all boundaries if any cannot be safely patched (DRIFT) or introspected
        # (UNAVAILABLE), so a per-method failure can never leave a half-patched model.
        if any(status in ("DRIFT", "UNAVAILABLE") for _, status in statuses):
            for (cls, method), status in statuses:
                if status == "DRIFT":
                    logger.warning(
                        f"Unsloth: Gemma4 PLE dtype patch skipped for "
                        f"`{cls.__name__}.{method}` (transformers source drift); skipping "
                        f"all PLE boundaries to avoid a half-patched model."
                    )
            return
        for cls, method, targets in specs:
            _patch_gemma4_ple_dtype_on_method(cls, method, targets)
    except Exception as e:
        return raise_error("Gemma4 PLE Linear input dtype", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4PLEInputDtype)


def patch_Gemma4TextScaledWordEmbedding():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding
    except Exception as e:
        return raise_error("Gemma4TextScaledWordEmbedding.forward", e)

    def forward(self, input_ids: torch.Tensor):
        input_embeds = torch.nn.functional.embedding(
            input_ids,
            weight = self.weight,
            padding_idx = self.padding_idx,
        )
        # float32 residual stream (embed_scale ~ sqrt(hidden_size))
        return input_embeds.to(torch.float32) * self.embed_scale.to(torch.float32)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding, "forward", forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextScaledWordEmbedding)


def patch_Gemma4RMSNorm():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
    except Exception as e:
        return raise_error("Gemma4RMSNorm.forward", e)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # fp32 (residual) or fp16 (sub-layer)
        # Gemma4 scales by `weight` directly (no 1.0 + weight) and only when with_scale.
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed_fp32 = x_fp32 * torch.pow(variance + self.eps, -0.5)
        if self.with_scale:
            normed_fp32 = normed_fp32 * self.weight.to(torch.float32)

        # Clamp to fp16 range before casting back so a large residual never becomes inf.
        fp16_max = torch.finfo(torch.float16).max
        return torch.clamp(normed_fp32, min = -fp16_max, max = fp16_max).to(torch.float16)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm, "forward", forward, fullgraph = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4RMSNorm)


def patch_Gemma4TextAttention():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("Gemma4TextAttention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive = True)

    def forward(
        self,
        hidden_states,
        position_embeddings = None,
        attention_mask = None,
        shared_kv_states = None,
        past_key_values = None,
        **kwargs,
    ):
        # Shared-KV carrier (5.5.0-5.5.1, no native shared_kv_states); a no-op
        # for 31B / 26B-A4B where num_kv_shared_layers == 0 and no carrier is attached.
        carrier = getattr(self, "_unsloth_shared_kv_carrier", None)
        if carrier is not None and past_key_values is None:
            past_key_values = carrier

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        # q: fp16 proj -> fp16 q_norm -> float32 -> RoPE (float32) -> (b, h, q, d)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states).to(torch.float32)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim = 2)
        query_states = query_states.transpose(1, 2).contiguous()

        if self.is_kv_shared_layer and shared_kv_states is not None:
            # Native shared-KV (5.5.2+): dict keyed by layer_type, read even with a
            # cache (sliding caches forget past the window). Shared layers define no
            # k/v projections on these builds, so this branch must win.
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device, torch.float32)
            value_states = value_states.to(query_states.device, torch.float32)
        elif self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device, torch.float32)
            value_states = value_states.to(query_states.device, torch.float32)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            # attention_k_eq_v global layers have v_proj is None -> value shares the
            # raw k_proj output (before k_norm / RoPE rebind key_states below).
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states).to(torch.float32)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim = 2)
            key_states = key_states.transpose(1, 2).contiguous()

            value_states = self.v_norm(value_states).to(torch.float32)
            value_states = value_states.transpose(1, 2).contiguous()

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if getattr(self, "store_full_length_kv", False):
            if shared_kv_states is not None:
                # Native producer stores unconditionally (the dict is passed every call).
                shared_kv_states[self.layer_type] = key_states, value_states
            elif past_key_values is not None:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        # Core attention in float32. Gemma4 uses scaling = 1.0 (QK-norm bakes the scale in).
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa") or "sdpa"
        if attn_impl == "flex_attention" and _UNSLOTH_FLEX_ATTENTION_DISABLED:
            attn_impl = "sdpa"
        attention_interface = None
        if attn_impl not in ("sdpa", "eager"):
            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            except KeyError:
                attention_interface = None  # unknown backend -> raw SDPA below
        attn_weights = None

        if attn_impl == "eager":
            # Mirror upstream eager (additive mask, fp32 softmax) and return weights.
            g = getattr(self, "num_key_value_groups", 1) or 1
            k_r = key_states.repeat_interleave(g, dim = 1) if g != 1 else key_states
            v_r = value_states.repeat_interleave(g, dim = 1) if g != 1 else value_states
            attn_weights = torch.matmul(query_states, k_r.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                m = attention_mask[..., : k_r.shape[-2]]
                if m.dtype == torch.bool:
                    m = torch.zeros_like(attn_weights).masked_fill_(~m, torch.finfo(torch.float32).min)
                attn_weights = attn_weights + m.to(torch.float32)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim = -1)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p = self.attention_dropout if self.training else 0.0, training = self.training,
            )
            attn_output = torch.matmul(attn_weights, v_r).transpose(1, 2)
        elif attention_interface is not None:
            # Registered backend (flex / flash) owns the mask semantics; pass the
            # mask exactly as upstream prepared it (FA uses a 2-D padding mask,
            # flex a BlockMask). Flash kernels need fp16 inputs and accumulate in
            # float32 internally, so stability is preserved; flex handles fp32.
            q, k, v = query_states, key_states, value_states
            if attn_impl != "flex_attention":
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            attn_output, attn_weights = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout = self.attention_dropout if self.training else 0.0,
                scaling = self.scaling,
                sliding_window = getattr(self, "sliding_window", None),
                **kwargs,
            )  # returns (b, q, h, d), already transposed
        else:
            attn_mask = attention_mask
            if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(torch.float32)
            is_causal = attn_mask is None and query_states.shape[2] > 1 and getattr(self, "is_causal", True)
            attn_output = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask = attn_mask,
                dropout_p = self.attention_dropout if self.training else 0.0,
                is_causal = is_causal,
                scale = self.scaling,
                enable_gqa = getattr(self, "num_key_value_groups", 1) != 1,
            )
            attn_output = attn_output.transpose(1, 2)  # (b, h, q, d) -> (b, q, h, d)

        # (b, q, h, d) -> (b, q, h*d), down-cast to fp16 for o_proj.
        attn_output = attn_output.contiguous().reshape(*input_shape, -1)
        attn_output = attn_output.to(torch.float16)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass
    # force = True: gemma4.py's shared-KV carrier already wrapped forward as
    # (self, *args, **kwargs); a signature check would reject this replacement (capturing
    # the carrier's q/k-fp32-vs-v-fp16 SDPA mismatch). Carrier handled inline above, so safe.
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention, "forward", forward, force = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextAttention)
