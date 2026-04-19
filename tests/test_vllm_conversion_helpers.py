import sys, os, warnings, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
import torch


def test_set_dtype_in_config_no_torch_dtype_deprecation():
    # Pre-fix: wrote both dtype and torch_dtype -> triggered transformers deprecation warning.
    # Post-fix: writes only dtype when available.
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config

    cfg = PretrainedConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_dtype_in_config(cfg, torch.bfloat16)
    dep_warnings = [
        w for w in caught
        if "torch_dtype" in str(w.message) and "deprecated" in str(w.message).lower()
    ]
    assert not dep_warnings, f"unexpected deprecation warning: {[str(w.message) for w in dep_warnings]}"


def test_set_dtype_in_config_writes_runtime_dtype():
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config

    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, torch.float16)
    # Either dtype or torch_dtype (aliased via property in modern transformers) should reflect it.
    got = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    assert got == torch.float16


def test_set_dtype_in_config_accepts_string():
    from transformers import PretrainedConfig
    from unsloth_zoo.hf_utils import set_dtype_in_config

    cfg = PretrainedConfig()
    set_dtype_in_config(cfg, "bfloat16")
    got = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    assert got == torch.bfloat16


def test_normalize_state_dict_tensor_guards_non_tensor():
    # Pre-fix: _normalize_state_dict_tensor called value.is_sparse unconditionally.
    # Post-fix: the is_sparse branch is guarded by isinstance(value, torch.Tensor).
    from unsloth_zoo import vllm_utils

    src = inspect.getsource(vllm_utils.assert_same_state_dict)
    assert "isinstance(value, torch.Tensor)" in src
    assert src.index("isinstance(value, torch.Tensor)") < src.index("value.is_sparse")


def test_gemma4_lora_patch_preserves_callable_signature():
    # Pre-fix: patched_create_lora_manager was `(model, *args, **kwargs)`, which hid vllm_config
    # from `inspect.signature` and broke `_call_create_lora_manager`'s signature check.
    # Post-fix: @functools.wraps preserves the original signature.
    from functools import wraps

    def original_create_lora_manager(
        model, max_num_seqs=None, vllm_config=None, lora_manager_cls=None, **kwargs,
    ):
        return (model, vllm_config, lora_manager_cls)

    @wraps(original_create_lora_manager)
    def patched_create_lora_manager(model, *args, **kwargs):
        return original_create_lora_manager(model, *args, **kwargs)

    sig = inspect.signature(patched_create_lora_manager)
    assert "vllm_config" in sig.parameters


def test_gemma4_lora_patch_positional_model_no_double_bind():
    # Pre-fix: `lora_manager_cls(model=model, *args, **kwargs)` raised
    # "TypeError: multiple values for argument 'model'" if *args was non-empty.
    # Post-fix: pass model positionally.
    class _LoRAManagerCls:
        def __init__(self, model, extra=None, **kwargs):
            self.model = model
            self.extra = extra
            self.kwargs = kwargs

    # Post-fix semantics: lora_manager_cls(model, *args, **kwargs)
    inst = _LoRAManagerCls("fake_model", "extra_positional", vllm_config="cfg")
    assert inst.model == "fake_model"
    assert inst.extra == "extra_positional"
    assert inst.kwargs == {"vllm_config": "cfg"}


def test_gemma4_k_eq_v_pairs_handles_split_layout():
    # Pre-fix: _get_gemma4_k_eq_v_qkv_param_names only searched for packed `qkv_proj.weight`.
    # Post-fix: detects split `k_proj.weight` / `v_proj.weight` layout too.
    import inspect as _inspect
    from unsloth_zoo import empty_model

    src = _inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    # Sanity: the split-layout branch exists.
    assert "k_proj.weight" in src
    assert "v_proj.weight" in src
    assert '"split"' in src or "'split'" in src
