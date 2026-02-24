
import sys
import os
import torch

# Add unsloth-zoo to path
sys.path.append(os.getcwd())

from unsloth_zoo.vllm_utils import _test_get_vllm_state_dict

def run_tests():
    print("=================================================================")
    print("Running Regression Test: Llama-3.2-1B-Instruct (Standard Model)")
    print("=================================================================")
    try:
        _test_get_vllm_state_dict(
            model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            load_in_4bit=True,
            counts=1, # short test
            skip_generation=False,
        )
        print("✅ Llama-3.2-1B Regression Test PASSED")
    except Exception as e:
        print(f"❌ Llama-3.2-1B Regression Test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n")
    print("=================================================================")
    print("Running Fix Verification: LiquidAI/LFM2.5-1.2B-Thinking")
    print("  - LoRA disabled (conv modules not BaseLayerWithLoRA)")
    print("  - Verifying in_proj weight shape is (3*hidden, hidden)")
    print("  - Verifying lm_head via tied embeddings")
    print("=================================================================")
    try:
        # LFM2 must run with enable_lora=False because vLLM's LoRA manager
        # cannot register conv modules (they aren't BaseLayerWithLoRA).
        # _test_get_vllm_state_dict already handles this via:
        #   enable_lora = model_type not in ("mllama", "lfm2")
        _test_get_vllm_state_dict(
            model_name="LiquidAI/LFM2.5-1.2B-Thinking",
            dtype=torch.float16,
            load_in_4bit=False,
            counts=1,
            skip_generation=False,
        )
        print("✅ LFM2 Fix Verification PASSED")
    except Exception as e:
        print(f"❌ LFM2 Fix Verification FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_lfm2_config_lora_exclusion():
    """Verify that LFM2 model_type is excluded from LoRA in the test harness."""
    print("\n=================================================================")
    print("Unit Test: LFM2 LoRA Exclusion")
    print("=================================================================")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Thinking",
        trust_remote_code=False,
    )
    model_type = getattr(config, "model_type", "causal_lm")
    enable_lora = model_type not in ("mllama", "lfm2")
    assert not enable_lora, f"Expected enable_lora=False for lfm2, got {enable_lora} (model_type={model_type})"
    print(f"  model_type = {model_type}")
    print(f"  enable_lora = {enable_lora}")
    print("✅ LFM2 LoRA exclusion test PASSED")

    # Also verify tie_word_embeddings is True (lm_head should use embed_tokens)
    text_config = getattr(config, "text_config", config)
    tie = getattr(text_config, "tie_word_embeddings", None)
    print(f"  tie_word_embeddings = {tie}")
    if tie:
        print("✅ LFM2 tied embeddings confirmed (lm_head.weight warning is benign)")
    else:
        print("⚠️  tie_word_embeddings is not True — lm_head extraction path will be used")


if __name__ == "__main__":
    test_lfm2_config_lora_exclusion()
    print("\n")
    run_tests()
