"""mlx_lm.load — top-level entry that returns (model, tokenizer)."""

from __future__ import annotations


def load(repo_path, *args, **kwargs):
    """Phase 1: try transformers HF as the actual backend."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_path, **{
        k: v for k, v in kwargs.items() if k in ("trust_remote_code", "revision")
    })
    model = AutoModelForCausalLM.from_pretrained(repo_path, **{
        k: v for k, v in kwargs.items()
        if k in ("trust_remote_code", "revision", "torch_dtype")
    })
    return model, tokenizer
