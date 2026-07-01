"""End-to-end: train_on_responses_only must actually mask for vision models.

For one representative model per marker style this builds a REAL
UnslothVisionDataCollator and runs a real image+text example through __call__,
two ways, asserting the labels are content-exact (assistant response trained, user
text masked, image/pad tokens masked):
  (A) constructor:  UnslothVisionDataCollator(..., train_on_responses_only=True, parts)
  (B) in-place:     bare UnslothVisionDataCollator(model, processor), then
                    train_on_responses_only(trainer) enables masking on the collator.
Path (B) is the case where a user keeps the default vision collator and calls
train_on_responses_only(trainer) separately (most vision notebooks build the bare
collator).

Network/integration test, OFF by default. Enable with
    UNSLOTH_TEST_VLM_COLLATOR=1 pytest tests/test_vlm_collator_masking.py
Processors are fetched config-only; set UNSLOTH_VLM_PROC_CACHE to reuse a cache dir.
Models that need a newer transformers to load are skipped, not failed.
"""
import os, sys, types, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# NOTE: unsloth_zoo is imported lazily inside _check (not at module top). Importing it
# runs unsloth_zoo/__init__.py, which raises ImportError when the separate `unsloth`
# package is absent; at module scope that would break pytest collection before the
# "OFF by default" skip gate in test_vlm_collator_masking() can run.

ENABLED = os.environ.get("UNSLOTH_TEST_VLM_COLLATOR", "") not in ("", "0", "false")
CACHE = os.environ.get("UNSLOTH_VLM_PROC_CACHE", os.path.join(tempfile.gettempdir(), "unsloth_vlm_proc"))
ALLOW = ["*.json", "*.jinja", "*.model", "*.txt", "tokenizer*", "merges*", "vocab*"]

# one model per marker style: ChatML, Gemma-3-turn, Gemma-4-turn, Llama-header,
# Mistral-[INST]. The Gemma-4 / Qwen3.5 entries need a newer transformers and skip
# (not fail) where it is too old to load them.
MODELS = [
    ("unsloth/Qwen2-VL-2B-Instruct", "<|im_start|>user\n", "<|im_start|>assistant\n"),
    ("unsloth/Qwen3.5-2B", "<|im_start|>user\n", "<|im_start|>assistant\n"),
    ("unsloth/gemma-3-4b-it", "<start_of_turn>user\n", "<start_of_turn>model\n"),
    ("unsloth/gemma-3n-E4B-it", "<start_of_turn>user\n", "<start_of_turn>model\n"),
    ("unsloth/gemma-4-E2B-it", "<|turn>user\n", "<|turn>model\n"),
    ("unsloth/Llama-3.2-11B-Vision-Instruct",
     "<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("unsloth/Pixtral-12B-2409", "[INST]", "[/INST]"),
]
USER_TXT, ASST_TXT = "ZEBRAQUESTION", "PENGUINANSWER"


def _fetch(repo):
    path = os.path.join(CACHE, repo.replace("/", "__"))
    cached = os.path.isdir(path) and any(f.startswith("tokenizer") for f in os.listdir(path))
    if not cached:
        # Downloads are opt-in. Without UNSLOTH_TEST_VLM_COLLATOR an uncached repo is
        # skipped (raise -> _check returns SKIP), never fetched, even when CACHE already
        # exists from a prior partial run - so the sweep stays truly off by default.
        if not ENABLED:
            raise RuntimeError(f"{repo} not cached; set UNSLOTH_TEST_VLM_COLLATOR=1 to download")
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=path, allow_patterns=ALLOW, token=os.environ.get("HF_TOKEN"))
    return path


def _mock_model(config):
    import torch
    vc = getattr(config, "vision_config", None)
    ns = types.SimpleNamespace
    vcfg = ns(patch_size=getattr(vc, "patch_size", 14) if vc is not None else 14,
              image_size=getattr(vc, "image_size", 336) if vc is not None else 336,
              model_type=getattr(vc, "model_type", "") if vc is not None else "")
    return ns(config=ns(torch_dtype="float16", vision_config=vcfg), max_seq_length=2048,
              get_input_embeddings=lambda: ns(weight=ns(dtype=torch.float16)))


def _example():
    from PIL import Image
    img = Image.new("RGB", (280, 280), (120, 130, 140))
    return {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TXT}]},
        {"role": "assistant", "content": [{"type": "text", "text": ASST_TXT}]}], "images": [img]}


def _content_exact(collator, proc):
    batch = collator([_example()])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    trained = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    img_ids = set(collator.padding_token_ids.tolist())
    img_leak = any((i in img_ids) and (l != -100) for i, l in zip(ids, labels))
    return (ASST_TXT in trained) and (USER_TXT not in trained) and not img_leak


def _check(repo, ins, res):
    from transformers import AutoProcessor, AutoConfig
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    from unsloth_zoo.dataset_utils import train_on_responses_only
    try:
        path = _fetch(repo)
        proc = AutoProcessor.from_pretrained(path)
        model = _mock_model(AutoConfig.from_pretrained(path))
    except Exception as e:
        return "SKIP", f"{type(e).__name__}"
    if not hasattr(proc, "image_processor"):
        return "SKIP", "no image_processor"
    # (A) constructor
    cA = UnslothVisionDataCollator(model, proc, train_on_responses_only=True,
                                   instruction_part=ins, response_part=res)
    if not _content_exact(cA, proc):
        return "FAIL", "constructor masking wrong"
    # (B) in-place
    cB = UnslothVisionDataCollator(model, proc)
    if cB.train_on_responses_only is not None:
        return "FAIL", "bare collator already had masking"
    trainer = types.SimpleNamespace(processing_class=proc, data_collator=cB,
                                    train_dataset=None, eval_dataset=None,
                                    args=types.SimpleNamespace(packing=False, max_length=2048))
    train_on_responses_only(trainer, instruction_part=ins, response_part=res)
    if not callable(getattr(cB, "train_on_responses_only", None)):
        return "FAIL", "train_on_responses_only(trainer) did not configure the collator"
    if not _content_exact(cB, proc):
        return "FAIL", "in-place masking wrong"
    return "PASS", ""


def run():
    out = {}
    for repo, ins, res in MODELS:
        cat, why = _check(repo, ins, res)
        out[repo] = (cat, why)
        print(f"{repo:42s} {cat}  {why}", flush=True)
    return out


def test_vlm_collator_masking():
    import pytest
    if not ENABLED and not os.path.isdir(CACHE):
        pytest.skip("set UNSLOTH_TEST_VLM_COLLATOR=1 (network) to run the vision collator e2e")
    out = run()
    checked = sum(1 for c, _ in out.values() if c in ("PASS", "FAIL"))
    if checked == 0:
        pytest.skip(f"no vision processors were loadable ({out})")
    fails = {r: w for r, (c, w) in out.items() if c == "FAIL"}
    assert not fails, f"vision masking failed: {fails}"


if __name__ == "__main__":
    out = run()
    fails = [r for r, (c, _) in out.items() if c == "FAIL"]
    print("\nFAILS:", fails or "none")
    sys.exit(1 if fails else 0)
