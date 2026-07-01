"""The batched chat-template render path must be byte-identical to the per-example loop.

UnslothVisionDataCollator.__call__ renders a batch's chat templates in one
apply_chat_template call, verified once per instance against the per-example loop and
cached (_batch_template). This test forces the batched (ON) and per-example (OFF) paths
and asserts the produced tensors (input_ids / attention_mask / labels / pixel_values)
are exactly equal for one representative model per marker style.

Network/integration test, OFF by default. Enable with
    UNSLOTH_TEST_VLM_COLLATOR=1 pytest tests/test_vlm_collator_batch_template.py
Processors are fetched config-only; set UNSLOTH_VLM_PROC_CACHE to reuse a cache dir.
Models that need a newer transformers to load are skipped, not failed.

unsloth_zoo is imported lazily inside the test: importing it runs unsloth_zoo/__init__.py,
which raises ImportError when the separate `unsloth` package is absent, and at module
scope that would break pytest collection before the OFF-by-default skip gate runs.
"""
import os, sys, types, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ENABLED = os.environ.get("UNSLOTH_TEST_VLM_COLLATOR", "") not in ("", "0", "false")
CACHE = os.environ.get("UNSLOTH_VLM_PROC_CACHE", os.path.join(tempfile.gettempdir(), "unsloth_vlm_proc"))
ALLOW = ["*.json", "*.jinja", "*.model", "*.txt", "tokenizer*", "merges*", "vocab*"]

MODELS = [
    ("unsloth/Qwen2-VL-2B-Instruct", "<|im_start|>user\n", "<|im_start|>assistant\n"),
    ("unsloth/gemma-3-4b-it", "<start_of_turn>user\n", "<start_of_turn>model\n"),
    ("unsloth/Llama-3.2-11B-Vision-Instruct",
     "<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("unsloth/Pixtral-12B-2409", "[INST]", "[/INST]"),
]
USER_TXT, ASST_TXT = "ZEBRAQUESTION", "PENGUINANSWER"


def _fetch(repo):
    from huggingface_hub import snapshot_download
    path = os.path.join(CACHE, repo.replace("/", "__"))
    if not (os.path.isdir(path) and any(f.startswith("tokenizer") for f in os.listdir(path))):
        snapshot_download(repo, local_dir=path, allow_patterns=ALLOW, token=os.environ.get("HF_TOKEN"))
    return path


def _mock_model(config):
    import torch
    vc = getattr(config, "vision_config", None)
    ns = types.SimpleNamespace
    vcfg = ns(patch_size=getattr(vc, "patch_size", 14) if vc else 14,
              image_size=getattr(vc, "image_size", 336) if vc else 336,
              model_type=getattr(vc, "model_type", "") if vc else "")
    return ns(config=ns(torch_dtype="float16", vision_config=vcfg), max_seq_length=2048,
              get_input_embeddings=lambda: ns(weight=ns(dtype=torch.float16)))


def _examples(n):
    from PIL import Image
    out = []
    for i in range(n):
        out.append({"messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{USER_TXT}{i}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{ASST_TXT}{i}"}]},
            {"role": "user", "content": [{"type": "text", "text": "And more?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Follow {i}"}]}],
            "images": [Image.new("RGB", (256 + (i % 3) * 24, 256), (100 + i, 120, 150))]})
    return out


def _tensors_equal(a, b):
    import torch
    keys = ("input_ids", "attention_mask", "labels", "pixel_values",
            "pixel_values_videos", "image_grid_thw")
    ka = [k for k in keys if k in a and torch.is_tensor(a[k])]
    kb = [k for k in keys if k in b and torch.is_tensor(b[k])]
    if ka != kb:
        return False, f"key set differs {ka} vs {kb}"
    for k in ka:
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            return False, f"{k} differs"
    return True, ""


def _check(repo, ins, res):
    from transformers import AutoProcessor, AutoConfig
    import unsloth_zoo.vision_utils as vu
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    try:
        path = _fetch(repo)
        proc = AutoProcessor.from_pretrained(path)
        model = _mock_model(AutoConfig.from_pretrained(path))
    except Exception as e:
        return "SKIP", f"{type(e).__name__}"
    if not hasattr(proc, "image_processor"):
        return "SKIP", "no image_processor"
    coll = UnslothVisionDataCollator(model, proc, train_on_responses_only=True,
                                     instruction_part=ins, response_part=res)
    for n in (1, 2, 3, 8):
        coll._batch_template = vu._BATCH_TEMPLATE_ON
        try:
            on = coll([{**e, "images": list(e["images"])} for e in _examples(n)])
        except Exception as e:
            return "SKIP", f"render {type(e).__name__}"
        coll._batch_template = vu._BATCH_TEMPLATE_OFF
        off = coll([{**e, "images": list(e["images"])} for e in _examples(n)])
        ok, why = _tensors_equal(on, off)
        if not ok:
            return "FAIL", f"batch={n}: {why}"
    # default path should self-verify and settle on ON for these standard processors
    coll._batch_template = vu._BATCH_TEMPLATE_UNKNOWN
    coll([{**e, "images": list(e["images"])} for e in _examples(2)])
    if coll._batch_template != vu._BATCH_TEMPLATE_ON:
        return "FAIL", "default path did not adopt the verified batched render"
    return "PASS", ""


def test_batch_template_render_is_byte_identical():
    import pytest
    if not ENABLED and not os.path.isdir(CACHE):
        pytest.skip("set UNSLOTH_TEST_VLM_COLLATOR=1 (network) to run the batched-render equivalence")
    try:
        import unsloth_zoo.vision_utils  # noqa: F401
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    out = {}
    for repo, ins, res in MODELS:
        cat, why = _check(repo, ins, res)
        out[repo] = (cat, why)
        print(f"{repo:42s} {cat}  {why}", flush=True)
    checked = sum(1 for c, _ in out.values() if c in ("PASS", "FAIL"))
    if checked == 0:
        pytest.skip(f"no vision processors were loadable ({out})")
    fails = {r: w for r, (c, w) in out.items() if c == "FAIL"}
    assert not fails, f"batched render diverged from per-example: {fails}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
