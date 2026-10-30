"""Offline guard that image/video/audio placeholder tokens of the vision and omni families in unslothai/notebooks stay
in vlm_tokens for the CUDA collator and MLX loss mask; live per-model id check is in the network sweep
(test_notebook_chat_templates.py / manual)."""
import os, sys, importlib.util

# Load vlm_tokens.py by file path: importing unsloth_zoo runs __init__, which raises ImportError without the unsloth package; the file is pure data.
def _load_vlm_tokens():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "unsloth_zoo", "vlm_tokens.py")
    spec = importlib.util.spec_from_file_location("unsloth_zoo_vlm_tokens_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_vt = _load_vlm_tokens()
IMAGE_TOKENS, AUDIO_TOKENS, VLM_PLACEHOLDER_TOKENS = (
    _vt.IMAGE_TOKENS, _vt.AUDIO_TOKENS, _vt.VLM_PLACEHOLDER_TOKENS,
)

COVERED = set(IMAGE_TOKENS) | set(AUDIO_TOKENS)

# family -> media placeholder tokens that must never train; verified against the actual tokenizers and configs of these models.
EXPECTED = {
    "Llama 3.2 Vision / Phi 3.5": ["<|image|>"],
    "Qwen2-VL / Qwen2.5-VL": ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"],
    "Qwen2.5-Omni": ["<|IMAGE|>", "<|VIDEO|>", "<|AUDIO|>", "<|vision_bos|>", "<|vision_eos|>",
                     "<|vision_pad|>", "<|audio_bos|>", "<|audio_eos|>"],
    "PaliGemma / Llava / InternVL": ["<image>"],
    "InternVL / Nemotron Nano Omni (image)": ["<img>", "</img>", "<IMG_CONTEXT>", "<video>"],
    "Nemotron Nano Omni (sound)": ["<so_embedding>", "<so_start>", "<so_end>"],
    "Pixtral / Mistral": ["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
    "Gemma 3 / 3n (image)": ["<image_soft_token>", "<start_of_image>", "<end_of_image>"],
    "Gemma 3n (audio)": ["<audio_soft_token>", "<start_of_audio>", "<end_of_audio>"],
    "Gemma 4 (image/video)": ["<|image|>", "<|image>", "<image|>", "<|video|>"],
    "Gemma 4 (audio)": ["<|audio|>", "<|audio>", "<audio|>"],
    "Cohere Aya Vision": ["<|START_OF_IMG|>", "<|END_OF_IMG|>", "<|IMG_LINE_BREAK|>", "<|IMG_PATCH|>"],
}


def test_expected_media_tokens_are_covered():
    missing = {fam: [t for t in toks if t not in COVERED] for fam, toks in EXPECTED.items()}
    missing = {fam: m for fam, m in missing.items() if m}
    assert not missing, f"media tokens missing from vlm_tokens: {missing}"


def test_lists_are_consistent():
    assert VLM_PLACEHOLDER_TOKENS == IMAGE_TOKENS + AUDIO_TOKENS
    assert len(IMAGE_TOKENS) == len(set(IMAGE_TOKENS)), "duplicate image tokens"
    assert len(AUDIO_TOKENS) == len(set(AUDIO_TOKENS)), "duplicate audio tokens"


def test_mlx_mirror_matches():
    # The MLX path must source the same strings (no separate hand-maintained copy).
    import ast
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "unsloth_zoo", "mlx", "utils.py")
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_IMAGE_TOKEN_STRINGS" for t in node.targets):
            assert not isinstance(node.value, (ast.Tuple, ast.List)), \
                "_IMAGE_TOKEN_STRINGS should derive from vlm_tokens, not hardcode a list"
            return
    raise AssertionError("_IMAGE_TOKEN_STRINGS not found in mlx/utils.py")


if __name__ == "__main__":
    test_expected_media_tokens_are_covered()
    test_lists_are_consistent()
    test_mlx_mirror_matches()
    print("OK:", len(COVERED), "covered tokens across", len(EXPECTED), "families")
