"""MLX train_on_responses_only must auto-detect markers from the RIGHT template.

The MLX wrapper (unsloth_zoo/mlx/trainer.py) resolves trainer.tokenizer to a
callable HF tokenizer before handing it to the shared auto-detect helper. Two
MLX-specific hazards follow, both fixed by _resolve_autodetect_template_source:

  1. VLM processor-only templates. Many VLM processors keep the chat template on
     the processor, not the inner tokenizer. Unwrapping to the inner tokenizer
     first makes auto-detect raise "No chat_template" even though the processor
     could render. Detection must see the processor; the HF helper unwraps to the
     inner tokenizer for token matching on its own.

  2. Configured chat_template override. MLXTrainingConfig.chat_template is only
     applied later during batch creation, so auto-detecting from trainer.tokenizer
     first reads the OLD template. Markers must be detected from the override, or
     they will not match the batches the trainer renders with it.

Hermetic: builds synthetic templates on a tiny tokenizer already cached locally
and runs under the mlx-on-torch simulation shim. Skips (never fails) when no
tokenizer is cached offline or when unsloth_zoo is not importable, so a default
pytest run never reaches the network and never breaks collection.
"""
import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    # Lets unsloth_zoo.mlx.trainer import on Linux+CUDA (treats itself as Apple).
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ChatML vs Mistral-INST: two templates with unmistakably different markers.
CHATML = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '<|im_start|>user\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- else %}{{ '<|im_start|>assistant\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
)
INST = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '[INST]' + m['content'] + '[/INST]' }}"
    "{%- else %}{{ m['content'] + '</s>' }}"
    "{%- endif %}{%- endfor %}"
)


def _new_tok():
    """A small tokenizer cached offline, with ChatML-ish specials; None if absent."""
    from transformers import AutoTokenizer
    for repo in ("hf-internal-testing/llama-tokenizer", "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        except Exception:
            continue
        tok.add_special_tokens({"additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]"]})
        return tok
    return None


def _load():
    """Import the MLX trainer + shared auto-detect helper, skipping if unavailable."""
    if _new_tok() is None:
        pytest.skip("no tokenizer cached offline")
    try:
        import unsloth_zoo.mlx.trainer as T
        from unsloth_zoo.dataset_utils import get_chat_template_parts
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    return T, get_chat_template_parts


class _Args:
    def __init__(self, chat_template=None, vlm_chat_template=None):
        self.chat_template = chat_template
        self.vlm_chat_template = vlm_chat_template


class _Model:
    _hf_repo = "unsloth/fake"
    _config = {"model_type": "fake"}


class _Trainer:
    def __init__(self, tokenizer, args, is_vlm=False, processor=None):
        self.tokenizer = tokenizer
        self.args = args
        self._is_vlm = is_vlm
        self.processor = processor
        self.model = _Model()


class _ProcessorOnly:
    """VLM processor whose chat template lives ONLY on the processor: the inner
    tokenizer has no template, and apply_chat_template renders via the processor's."""
    def __init__(self, inner_tok, render_tok):
        self.tokenizer = inner_tok        # inner: chat_template is None
        self.image_processor = object()   # marks this as a VLM processor
        self.chat_template = render_tok.chat_template
        self._render_tok = render_tok

    def apply_chat_template(self, messages, **kwargs):
        return self._render_tok.apply_chat_template(messages, **kwargs)


def test_text_override_detects_from_configured_template():
    """Issue 2: with args.chat_template set, detect markers from the override."""
    T, get_parts = _load()

    native = _new_tok(); native.chat_template = CHATML
    override = _new_tok(); override.chat_template = INST
    native_markers = get_parts(native)
    override_markers = get_parts(override)
    assert native_markers != override_markers, (native_markers, override_markers)

    # No override -> resolver leaves the tokenizer's own template in force.
    tok0 = _new_tok(); tok0.chat_template = CHATML
    tr0 = _Trainer(tok0, _Args(chat_template=None))
    resolved0 = T._resolve_response_mask_tokenizer(tok0)
    detect0 = T._resolve_autodetect_template_source(tr0, tok0, resolved0)
    assert get_parts(detect0) == native_markers

    # Override set -> resolver applies it, so detection reflects the override.
    tok1 = _new_tok(); tok1.chat_template = CHATML
    tr1 = _Trainer(tok1, _Args(chat_template=INST))
    resolved1 = T._resolve_response_mask_tokenizer(tok1)
    detect1 = T._resolve_autodetect_template_source(tr1, tok1, resolved1)
    assert get_parts(detect1) == override_markers


def test_vlm_processor_only_template_is_used_for_detection():
    """Issue 1: a processor-only template must drive auto-detection, not the
    unwrapped inner tokenizer (which would raise 'No chat_template')."""
    T, get_parts = _load()

    inner = _new_tok(); inner.chat_template = None
    render = _new_tok(); render.chat_template = INST
    proc = _ProcessorOnly(inner, render)
    expected = get_parts(proc)  # e.g. ('[INST]', '[/INST]')

    # OLD behavior: unwrapping to the inner tokenizer loses the template.
    resolved = T._resolve_response_mask_tokenizer(proc)
    assert resolved is inner
    with pytest.raises(Exception):
        get_parts(resolved)

    # NEW behavior: the resolver hands back the processor for detection.
    tr = _Trainer(proc, _Args(), is_vlm=True, processor=proc)
    detect = T._resolve_autodetect_template_source(tr, proc, resolved)
    assert detect is proc
    assert get_parts(detect) == expected

    # And the full wrapper builds a mask function without raising.
    fn = T.train_on_responses_only(tr, return_function=True)
    assert callable(fn)


def test_explicit_markers_bypass_template_resolution():
    """Explicit instruction/response parts need no template; the wrapper must not
    require a chat_template and must return a working mask function."""
    T, _ = _load()
    inner = _new_tok(); inner.chat_template = None
    proc = _ProcessorOnly(inner, _new_tok())  # render tok irrelevant here
    proc.chat_template = None                  # no template anywhere
    tr = _Trainer(proc, _Args(), is_vlm=True, processor=proc)
    fn = T.train_on_responses_only(
        tr, instruction_part="[INST]", response_part="[/INST]",
        return_function=True,
    )
    assert callable(fn)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
