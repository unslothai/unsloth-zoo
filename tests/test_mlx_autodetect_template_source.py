"""_resolve_autodetect_template_source must auto-detect markers from the right template:
the VLM processor (whose template the inner tokenizer lacks) and after any configured
chat_template override, so markers match the rendered batches.

Hermetic: synthetic templates on a tiny offline tokenizer, under the mlx-on-torch shim;
skips (never fails) when the tokenizer cache or unsloth_zoo is unavailable.
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


def _new_tok_with(template):
    tok = _new_tok(); tok.chat_template = template; return tok


def _new_tok_none():
    tok = _new_tok(); tok.chat_template = None; return tok


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
    """VLM processor whose chat template lives only on the processor (inner tokenizer has none)."""
    def __init__(self, inner_tok, render_tok):
        self.tokenizer = inner_tok
        self.image_processor = object()
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


def test_vlm_explicit_processor_override_wins_over_trainer_processor():
    """An explicit tokenizer=/processor override must drive detection over
    trainer.processor, matching the wrapper's kwarg precedence."""
    T, get_parts = _load()

    # trainer.processor renders ChatML; the caller overrides with an INST processor.
    trainer_proc = _ProcessorOnly(_new_tok_none(), _new_tok_with(CHATML))
    override_proc = _ProcessorOnly(_new_tok_none(), _new_tok_with(INST))
    tr = _Trainer(trainer_proc, _Args(), is_vlm=True, processor=trainer_proc)

    resolved = T._resolve_response_mask_tokenizer(override_proc)
    detect = T._resolve_autodetect_template_source(tr, override_proc, resolved)
    assert detect is override_proc
    assert get_parts(detect) == get_parts(override_proc)  # INST markers, not ChatML


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


MISTRAL = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'user' %}{{ '[INST] ' + m['content'] + ' [/INST]' }}"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] + eos_token }}{% endif %}{% endfor %}"
)


def test_headerless_template_marker_excludes_eos():
    """A headerless template (add_generation_prompt is a no-op) must reach the fallback and
    detect [INST] without gluing the preceding eos into instruction_part."""
    _, get_parts = _load()
    tok = _new_tok(); tok.chat_template = MISTRAL
    ins, res = get_parts(tok)
    assert tok.eos_token not in ins, (ins, res)
    assert ins.strip() == "[INST]" and res.strip() == "[/INST]"


def test_chat_template_override_clears_stale_cached_markers():
    """A chat_template override must drop markers cached from a prior Unsloth template so the
    HF helper re-detects, instead of masking with the old markers."""
    T, _ = _load()
    tok = _new_tok(); tok.chat_template = CHATML
    tok._unsloth_input_part = "OLD_USER"; tok._unsloth_output_part = "OLD_ASST"
    tr = _Trainer(tok, _Args(chat_template=INST))
    resolved = T._resolve_response_mask_tokenizer(tok)
    out = T._resolve_autodetect_template_source(tr, tok, resolved)
    assert not hasattr(out, "_unsloth_input_part")
    assert not hasattr(out, "_unsloth_output_part")


def test_processor_template_preferred_over_inner_when_both_present():
    """When both the processor and its inner tokenizer carry a chat_template, detection must
    render with the processor's (what VLM batching uses), not the inner tokenizer's."""
    _, get_parts = _load()
    proc = _ProcessorOnly(_new_tok_with(CHATML), _new_tok_with(INST))  # inner=ChatML, render=INST
    proc.tokenizer.chat_template = CHATML   # inner tokenizer ALSO has a (different) template
    assert get_parts(proc) == get_parts(_new_tok_with(INST))  # processor (INST) wins


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
