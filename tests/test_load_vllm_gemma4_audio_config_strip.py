import inspect
from copy import deepcopy

from unsloth_zoo import vllm_utils


def _gate_body():
    src = inspect.getsource(vllm_utils.load_vllm)
    lines = src.splitlines()
    gate_idx = next(
        i for i, line in enumerate(lines)
        if '_outer_model_type == "gemma4" and getattr(config, "audio_config"' in line
    )
    gate_indent = len(lines[gate_idx]) - len(lines[gate_idx].lstrip())
    body = []
    for line in lines[gate_idx + 1:]:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent <= gate_indent:
            break
        body.append(stripped)
    return body


def test_audio_gate_deepcopies_then_strips_audio_config():
    body = _gate_body()
    dc_idx = next(
        (i for i, line in enumerate(body) if "config = deepcopy(config)" in line),
        None,
    )
    strip_idx = next(
        (i for i, line in enumerate(body) if "config.audio_config = None" in line),
        None,
    )
    assert dc_idx is not None, (
        "Gemma4 audio gate must rebind config to a deepcopy before mutating audio_config"
    )
    assert strip_idx is not None, (
        "Gemma4 audio gate must strip audio_config so the rebuilt HF model does not "
        "instantiate a silently-uninitialized audio_tower"
    )
    assert dc_idx < strip_idx, (
        "deepcopy must precede audio_config=None; otherwise the caller's config is mutated in place"
    )


def test_simulated_strip_preserves_original_config():
    class _AudioCfg:
        pass

    class _Cfg:
        def __init__(self):
            self.model_type = "gemma4"
            self.audio_config = _AudioCfg()

    original = _Cfg()
    original_audio = original.audio_config

    if getattr(original, "audio_config", None) is not None:
        local = deepcopy(original)
        local.audio_config = None

    assert original.audio_config is original_audio, (
        "load_vllm must not mutate the caller's config; deepcopy is required before clearing audio_config"
    )
    assert local.audio_config is None
    assert local is not original
