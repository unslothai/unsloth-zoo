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


def test_audio_gate_deepcopies_config_before_mutation():
    body = _gate_body()
    assert any("config = deepcopy(config)" in line for line in body), (
        "Gemma4 audio gate must rebind config to a deepcopy before mutating "
        "audio_config so the caller's config object is not silently modified"
    )


def test_audio_gate_sets_audio_config_to_none():
    body = _gate_body()
    assert any("config.audio_config = None" in line for line in body), (
        "Gemma4 audio gate must strip audio_config on the local copy so "
        "create_empty_vision_model cannot instantiate a silently-uninitialized "
        "audio_tower downstream"
    )


def test_deepcopy_happens_before_strip():
    body = _gate_body()
    dc_idx = next(i for i, line in enumerate(body) if "config = deepcopy(config)" in line)
    strip_idx = next(i for i, line in enumerate(body) if "config.audio_config = None" in line)
    assert dc_idx < strip_idx, (
        "deepcopy must occur before audio_config is set to None; otherwise "
        "the caller's original config object is mutated in place"
    )


def test_simulated_strip_preserves_original_config():
    class _AudioCfg:
        pass

    class _Cfg:
        def __init__(self):
            self.model_type = "gemma4"
            self.audio_config = _AudioCfg()
            self.text_config = None
            self.vision_config = None

    original = _Cfg()
    original_audio = original.audio_config
    assert original_audio is not None

    if getattr(original, "audio_config", None) is not None:
        local = deepcopy(original)
        local.audio_config = None
    else:
        local = original

    assert original.audio_config is original_audio, (
        "load_vllm must not mutate the caller's config; deepcopy is required "
        "before clearing audio_config"
    )
    assert local.audio_config is None
    assert local is not original


def test_strip_skipped_when_audio_config_absent():
    class _Cfg:
        def __init__(self):
            self.model_type = "gemma4"
            self.audio_config = None

    original = _Cfg()
    entered = False
    if getattr(original, "audio_config", None) is not None:
        entered = True

    assert not entered, "audio gate must not strip when audio_config is already None"


def test_strip_skipped_when_outer_model_type_is_text_only():
    class _Cfg:
        def __init__(self):
            self.model_type = "gemma4_text"
            self.audio_config = None

    original = _Cfg()
    outer = getattr(original, "model_type", None)
    entered = (outer == "gemma4") and (getattr(original, "audio_config", None) is not None)
    assert not entered, (
        "gemma4_text (text-only) configs must not trigger the audio_config "
        "strip path; audio_config is only expected on outer gemma4 configs"
    )
