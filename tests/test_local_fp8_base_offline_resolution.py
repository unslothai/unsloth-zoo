# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""An unreachable Hub must not block a 16bit merge of a local FP8 base.

Companion to `test_local_base_model_resolution_offline.py`. That file fixed the
half of the problem a string test can reach: a leading `.`, `/` or `~`, or more
than one `/`, is a filesystem path and is never offered to the Hub, and local
unquantized / local mxfp4 are resolved before the probe because they outrank
every Hub answer anyway.

`outputs/mymodel` survives all of that. It is simultaneously a valid repo id and
an ordinary directory, so the string gate lets it through, and an FP8 directory
is neither priority 1 nor priority 2, so it reaches the probe. With the Hub down
the probe raised and priority 5 was never consulted, even though
`merge_and_overwrite_lora` dequantizes an FP8 base for `merged_16bit` via
`_merge_and_overwrite_lora_fp8` and therefore needs nothing from the network.

Measured on huggingface_hub 1.24.0 with `outputs/fp8` present on disk and
`HfFileSystem.ls` raising OfflineModeIsEnabled:

    local unquantized   ('.../outputs/unquantized', True, 'local_unquantized', ...)
    local mxfp4         ('.../outputs/mxfp4',       True, 'local_mxfp4', ...)
    local fp8           RuntimeError                <- the regression
    local nf4           RuntimeError                <- correct, see below

nf4/fp4 deliberately keeps raising. For those `merge_and_overwrite_lora` answers
`warnings.warn` plus `return None` and writes nothing, so falling back to the
local copy would trade a loud failure for exactly the silent no-op this whole
change exists to remove. Falling back is not automatically the safer choice.
"""

import json
import os
import warnings

import pytest
import requests
from huggingface_hub.errors import OfflineModeIsEnabled

from unsloth_zoo import saving_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FP8_CONFIG        = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
FBGEMM_FP8_CONFIG = {"quant_method": "fbgemm_fp8"}
NF4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
FP4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}
BNB_CONFIG        = {"load_in_4bit": True}
MXFP4_CONFIG      = {"quant_method": "mxfp4"}


def _make_local_model(directory, quantization_config = None):
    """`check_local_model_exists` keys off a `.safetensors` file,
    `check_model_quantization_status` off config.json."""
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    config = {"model_type": "llama"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return directory


def _hub_raises(monkeypatch, error):
    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _forbid_hub(monkeypatch):
    def no_network(self, path, detail = True, **kwargs):
        raise AssertionError(
            f"HfFileSystem.ls({path!r}) was called; this path must resolve "
            f"without probing the Hub"
        )
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", no_network, raising = True)


_TRANSPORT_ERRORS = [
    pytest.param(
        OfflineModeIsEnabled("Cannot reach https://huggingface.co: offline mode is enabled."),
        id = "offline-mode",
    ),
    pytest.param(requests.exceptions.ConnectionError("dns failure"), id = "dns-failure"),
    pytest.param(requests.exceptions.ReadTimeout("read timed out"),  id = "read-timeout"),
    pytest.param(requests.exceptions.ProxyError("proxy refused"),    id = "proxy-error"),
]


def _same_path(a, b):
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


# ---------------------------------------------------------------------------
# The regression: a repo-id shaped local FP8 directory with the Hub down.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("quant_config", [
    pytest.param(FP8_CONFIG,        id = "finegrained-fp8"),
    pytest.param(FBGEMM_FP8_CONFIG, id = "fbgemm-fp8"),
])
def test_local_fp8_resolves_when_the_hub_is_unreachable(monkeypatch, tmp_path, error, quant_config):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")

    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == "local_fp8"
    assert is_quantized is True
    assert quant_type == "fp8"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_local_fp8_fallback_never_lands_on_the_silent_no_op(monkeypatch, tmp_path, error):
    """`merge_and_overwrite_lora` warns and returns None only for nf4/fp4 on a
    16bit merge. The quant type handed back here must therefore not be one of
    those, or the fallback would have swapped a loud failure for a merge that
    writes nothing."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    _, _, _, is_quantized, quant_type = saving_utils.determine_base_model_source("outputs/mymodel")
    assert is_quantized is True
    assert quant_type not in ("nf4", "fp4", "bitsandbytes")


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_local_fp8_fallback_emits_no_warning(monkeypatch, tmp_path, error):
    """A resolution that succeeds is not something to warn about, and a warning
    is exactly the signal that scrolls past unnoticed."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


# ---------------------------------------------------------------------------
# Everything the merge cannot complete offline must keep raising.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("quant_config, label", [
    pytest.param(NF4_CONFIG, "nf4", id = "nf4"),
    pytest.param(FP4_CONFIG, "fp4", id = "fp4"),
    pytest.param(BNB_CONFIG, "bitsandbytes", id = "bitsandbytes"),
])
def test_local_4bit_still_raises_when_the_hub_is_unreachable(
    monkeypatch, tmp_path, error, quant_config, label,
):
    """Falling back here would put a 16bit merge straight onto
    `warnings.warn` plus `return None`, which writes nothing."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert "connectivity" in str(excinfo.value)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_no_local_copy_still_raises(monkeypatch, tmp_path, error):
    monkeypatch.chdir(tmp_path)
    _hub_raises(monkeypatch, error)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("unsloth/does-not-exist")


# ---------------------------------------------------------------------------
# A reachable Hub stays authoritative: nothing that resolved before changed.
# ---------------------------------------------------------------------------

def test_reachable_hub_16bit_repo_still_outranks_the_local_fp8_copy(monkeypatch, tmp_path):
    """Catching the failure rather than hoisting priority 5 is what preserves
    this: with the Hub up, the 16bit repo still wins at priority 3."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        # The local copy is consulted by its resolved absolute path, the Hub by
        # the name as given, which is what tells the two calls apart here.
        lambda name, token = None: (True, "fp8") if os.path.isabs(str(name)) else (False, None),
    )

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert name == "outputs/mymodel"
    assert is_local is False
    assert source == "HF_unquantized"


def test_reachable_hub_absent_repo_still_falls_to_the_local_fp8_copy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    # fsspec converts RepositoryNotFoundError into a plain FileNotFoundError.
    _hub_raises(monkeypatch, FileNotFoundError("outputs/mymodel (repository not found)"))

    name, is_local, source, _, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert source == "local_fp8"
    assert quant_type == "fp8"


@pytest.mark.parametrize("shape", ["absolute", "dot-relative", "deep-relative"])
def test_path_shaped_local_fp8_never_probes_the_hub_at_all(monkeypatch, tmp_path, shape):
    """The string gate already spares these; pin that it still does, so the new
    fallback is only ever reached by the repo-id shaped case that needs it."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "nested", "mymodel"), FP8_CONFIG)
    given = {
        "absolute":      str(tmp_path / "outputs" / "nested" / "mymodel"),
        "dot-relative":  os.path.join(".", "outputs", "nested", "mymodel"),
        "deep-relative": os.path.join("outputs", "nested", "mymodel"),
    }[shape]
    _forbid_hub(monkeypatch)

    name, is_local, source, _, quant_type = saving_utils.determine_base_model_source(given)
    assert _same_path(name, tmp_path / "outputs" / "nested" / "mymodel")
    assert is_local is True
    assert source == "local_fp8"
    assert quant_type == "fp8"


def test_local_mxfp4_still_resolves_before_the_probe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), MXFP4_CONFIG)
    _forbid_hub(monkeypatch)
    _, _, source, _, quant_type = saving_utils.determine_base_model_source("outputs/mymodel")
    assert source == "local_mxfp4"
    assert quant_type == "mxfp4"


# ---------------------------------------------------------------------------
# The FP8 16bit sibling lookup must not report "no sibling" for a Hub it
# could not reach.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_sibling_lookup_says_so_when_the_hub_is_unreachable(monkeypatch, tmp_path, error):
    """Returning None stays right, because the merge can still dequantize the
    FP8 weights. Doing it silently is not: the base actually used is then not
    the one a reachable Hub would have chosen."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8") is None
    messages = [str(w.message) for w in caught]
    assert any("16bit sibling" in m for m in messages), messages
    assert any("FP8 weights instead" in m for m in messages), messages


def test_sibling_lookup_is_silent_and_network_free_when_a_local_sibling_exists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    _make_local_model(os.path.join("outputs", "GLM-5.2"), None)
    _forbid_hub(monkeypatch)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8")
    assert _same_path(sibling, tmp_path / "outputs" / "GLM-5.2")
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


def test_sibling_lookup_stays_silent_for_a_genuinely_absent_sibling(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    # fsspec converts RepositoryNotFoundError into a plain FileNotFoundError.
    _hub_raises(monkeypatch, FileNotFoundError("outputs/mymodel (repository not found)"))

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8") is None
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []
