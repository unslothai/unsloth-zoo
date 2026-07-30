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

"""A 4bit merge of a local 4bit base must not depend on the Hub being up.

`merged_4bit` and `forced_merged_4bit` hand the live PeftModel to
`merge_and_unload()` and write the result out, never reading base weights from
anywhere: `determine_base_model_source` only supplies a directory to size the shards
from. So for a local nf4 directory such as `outputs/mymodel`, which is both a valid
repo id and an ordinary directory and so cannot be spared the probe by any string
test, a Hub outage used to abort a merge that needed nothing from the Hub.

Measured end to end on huggingface_hub 1.24.0 / transformers 5.14.1 with a real
bitsandbytes nf4 base in `outputs/mymodel` and both `ls` and `hf_hub_download`
raising ConnectionError:

    before the Hub-outage fix   Detected local model directory -> Merging finished
    with the fix, no save_method RuntimeError, nothing written, merge never ran
    with the fix, merged_4bit    Detected local model directory -> Merging finished

Exactly one Hub call is attempted across that run, the probe inside
`determine_base_model_source`, which is what says the rest of the path is already
offline.

nf4/fp4 under `merged_16bit` keeps raising, which is why the save method is passed
rather than the fallback simply widened: there the merge writes nothing, so falling
back would trade a loud failure for the silent no-op.
`test_the_fallback_can_never_reach_the_silent_no_op` pins that disjointness.
"""

import ast
import inspect
import json
import os
import warnings

import pytest
import requests
from huggingface_hub.errors import OfflineModeIsEnabled

from unsloth_zoo import saving_utils


NF4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
FP4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}
BNB_CONFIG        = {"load_in_4bit": True}
FP8_CONFIG        = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
MXFP4_CONFIG      = {"quant_method": "mxfp4"}

# The two save methods that fold LoRA into the weights already in memory.
FOUR_BIT_SAVE_METHODS = ["merged_4bit", "forced_merged_4bit"]

# Everything else keeps propagating an unreachable Hub. None is the default, standing
# for callers that do not know the save method (unsloth's
# `_prewarm_base_model_hub_cache`).
HUB_DEPENDENT_SAVE_METHODS = ["merged_16bit", "mxfp4", "lora", None]


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

_FOUR_BIT_QUANTS = [
    pytest.param(NF4_CONFIG, "nf4",          id = "nf4"),
    pytest.param(FP4_CONFIG, "fp4",          id = "fp4"),
    pytest.param(BNB_CONFIG, "bitsandbytes", id = "bitsandbytes"),
]


def _same_path(a, b):
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


# The regression: a repo-id shaped local 4bit directory, Hub down, 4bit merge.

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS)
@pytest.mark.parametrize("quant_config, label", _FOUR_BIT_QUANTS)
def test_local_4bit_resolves_when_the_hub_is_unreachable(
    monkeypatch, tmp_path, error, save_method, quant_config, label,
):
    """The 4bit merge reads no base weights, so nothing about it is the Hub's to
    decide. This is priority 5's answer, unchanged."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)

    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == f"local_{label}"
    assert is_quantized is True
    assert quant_type == label


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS)
def test_local_4bit_fallback_emits_no_warning(monkeypatch, tmp_path, error, save_method):
    """A resolution that succeeds is not something to warn about."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _hub_raises(monkeypatch, error)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS)
def test_local_4bit_resolves_positionally_too(monkeypatch, tmp_path, save_method):
    """`merge_and_overwrite_lora` passes the save method positionally after the token,
    so pin that order against a call site drifting into passing it as the token."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _hub_raises(monkeypatch, requests.exceptions.ConnectionError("dns failure"))

    name, is_local, source, _, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel", None, save_method)
    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == "local_nf4"
    assert quant_type == "nf4"


# Everything the merge cannot complete offline must keep raising. Nothing that used to
# raise may now answer None, and nothing may now reach the silent no-op.

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", HUB_DEPENDENT_SAVE_METHODS)
@pytest.mark.parametrize("quant_config, label", _FOUR_BIT_QUANTS)
def test_local_4bit_still_raises_for_every_other_save_method(
    monkeypatch, tmp_path, error, save_method, quant_config, label,
):
    """`merged_16bit` on an nf4/fp4 base writes nothing, so falling back would swap a
    loud failure for that silent no-op. Same for every save method whose weights the
    Hub really does supply."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert "connectivity" in str(excinfo.value)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_default_call_is_unchanged(monkeypatch, tmp_path, error):
    """Two positional arguments is what every caller outside this module uses, and it
    must behave as it did before the save method existed."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("outputs/mymodel", None)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS + HUB_DEPENDENT_SAVE_METHODS)
def test_no_local_copy_still_raises(monkeypatch, tmp_path, error, save_method):
    """The fallback hands back a directory that exists. With nothing on disk, "I could
    not tell" must not become "absent" for any save method."""
    monkeypatch.chdir(tmp_path)
    _hub_raises(monkeypatch, error)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source(
            "unsloth/does-not-exist", save_method = save_method,
        )


def test_the_fallback_can_never_reach_the_silent_no_op(monkeypatch, tmp_path):
    """The sink in `merge_and_overwrite_lora` reads the same save method variable the
    fallback fires on, so walk the whole product and check the two are disjoint."""
    # Mirrors the gate in merge_and_overwrite_lora; pinned against the source below.
    def hits_the_silent_no_op(base_model_is_quantized, quant_type, save_method):
        return (
            base_model_is_quantized
            and (quant_type == "nf4" or quant_type == "fp4")
            and save_method == "merged_16bit"
        )

    # Whitespace-normalised fragments, not one exact line: the property pinned is which
    # inputs the sink reads, and an exact match also failed on a reflow. This only
    # guards that the walk below is computed against the right gate; the sink's own
    # behaviour is covered by the end-to-end merge tests.
    normalised = " ".join(inspect.getsource(saving_utils.merge_and_overwrite_lora).split())
    for fragment in (
        "base_model_is_quantized and",
        'quant_type == "nf4"',
        'quant_type == "fp4"',
        'save_method == "merged_16bit"',
    ):
        assert fragment in normalised, (
            f"the silent no-op gate no longer reads {fragment}; "
            "re-check what the fallback can reach"
        )

    all_save_methods = FOUR_BIT_SAVE_METHODS + HUB_DEPENDENT_SAVE_METHODS
    resolved, raised = 0, 0
    for quant_config, label in [(NF4_CONFIG, "nf4"), (FP4_CONFIG, "fp4"),
                                (BNB_CONFIG, "bitsandbytes"), (FP8_CONFIG, "fp8")]:
        for save_method in all_save_methods:
            with monkeypatch.context() as patch:
                work = tmp_path / f"{label}-{save_method}"
                work.mkdir()
                patch.chdir(work)
                _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
                _hub_raises(patch, requests.exceptions.ConnectionError("dns failure"))
                try:
                    _, _, _, is_quantized, quant_type = \
                        saving_utils.determine_base_model_source(
                            "outputs/mymodel", save_method = save_method,
                        )
                except RuntimeError:
                    raised += 1
                    continue
            resolved += 1
            assert not hits_the_silent_no_op(is_quantized, quant_type, save_method), (
                f"{label} + {save_method} fell back into the silent no-op"
            )
    # 4 quant types x 6 save methods. fp8 resolves for the two 4bit merges and for
    # `merged_16bit`, the one save method that applies its companion scales; the three
    # 4bit types resolve for the two 4bit save methods only.
    assert (resolved, raised) == (3 + 3 * 2, 3 + 3 * 4), (resolved, raised)


def test_every_merge_call_site_passes_the_save_method():
    """*Every* resolution must forward it, not exactly two of them. Forwarding changes
    nothing for the FP8 16bit sibling, reachable only under `merged_16bit`, but omitting
    it would trap whoever next widens the fallback. Asserted as a property, since a
    count assertion would fail on a legitimate third resolution."""
    tree = ast.parse(inspect.getsource(saving_utils.merge_and_overwrite_lora).lstrip())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "determine_base_model_source"
    ]
    assert calls, "the merge no longer resolves a base model source at all"
    for call in calls:
        passed = [
            arg.id for arg in call.args if isinstance(arg, ast.Name)
        ] + [
            kw.arg for kw in call.keywords
        ]
        assert "save_method" in passed, ast.dump(call)


# A reachable Hub stays authoritative: nothing that resolved before changed.

@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS)
def test_reachable_hub_16bit_repo_still_outranks_the_local_4bit_copy(
    monkeypatch, tmp_path, save_method,
):
    """The fallback lives on the exception path only, so with the Hub up the 16bit repo
    still wins at priority 3 for a 4bit merge."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        # The local copy is consulted by resolved absolute path, the Hub by the name as
        # given, which is what tells the two calls apart.
        lambda name, token = None: (True, "nf4") if os.path.isabs(str(name)) else (False, None),
    )

    name, is_local, source, _, _ = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = save_method,
    )
    assert name == "outputs/mymodel"
    assert is_local is False
    assert source == "HF_unquantized"


@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS + HUB_DEPENDENT_SAVE_METHODS)
def test_reachable_hub_absent_repo_still_falls_to_the_local_4bit_copy(
    monkeypatch, tmp_path, save_method,
):
    """A genuinely absent repo was always priority 5 for every save method, and the new
    fallback must not have narrowed that."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    # fsspec converts RepositoryNotFoundError into a plain FileNotFoundError.
    _hub_raises(monkeypatch, FileNotFoundError("outputs/mymodel (repository not found)"))

    name, is_local, source, _, quant_type = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = save_method,
    )
    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == "local_nf4"
    assert quant_type == "nf4"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS + ["merged_16bit"])
def test_local_fp8_serves_the_save_methods_that_can_read_it(
    monkeypatch, tmp_path, error, save_method,
):
    """The 4bit merges take the weights from memory, and `merged_16bit` dequantizes the
    stored ones through `_merge_and_overwrite_lora_fp8`. Both can finish from the local
    copy, so an unreachable Hub must not stop them."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == "local_fp8"
    assert is_quantized is True
    assert quant_type == "fp8"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", [m for m in HUB_DEPENDENT_SAVE_METHODS if m != "merged_16bit"])
def test_local_fp8_does_not_serve_the_rest(monkeypatch, tmp_path, error, save_method):
    """Only `merged_16bit` applies the companion scales. The other save methods reach the
    in place writer, which reads the stored tensor raw and writes it back at its FP8 dtype,
    so a fallback there would fold the delta into scaled space and export a wrong base."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


@pytest.mark.parametrize("save_method", FOUR_BIT_SAVE_METHODS + HUB_DEPENDENT_SAVE_METHODS)
@pytest.mark.parametrize("quant_config, source_info", [
    pytest.param(None,         "local_unquantized", id = "unquantized"),
    pytest.param(MXFP4_CONFIG, "local_mxfp4",       id = "mxfp4"),
])
def test_priorities_1_and_2_still_never_probe_the_hub(
    monkeypatch, tmp_path, save_method, quant_config, source_info,
):
    """These outrank every Hub answer and resolve before the probe, so the save method
    must not have introduced a reason to consult it."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _forbid_hub(monkeypatch)

    _, is_local, source, _, _ = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = save_method,
    )
    assert is_local is True
    assert source == source_info
