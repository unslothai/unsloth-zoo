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

"""Guards two version-skew defects: an empty import list from patch_merge_quantization_configs ended import unsloth
(unsloth issue 3130, transformers 4.55.0); triton 3.3.0/3.3.1 hit LLVM ERROR lowering cut_cross_entropy on compute
capability 7.5 (unsloth issue 2491)."""

import importlib

import pytest


# ---------------------------------------------------------------- #3130


ATTRIBUTE = "merge_quantization_configs"


def _strip_generated_globals(namespace, before):
    """Drop whatever an exec added to `namespace`, keeping the rest untouched."""
    for name in set(namespace) - before:
        del namespace[name]


def test_a_generated_global_is_removed_rather_than_restored():
    """The test strips generated globals itself: monkeypatch.delitem after the exec would record the generated object to
    restore, leaving a stale merge_quantization_configs for the next call."""
    namespace = {"kept": 1}
    before = set(namespace)
    namespace[ATTRIBUTE] = lambda *a, **k: None       # as the exec would leave it

    undone = pytest.MonkeyPatch()
    undone.delitem(namespace, ATTRIBUTE)
    assert ATTRIBUTE not in namespace
    undone.undo()
    assert ATTRIBUTE in namespace, "monkeypatch put it back, which is the trap"

    _strip_generated_globals(namespace, before)
    assert ATTRIBUTE not in namespace
    assert namespace == {"kept": 1}


@pytest.fixture
def unpatched_quantizer(monkeypatch):
    """Gives the test a pristine classmethod on transformers.quantizers.auto (the patch fast-returns once run), restores
    both attributes after, and records patch_function."""
    auto = pytest.importorskip("transformers.quantizers.auto")
    from unsloth_zoo.temporary_patches import misc
    from unsloth_zoo.temporary_patches.utils import _get_unique_storage_name

    quantizer = auto.AutoHfQuantizer
    sentinel = _get_unique_storage_name(quantizer, ATTRIBUTE)
    names = (ATTRIBUTE, sentinel)
    snapshot = {name: quantizer.__dict__[name] for name in names if name in quantizer.__dict__}

    if sentinel in quantizer.__dict__:
        # patch_function stashes the original it replaced under `sentinel`.
        setattr(quantizer, ATTRIBUTE, quantizer.__dict__[sentinel])
        delattr(quantizer, sentinel)

    recorded = []
    monkeypatch.setattr(
        misc, "patch_function", lambda *args, **kwargs: recorded.append(args)
    )
    # A previous test's exec would otherwise leave the name in these globals.
    monkeypatch.delitem(misc.__dict__, ATTRIBUTE, raising = False)
    globals_before = set(misc.__dict__)

    def set_module_dir(returns):
        """PEP 562 __dir__ must be set by hand: monkeypatch undo would put the inherited module __dir__ into the module
        dict, and every later dir() on it recurses forever."""
        auto.__dict__["__dir__"] = lambda: list(returns)

    try:
        yield auto, misc, recorded, set_module_dir
    finally:
        # Strip every name the patch exec'd into the module globals: monkeypatch.delitem would reinstate the generated
        # object at teardown, leaving a stale callable for later tests; restoring pre-existing names stays with
        # monkeypatch.
        _strip_generated_globals(misc.__dict__, globals_before)
        auto.__dict__.pop("__dir__", None)
        for name in names:
            if name in snapshot:
                setattr(quantizer, name, snapshot[name])
            elif name in quantizer.__dict__:
                delattr(quantizer, name)


def test_an_empty_name_list_no_longer_ends_the_import(monkeypatch, unpatched_quantizer):
    """No empty from-import is built, so nothing raises; if a release's annotations need names that import would supply,
    the patch declines rather than ending import."""
    auto, misc, recorded, set_module_dir = unpatched_quantizer
    # PEP 562: a module may define __dir__, which transformers 4.55.0 amounted to here; reproduced without pinning an old transformers.
    set_module_dir([])
    assert dir(auto) == []

    # A module global shadows the builtin inside that module's functions, so
    # this reads exactly what the patch asked to execute. Set by hand: the
    # fixture takes back every global that was not there before it, and
    # monkeypatch's undo would then try to remove an attribute that has already
    # gone and fail the teardown.
    execd = []
    def recording_exec(source, *args, **kwargs):
        execd.append(source)
        return exec(source, *args, **kwargs)
    misc.__dict__["exec"] = recording_exec

    misc.patch_merge_quantization_configs()

    assert execd, "the rewritten source was never reached"
    assert not any(
        source.lstrip().startswith("from transformers.quantizers.auto import")
        for source in execd
    ), "an empty name list still produced an import statement"


def merge_quantization_configs(cls, quantization_config, quantization_config_from_args):
    """Stands in for the real classmethod; its source needs no imported names before it can be defined, which is the one property that matters."""
    return quantization_config


def test_a_rewrite_that_needs_no_names_is_still_applied(monkeypatch, unpatched_quantizer):
    """An empty name list still patches when the rewritten source needs no names: the import only supplies names the source uses."""
    auto, misc, recorded, set_module_dir = unpatched_quantizer
    monkeypatch.setattr(auto.AutoHfQuantizer, ATTRIBUTE, merge_quantization_configs)
    set_module_dir([])

    misc.patch_merge_quantization_configs()

    assert len(recorded) == 1
    target, attribute, replacement = recorded[0]
    assert target is auto.AutoHfQuantizer
    assert attribute == ATTRIBUTE
    assert callable(replacement)
    # Asserts on the exec'd copy rather than the object passed in, since the defined-something-else guard reads the patch's own globals.
    assert replacement is not merge_quantization_configs
    assert replacement.__name__ == ATTRIBUTE


def test_an_import_that_fails_is_reported_not_raised(monkeypatch, unpatched_quantizer):
    """A name in the source but missing from the module is a patch failure, reported rather than raised as an import failure."""
    auto, misc, recorded, set_module_dir = unpatched_quantizer
    # A parameter name of merge_quantization_configs passes the `x in source` filter but is not a module attribute, so the generated import raises ImportError.
    unimportable = "quantization_config_from_args"
    assert not hasattr(auto, unimportable)
    set_module_dir([unimportable])

    misc.patch_merge_quantization_configs()

    assert recorded == []


def _defines_another_name(cls, quantization_config, quantization_config_from_args):
    """Stands in for AutoHfQuantizer.merge_quantization_configs; its source defines a function under a different name, which is the one property that matters."""
    return quantization_config


def test_a_rewrite_that_defines_nothing_is_reported_not_a_name_error(
    monkeypatch, unpatched_quantizer
):
    """When the rewritten source runs cleanly but defines another name, the patch must report that rather than raise NameError from itself."""
    auto, misc, recorded, _ = unpatched_quantizer
    monkeypatch.setattr(auto.AutoHfQuantizer, ATTRIBUTE, _defines_another_name)

    misc.patch_merge_quantization_configs()

    assert recorded == []
    assert ATTRIBUTE not in misc.__dict__
    assert "_defines_another_name" in misc.__dict__


# ---------------------------------------------------------------- #2491


@pytest.fixture
def loss_utils():
    pytest.importorskip("triton")
    return importlib.import_module("unsloth_zoo.loss_utils")


@pytest.mark.parametrize(
    "version,rejected",
    [
        ("3.1.0", False),
        ("3.2.0", False),
        ("3.3.0", True),
        ("3.3.1", True),
        # pytorch-triton nightlies carry a local version; 3.3.x is still 3.3.x.
        ("3.3.1+git1234abcd", True),
        # Windows Triton (triton-windows) installs as the triton package with a .postN suffix on the same upstream
        # version, so a repackaged 3.3.x must read as 3.3.x, not as a new release.
        ("3.3.1.post19", True),
        ("3.3.0.post1", True),
        ("3.4.0", False),
        ("3.4.0.post28", False),
        ("3.4.0+git1234abcd", False),
        ("3.5.1", False),
        ("3.6.0", False),
        ("3.8.0", False),
        # Not a version we can read, which is not evidence of a defect.
        ("not a version", False),
    ],
)
def test_only_the_two_broken_triton_releases_are_rejected(loss_utils, version, rejected):
    """The cut_cross_entropy kernel compiled ahead of time for cuda:75 on every published release from 3.1.0 to 3.8.0, and only 3.3.0 and 3.3.1 abort."""
    assert loss_utils._triton_miscompiles_cce_on_sm75(7, 5, version) is rejected


@pytest.mark.parametrize(
    "capability", [(7, 0), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
)
def test_every_other_gpu_keeps_cut_cross_entropy_on_the_broken_triton(
    loss_utils, capability
):
    """The sm_80 control compiled cleanly on all of 3.1.0 to 3.8.0, so only compute capability 7.5 may be gated off."""
    assert loss_utils._triton_miscompiles_cce_on_sm75(*capability, "3.3.1") is False


@pytest.mark.parametrize(
    "version,expected", [("3.3.1", False), ("3.4.0", True), ("3.2.0", True)]
)
def test_the_gate_reaches_has_cut_cross_entropy(monkeypatch, version, expected):
    """Wiring, not arithmetic: the flag the loss path reads has to change."""
    import torch

    pytest.importorskip("cut_cross_entropy")
    triton = pytest.importorskip("triton")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    # device_count must be stubbed too since the gate enumerates visible devices; on a CPU-only runner it answers 0, so
    # the 3.3.1 case would pass for the wrong reason.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(triton, "__version__", version)

    module = importlib.import_module("unsloth_zoo.loss_utils")
    try:
        reloaded = importlib.reload(module)
        assert reloaded.HAS_CUT_CROSS_ENTROPY is expected
    finally:
        # Leave sys.modules holding a loss_utils built from the real versions.
        monkeypatch.undo()
        importlib.reload(module)


def _causal_lm_templates():
    """The two generated causal-LM branches, read out of the compiler's own source."""
    import re

    from pathlib import Path

    source = (
        Path(importlib.import_module("unsloth_zoo.compiler").__file__)
    ).read_text(encoding = "utf-8")
    branches = re.findall(r"^elif .*fused_linear_cross_entropy.*?$", source, re.M)
    if not branches:
        branches = [
            line
            for line in source.splitlines()
            if line.startswith("elif ") and "UNSLOTH_ENABLE_CCE" in line
        ]
    return source, branches


def test_the_compiled_branch_consults_has_cut_cross_entropy():
    """Disabling CCE must reach the compiled forward: unsloth_zoo/__init__.py force-disables UNSLOTH_ENABLE_CCE only for
    torch >= 2.8 and HIP, so on torch 2.7 the branch must also check HAS_CUT_CROSS_ENTROPY, else it raises NameError."""
    source, branches = _causal_lm_templates()
    assert branches, "the generated causal-LM CCE branch is no longer recognisable"
    for branch in branches:
        assert "UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY" in branch, branch
    assert "    HAS_CUT_CROSS_ENTROPY,\n" in source, (
        "the generated preamble must import HAS_CUT_CROSS_ENTROPY alongside "
        "fused_linear_cross_entropy, or the branch above is a NameError of its own"
    )


def test_fused_linear_cross_entropy_needs_the_symbol_the_gate_skips():
    """With the gate false the symbol is really absent, so a branch that ignores the gate cannot work; guards against
    the two flags quietly ceasing to be independent."""
    import inspect

    loss_utils = importlib.import_module("unsloth_zoo.loss_utils")
    body = inspect.getsource(loss_utils.fused_linear_cross_entropy)
    assert "linear_cross_entropy(" in body
    assert "HAS_CUT_CROSS_ENTROPY" not in body, (
        "fused_linear_cross_entropy now checks the flag itself; retarget this test"
    )


def _sm75_warning_text():
    import inspect
    import re

    source = inspect.getsource(importlib.import_module("unsloth_zoo.loss_utils"))
    match = re.search(r"miscompiles the cut cross entropy.*?\"\n\s*\)", source, re.S)
    assert match, "the sm75 CCE warning is no longer recognisable"
    return match.group(0)


def test_the_sm75_warning_does_not_recommend_a_torch_that_still_disables_cce():
    """The remedy must clear both gates: torch 2.8 with triton 3.4.0 still sets UNSLOTH_ENABLE_CCE=0 for torch >= 2.8,
    so torch 2.6.0 (triton 3.2.0) is the version that restores CCE."""
    text = _sm75_warning_text()
    assert "torch 2.6.0" in text
    assert "UNSLOTH_ENABLE_CCE=0" in text, (
        "the warning names torch 2.8 without saying that the flag keeps CCE off there"
    )
    for claim in ("2.8.0 or later, which carries triton 3.4.0, restores it",):
        assert claim not in text, f"the warning still promises {claim!r}"


def test_the_flag_the_warning_now_names_really_is_set_for_torch_28():
    """Negative control: reads the unsloth_zoo __init__ gate rather than the prose; fails if torch 2.8 stops
    force-disabling CCE, letting the warning go back to recommending it."""
    import inspect
    import re

    source = inspect.getsource(importlib.import_module("unsloth_zoo"))
    gate = re.search(
        r"if \(major_torch >= 2 and minor_torch >= 8\) or \(major_torch > 2\):\s*\n"
        r"\s*os\.environ\[\"UNSLOTH_ENABLE_CCE\"\] = \"0\"",
        source,
    )
    assert gate, (
        "unsloth_zoo no longer force-disables UNSLOTH_ENABLE_CCE for torch >= 2.8; the "
        "sm75 warning's advice depends on this and has to be revisited"
    )


@pytest.mark.parametrize(
    "capabilities, affected_indices",
    [
        # Heterogeneous host from the report: device 0 is an A100 and the current device at import, but device_map can put lm_head on the T4 at index 1.
        ([(8, 0), (7, 5)], [1]),
        ([(7, 5), (8, 0)], [0]),
        ([(7, 5), (7, 5)], [0, 1]),
        # Negative controls: a homogeneous host that is not sm_75, and a host with no devices at all, are both left untouched.
        ([(8, 0), (9, 0)], []),
        ([], []),
    ],
)
def test_every_visible_device_is_asked_not_just_the_current_one(
    monkeypatch, capabilities, affected_indices
):
    """get_device_capability() with no argument only describes the current device, but device_map can put lm_head on
    another GPU; guessing wrong aborts the process with an uncatchable LLVM error."""
    import torch

    loss_utils = importlib.import_module("unsloth_zoo.loss_utils")
    monkeypatch.setattr(loss_utils, "triton_version", "3.3.1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = 0: capabilities[index])

    affected = loss_utils._triton_miscompiles_cce_on_any_visible_device()
    assert [index for index, _major, _minor in affected] == affected_indices


def test_a_working_triton_leaves_every_device_alone(monkeypatch):
    """sm_75 is only a problem on the two broken triton releases, so a T4 host on triton 3.4.0 must report no devices and keep CCE."""
    import torch

    loss_utils = importlib.import_module("unsloth_zoo.loss_utils")
    monkeypatch.setattr(loss_utils, "triton_version", "3.4.0")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = 0: (7, 5))

    assert loss_utils._triton_miscompiles_cce_on_any_visible_device() == []


def test_a_device_that_cannot_be_queried_is_skipped_not_fatal(monkeypatch):
    """Import-time code on a partly broken driver must not take the import down with it."""
    import torch

    loss_utils = importlib.import_module("unsloth_zoo.loss_utils")
    monkeypatch.setattr(loss_utils, "triton_version", "3.3.0")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def _capability(index = 0):
        if index == 0:
            raise RuntimeError("device 0 is unavailable")
        return (7, 5)

    monkeypatch.setattr(torch.cuda, "get_device_capability", _capability)
    assert [i for i, _, _ in loss_utils._triton_miscompiles_cce_on_any_visible_device()] == [1]
