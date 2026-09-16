# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Two version-skew defects that used to surface as something else.

unsloth #3130: `patch_merge_quantization_configs` builds
`from transformers.quantizers.auto import (...)` from the names of that module
that happen to occur in the source it rewrites. On transformers 4.55.0 no name
did, so it exec'd `from transformers.quantizers.auto import ()`, a SyntaxError,
outside any try, and `import unsloth` ended there.

unsloth #2491: triton 3.3.0 and 3.3.1 cannot lower cut_cross_entropy's forward
kernel for compute capability 7.5, so a T4 or a 2080 Ti aborted the process with
`LLVM ERROR: Unsupported rounding mode for conversion.`
"""

import importlib

import pytest


# ---------------------------------------------------------------- #3130


ATTRIBUTE = "merge_quantization_configs"


@pytest.fixture
def unpatched_quantizer(monkeypatch):
    """transformers.quantizers.auto with nothing of ours applied to it yet.

    The patch fast-returns once it has already run, and anything that imports
    unsloth in the same interpreter will have run it, so put the pristine
    classmethod back for the duration of the test and restore both attributes
    exactly as they were afterwards. A recorder stands in for `patch_function`,
    so the test never mutates transformers itself.
    """
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

    def set_module_dir(returns):
        """PEP 562 module __dir__, set by hand rather than with monkeypatch.

        A module inherits a `__dir__` method from the module type, so
        monkeypatch's undo puts that bound method into the module's own __dict__
        and every later dir() on it recurses forever. Measured, and it silently
        poisoned the next test in the file.
        """
        auto.__dict__["__dir__"] = lambda: list(returns)

    try:
        yield auto, misc, recorded, set_module_dir
    finally:
        auto.__dict__.pop("__dir__", None)
        for name in names:
            if name in snapshot:
                setattr(quantizer, name, snapshot[name])
            elif name in quantizer.__dict__:
                delattr(quantizer, name)


def test_an_empty_name_list_no_longer_ends_the_import(monkeypatch, unpatched_quantizer):
    """The reported case: nothing in dir() occurs in the source."""
    auto, misc, recorded, set_module_dir = unpatched_quantizer
    # PEP 562: a module may define __dir__, which is what transformers 4.55.0
    # amounted to here, without pinning an old transformers to find out.
    set_module_dir([])
    assert dir(auto) == []

    misc.patch_merge_quantization_configs()

    # It still patches: the import was only ever there to supply names the
    # rewritten source needs, and an empty list means it needs none.
    assert len(recorded) == 1
    target, attribute, replacement = recorded[0]
    assert target is auto.AutoHfQuantizer
    assert attribute == ATTRIBUTE
    assert callable(replacement)


def test_an_import_that_fails_is_reported_not_raised(monkeypatch, unpatched_quantizer):
    """A name that occurs in the source but is not importable from the module is a
    patch failure, not an import failure."""
    auto, misc, recorded, set_module_dir = unpatched_quantizer
    # A parameter name of merge_quantization_configs, so it passes the
    # `x in source` filter, and not an attribute of the module, so the generated
    # import raises ImportError.
    unimportable = "quantization_config_from_args"
    assert not hasattr(auto, unimportable)
    set_module_dir([unimportable])

    misc.patch_merge_quantization_configs()

    assert recorded == []


def _defines_another_name(cls, quantization_config, quantization_config_from_args):
    """Stands in for AutoHfQuantizer.merge_quantization_configs, with the one
    property that matters: its source defines a function under a different name."""
    return quantization_config


def test_a_rewrite_that_defines_nothing_is_reported_not_a_name_error(
    monkeypatch, unpatched_quantizer
):
    """The name in the issue title: the rewritten source exec'd cleanly but under
    another name, and the patch then raised NameError out of itself."""
    auto, misc, recorded, _ = unpatched_quantizer
    monkeypatch.setattr(auto.AutoHfQuantizer, ATTRIBUTE, _defines_another_name)

    misc.patch_merge_quantization_configs()

    assert recorded == []
    assert ATTRIBUTE not in misc.__dict__
    assert "_defines_another_name" in misc.__dict__
    monkeypatch.delitem(misc.__dict__, "_defines_another_name", raising = False)


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
        # Windows gets Triton from the separate `triton-windows` distribution, which
        # installs as the `triton` package and carries a `.postN` suffix on the same
        # upstream version. `from triton import __version__` reads it either way, so
        # the gate has to treat a repackaged 3.3.x as 3.3.x and a repackaged 3.4.x as
        # 3.4.x, rather than reading the post segment as a different release.
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
    """Both ends measured: the real cut_cross_entropy kernel was compiled ahead of
    time for cuda:75 on every published release from 3.1.0 to 3.8.0, and only
    3.3.0 and 3.3.1 abort."""
    assert loss_utils._triton_miscompiles_cce_on_sm75(7, 5, version) is rejected


@pytest.mark.parametrize(
    "capability", [(7, 0), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
)
def test_every_other_gpu_keeps_cut_cross_entropy_on_the_broken_triton(
    loss_utils, capability
):
    """The sm_80 control compiled cleanly on all of 3.1.0 to 3.8.0, so nothing but
    7.5 may be gated off."""
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
    """Disabling CCE has to reach the compiled forward, not just the import.

    The sm75 gate above only makes HAS_CUT_CROSS_ENTROPY false and skips
    `from cut_cross_entropy import linear_cross_entropy`. UNSLOTH_ENABLE_CCE is a separate
    env flag defaulting to "1" that unsloth_zoo/__init__.py force-disables only for torch
    >= 2.8 and for HIP -- so on torch 2.7, which pins exactly the triton 3.3.x the kernel
    miscompiles under, the flag stayed on and the generated branch called
    fused_linear_cross_entropy, which references the symbol that was never imported and
    raises NameError. The warning promises "the standard loss is used instead", and the
    elif below the branch is that standard loss.
    """
    source, branches = _causal_lm_templates()
    assert branches, "the generated causal-LM CCE branch is no longer recognisable"
    for branch in branches:
        assert "UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY" in branch, branch
    assert "    HAS_CUT_CROSS_ENTROPY,\n" in source, (
        "the generated preamble must import HAS_CUT_CROSS_ENTROPY alongside "
        "fused_linear_cross_entropy, or the branch above is a NameError of its own"
    )


def test_fused_linear_cross_entropy_needs_the_symbol_the_gate_skips():
    """The premise: with the gate false the symbol genuinely is not there, so a branch
    that ignores the gate cannot work. Guards against the test above passing while the
    two flags have quietly stopped being independent."""
    import inspect

    loss_utils = importlib.import_module("unsloth_zoo.loss_utils")
    body = inspect.getsource(loss_utils.fused_linear_cross_entropy)
    assert "linear_cross_entropy(" in body
    assert "HAS_CUT_CROSS_ENTROPY" not in body, (
        "fused_linear_cross_entropy now checks the flag itself; retarget this test"
    )
