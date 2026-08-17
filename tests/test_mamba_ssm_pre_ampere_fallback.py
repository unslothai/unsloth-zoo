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

"""Tests patch_mamba_ssm_pre_ampere_fallback in temporary_patches/misc.py.

mamba_ssm's Triton kernels need sm_80+. On a T4 the package imports fine and
is_fast_path_available is True, so the layer routes into cuda_kernels_forward
and Triton fails once training starts. The patch must fire ONLY on pre-Ampere
NVIDIA CUDA: ROCm, XPU, MPS and CPU are left alone, Ampere+ keeps the fast path.

Extracted by AST so the test needs neither a GPU nor a transformers import.
"""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"
_SRC = MISC.read_text(encoding = "utf-8")


def _load():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_mamba_ssm_pre_ampere_fallback":
            ns = {"torch": torch}
            exec(ast.get_source_segment(_SRC, node), ns)
            return ns[node.name]
    raise AssertionError("patch_mamba_ssm_pre_ampere_fallback not found")


patch = _load()

MODEL_MOD = "transformers.models.granitemoehybrid.modeling_granitemoehybrid"

CONFIG_MODS = (
    "transformers.models.jamba.configuration_jamba",
    "transformers.models.zamba.configuration_zamba",
)


class _FakeCuda:
    def __init__(self, available = True, capability = (7, 5)):
        self._available, self._capability = available, capability
    def is_available(self): return self._available
    def get_device_capability(self, *a, **k): return self._capability


@pytest.fixture
def env(monkeypatch):
    """Pre-Ampere NVIDIA CUDA with mamba_ssm installed and a model imported."""
    monkeypatch.setattr(torch, "cuda", _FakeCuda(), raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)

    saved = {k: sys.modules.get(k) for k in ("mamba_ssm", MODEL_MOD,
                                             "transformers.utils.import_utils")}
    sys.modules["mamba_ssm"] = types.ModuleType("mamba_ssm")

    model_mod = types.ModuleType(MODEL_MOD)
    model_mod.is_fast_path_available = True
    model_mod.selective_state_update = lambda *a, **k: None
    model_mod.mamba_chunk_scan_combined = lambda *a, **k: None
    model_mod.mamba_split_conv1d_scan_combined = lambda *a, **k: None
    sys.modules[MODEL_MOD] = model_mod

    # transformers 5 also resolves these from the Hub, through
    # integrations.hub_kernels.lazy_load_kernel, which asks no predicate.
    hk = types.ModuleType("transformers.integrations.hub_kernels")
    hk._HUB_KERNEL_MAPPING = {
        "causal-conv1d": {"repo_id": "kernels-community/causal-conv1d"},
        "mamba-ssm": {"repo_id": "kernels-community/mamba-ssm"},
        "falcon_mamba-ssm": {"repo_id": "kernels-community/mamba-ssm"},
        "deep-gemm": {"repo_id": "kernels-community/deep-gemm"},
    }
    hk._KERNEL_MODULE_MAPPING = {"mamba-ssm": types.ModuleType("already_loaded")}
    for name in ("transformers.integrations", "transformers.integrations.hub_kernels"):
        saved[name] = sys.modules.get(name)
    integrations = types.ModuleType("transformers.integrations")
    integrations.__path__ = []
    integrations.hub_kernels = hk
    sys.modules["transformers.integrations"] = integrations
    sys.modules["transformers.integrations.hub_kernels"] = hk

    # The config step imports these two by name. Park empty stubs on them so no
    # test reaches out to the installed transformers and permanently rewrites
    # the real ZambaConfig.__init__; a module with no config class in it is
    # skipped. Tests that want the class install their own over the top, and
    # this fixture's teardown restores whatever was there.
    for cfg_mod_name in CONFIG_MODS:
        saved[cfg_mod_name] = sys.modules.get(cfg_mod_name)
        sys.modules[cfg_mod_name] = types.ModuleType(cfg_mod_name)

    iu = types.ModuleType("transformers.utils.import_utils")
    iu.is_mamba_ssm_available = lambda: True
    iu.is_mamba_2_ssm_available = lambda: True
    sys.modules["transformers.utils.import_utils"] = iu
    # `import transformers.utils.import_utils as _iu` needs the parents too.
    for parent in ("transformers", "transformers.utils"):
        if parent not in sys.modules:
            m = types.ModuleType(parent)
            m.__path__ = []
            sys.modules[parent] = m
            saved.setdefault(parent, None)
    sys.modules["transformers"].utils = sys.modules["transformers.utils"]
    # `import a.b.c as x` resolves through getattr(a.b, "c"), so this attribute
    # outlives a sys.modules-only teardown and hands the stub to every later
    # `import transformers.utils.import_utils as iu` (test_vendor_fla.py:219,
    # fla_vendor.py:473). Restore it too.
    utils_mod = sys.modules["transformers.utils"]
    had_attr = hasattr(utils_mod, "import_utils")
    saved_attr = getattr(utils_mod, "import_utils", None)
    utils_mod.import_utils = iu

    yield model_mod, iu, hk

    if had_attr: utils_mod.import_utils = saved_attr
    elif hasattr(utils_mod, "import_utils"): del utils_mod.import_utils
    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v


def _no_local_wheel():
    """Make `import mamba_ssm` raise, for the not-installed branch.

    None in sys.modules does that even where the real package is installed,
    which popping the fake would not. Assigned rather than
    `monkeypatch.setitem`: monkeypatch undoes after the `env` teardown and
    would put the fake back, and a `types.ModuleType` has `__spec__` None, so
    a leaked one makes every later `importlib.util.find_spec("mamba_ssm")`
    raise ValueError. `env` owns the key and restores it.
    """
    sys.modules["mamba_ssm"] = None


def test_pre_ampere_disables_fast_path(env):
    model_mod, iu, _hk = env
    assert patch() is True
    assert model_mod.is_fast_path_available is False
    assert model_mod.selective_state_update is None
    assert model_mod.mamba_chunk_scan_combined is None
    assert model_mod.mamba_split_conv1d_scan_combined is None
    # Modules imported later must also see it as unavailable.
    assert iu.is_mamba_ssm_available() is False
    assert iu.is_mamba_2_ssm_available() is False


def _fake_mixer_module(mod_name, cls_name, weight_attr):
    """A stand-in for jamba/zamba: the flag plus a mixer that raises on it.

    Both bind `self.use_fast_kernels = config.use_mamba_kernels` (True by
    default) at construction, and their forward raises rather than falling back
    when the module flag is False (transformers 4.57.6
    models/jamba/modeling_jamba.py:815-821, models/zamba/modeling_zamba.py:553-560;
    zamba still does in 5.5.0 and 5.14.1).
    """
    mod = types.ModuleType(mod_name)
    mod.is_fast_path_available = True

    class Mixer:
        def __init__(self):
            self.use_fast_kernels = True
            setattr(self, weight_attr,
                    types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda")))
        def forward(self, *args, **kwargs):
            if self.use_fast_kernels:
                if not mod.is_fast_path_available:
                    raise ValueError("Fast Mamba kernels are not available.")
                return "FAST PATH"
            return "SLOW PATH"

    Mixer.__name__ = cls_name
    setattr(mod, cls_name, Mixer)
    sys.modules[mod_name] = mod
    return mod, Mixer


@pytest.mark.parametrize("mod_name, cls_name, weight_attr", [
    ("transformers.models.jamba.modeling_jamba", "JambaMambaMixer", "x_proj"),
    ("transformers.models.zamba.modeling_zamba", "ZambaMambaMixer", "x_proj_weight"),
])
def test_jamba_and_zamba_reach_the_slow_path_not_a_valueerror(
    env, mod_name, cls_name, weight_attr,
):
    saved = sys.modules.get(mod_name)
    mod, Mixer = _fake_mixer_module(mod_name, cls_name, weight_attr)
    try:
        mixer = Mixer()
        assert mixer.forward() == "FAST PATH"
        assert patch() is True
        assert mod.is_fast_path_available is False
        # The guard promises the slow path; it must not hand back a ValueError.
        assert mixer.forward() == "SLOW PATH"
        assert Mixer().forward() == "SLOW PATH", "instances built later too"
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def test_the_mixer_wrapper_is_applied_once(env):
    mod_name = "transformers.models.zamba.modeling_zamba"
    saved = sys.modules.get(mod_name)
    mod, Mixer = _fake_mixer_module(mod_name, "ZambaMambaMixer", "x_proj_weight")
    try:
        assert patch() is True
        first = Mixer.forward
        mod.is_fast_path_available = True   # a later phase sees it truthy again
        assert patch() is True
        assert Mixer.forward is first, "must not stack on every phase"
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def _lazy_flag_mixer_module(mod_name, cls_name, weight_attr, hk):
    """transformers >= 5.3's shape, where the module flag does not exist yet.

    5.3 moved the kernel resolution into the mixer's `__init__`, so the module
    carries no `is_fast_path_available` at all until a mixer has been built
    (`global is_fast_path_available`, models/zamba/modeling_zamba.py:257 in
    5.5.0) and `forward` recomputes it as a local (line 449). Verified against
    the real 5.5.0 wheel: right after
    `import transformers.models.zamba.modeling_zamba`,
    `"is_fast_path_available" in module.__dict__` is False.

    Availability is read back from `_HUB_KERNEL_MAPPING` the way
    `lazy_load_kernel` does, so unregistering the Hub kernels is what makes the
    fast path unavailable here, exactly as on a real 5.5 install.
    """
    mod = types.ModuleType(mod_name)

    def _kernels_available():
        return "mamba-ssm" in hk._HUB_KERNEL_MAPPING

    class Mixer:
        def __init__(self):
            self.use_fast_kernels = True                  # config.use_mamba_kernels
            setattr(self, weight_attr,
                    types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda")))
            mod.is_fast_path_available = _kernels_available()

        def forward(self, *args, **kwargs):
            is_fast_path_available = _kernels_available()  # a local, as in 5.5
            if self.use_fast_kernels:
                if not is_fast_path_available:
                    raise ValueError("Fast Mamba kernels are not available.")
                return "FAST PATH"
            return "SLOW PATH"

    Mixer.__name__ = cls_name
    setattr(mod, cls_name, Mixer)
    sys.modules[mod_name] = mod
    return mod, Mixer


@pytest.mark.parametrize("mod_name, cls_name, weight_attr", [
    ("transformers.models.jamba.modeling_jamba", "JambaMambaMixer", "x_proj"),
    ("transformers.models.zamba.modeling_zamba", "ZambaMambaMixer", "x_proj_weight"),
])
def test_the_mixer_is_wrapped_before_any_mixer_exists(
    env, mod_name, cls_name, weight_attr,
):
    """The wrapper must not hang off the module's `is_fast_path_available`.

    On transformers >= 5.3 that name is absent from the module dict until a
    mixer has been constructed, and on 4.x the predicate flip in step 1 has
    already made it False by the time the modeling module is imported. Gating
    the wrapper on it left the mixer unwrapped in both orderings, and the guard
    then handed back `ValueError: Fast Mamba kernels are not available` instead
    of the slow path. Reproduced on real transformers 5.5.0 and 4.57.6.
    """
    _model_mod, _iu, hk = env
    saved = sys.modules.get(mod_name)
    mod, Mixer = _lazy_flag_mixer_module(mod_name, cls_name, weight_attr, hk)
    try:
        assert "is_fast_path_available" not in mod.__dict__, "the 5.3+ shape"
        patch()
        assert Mixer.__dict__.get("_unsloth_slow_only", False) is True
        mixer = Mixer()
        assert mod.is_fast_path_available is False, "step 0 removed the kernels"
        assert mixer.forward() == "SLOW PATH"
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def test_the_mixer_is_wrapped_without_a_local_mamba_ssm_wheel(env):
    """Unregistering the Hub kernels is on its own enough to make Zamba raise,
    so the wrapper has to be installed before the local-wheel early return."""
    _model_mod, _iu, hk = env
    mod_name = "transformers.models.zamba.modeling_zamba"
    saved = sys.modules.get(mod_name)
    mod, Mixer = _lazy_flag_mixer_module(
        mod_name, "ZambaMambaMixer", "x_proj_weight", hk)
    try:
        _no_local_wheel()
        assert patch() is None
        assert Mixer().forward() == "SLOW PATH"
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def test_the_lazy_flag_wrapper_is_applied_once(env):
    _model_mod, _iu, hk = env
    mod_name = "transformers.models.zamba.modeling_zamba"
    saved = sys.modules.get(mod_name)
    _mod, Mixer = _lazy_flag_mixer_module(
        mod_name, "ZambaMambaMixer", "x_proj_weight", hk)
    try:
        patch()
        first = Mixer.forward
        patch()
        assert Mixer.forward is first, "must not stack on every phase"
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def test_the_config_flag_is_cleared_for_jamba_5_5(env):
    """Jamba on 5.5 routes on `self.config.use_mamba_kernels`, not on the
    instance flag (models/jamba/modeling_jamba.py:462-472), and clears exactly
    that field on its own fallback."""
    _model_mod, _iu, _hk = env
    mod_name = "transformers.models.jamba.modeling_jamba"
    saved = sys.modules.get(mod_name)
    mod = types.ModuleType(mod_name)

    class JambaMambaMixer:
        def __init__(self):
            self.config = types.SimpleNamespace(use_mamba_kernels = True)
        def forward(self, *args, **kwargs):
            return "FAST PATH" if self.config.use_mamba_kernels else "SLOW PATH"

    mod.JambaMambaMixer = JambaMambaMixer
    sys.modules[mod_name] = mod
    try:
        assert JambaMambaMixer().forward() == "FAST PATH"
        patch()
        mixer = JambaMambaMixer()
        assert mixer.forward() == "SLOW PATH"
        assert mixer.config.use_mamba_kernels is False
    finally:
        if saved is None: sys.modules.pop(mod_name, None)
        else: sys.modules[mod_name] = saved


def _fake_config_module(mod_name, cls_name):
    """A stand-in for jamba/zamba's configuration module.

    `use_mamba_kernels` defaults to True and is also written out into every
    checkpoint's config.json, so both the default and an explicit True have to
    end up cleared (transformers 4.57.6 configuration_zamba.py:151/193, and the
    strict-dataclass field `use_mamba_kernels: bool = True` on 5.5.0:81).
    """
    mod = types.ModuleType(mod_name)

    class Config:
        def __init__(self, use_mamba_kernels = True, **kwargs):
            self.use_mamba_kernels = use_mamba_kernels

    Config.__name__ = cls_name
    setattr(mod, cls_name, Config)
    sys.modules[mod_name] = mod
    return mod, Config


class _ConfigBoundMixer:
    """Both families bind the config flag once, in the mixer's __init__."""
    def __init__(self, config):
        self.use_fast_kernels = config.use_mamba_kernels
    def forward(self, *args, **kwargs):
        if self.use_fast_kernels:
            raise ValueError("Fast Mamba kernels are not available.")
        return "SLOW PATH"


@pytest.mark.parametrize("cfg_mod_name, cls_name, modeling", [
    (CONFIG_MODS[0], "JambaConfig", "transformers.models.jamba.modeling_jamba"),
    (CONFIG_MODS[1], "ZambaConfig", "transformers.models.zamba.modeling_zamba"),
])
def test_the_config_flag_is_cleared_when_modeling_is_imported_later(
    env, cfg_mod_name, cls_name, modeling,
):
    """The mixer wrapper alone cannot fire when nothing has imported modeling.

    `unsloth_compile_transformers` returns at models/_utils.py:3277 when
    `trust_remote_code = True`, before both the pre_compile and the post_compile
    phase, so the only phase that ran is "init", at `import unsloth`, and
    transformers imports modeling_zamba later, from inside `from_pretrained`.
    Measured: with trust_remote_code = True there is no
    `_run_temporary_patches` call at all, and the module is still absent from
    sys.modules when the call returns. Step 0 has already unregistered the Hub
    kernels by then, so an unwrapped mixer raises instead of taking the slow
    path this guard promises. Clearing the config flag is what covers it.
    """
    saved = sys.modules.get(modeling)
    sys.modules.pop(modeling, None)
    _mod, Config = _fake_config_module(cfg_mod_name, cls_name)
    try:
        with pytest.raises(ValueError):
            _ConfigBoundMixer(Config()).forward()
        patch()
        assert Config().use_mamba_kernels is False
        assert Config(use_mamba_kernels = True).use_mamba_kernels is False, \
            "a checkpoint's config.json spells the flag out"
        assert _ConfigBoundMixer(Config()).forward() == "SLOW PATH"
        assert modeling not in sys.modules, "and without importing modeling"
    finally:
        if saved is None: sys.modules.pop(modeling, None)
        else: sys.modules[modeling] = saved


def test_the_config_wrapper_is_applied_once(env):
    _mod, Config = _fake_config_module(CONFIG_MODS[1], "ZambaConfig")
    patch()
    first = Config.__init__
    patch()
    assert Config.__init__ is first, "must not stack on every phase"
    assert Config().use_mamba_kernels is False


def test_the_config_step_leaves_no_attribute_on_the_config_class(env):
    """transformers 5 configs are strict dataclasses, so the idempotency marker
    goes on the wrapper function rather than on the class."""
    _mod, Config = _fake_config_module(CONFIG_MODS[1], "ZambaConfig")
    before = set(Config.__dict__)
    patch()
    assert set(Config.__dict__) == before, "no new class attribute"
    assert Config.__init__._unsloth_slow_only is True, "the marker rides the wrapper"


def test_the_config_step_does_not_walk_sys_modules(env):
    """Two import_module calls on names we already know, not a third pass over
    every loaded module."""
    assert _SRC.count("for _name, _mod in list(sys.modules.items()):") == 1


def test_the_mixer_scan_does_not_walk_sys_modules(env):
    """The two mixers are reached by direct `sys.modules` lookup, not by another
    pass over every loaded module, so the extra work is two dict gets."""
    assert 'sys.modules.get(_name, None)' in _SRC
    assert _SRC.count("for _name, _mod in list(sys.modules.items()):") == 1


def test_alias_modules_are_not_probed_through_getattr():
    """transformers 5 registers ~200 `image_processing_<m>_fast` alias modules
    whose catch-all __getattr__ imports the real image processor. Probing them
    with getattr costs seconds per phase and can propagate an ImportError out
    of `import unsloth`, so the scan must read __dict__ instead."""
    assert '_mod.__dict__.get("is_fast_path_available", False)' in _SRC
    assert 'getattr(_mod, "is_fast_path_available", False)' not in _SRC


def test_ampere_keeps_fast_path(env, monkeypatch):
    model_mod, iu, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (8, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_hopper_keeps_fast_path(env, monkeypatch):
    model_mod, _, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (9, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_rocm_is_untouched(env, monkeypatch):
    model_mod, iu, _hk = env
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_no_cuda_is_untouched(env, monkeypatch):
    model_mod, _, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(available = False), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_no_mamba_ssm_is_untouched(env):
    model_mod, iu, _hk = env
    _no_local_wheel()
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


# ---- transformers 5's Hub kernels ----------------------------------------
#
# lazy_load_kernel resolves mamba-ssm from kernels-community when the `kernels`
# package is installed. It needs no local wheel and never calls
# is_mamba_ssm_available, so the predicates above cannot reach it.

def test_pre_ampere_unregisters_the_hub_kernels(env):
    _model_mod, _iu, hk = env
    patch()
    for name in ("mamba-ssm", "falcon_mamba-ssm", "causal-conv1d"):
        assert name not in hk._HUB_KERNEL_MAPPING
        # A module already resolved into the cache must stop being handed out.
        assert hk._KERNEL_MODULE_MAPPING[name] is None


def test_unrelated_hub_kernels_are_left_registered(env):
    _model_mod, _iu, hk = env
    patch()
    assert "deep-gemm" in hk._HUB_KERNEL_MAPPING


def test_hub_kernels_go_even_without_the_local_wheel(env):
    """The Hub path is the one that reaches a pre-Ampere GPU with no wheel
    installed, so unregistering has to precede the mamba_ssm import."""
    _model_mod, _iu, hk = env
    _no_local_wheel()
    assert patch() is None
    assert "mamba-ssm" not in hk._HUB_KERNEL_MAPPING


def test_ampere_keeps_the_hub_kernels(env, monkeypatch):
    _model_mod, _iu, hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (8, 0)), raising = False)
    assert patch() is None
    assert "mamba-ssm" in hk._HUB_KERNEL_MAPPING


def test_non_transformers_modules_are_left_alone(env):
    vllm = types.ModuleType("vllm.model_executor.layers.mamba")
    vllm.is_fast_path_available = True
    sys.modules["vllm.model_executor.layers.mamba"] = vllm
    try:
        patch()
        assert vllm.is_fast_path_available is True
    finally:
        sys.modules.pop("vllm.model_executor.layers.mamba", None)


def test_registered_as_a_temporary_patch():
    assert "TEMPORARY_PATCHES.append(patch_mamba_ssm_pre_ampere_fallback)" in _SRC


# Keep last: it checks what the `env` fixture left behind after teardown.
def test_the_fixture_leaves_no_stub_bound_on_transformers_utils():
    """The stub must not outlive the fixture. `import a.b.c as x` goes through
    getattr(a.b, "c"), so a leaked attribute is handed to every later
    `import transformers.utils.import_utils as iu` -- the form used by
    tests/test_vendor_fla.py:219 and unsloth_zoo/temporary_patches/fla_vendor.py:473."""
    utils_mod = sys.modules.get("transformers.utils")
    if utils_mod is None or getattr(utils_mod, "__file__", None) is None:
        pytest.skip("real transformers.utils was never imported in this session")
    import importlib
    real = importlib.import_module("transformers.utils.import_utils")
    assert getattr(utils_mod, "import_utils", real) is real
    import transformers.utils.import_utils as reimported
    assert reimported is real


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
