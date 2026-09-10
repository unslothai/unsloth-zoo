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

"""Class-valued symbols of the Apple Silicon triton stub.

Unseeded ``triton.*`` attributes come back as ``_Noop`` instances, and an
instance is not a type, so ``isinstance`` / ``issubclass`` / ``except`` on one
raises ``TypeError: isinstance() arg 2 must be a type``. torch >= 2.10 imports
``triton.runtime.jit.JITFunction`` at import time of ``torch.utils.flop_counter``
and isinstance()s it: with triton truly absent that import fails and torch falls
back to NoneType, but the stub makes it succeed, so every name real triton
exposes as a class has to be seeded as a real class.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STUB_PATH = REPO_ROOT / "unsloth_zoo" / "stubs" / "triton_stub.py"


def _load_stub_module():
    """Load triton_stub.py standalone, without importing unsloth_zoo itself."""
    spec = importlib.util.spec_from_file_location("_triton_stub_under_test", STUB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Real triton's class-valued public surface, per module.
_TRITON_CLASSES = {
    "triton.runtime.jit": ["JITFunction", "JITCallable", "KernelInterface",
                           "KernelParam", "MockTensor", "TensorWrapper"],
    "triton.runtime.autotuner": ["Autotuner", "Heuristics", "Config",
                                 "OutOfResources", "PTXASError"],
    "triton.runtime.errors": ["TritonError", "InterpreterError", "OutOfResources",
                              "PTXASError", "AutotunerError", "IntelGPUError"],
    "triton.runtime.interpreter": ["InterpretedFunction"],
    "triton.errors": ["TritonError"],
    "triton.compiler": ["CompiledKernel", "ASTSource", "IRSource", "LazyDict",
                        "CompilationError"],
    "triton.compiler.compiler": ["CompiledKernel", "ASTSource", "IRSource", "LazyDict"],
    "triton.compiler.errors": ["CompilationError", "CompileTimeAssertionFailure",
                               "UnsupportedLanguageConstruct"],
    "triton.backends.compiler": ["GPUTarget", "BaseBackend"],
    "triton.tools.tensor_descriptor": ["TensorDescriptor"],
    "triton.language": ["constexpr", "dtype", "tensor", "block_type", "pointer_type"],
    "triton.language.core": ["constexpr", "dtype", "tensor", "block_type", "pointer_type"],
}

# Exceptions must be catchable: `except <non-class>` raises TypeError too.
_TRITON_EXCEPTIONS = ["TritonError", "InterpreterError", "OutOfResources",
                      "PTXASError", "AutotunerError", "CompilationError",
                      "IntelGPUError"]


@pytest.mark.parametrize("module_path,names", sorted(_TRITON_CLASSES.items()))
def test_seeded_symbols_are_real_classes(module_path, names):
    stub = _load_stub_module()
    module = stub
    for part in module_path.split(".")[1:]:
        module = getattr(module, part)
    for name in names:
        attr = getattr(module, name)
        assert isinstance(attr, type), f"{module_path}.{name} is {attr!r}, not a class"
        # The check that actually crashed torch.
        assert isinstance(object(), attr) is False


@pytest.mark.parametrize("name", _TRITON_EXCEPTIONS)
def test_error_symbols_are_catchable(name):
    stub = _load_stub_module()
    error = getattr(stub, name)
    assert issubclass(error, Exception)
    with pytest.raises(error):
        raise error("stub")


def test_stub_classes_stay_permissive():
    """Seeding must not cost the stub its permissive attribute access."""
    stub = _load_stub_module()
    assert isinstance(stub.JITFunction.anything_unknown, stub._Noop)
    assert isinstance(stub.JITFunction().anything_unknown, stub._Noop)


_SUBPROCESS = textwrap.dedent(
    """
    import importlib.util, sys
    sys.path.insert(0, {tests!r})
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()          # spoof Darwin/arm64, as on a Mac host

    spec = importlib.util.spec_from_file_location("triton_stub", {stub!r})
    stub = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stub
    spec.loader.exec_module(stub)
    stub.inject_into_sys_modules()

    from triton.runtime.jit import JITFunction
    assert isinstance(JITFunction, type), JITFunction
    import torch.utils.flop_counter          # crashed here pre-fix
    import torch.utils._runtime_estimation   # the import unsloth.save reaches
    print("OK")
    """
)


# inductor's two other class-sensitive uses, verbatim from torch main:
# triton_compat imports IntelGPUError and triton_heuristics catches it;
# _interpret_args_grid isinstance()s InterpretedFunction.
_INDUCTOR_SUBPROCESS = textwrap.dedent(
    """
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("triton_stub", {stub!r})
    stub = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stub
    spec.loader.exec_module(stub)
    stub.inject_into_sys_modules()

    try:
        from triton.runtime.errors import IntelGPUError
    except ImportError:
        class IntelGPUError(Exception): pass
    from triton.runtime.errors import OutOfResources, PTXASError
    try:
        raise RuntimeError("compile failed")
    except (OutOfResources, PTXASError, IntelGPUError):
        raise AssertionError("stub error classes must not swallow RuntimeError")
    except RuntimeError:
        pass

    from triton.runtime.interpreter import InterpretedFunction
    assert isinstance(object(), InterpretedFunction) is False
    print("OK")
    """
)


def test_inductor_error_and_interpreter_symbols():
    code = _INDUCTOR_SUBPROCESS.format(stub=str(STUB_PATH))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def _torch_flop_counter_imports_jitfunction() -> bool:
    spec = importlib.util.find_spec("torch.utils.flop_counter")
    if spec is None or spec.origin is None:
        return False
    # Read the source rather than import it: importing is what used to crash.
    return "from triton.runtime.jit import JITFunction" in \
        pathlib.Path(spec.origin).read_text(encoding="utf-8")


def test_flop_counter_imports_under_the_stub():
    """torch.utils.flop_counter must import with the stub installed."""
    pytest.importorskip("torch")
    if not _torch_flop_counter_imports_jitfunction():
        pytest.skip("this torch does not import triton.runtime.jit.JITFunction")
    code = _SUBPROCESS.format(tests=str(REPO_ROOT / "tests"), stub=str(STUB_PATH))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
