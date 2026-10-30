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

"""The up-front CPU Inductor probe skips only on a missing C++ compiler."""

import json
import os

import pytest

import _cpu_compiler_skip as skip


def _report(*chain, stage = "compiler"):
    return "PROBE " + json.dumps({"stage": stage, "chain": [list(x) for x in chain]}) + "\n"


@pytest.mark.parametrize("chain", [
    [("InductorError", "InvalidCxxCompiler: No working C++ compiler found"), ("InvalidCxxCompiler", "No working")],
    [("BackendCompilerFailed", "backend='inductor' raised:"), ("InvalidCxxCompiler", "No working C++ compiler")],
    [("InductorError", "RuntimeError: Compiler: cl is not found."), ("RuntimeError", "Compiler: cl is not found.")],
])
def test_a_missing_compiler_skips(chain, monkeypatch):
    monkeypatch.setattr(skip, "_probe", lambda env: (3, _report(*chain)))
    with pytest.raises(pytest.skip.Exception, match = "no C\\+\\+ compiler"):
        skip.skip_without_cpu_inductor({})


@pytest.mark.parametrize("result", [
    (0, ""),  # compiles
    (3, _report(("ImportError", "cannot import name 'InvalidCxxCompiler'"))),  # names it, is not it
    (3, _report(("AssertionError", "InvalidCxxCompiler was expected"))),
    (3, _report(("RuntimeError", "Compiler: cl is not found. Retrying"))),
    (1, _report(("InvalidCxxCompiler", ""))),  # not the probe's own except branch
    (3, _report(("InvalidCxxCompiler", ""), stage = "import")),  # Inductor itself is missing
    (3, "PROBE not json"),
    (3, "PROBE null"),
    (3, "PROBE 7"),
    (3, "PROBE " + json.dumps([["InvalidCxxCompiler", ""]])),  # not the report shape
    (3, "PROBE " + json.dumps({"stage": "compiler", "chain": None})),
    (3, "PROBE " + json.dumps({"stage": "compiler", "chain": [None, 3, ["InvalidCxxCompiler"], [1, 2]]})),
    (3, "PROBE " + json.dumps({"stage": "compiler", "chain": [["InvalidCxxCompiler", None]]})),
    (None, ""),  # timed out
])
def test_any_other_probe_result_runs_the_check(result, monkeypatch):
    monkeypatch.setattr(skip, "_probe", lambda env: result)
    skip.skip_without_cpu_inductor({})


def test_the_probe_is_cached_per_child_environment(monkeypatch):
    calls = []

    class _Proc:
        returncode, stdout = 0, ""

    monkeypatch.setattr(skip, "_probes", {})
    monkeypatch.setattr(skip.subprocess, "run", lambda *a, **k: calls.append(k["env"]) or _Proc())
    envs = ({"CXX": "g++"}, {"CXX": "g++"}, {"CXX": "g++", "HOME": "/x"}, {"CXX": "g++", "TORCH_LOGS": "+x"})
    for env in envs:
        skip.skip_without_cpu_inductor(dict(env))
    # The child receives the whole env, so every distinct env is probed once and an identical one is reused.
    assert calls == [envs[0], envs[2], envs[3]], calls


def test_the_probe_cache_ignores_pytest_bookkeeping_and_keys_on_the_cwd(monkeypatch, tmp_path):
    calls = []

    class _Proc:
        returncode, stdout = 0, ""

    monkeypatch.setattr(skip, "_probes", {})
    monkeypatch.setattr(skip.subprocess, "run", lambda *a, **k: calls.append(os.getcwd()) or _Proc())
    skip.skip_without_cpu_inductor({"CXX": "g++", "PYTEST_CURRENT_TEST": "a (call)"})
    skip.skip_without_cpu_inductor({"CXX": "g++", "PYTEST_CURRENT_TEST": "b (call)"})
    assert len(calls) == 1, calls
    monkeypatch.chdir(tmp_path)
    skip.skip_without_cpu_inductor({"CXX": "g++", "PYTEST_CURRENT_TEST": "c (call)"})
    assert len(calls) == 2, calls


def _bogus_compiler_env(**extra):
    # Keep the rest of the env (e.g. TORCHINDUCTOR_CACHE_DIR); only drop the settings that skip compilation.
    drop = {"TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_SUPPRESS_ERRORS"}
    env = {k: v for k, v in os.environ.items() if k.upper() not in drop}
    env["CXX"] = os.path.join(os.sep, "nonexistent", "cxx-for-test")
    env.update(extra)
    return env


def _classified(env):
    pytest.importorskip("torch")
    returncode, stdout = skip._probe(env)
    if returncode == 0:
        pytest.skip("this torch finds a C++ compiler without the configured one, so there is nothing to classify")
    report = skip._report(stdout)
    if report is not None and report.get("stage") == "import":
        pytest.skip(f"no CPU Inductor here: {str(report['chain'])[:300]}")
    return skip.missing_compiler(returncode, stdout), stdout


def test_a_real_probe_with_a_bogus_compiler_skips():
    missing, stdout = _classified(_bogus_compiler_env())
    assert missing, stdout[-2000:]


def test_inherited_compile_disable_settings_cannot_hide_a_missing_compiler():
    # suppress_errors / compile-disable used to let a trivial compile succeed eagerly while the checks still failed.
    missing, stdout = _classified(_bogus_compiler_env())  # precondition: the bogus compiler is seen as missing
    assert missing, stdout[-2000:]
    env = _bogus_compiler_env(TORCHDYNAMO_SUPPRESS_ERRORS = "1", TORCH_COMPILE_DISABLE = "1", TORCHDYNAMO_DISABLE = "1")
    returncode, stdout = skip._probe(env)
    assert skip.missing_compiler(returncode, stdout), (returncode, stdout[-2000:])
