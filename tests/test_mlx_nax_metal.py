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

"""NAX detection, GPU identity, MLX gap tracking and the kernel probe, on real MLX."""

from __future__ import annotations

import subprocess

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX: the shim has no Metal device or metallib", allow_module_level = True)


@pytest.mark.parametrize("architecture,release,expected", [
    ("applegpu_g17s", "26.2", True),
    ("applegpu_g17d", "27.0", True),
    ("applegpu_g18p", "26.5.1", True),
    ("applegpu_g17p", "26.5", False),   # the `p` class needs generation 18
    ("applegpu_g15s", "26.5", False),
    ("applegpu_g17s", "26.1", False),
    ("air64_v27", "26.5", False),       # paravirtual runners
])
def test_nax_detection_mirrors_mlx(monkeypatch, architecture, release, expected):
    from unsloth_zoo.mlx import nax

    monkeypatch.setattr(nax.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(nax.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(nax.platform, "mac_ver", lambda: (release, ("", "", ""), ""))
    monkeypatch.setattr(nax.mx, "device_info", lambda: {"architecture": architecture})
    nax._nax_gpu.cache_clear()
    try:
        assert nax._nax_gpu() is expected
    finally:
        nax._nax_gpu.cache_clear()


@pytest.mark.parametrize("metallib,env,expected", [
    (b"..._nax_...", "1", True),
    (b"...steel...", "1", False),   # a wheel built below macOS 26.2 ships no NAX kernels
    (None, "1", False),
    (b"..._nax_...", "0", False),
])
def test_nax_available_needs_stock_kernels_and_honors_kill_switch(
        monkeypatch, tmp_path, metallib, env, expected):
    from unsloth_zoo.mlx import nax

    if metallib is not None:
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "mlx.metallib").write_bytes(metallib)
    monkeypatch.setattr(nax.mx, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(nax, "_nax_gpu", lambda: True)
    monkeypatch.setenv("UNSLOTH_MLX_NAX", env)
    nax._stock_nax_kernels.cache_clear()
    try:
        assert nax.nax_available() is expected
    finally:
        nax._stock_nax_kernels.cache_clear()


def test_every_gap_names_the_pinned_mlx_release():
    import re
    from pathlib import Path
    from unsloth_zoo.mlx import nax

    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    pinned = set(re.findall(r'"mlx==([0-9.]+)', pyproject))
    assert len(pinned) == 1
    for name, gap in nax._GAPS.items():
        assert pinned <= set(gap.open_in), f"re-measure {name} for mlx {pinned}"


@pytest.mark.parametrize("version,closed_on_main,expected", [
    ("0.32.2", True, True),
    ("0.33.0", True, False),   # the release carrying the upstream fix
    ("0.31.1", True, False),   # unmeasured older release
    ("0.33.0", False, True),
])
def test_gap_closes_off_the_measured_releases(monkeypatch, version, closed_on_main, expected):
    from unsloth_zoo.mlx import nax

    monkeypatch.setitem(nax._GAPS, "probe", nax.Gap(open_in = ("0.32.2",), closed_on_main = closed_on_main))
    monkeypatch.setattr(nax.mx, "__version__", version)
    assert nax.gap_open("probe") is expected


def test_kernel_probe_runs_once_per_build_and_survives_a_bad_cache(monkeypatch, tmp_path):
    from unsloth_zoo.mlx import nax

    (tmp_path / "unsloth_probe_target.py").write_text(
        "import os, sys\n"
        "def good():\n    open(os.environ['PROBE_LOG'], 'a').write('ran\\n')\n"
        "def bad():\n    sys.exit('kernel refused')\n")
    log, cache = tmp_path / "log", tmp_path / "probes.json"
    cache.write_text("{not json")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PROBE_LOG", str(log))
    monkeypatch.setattr(nax, "_PROBE_PATH", str(cache))
    monkeypatch.setattr(nax, "_PROBES", {})
    monkeypatch.setattr(nax, "nax_available", lambda: False)
    assert nax.kernel_probe_passed("good", "unsloth_probe_target", "good") is False
    assert not log.exists()
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    assert nax.kernel_probe_passed("good", "unsloth_probe_target", "good") is True
    assert nax.kernel_probe_passed("bad", "unsloth_probe_target", "bad") is False
    monkeypatch.setattr(nax, "_PROBES", {})   # a new process reads the disk cache
    assert nax.kernel_probe_passed("good", "unsloth_probe_target", "good") is True
    assert nax.kernel_probe_passed("bad", "unsloth_probe_target", "bad") is False
    assert log.read_text() == "ran\n"
    assert sorted(key.rsplit("|", 1)[1] for key in __import__("json").loads(cache.read_text())) == ["bad", "good"]
    monkeypatch.setattr(nax, "_run_probe", lambda *args: None)   # a timeout is retried by the next process
    assert nax.kernel_probe_passed("slow", "unsloth_probe_target", "good") is False
    assert "|slow" not in cache.read_text()
    monkeypatch.setattr(nax, "_stored_probes", lambda: 1 / 0)
    assert nax.kernel_probe_passed("broken", "unsloth_probe_target", "good") is False


def test_kernel_probe_does_not_import_from_the_working_directory(monkeypatch, tmp_path):
    from unsloth_zoo.mlx import nax

    (tmp_path / "unsloth_planted_probe.py").write_text("def good():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    assert nax._run_probe("unsloth_planted_probe", "good") != ""
    monkeypatch.syspath_prepend(str(tmp_path))   # on this process's own path it is importable
    assert nax._run_probe("unsloth_planted_probe", "good") == ""


def test_gpu_core_count_reads_the_ioregistry(monkeypatch):
    from unittest import mock
    from unsloth_zoo.mlx import nax

    # Paravirtual GPUs (CI VMs, `air64_*`) publish no core count.
    assert nax._gpu_generation() is None or nax._gpu_core_count() > 0
    ioreg = lambda code, out = '"gpu-core-count" = 12': subprocess.CompletedProcess([], code, out, "")
    for result, cores in ((ioreg(0), 12), (ioreg(1), None), (ioreg(0, '"gpu-core-count" = <0c>'), None),
                          (FileNotFoundError("ioreg"), None)):
        monkeypatch.setattr(nax.subprocess, "run", mock.Mock(side_effect = [result]))
        assert nax._gpu_core_count.__wrapped__() == cores


def test_gpu_generation_reads_the_architecture(monkeypatch):
    from unsloth_zoo.mlx import nax

    for architecture, generation in (("applegpu_g17s", 17), ("applegpu_g18p", 18), ("air64_v27", None), ("", None)):
        monkeypatch.setattr(nax, "_gpu_architecture", lambda: architecture)
        assert nax._gpu_generation() == generation, architecture
