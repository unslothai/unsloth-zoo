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

"""Two-rank checks for recovery coordination and temp-mode agreement."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import pathlib
import time

import pytest


def _gloo_worker(rank, root_text, scenario):
    import torch.distributed as dist
    from unsloth_zoo import compiler

    root = pathlib.Path(root_text)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    compiler.UNSLOTH_COMPILE_LOCATION = str(root / "shared")
    compiler.UNSLOTH_COMPILE_USE_TEMP = (
        scenario == "preinit-temp-divergence" and rank == 1
    )
    local_temp = root / f"rank{rank}-tmp"
    # Safe to patch stdlib module objects: each worker is a spawned interpreter.
    compiler.tempfile.gettempdir = lambda: str(local_temp)

    try:
        dist.init_process_group(
            "gloo",
            init_method=(root / "init").resolve().as_uri(),
            rank=rank,
            world_size=2,
        )

        if scenario == "preinit-temp-divergence":
            location, use_temp = compiler.get_compile_folder(use_tempfile=False)
            expected = local_temp / "shared"
            assert use_temp and pathlib.Path(location) == expected
        else:
            name = f"pr967_gloo_{scenario.replace('-', '_')}"
            if scenario == "one-rank-import-failure":
                real_import = compiler.importlib.import_module
                failed = False

                def fail_once(module_name, package=None):
                    nonlocal failed
                    if rank == 1 and module_name == name and not failed:
                        failed = True
                        raise ImportError("simulated rank-local import failure")
                    return real_import(module_name, package)

                # Safe to patch globally: every worker is a spawned interpreter.
                compiler.importlib.import_module = fail_once

            module = compiler.create_new_function(
                name,
                f"def {name}_fn(x):\n    return x * 2\n",
                "gloo_probe",
                {},
                overwrite=True,
            )
            assert getattr(module, f"{name}_fn")(21) == 42

        (root / f"result-{rank}").write_text("ok", encoding="utf-8")
        dist.barrier()
        dist.destroy_process_group()
    except Exception as error:
        (root / f"result-{rank}").write_text(
            f"{type(error).__name__}: {error}",
            encoding="utf-8",
        )
        raise


@pytest.mark.parametrize(
    "scenario",
    ["preinit-temp-divergence", "one-rank-import-failure"],
)
def test_two_rank_gloo_recovery_coordination(tmp_path, scenario):
    if importlib.util.find_spec("unsloth") is None:
        pytest.skip("requires the companion `unsloth` package")
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires an available torch.distributed gloo backend")

    (tmp_path / "shared").mkdir()
    for rank in range(2):
        (tmp_path / f"rank{rank}-tmp").mkdir()

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_gloo_worker, args=(rank, str(tmp_path), scenario))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 30
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail(f"{scenario} hung in a collective")
        assert process.exitcode == 0

    assert [
        (tmp_path / f"result-{rank}").read_text(encoding="utf-8")
        for rank in range(2)
    ] == ["ok", "ok"]
