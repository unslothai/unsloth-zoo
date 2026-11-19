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

"""MLX startup skips the GPU-only temporary_patches, so it must install the folder-dataset metadata guard itself
(CVE-2026-66007: a metadata file_name of "../x" or an absolute path makes imagefolder read any local file).

The runtime module is loaded from its file in a fresh interpreter, so unsloth_zoo's GPU init (which installs the
guard by itself) never runs and cannot mask the result."""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("datasets")
pytest.importorskip("PIL")

SCRIPT = textwrap.dedent(
    """
    import importlib.util, json, sys
    runtime_path, data_dir, pre_import = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    if pre_import:
        import datasets.packaged_modules.folder_based_builder.folder_based_builder
    spec = importlib.util.spec_from_file_location("mlx_runtime_under_test", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.install_mlx_security_patches()
    runtime.install_mlx_security_patches()
    assert "torch" not in sys.modules
    from datasets import load_dataset
    try:
        ds = load_dataset("imagefolder", data_dir = data_dir, split = "train", cache_dir = data_dir + "_cache")
        ds = ds.cast_column("image", __import__("datasets").Image(decode = False))
        print(json.dumps({"loaded": [row["image"]["path"] for row in ds]}))
    except Exception as exc:
        print(json.dumps({"refused": str(exc)[:300]}))
    """
)


def _dataset(tmp_path):
    from PIL import Image

    secret = tmp_path / "secret.png"
    Image.new("RGB", (2, 2)).save(secret)
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (2, 2)).save(data / "ok.png")
    rows = [{"file_name": "ok.png", "label": 0}, {"file_name": str(secret), "label": 1}]
    (data / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return data


@pytest.mark.parametrize("pre_import", ["0", "1"])
def test_outside_file_name_refused(tmp_path, pre_import):
    data = _dataset(tmp_path)
    runtime = ROOT / "unsloth_zoo" / "mlx" / "runtime.py"
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(runtime), str(data), pre_import],
        capture_output = True, text = True, timeout = 300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert "refused" in result and "must be a relative path" in result["refused"], result


def test_mlx_branch_installs_guard():
    tree = ast.parse((ROOT / "unsloth_zoo" / "__init__.py").read_text(encoding = "utf-8"))
    branch = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "_is_mlx_only")
    calls = [ast.unparse(n.func) for n in ast.walk(ast.Module(body = branch.body, type_ignores = [])) if isinstance(n, ast.Call)]
    assert "install_mlx_security_patches" in calls


def test_normal_folder_dataset_still_loads(tmp_path):
    from PIL import Image

    data = tmp_path / "data"
    (data / "sub").mkdir(parents = True)
    Image.new("RGB", (2, 2)).save(data / "sub" / "ok.png")
    (data / "metadata.jsonl").write_text(json.dumps({"file_name": "sub/ok.png", "label": 0}) + "\n")
    runtime = ROOT / "unsloth_zoo" / "mlx" / "runtime.py"
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(runtime), str(data), "0"], capture_output = True, text = True, timeout = 300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"loaded": [str(data / "sub" / "ok.png")]}, result
