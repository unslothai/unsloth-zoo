import json, os, subprocess, sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


def test_mlx_launch_ordered_text_batches_are_rank_sharded(tmp_path):
    launcher = Path(sys.executable).with_name("mlx.launch")
    if not launcher.exists():
        pytest.skip("mlx.launch not found beside current Python executable")

    script = tmp_path / "ddp_batch_probe.py"
    script.write_text("""import json, sys
from pathlib import Path

import mlx.core as mx
from unsloth_zoo.mlx.trainer import _create_labeled_batches
from unsloth_zoo.mlx.utils import create_batches, create_ordered_batches

class TinyTokenizer:
    pad_token_id = eos_token_id = 2
    def encode(self, text): return [int(part) for part in str(text).split()]

def keep_all_labels(d): return {"labels": [list(d["input_ids"][0])]}
def first_token_rows(batches):
    return [[int(row[0]) for row in batch.tolist()] for batch, _, _ in batches]

world = mx.distributed.init()
data = [{"text": f"{i} {i + 20} {i + 30}"} for i in range(10, 15)]
common = dict(batch_size=2, max_seq_length=8, comm_group=world)
payload = {
    "rank": int(world.rank()),
    "size": int(world.size()),
    "default": first_token_rows(create_batches(data, TinyTokenizer(), seed=1, **common)),
    "ordered": first_token_rows(
        create_ordered_batches(data, TinyTokenizer(), dataset_order="sequential", **common)
    ),
    "labeled": first_token_rows(
        _create_labeled_batches(
            data, TinyTokenizer(), keep_all_labels,
            dataset_order="sequential", **common,
        )
    ),
}
Path(sys.argv[1], f"rank{world.rank()}.json").write_text(json.dumps(payload))
""",
        encoding="utf-8")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        str(launcher), "-n", "2", "--backend", "ring",
        "--python", sys.executable, "--cwd", str(repo_root),
        str(script), str(tmp_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, env=env, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    outputs = [tmp_path / f"rank{rank}.json" for rank in (0, 1)]
    assert all(path.exists() for path in outputs), proc.stdout + proc.stderr
    ranks = [json.loads(path.read_text()) for path in outputs]
    expected = [[[10, 12], [14, 11]], [[11, 13], [10, 12]]]
    assert [rank["size"] for rank in ranks] == [2, 2]
    assert [rank["default"] for rank in ranks] == expected
    assert [rank["ordered"] for rank in ranks] == expected
    assert [rank["labeled"] for rank in ranks] == expected
