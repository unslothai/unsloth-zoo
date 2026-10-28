"""The compressed int / MXFP4 export planners open each shard once for its cross-shard companions."""

import gc
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unsloth_zoo import saving_utils
from unsloth_zoo.saving_utils import _plan_compressed_int_rewrite, _plan_compressed_mxfp4_rewrite

N = 40
INT_SCHEMES = [{"num_bits": 4, "type": "int", "strategy": "channel", "symmetric": True}]


def _counting_safe_open(monkeypatch):
    calls, live = [], [0, 0]

    class Handle:
        def __init__(self, path, *args, **kwargs):
            calls.append(os.path.basename(path))
            self._inner = safe_open(path, *args, **kwargs)

        def __enter__(self):
            live[0] += 1
            live[1] = max(live[1], live[0])
            return self._inner.__enter__()

        def __exit__(self, *exc):
            live[0] -= 1
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(saving_utils, "safe_open", Handle)
    return calls, live


def _write_int_shards(root, shards = 4):
    """Every module's packed weight in shard 0; its scale and shape spread round robin over the others."""
    gen = torch.Generator().manual_seed(0)
    files = {f"s{i}.safetensors": {} for i in range(shards)}
    for m in range(N):
        base = f"model.layers.{m}.mlp.up_proj"
        out_f, in_f = 8 + (m % 3) * 8, 64
        files["s0.safetensors"][base + ".weight_packed"] = torch.randint(-2**31, 2**31 - 1, (out_f, in_f // 8), generator = gen, dtype = torch.int32)
        other = files[f"s{1 + m % (shards - 1)}.safetensors"]
        other[base + ".weight_scale"] = torch.rand(out_f, 1, generator = gen).to(torch.bfloat16)
        other[base + ".weight_shape"] = torch.tensor([out_f, in_f])
    files["s0.safetensors"]["model.embed_tokens.weight"] = torch.zeros(4, 4)
    for name, tensors in files.items():
        save_file(tensors, os.path.join(root, name))
    return sorted(files)


def _write_mxfp4_shards(root, shards = 4):
    gen = torch.Generator().manual_seed(1)
    files = {f"s{i}.safetensors": {} for i in range(shards)}
    for m in range(N):
        base = f"model.layers.{m}.mlp.experts.0.w1"
        out_f, in_f = 4, 64
        files["s0.safetensors"][base + ".weight_packed"] = torch.randint(0, 256, (out_f, in_f // 2), generator = gen, dtype = torch.uint8)
        target = "s0.safetensors" if m % 5 == 0 else f"s{1 + m % (shards - 1)}.safetensors"
        files[target][base + ".weight_scale"] = torch.randint(0, 256, (out_f, in_f // 32), generator = gen, dtype = torch.uint8)
    for name, tensors in files.items():
        save_file(tensors, os.path.join(root, name))
    return sorted(files)


def _direct(root, filenames, names):
    out = {}
    for filename in filenames:
        with safe_open(os.path.join(root, filename), framework = "pt", device = "cpu") as f:
            for key in f.keys():
                if key in names:
                    out[key] = f.get_tensor(key)
    return out


def _assert_same(got, want):
    assert list(got) == list(want)
    for key in want:
        assert got[key].dtype == want[key].dtype and torch.equal(got[key], want[key]), key


def _mapped(root):
    if not sys.platform.startswith("linux"):
        return []
    with open("/proc/self/maps") as maps:
        return [line for line in maps if root in line]


def test_int_planner_opens_each_shard_once_for_companions(tmp_path, monkeypatch):
    root = str(tmp_path)
    filenames = _write_int_shards(root)
    want = _plan_compressed_int_rewrite(root, filenames, INT_SCHEMES)
    calls, live = _counting_safe_open(monkeypatch)
    plan = _plan_compressed_int_rewrite(root, filenames, INT_SCHEMES)
    # One inventory pass plus one companion pass per shard holding companions (s1..s3), not one per companion.
    assert len(calls) == len(filenames) + 3, calls
    assert live == [0, 1]
    assert len(plan["elsewhere"]) == 2 * N
    _assert_same(plan["elsewhere"], want["elsewhere"])
    assert plan["bases"] == want["bases"] and plan["logical"] == want["logical"]
    names = set(want["elsewhere"])
    direct = _direct(root, filenames, names)
    _assert_same(plan["elsewhere"], {k: direct[k] for k in want["elsewhere"]})
    # No reader outlives planning: once the tensors go, nothing maps a shard, so every one can be replaced.
    del plan, want, direct
    gc.collect()
    assert _mapped(root) == []
    for filename in filenames:
        os.replace(os.path.join(root, filename), os.path.join(root, filename + ".moved"))
        os.remove(os.path.join(root, filename + ".moved"))


def test_mxfp4_planner_opens_each_shard_once_for_scales(tmp_path, monkeypatch):
    root = str(tmp_path)
    filenames = _write_mxfp4_shards(root)
    model = torch.nn.Linear(2, 2)
    want = _plan_compressed_mxfp4_rewrite(root, filenames, model)
    calls, live = _counting_safe_open(monkeypatch)
    plan = _plan_compressed_mxfp4_rewrite(root, filenames, model)
    assert len(calls) == len(filenames) + 3, calls
    assert live == [0, 1]
    assert len(plan["elsewhere"]) == N - N // 5
    _assert_same(plan["elsewhere"], want["elsewhere"])
    direct = _direct(root, filenames, set(want["elsewhere"]))
    _assert_same(plan["elsewhere"], {k: direct[k] for k in want["elsewhere"]})
    assert plan["keys_of"] == want["keys_of"] and plan["shapes"] == want["shapes"]
    del plan, want, direct
    gc.collect()
    assert _mapped(root) == []
    for filename in filenames:
        os.remove(os.path.join(root, filename))


def test_a_refusal_reads_no_companion(tmp_path, monkeypatch):
    root = str(tmp_path)
    filenames = _write_int_shards(root)
    calls, _ = _counting_safe_open(monkeypatch)
    with pytest.raises(RuntimeError, match = "matches none"):
        _plan_compressed_int_rewrite(root, filenames, [dict(INT_SCHEMES[0], symmetric = False)])
    assert len(calls) == len(filenames)
