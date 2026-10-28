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

"""The reversed-key index must return exactly the linear scan's candidates: a wrong list routes a LoRA delta onto the wrong tensor."""

from __future__ import annotations

import collections
import importlib.util
import random
import sys
import types

if importlib.util.find_spec("bitsandbytes") is None:
    from importlib.machinery import ModuleSpec
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = ModuleSpec("bitsandbytes", loader=None, is_package=True)
    _bnb.__path__ = []
    _bnb_nn = types.ModuleType("bitsandbytes.nn")
    _bnb_nn.__spec__ = ModuleSpec("bitsandbytes.nn", loader=None)
    for _cls in ("Linear8bitLt", "Linear4bit", "Int8Params", "Params4bit"):
        setattr(_bnb_nn, _cls, type(_cls, (object,), {}))
    _bnb.nn = _bnb_nn
    sys.modules["bitsandbytes"] = _bnb
    sys.modules["bitsandbytes.nn"] = _bnb_nn

import pytest  # noqa: E402

import unsloth_zoo.saving_utils as su  # noqa: E402


def _linear_candidates(keys, k):
    """The linear scan the index replaces, verbatim."""
    return list(dict.fromkeys(
        sf_key[: -len(suffix)]
        for suffix in (k + ".weight", k + ".linear.weight")
        for sf_key in keys
        if sf_key.endswith(suffix) and sf_key[: -len(suffix)]
    ))


def _indexed(keys, k):
    built = su._build_reversed_key_index(keys)
    assert built is not None
    return su._indexed_prefix_candidates(*built, k)


_COMP = ["model", "language_model", "layers", "0", "1", "q_proj", "xmodel", "odel",
         "linear", "weight", "experts", "", "é", "\U0001f600"]
_PREFIX = ["", "model.", "language_model.", "x", "m", "model.language_model.", "a.b."]
_SUFFIX = [".weight", ".linear.weight", ".bias", "", ".weight.weight", "weight"]


def test_index_matches_linear_scan_randomized():
    r = random.Random(1234)
    for _ in range(3000):
        lora = [".".join(r.choice(_COMP) for _ in range(r.randint(1, 4))) for _ in range(r.randint(1, 6))]
        keys = [r.choice(_PREFIX) + r.choice(lora) + r.choice(_SUFFIX) for _ in range(r.randint(1, 12))]
        keys += r.sample(keys, r.randint(0, min(3, len(keys))))
        r.shuffle(keys)
        for k in lora + [r.choice(keys)]:
            assert _indexed(keys, k) == _linear_candidates(keys, k), (keys, k)


@pytest.mark.parametrize("keys, k", [
    (["xmodel.layers.0.q.weight"], "model.layers.0.q"),
    (["model.layers.0.q.weight", "xmodel.layers.0.q.weight"], "odel.layers.0.q"),
    (["a.model.q.weight", "b.model.q.weight", "a.model.q.weight"], "model.q"),
    (["z.m.q.linear.weight", "a.m.q.weight", "y.m.q.weight"], "m.q"),
    (["p.m.q.linear.weight", "p.m.q.weight"], "m.q"),
    (["m.q.weightx", "xm.q.weight", "m.qq.weight", "m.q.weight.weight"], "m.q"),
    (["\U0001f600.m.q.weight", "ém.q.weight"], "m.q"),
])
def test_index_matches_linear_scan_adversarial(keys, k):
    assert _indexed(keys, k) == _linear_candidates(keys, k)
    assert _indexed(tuple(keys), k) == _linear_candidates(keys, k)


def _lw(keys):
    d = collections.defaultdict(lambda: "default")
    for k in keys:
        d[k] = "v:" + k
    return d


def _items(out):
    return None if out is None else list(out.items())


def _reference_remap(lw, keys, monkeypatch):
    """Head function with the index disabled, i.e. the original linear scan."""
    with monkeypatch.context() as m:
        m.setattr(su, "_build_reversed_key_index", lambda keys: None)
        return _items(su._infer_prefix_and_remap(lw, keys))


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_remap_unchanged_by_index(container, monkeypatch):
    lora = ["model.layers.0.q", "model.layers.0.k", "model.layers.1.q", "model.q", "lm_head"]
    disk = [
        "language_model.model.layers.0.q.weight",
        "language_model.model.layers.0.k.linear.weight",
        "a.model.layers.1.q.weight", "b.model.layers.1.q.weight",
        "xmodel.q.weight",
        "lm_head.weight",
    ]
    expected = _reference_remap(_lw(lora), container(disk), monkeypatch)
    assert expected is not None
    assert _items(su._infer_prefix_and_remap(_lw(lora), container(disk))) == expected


def test_index_built_once_and_only_when_needed(monkeypatch):
    calls = []
    real = su._build_reversed_key_index
    monkeypatch.setattr(su, "_build_reversed_key_index", lambda keys: calls.append(1) or real(keys))
    disk = [f"language_model.model.layers.{i}.q.weight" for i in range(8)]
    out = su._infer_prefix_and_remap(_lw([f"model.layers.{i}.q" for i in range(8)]), disk)
    assert list(out) == [f"language_model.model.layers.{i}.q" for i in range(8)]
    assert len(calls) == 1
    calls.clear()
    assert su._infer_prefix_and_remap(_lw(["language_model.model.layers.0.q"]), disk) is None
    assert calls == []


def test_unsupported_inputs_keep_linear_scan(monkeypatch):
    class S(str):
        pass
    assert su._build_reversed_key_index(iter(["a.weight"])) is None
    assert su._build_reversed_key_index({"a.weight": 1}.keys()) is None
    assert su._build_reversed_key_index(["a.weight", S("b.weight")]) is None
    assert su._build_reversed_key_index(["a.weight", 3]) is None
    with pytest.raises(AttributeError):
        su._infer_prefix_and_remap(_lw(["m.q"]), ["x.m.q.weight", 3])
