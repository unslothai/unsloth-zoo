# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Covers rl_environments.py: the create_locked_down_function allowlist is the only barrier to __import__;
_reject_dunder_access misses attributes spelled as bare identifiers, such as match keywords; launch_openenv must bind
loopback and never adopt a foreign /health server."""

import contextlib
import http.server
import io
import multiprocessing
import os
import socket
import sys
import subprocess

"""A generated strategy must not rewrite host classes: allowlisted modules return the real
collections.Counter / random.Random, and type(x) reaches any host object's class, so an attribute
store there changes every later evaluation and the reward code in the same interpreter."""

import collections
import random

import pytest

from unsloth_zoo.rl_environments import create_locked_down_function

MUTATIONS = {
    "module_class": "def s(b):\n    import collections\n    collections.Counter.most_common = lambda self, n = None: [('x', 999)]\n    return 1\n",
    "from_import": "def s(b):\n    from random import Random\n    Random.choice = lambda self, seq: seq[0]\n    return 1\n",
    "type_of_host_object": "def s(b):\n    import collections\n    type(collections.Counter()).elements = len\n    return 1\n",
    "alias": "def s(b):\n    import collections\n    C = collections.Counter\n    C.update = len\n    return 1\n",
    "delete": "def s(b):\n    import collections\n    del collections.Counter.most_common\n    return 1\n",
    "augassign": "def s(b):\n    import random\n    random.Random.VERSION += 1\n    return 1\n",
    "tuple_target": "def s(b):\n    import collections\n    collections.Counter.subtract, x = len, 1\n    return 1\n",
}


def _snapshot():
    return dict(vars(collections.Counter)), dict(vars(random.Random))


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_strategy_cannot_rewrite_host_class(name):
    before = _snapshot()
    try:
        fn = create_locked_down_function(MUTATIONS[name])
        with pytest.raises(AttributeError, match = "not allowed in generated code"):
            fn(None)
        assert _snapshot() == before
    finally:
        counter, rand = before
        for cls, saved in ((collections.Counter, counter), (random.Random, rand)):
            for key in set(vars(cls)) - set(saved):
                delattr(cls, key)
            for key, value in saved.items():
                if vars(cls).get(key) is not value:
                    setattr(cls, key, value)


def test_own_classes_and_instances_stay_assignable():
    source = (
        "def s(values):\n"
        "    import collections\n"
        "    class Node:\n"
        "        count = 0\n"
        "        def __init__(self, v):\n"
        "            self.v = v\n"
        "            self.next = None\n"
        "    head = Node(values[0])\n"
        "    head.next = Node(values[1])\n"
        "    head.next.v += 10\n"
        "    Node.count = 2\n"
        "    del head.next.next\n"
        "    c = collections.Counter(values)\n"
        "    c.label = 'ok'\n"
        "    return head.v, head.next.v, Node.count, c.label, c.most_common(1)\n"
    )
    fn = create_locked_down_function(source, builtins_policy = "allowlist")
    assert fn([1, 2]) == (1, 12, 2, "ok", [(1, 1)])


def test_factory_made_classes_stay_assignable():
    source = (
        "def s(values):\n"
        "    import collections\n"
        "    Node = type('Node', (), {})\n"
        "    Node.count = len(values)\n"
        "    Point = collections.namedtuple('Point', 'x y')\n"
        "    Point.label = 'p'\n"
        "    return Node.count, Point.label, Point(1, 2).x\n"
    )
    fn = create_locked_down_function(source, builtins_policy = "allowlist")
    assert fn([1, 2, 3]) == (3, "p", 1)


SHADOWS = {
    "parameter": "def s(b, __unsloth_attr_target__ = None):\n    import collections\n    collections.Counter.update = len\n    return 1\n",
    "import_alias": "def s(b):\n    import collections as __unsloth_attr_target__\n    return 1\n",
    "except_name": "def s(b):\n    try:\n        pass\n    except Exception as __unsloth_attr_target__:\n        pass\n    return 1\n",
    "lambda_arg": "def s(b):\n    f = lambda __unsloth_attr_target__: 1\n    return f(1)\n",
}


@pytest.mark.parametrize("name", sorted(SHADOWS))
def test_guard_name_cannot_be_shadowed(name):
    with pytest.raises(RuntimeError, match = "is not allowed in generated code"):
        create_locked_down_function(SHADOWS[name], builtins_policy = "allowlist")


def test_full_policy_unchanged():
    # "full" is the documented permissive mode: no rewrite there.
    fn = create_locked_down_function("def s(x):\n    class A:\n        pass\n    A.y = x\n    return A.y\n", builtins_policy = "full")
    assert fn(3) == 3
