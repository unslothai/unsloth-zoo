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

"""A generated strategy must not rewrite module-level registries of allowlisted modules: `copy.dispatch_table` is
copyreg's real dict, so an item store there changes every later copy.copy in the host process."""

import copy
import copyreg

import pytest

from unsloth_zoo.rl_environments import create_locked_down_function


class _Host:
    pass


MUTATIONS = {
    "item_store": "def s(b):\n    import copy\n    copy.dispatch_table[b] = lambda o: (len, ([1, 2, 3],))\n    return 1\n",
    "from_import": "def s(b):\n    from copy import dispatch_table\n    dispatch_table[b] = lambda o: (len, ([1],))\n    return 1\n",
    "update": "def s(b):\n    import copy\n    copy.dispatch_table.update({b: lambda o: (len, ([1],))})\n    return 1\n",
    "setdefault": "def s(b):\n    import copy\n    copy.dispatch_table.setdefault(b, lambda o: (len, ([1],)))\n    return 1\n",
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_strategy_cannot_rewrite_copy_registry(name):
    before = dict(copyreg.dispatch_table)
    try:
        fn = create_locked_down_function(MUTATIONS[name])
        with pytest.raises((TypeError, AttributeError)):
            fn(_Host)
        assert dict(copyreg.dispatch_table) == before
        assert isinstance(copy.copy(_Host()), _Host)
    finally:
        copyreg.dispatch_table.clear()
        copyreg.dispatch_table.update(before)


def test_registry_still_readable_and_copy_works():
    source = (
        "def s(values):\n"
        "    import copy\n"
        "    out = copy.deepcopy(values)\n"
        "    out.append(len(copy.dispatch_table) >= 0)\n"
        "    own = {'a': 1}\n"
        "    own['b'] = 2\n"
        "    return out, own\n"
    )
    assert create_locked_down_function(source)([1, [2]]) == ([1, [2], True], {"a": 1, "b": 2})
