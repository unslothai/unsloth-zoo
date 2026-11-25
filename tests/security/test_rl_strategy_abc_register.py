# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A generated strategy must not register into host ABCs: it changes trusted isinstance checks."""

import collections.abc
import numbers

import pytest

from unsloth_zoo.rl_environments import create_locked_down_function


class _Host:
    pass


MUTATIONS = {
    "from_import": "def s(b):\n    from collections.abc import Sequence\n    Sequence.register(b)\n    return 1\n",
    "module_attr": "def s(b):\n    import collections.abc\n    collections.abc.Sequence.register(b)\n    return 1\n",
    "metaclass": "def s(b):\n    import abc, numbers\n    abc.ABCMeta.register(numbers.Number, b)\n    return 1\n",
    "type_of": "def s(b):\n    import numbers\n    type(numbers.Number).register(numbers.Number, b)\n    return 1\n",
    "decorator_name": "def s(b):\n    from collections.abc import Mapping\n    r = Mapping.register\n    r(b)\n    return 1\n",
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_strategy_cannot_register_virtual_subclass(name):
    try:
        fn = create_locked_down_function(MUTATIONS[name])
    except RuntimeError:
        fn = None
    if fn is not None:
        with pytest.raises(Exception):
            fn(_Host)
    assert not issubclass(_Host, collections.abc.Sequence)
    assert not issubclass(_Host, collections.abc.Mapping)
    assert not issubclass(_Host, numbers.Number)


def test_isinstance_checks_still_work():
    source = (
        "def s(values):\n"
        "    from collections.abc import Sequence\n"
        "    import numbers\n"
        "    return [isinstance(values, Sequence), isinstance(3, numbers.Number), issubclass(list, Sequence)]\n"
    )
    assert create_locked_down_function(source)([1, 2]) == [True, True, True]
