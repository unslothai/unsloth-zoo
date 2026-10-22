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

"""unsloth#409: the compiled trainers' EMPTY_LOGITS must not claim protocol dunders.

A catch-all ``__getattr__`` made ``hasattr`` true for every name, so torch's
``_apply_to_tensors`` saw ``__dataclass_fields__``, called ``dataclasses.replace``
and killed every FSDP2 mixed-precision step. Under test is the source text in
``compiler._cross_entropy_code`` (never imported, so exec'd); _utils.py has a twin.
"""

import dataclasses

import pytest
import torch

from unsloth_zoo.compiler import _cross_entropy_code


def _exec_sentinel():
    start = _cross_entropy_code.index("LOGITS_ERROR_STRING = ")
    end = _cross_entropy_code.index("EMPTY_LOGITS = EmptyLogits()")
    namespace = {"torch": torch}
    exec(_cross_entropy_code[start:end], namespace)
    namespace["EMPTY_LOGITS"] = namespace["EmptyLogits"]()
    return namespace


_NS = _exec_sentinel()
EMPTY_LOGITS = _NS["EMPTY_LOGITS"]
LOGITS_ERROR_STRING = _NS["LOGITS_ERROR_STRING"]


# Not tensor dunders: the loop after the definition binds those as real attributes.
PROTOCOL_DUNDERS = (
    "__dataclass_fields__",
    "__fields__",
    "__attrs_attrs__",
    "__get_validators__",
    "__pydantic_fields__",
    "__dataclass_params__",
)


@pytest.mark.parametrize("name", PROTOCOL_DUNDERS)
def test_the_sentinel_does_not_claim_a_protocol_dunder(name):
    assert not hasattr(EMPTY_LOGITS, name), (
        f"EMPTY_LOGITS answers hasattr({name!r}); a library that duck-types on it "
        f"will take a branch the sentinel cannot honour"
    )


def test_the_torch_distributed_output_walk_leaves_the_sentinel_alone():
    assert not hasattr(EMPTY_LOGITS, "__dataclass_fields__")
    with pytest.raises(TypeError):
        dataclasses.replace(EMPTY_LOGITS)


def test_an_ordinary_attribute_still_explains_how_to_get_real_logits():
    """Protocol probes only, or the fix trades the TypeError for `AttributeError: shape`."""
    raiser = EMPTY_LOGITS.shape
    with pytest.raises(NotImplementedError) as excinfo:
        raiser()
    assert "UNSLOTH_RETURN_LOGITS" in str(excinfo.value)


def test_to_is_still_the_no_op_accelerate_needs():
    assert EMPTY_LOGITS.to("cpu") is None


def test_dunder_names_the_class_really_defines_are_untouched():
    """`__getattr__` is consulted only after the normal lookup fails."""
    assert EMPTY_LOGITS == EMPTY_LOGITS
    assert repr(EMPTY_LOGITS) == LOGITS_ERROR_STRING
    assert EMPTY_LOGITS.__reduce__() == (type(EMPTY_LOGITS), ())
