"""unsloth#409: the compiled trainers' EMPTY_LOGITS must not claim protocol dunders.

``EmptyLogits.__getattr__`` answers every name, so ``hasattr`` on the sentinel
used to be true for every name too. Libraries duck-type on dunders:
``torch.distributed.utils._apply_to_tensors`` tests
``hasattr(x, "__dataclass_fields__")`` and then calls ``dataclasses.replace(x)``
on whatever said yes, so FSDP2's mixed-precision output cast
(``_fsdp_state._cast_output_dtype``) killed every training step with
``TypeError: replace() should be called on dataclass instances``.

This file tests the copy that lives in ``compiler._cross_entropy_code``, the
source text written into ``unsloth_compiled_cache``. It is never imported, so
the test execs it the way a generated module does. unsloth/models/_utils.py
carries the twin; both have to answer the probe the same way or the failure
comes back through whichever copy the run happens to use.

Measured on 2 GPUs with ``accelerate launch`` and an FSDP2 config: before, every
step of a Qwen2.5-0.5B LoRA SFT died in that TypeError; after, the same run
logged ``[3.5781, 3.9453, 3.5156, 3.4297]`` on both ranks.

No GPU and no distributed launcher: the probe below is exactly torch's.
"""

import dataclasses

import pytest
import torch

from unsloth_zoo.compiler import _cross_entropy_code


def _exec_sentinel():
    """Run the emitted sentinel definition on its own, as a generated module does."""
    start = _cross_entropy_code.index("LOGITS_ERROR_STRING = ")
    end = _cross_entropy_code.index("EMPTY_LOGITS = EmptyLogits()")
    namespace = {"torch": torch}
    exec(_cross_entropy_code[start:end], namespace)
    namespace["EMPTY_LOGITS"] = namespace["EmptyLogits"]()
    return namespace


_NS = _exec_sentinel()
EMPTY_LOGITS = _NS["EMPTY_LOGITS"]
LOGITS_ERROR_STRING = _NS["LOGITS_ERROR_STRING"]


# Deliberately none of `dir(torch.Tensor)`: the loop that follows the definition
# binds every tensor dunder as a real instance attribute, so those never reach
# `__getattr__`. These are the ones only `__getattr__` could have invented.
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
    """The AttributeError is for protocol probes only, or the fix trades a
    TypeError for a silent `AttributeError: shape`."""
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
