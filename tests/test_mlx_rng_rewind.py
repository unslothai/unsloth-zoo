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

"""Capture and rewind of the MLX global PRNG key.

mlx 0.32.1 made `mx.random.state` a sentinel that refuses item assignment, so
the key is rewound by reseeding with the captured key's own two 32-bit words;
`mx.random.key` builds `{seed >> 32, (uint32) seed}`, making that exact over the
whole unsigned 64-bit range.

Deliberately not in tests/test_mlx_training_e2e_metal.py: none of this needs
Metal, and that lane is opt-in and usually skipped, so tests parked behind it
never run on a pull request. The Linux CPU lane gates these on every push.
"""

import contextlib
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# `importorskip` is not enough: most tests/test_mlx_*.py files install the torch-backed
# tests/mlx_simulation stub into sys.modules and never remove it, so in a full-suite run
# `mlx.core` can be that stub. Everything below pins real mlx >= 0.32.1 semantics (a
# `mx.random.state` sentinel refusing item assignment), which the stub does not model.
if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip("mlx.core is the tests/mlx_simulation stub, not real MLX",
                allow_module_level = True)

from mlx.utils import tree_map                                     # noqa: E402

from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig  # noqa: E402
from unsloth_zoo.mlx.utils import (                                # noqa: E402
    FiniteTextBatchPlan, _FiniteTextRow, _mlx_rng_key,
    _preserved_preprocessing_rng, _restore_mlx_rng_key,
)

# High word's top bit set, so a rewind that packs the seed as signed is visible.
_SPLIT_KEY_SEED = 0xFEDCBA9876543210


class _NoItemAssignment:
    """``mx.random.state`` as mlx >= 0.32.1 exposes it: readable, not writable."""

    def __init__(self, words = None):
        self._words = words

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError("random state index out of range")
        if self._words is None:
            return mx.random.key(_SPLIT_KEY_SEED)
        return mx.array(list(self._words), dtype = mx.uint32)

    def __iter__(self):
        return iter([self[0]])


@contextlib.contextmanager
def _shadowing_random_state(replacement):
    """Swap ``mx.random.state`` for a stand-in and put the original back.

    Which "back" depends on the mlx version, and getting it wrong poisons the
    rest of the pytest process rather than failing here. mlx >= 0.31.2 serves
    ``state`` from a module ``__getattr__``, so the shadow must be deleted or it
    hides the hook for the whole run; mlx < 0.31.2 keeps it as a real module
    attribute, so deleting it removes the PRNG state itself and every later read
    raises AttributeError, which ``_mlx_rng_key`` turns into None. pyproject
    declares ``mlx>=0.22.0``, so both eras are in support.
    """
    sentinel = object()
    original = mx.random.__dict__.get("state", sentinel)
    setattr(mx.random, "state", replacement)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(mx.random, "state")
        else:
            setattr(mx.random, "state", original)


def _draw():
    value = mx.random.uniform(shape = (8,))
    mx.eval(value)
    return value.tolist()


def _draw_after_seed(seed):
    mx.random.seed(seed)
    return _draw()


# --- Key arithmetic ---

@pytest.mark.parametrize("seed", [
    0, 1, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**64 - 1,
    _SPLIT_KEY_SEED,
])
def test_the_key_packing_is_exactly_invertible(seed):
    """Half of all real keys have the high bit set, so a signed packing anywhere
    would break those runs. Checked, not assumed."""
    words = mx.random.key(seed).tolist()
    assert words == [seed >> 32, seed & 0xFFFFFFFF]
    assert ((int(words[0]) << 32) | int(words[1])) == seed

    mx.random.seed(seed)
    assert _mlx_rng_key() == (seed >> 32, seed & 0xFFFFFFFF)


def test_restore_reproduces_the_draws_that_followed_the_capture():
    mx.random.seed(17)
    _draw()

    key = _mlx_rng_key()
    assert key is not None
    expected = _draw()

    _restore_mlx_rng_key(key)
    assert _draw() == expected


def test_restore_works_where_the_state_refuses_item_assignment():
    """A stand-in with mlx 0.32.1's shape (no ``__setitem__``) pins the contract
    on whichever mlx is installed, including ones where state is still a list."""
    expected = _draw_after_seed(_SPLIT_KEY_SEED)

    mx.random.seed(1)
    with _shadowing_random_state(_NoItemAssignment()):
        key = _mlx_rng_key()
        assert key == tuple(mx.random.key(_SPLIT_KEY_SEED).tolist())
        _restore_mlx_rng_key(key)

    assert _draw() == expected


def test_capture_reads_nothing_where_the_state_is_not_indexable():
    # The torch simulation shim's state is a callable: a no-op, not a raise.
    with _shadowing_random_state(lambda: {"counter": 0}):
        assert _mlx_rng_key() is None

    mx.random.seed(5)
    expected = _draw()
    mx.random.seed(5)
    _restore_mlx_rng_key(None)
    assert _draw() == expected


def test_shadowing_the_state_does_not_outlive_the_test():
    """Regression guard for the helper above: on mlx < 0.31.2 an unshadow by
    `delattr` deletes the PRNG state for the rest of the process."""
    before = _mlx_rng_key()
    assert before is not None

    with _shadowing_random_state(_NoItemAssignment((7, 9))):
        assert _mlx_rng_key() == (7, 9)

    assert _mlx_rng_key() == before
    mx.random.state[0]              # must not raise AttributeError
    _draw()


@pytest.mark.parametrize("words", [
    (-1, -2), (-2147483648, 5), (0, -1), (2**32, 0), (0, 2**32),
])
def test_restore_normalises_words_outside_the_unsigned_32_bit_range(words):
    """`mx.random.seed` takes a uint64 and hard-raises outside [0, 2**64).

    The restore is unguarded on purpose, since a blanket `except` would be a
    failure indistinguishable from an intentional no-op. That holds only if the
    words cannot put it out of range, and capture no longer type-checks the
    state, so the masking is what makes "cannot raise" true. A raise would land
    in `_preserved_preprocessing_rng`'s `finally`, or replace the compile error
    a fallback is recovering from.
    """
    _restore_mlx_rng_key(words)
    assert _mlx_rng_key() == (words[0] & 0xFFFFFFFF, words[1] & 0xFFFFFFFF)


def test_capture_masks_a_state_whose_words_are_not_unsigned():
    with _shadowing_random_state(_NoItemAssignment((0xFFFFFFFF, 0xFFFFFFFE))):
        words = _mlx_rng_key()
    assert words == (0xFFFFFFFF, 0xFFFFFFFE)
    _restore_mlx_rng_key(words)     # must not raise
    assert _mlx_rng_key() == words


@pytest.mark.parametrize("n", [1, 3, 4])
def test_a_key_that_is_not_two_words_is_reported_rather_than_swallowed(n):
    """A bare None here would leave every compile fallback silently not
    rewinding, as the old `isinstance(state, list)` guard did."""
    class _WideKey:
        def __len__(self): return 1
        def __getitem__(self, index): return mx.array([1] * n, dtype = mx.uint32)

    with _shadowing_random_state(_WideKey()):
        with pytest.warns(RuntimeWarning, match = "random key"):
            assert _mlx_rng_key() is None


def test_an_unreadable_state_does_not_warn():
    """The simulation shim is an intentional no-op, not a surprise; warning on
    every call would train operators to ignore the warning that matters."""
    import warnings
    with _shadowing_random_state(lambda: {"counter": 0}):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _mlx_rng_key() is None


# --- _preserved_preprocessing_rng ---

def test_preserved_preprocessing_rng_rewinds_the_mlx_key():
    mx.random.seed(99)
    expected = _draw()

    mx.random.seed(99)
    with _preserved_preprocessing_rng():
        _draw()
    assert _draw() == expected


def test_preserved_preprocessing_rng_rewinds_even_when_the_block_raises():
    mx.random.seed(64)
    expected = _draw()

    mx.random.seed(64)
    with pytest.raises(ValueError):
        with _preserved_preprocessing_rng():
            _draw()
            raise ValueError("preprocessing blew up")
    assert _draw() == expected


def test_preserved_preprocessing_rng_does_not_mask_the_block_s_exception():
    with pytest.raises(ValueError, match = "the real problem"):
        with _preserved_preprocessing_rng():
            raise ValueError("the real problem")


def test_preserved_preprocessing_rng_is_inert_when_the_state_is_unreadable():
    with _shadowing_random_state(lambda: {"counter": 0}):
        with _preserved_preprocessing_rng():
            pass
        with pytest.raises(KeyError, match = "inner"):
            with _preserved_preprocessing_rng():
                raise KeyError("inner")


# --- Compile fallback ---

class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(128, 4)
        self.proj = nn.Linear(4, 128, bias = False)
        self._config = {"model_type": "tiny"}

    def __call__(self, input_ids):
        return self.proj(self.embed(input_ids))

    def train(self, mode = True):
        return self

    @property
    def state(self):
        return []


def test_a_runtime_compile_failure_retries_eagerly_from_the_captured_key(tmp_path, monkeypatch):
    """Without the rewind the failed attempt's draws stay in the stream, and a
    run that falls back diverges from an eager run of the same seed.
    """
    keys = []
    failed = False

    def compile_spy(fn, **_kwargs):
        def compiled(*args):
            nonlocal failed
            if not failed:
                failed = True
                keys.append(_mlx_rng_key())
                # What the trainer has to undo.
                _draw()
                raise RuntimeError("compile runtime failure")
            return fn(*args)
        return compiled

    def value_and_grad_recording_key(model, fn):
        def wrapped(*args):
            keys.append(_mlx_rng_key())
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(mx, "compile", compile_spy)
    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_recording_key)

    rows = tuple(
        _FiniteTextRow(tuple(range(1, width + 1)), offset = 1, labels = None)
        for width in (10, 11)
    )
    trainer = MLXTrainer(
        _TinyLM(),
        SimpleNamespace(pad_token_id = 99, eos_token_id = 2),
        [],
        args = MLXTrainingConfig(
            max_steps = 2,
            gradient_accumulation_steps = 1,
            compile = True,
            use_cce = False,
            gradient_checkpointing = False,
            cast_norm_output_to_input_dtype = False,
            max_grad_norm = 0.0,
            max_grad_leaf_norm = 0.0,
            disable_memory_limits = True,
            logging_steps = 2,
            output_dir = str(tmp_path),
        ),
    )
    trainer._batches = FiniteTextBatchPlan(
        rows,
        tuple((index,) for index in range(len(rows))),
        max_seq_length = 64,
        pad_id = 99,
    )
    trainer._build_optimizer = lambda _total_steps: SimpleNamespace(
        learning_rate = mx.array(1e-5),
        state = {},
        update = lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None

    result = trainer.train()

    assert result["compile_scope"] == "fallback_eager"
    assert len(keys) >= 2, keys
    assert keys[1] == keys[0]


def test_every_compile_fallback_rewinds_the_rng_before_retrying_eagerly():
    """The single-process retry is exercised for real above; its DDP twin needs
    a distributed group no unit test can raise, so it is pinned structurally.
    On the parse tree, not substring counts: a harmless refactor must not fail
    this, and a misplaced restore must.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(MLXTrainer._train_inner))
    tree = ast.parse(source)

    def calls_named(name, root = tree):
        return [
            node for node in ast.walk(root)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    captures = calls_named("_mlx_rng_key")
    restores = calls_named("_restore_mlx_rng_key")
    assert len(captures) == 2, (
        f"expected both compile fallbacks to capture, found {len(captures)}"
    )
    assert len(restores) == 2, (
        f"expected both compile fallbacks to restore, found {len(restores)}"
    )

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node

    # Both restores must sit on a recovery branch and precede its eager retry.
    # The branch type is not pinned: single-process recovers inside `except`,
    # DDP inside an `if` after the try (it syncs ranks first).
    for restore in restores:
        recovery = [a for a in ancestors(restore)
                    if isinstance(a, (ast.ExceptHandler, ast.If))]
        assert recovery, "a restore sits on the happy path, not a recovery branch"
        branch = recovery[0]
        retries = calls_named("step_fn", branch)
        assert retries, "a recovery branch rewinds the RNG but never retries"
        for retry in retries:
            assert restore.lineno < retry.lineno, (
                "an eager fallback re-runs the step without rewinding the RNG"
            )
