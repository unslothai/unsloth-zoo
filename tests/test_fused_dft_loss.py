from __future__ import annotations

from contextlib import contextmanager
import importlib.machinery
import os
from pathlib import Path
import sys
import types

import pytest


torch = pytest.importorskip("torch")


@contextmanager
def _lightweight_unsloth_zoo_import():
    """Import fused losses without running unsloth_zoo.__init__."""
    root = Path(__file__).resolve().parents[1]
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "unsloth_zoo" or name.startswith("unsloth_zoo.")
    }
    for name in list(saved):
        sys.modules.pop(name, None)

    package = types.ModuleType("unsloth_zoo")
    package.__path__ = [str(root / "unsloth_zoo")]
    package.__package__ = "unsloth_zoo"
    package.__spec__ = importlib.machinery.ModuleSpec(
        "unsloth_zoo",
        loader=None,
        is_package=True,
    )
    package.DEVICE_TYPE = "cpu"
    sys.modules["unsloth_zoo"] = package

    device_type = types.ModuleType("unsloth_zoo.device_type")
    device_type.DEVICE_TYPE = "cpu"
    device_type.DEVICE_TYPE_TORCH = "cpu"
    device_type.DEVICE_COUNT = 0
    device_type.ALLOW_PREQUANTIZED_MODELS = False
    device_type.is_hip = lambda: False
    device_type.get_device_type = lambda: "cpu"
    device_type.get_device_count = lambda: 0
    sys.modules["unsloth_zoo.device_type"] = device_type

    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "unsloth_zoo" or name.startswith("unsloth_zoo."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def fused_losses():
    with _lightweight_unsloth_zoo_import():
        import unsloth_zoo.fused_losses.cross_entropy_loss as module

        yield module


@contextmanager
def _compile_flag_guard(fused_losses):
    previous = fused_losses._FUSED_CE_COMPILE_SUPPORTED
    proven = fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN
    previous_proven = set(proven)
    torch._dynamo.reset()
    fused_losses._FUSED_CE_COMPILE_SUPPORTED = None
    proven.clear()
    try:
        yield
    finally:
        fused_losses._FUSED_CE_COMPILE_SUPPORTED = previous
        proven.clear()
        proven.update(previous_proven)
        torch._dynamo.reset()


def _make_inputs(device, dtype):
    hidden = torch.tensor(
        [
            [[0.30, -0.70, 0.20], [1.10, 0.40, -0.30], [-0.50, 0.80, 0.60]],
            [[0.90, -0.20, 0.50], [-0.40, -0.60, 1.00], [0.70, 0.30, -0.80]],
        ],
        dtype=dtype,
        device=device,
    )
    weight = torch.tensor(
        [
            [0.20, -0.50, 0.70],
            [-0.30, 0.60, 0.10],
            [0.80, 0.20, -0.40],
            [-0.60, -0.10, 0.50],
            [0.40, 0.90, -0.20],
        ],
        dtype=dtype,
        device=device,
    )
    bias = torch.tensor(
        [0.10, -0.20, 0.30, -0.10, 0.20],
        dtype=dtype,
        device=device,
    )
    labels = torch.tensor(
        [[0, 1, -100], [2, 3, 4]],
        device=device,
    )
    return hidden, weight, bias, labels


def _shift_labels(labels, ignore_index=-100):
    shifted = torch.empty_like(labels)
    shifted[..., :-1] = labels[..., 1:]
    shifted[..., -1] = ignore_index
    return shifted


def _reference_dft(
    hidden,
    weight,
    bias,
    labels,
    *,
    shift_labels=True,
    ignore_index=-100,
    logit_scale_multiply=None,
    logit_scale_divide=None,
    logit_softcapping=None,
    detach_weight=True,
):
    """Materialized-logit oracle for finite inputs.

    Non-finite ignored rows are compared with a valid-row-only oracle because
    sanitizing those rows is the production behavior under test.
    """
    if shift_labels:
        labels = _shift_labels(labels, ignore_index)
    logits = torch.nn.functional.linear(
        hidden.to(dtype=weight.dtype, device=weight.device),
        weight,
        bias,
    )
    if logit_scale_multiply not in (None, 0):
        logits = logits * logit_scale_multiply
    if logit_scale_divide not in (None, 0):
        logits = logits / logit_scale_divide
    if logit_softcapping not in (None, 0):
        logits = torch.tanh(logits / logit_softcapping) * logit_softcapping

    flat_labels = labels.reshape(-1).to(device=weight.device)
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float().contiguous(),
        flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=0.0,
    )
    divisor = (flat_labels != ignore_index).to(dtype=token_nll.dtype).sum()
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    weight_nll = token_nll.detach() if detach_weight else token_nll
    return (torch.exp(-weight_nll) * token_nll).sum() / divisor


def _clone_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _collect_result(loss, hidden, weight, bias):
    loss.backward()
    return (
        loss.detach().clone(),
        hidden.grad.detach().clone(),
        weight.grad.detach().clone(),
        bias.grad.detach().clone(),
    )


def _run_reference(hidden, weight, bias, labels, **kwargs):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    loss = _reference_dft(hidden, weight, bias, labels.detach().clone(), **kwargs)
    return _collect_result(loss, hidden, weight, bias)


def _run_direct(fused_losses, hidden, weight, bias, labels, **kwargs):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    loss, _ = fused_losses.compute_fused_dft_loss(
        hidden,
        weight,
        bias,
        labels.detach().clone(),
        **kwargs,
    )
    return _collect_result(loss, hidden, weight, bias)


def _run_wrapper(loss_fn, hidden, weight, bias, labels, **kwargs):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    loss = loss_fn(
        trainer=None,
        hidden_states=hidden,
        lm_head_weight=weight,
        lm_head_bias=bias,
        labels=labels.detach().clone(),
        **kwargs,
    )
    return _collect_result(loss, hidden, weight, bias)


def _assert_result_close(actual, expected, *, rtol=1e-5, atol=1e-7):
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=rtol,
            atol=atol,
        )


def _skip_if_compile_unavailable():
    # Only skip a missing generic toolchain; fused-loss graph failures must fail.
    try:
        compiled = torch.compile(lambda value: value + 1, fullgraph=True)
        compiled(torch.ones(1))
    except Exception as error:
        torch._dynamo.reset()
        pytest.skip(f"torch.compile toolchain unavailable: {type(error).__name__}")
    torch._dynamo.reset()


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("n_chunks", [1, 2, 20])
def test_fused_dft_matches_reference(fused_losses, device_name, n_chunks):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device(device_name)
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    wrapper_rtol, direct_rtol, atol = (
        (1e-5, 1e-6, 1e-7)
        if device_name == "cpu"
        else (1e-4, 1e-4, 1e-6)
    )
    hidden, weight, bias, labels = _make_inputs(device, dtype)
    shifted_labels = _shift_labels(labels)
    valid_per_chunk = [
        int((chunk != -100).sum())
        for chunk in torch.chunk(shifted_labels.reshape(-1), n_chunks)
    ]
    assert valid_per_chunk == {
        1: [3],
        2: [1, 2],
        20: [1, 0, 0, 1, 1, 0],
    }[n_chunks]

    transforms = {
        "logit_scale_multiply": 1.7,
        "logit_scale_divide": 0.8,
        "logit_softcapping": 2.25,
    }
    expected = _run_reference(hidden, weight, bias, labels, **transforms)
    nondetached = _run_reference(
        hidden,
        weight,
        bias,
        labels,
        detach_weight=False,
        **transforms,
    )
    assert torch.max(torch.abs(expected[1] - nondetached[1])) > 1e-3
    direct = _run_direct(
        fused_losses,
        hidden,
        weight,
        bias,
        shifted_labels,
        shift_labels=False,
        **transforms,
    )
    chunked = _run_wrapper(
        fused_losses.unsloth_fused_dft_loss,
        hidden,
        weight,
        bias,
        labels,
        torch_compile=False,
        n_chunks=n_chunks,
        scaling=7.0,
        **transforms,
    )

    _assert_result_close(direct, expected, rtol=direct_rtol, atol=atol)
    _assert_result_close(chunked, expected, rtol=wrapper_rtol, atol=atol)


@pytest.mark.parametrize("ignore_index", [-100, 999])
def test_fused_dft_ignored_nonfinite_row_is_safe(fused_losses, ignore_index):
    max_float = torch.finfo(torch.float32).max
    hidden = torch.tensor(
        [[[0.20, -0.10], [max_float, max_float]]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[0.30, -0.10], [2.0, -2.0], [-2.0, 2.0]],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.10, -0.20, 0.30], dtype=torch.float32)
    labels = torch.tensor([[1, ignore_index]])
    assert torch.isnan(torch.nn.functional.linear(hidden, weight, bias)[0, 1]).any()

    actual = _run_direct(
        fused_losses,
        hidden,
        weight,
        bias,
        labels,
        shift_labels=False,
        ignore_index=ignore_index,
        logit_softcapping=1.0,
    )
    expected = _run_reference(
        hidden[:, :1],
        weight,
        bias,
        labels[:, :1],
        shift_labels=False,
        ignore_index=ignore_index,
        logit_softcapping=1.0,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual[1][:, :1], expected[1], rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(actual[1][:, 1:]) == 0
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual[3], expected[3], rtol=1e-5, atol=1e-7)
    assert all(torch.isfinite(tensor).all() for tensor in actual)


@pytest.mark.parametrize(
    "loss_name", ["unsloth_fused_ce_loss", "unsloth_fused_dft_loss"]
)
def test_fused_loss_mask_applies_to_shifted_targets(fused_losses, loss_name):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float64)
    labels = labels.masked_fill(labels == -100, 2)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
    expected_labels = torch.tensor([[1, -100, -100], [-100, 4, -100]])
    loss_fn = getattr(fused_losses, loss_name)

    masked = _run_wrapper(
        loss_fn, hidden, weight, bias, labels, mask=mask, torch_compile=False, n_chunks=2
    )
    explicit = _run_wrapper(
        loss_fn,
        hidden,
        weight,
        bias,
        expected_labels,
        shift_labels=False,
        torch_compile=False,
        n_chunks=2,
    )

    _assert_result_close(masked, explicit)


def test_fused_dft_all_ignored_returns_connected_zero(fused_losses):
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 3, dtype=torch.float64)
    weight = torch.randn(6, 3, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)
    labels = torch.full((1, 4), -100)

    result = _run_wrapper(
        fused_losses.unsloth_fused_dft_loss,
        hidden,
        weight,
        bias,
        labels,
        torch_compile=False,
        n_chunks=2,
    )

    assert torch.isfinite(result[0])
    assert float(result[0]) == 0.0
    assert all(torch.count_nonzero(gradient) == 0 for gradient in result[1:])


@pytest.mark.parametrize("ce_first", [False, True], ids=["fresh-dft", "ce-first"])
def test_fused_dft_compiled_matches_eager(fused_losses, ce_first):
    if os.environ.get("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "0") == "1":
        pytest.skip("UNSLOTH_FUSED_CE_COMPILE_DISABLE=1 disables fused-loss compile")
    _skip_if_compile_unavailable()

    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float32)
    eager = _run_wrapper(
        fused_losses.unsloth_fused_dft_loss,
        hidden,
        weight,
        bias,
        labels,
        torch_compile=False,
        n_chunks=2,
        shift_labels=False,
    )

    with _compile_flag_guard(fused_losses):
        if ce_first:
            _run_wrapper(
                fused_losses.unsloth_fused_ce_loss,
                hidden,
                weight,
                bias,
                labels,
                torch_compile=True,
                n_chunks=2,
                shift_labels=False,
            )
            assert fused_losses._FUSED_CE_COMPILE_SUPPORTED is True
            assert (
                fused_losses.compute_fused_ce_loss
                in fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN
            )

        compiled = _run_wrapper(
            fused_losses.unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=True,
            n_chunks=2,
            shift_labels=False,
        )
        assert fused_losses._FUSED_CE_COMPILE_SUPPORTED is True
        assert (
            fused_losses.compute_fused_dft_loss
            in fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN
        )

    _assert_result_close(compiled, eager)


def test_fused_dft_rejects_label_smoothing_without_poisoning_compile(fused_losses):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float32)

    with pytest.raises(ValueError, match="label_smoothing"):
        fused_losses.compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
            label_smoothing=0.1,
        )

    with _compile_flag_guard(fused_losses):
        with pytest.raises(ValueError, match="label_smoothing"):
            fused_losses.unsloth_fused_dft_loss(
                trainer=None,
                hidden_states=hidden,
                lm_head_weight=weight,
                lm_head_bias=bias,
                labels=labels,
                torch_compile=True,
                n_chunks=2,
                shift_labels=False,
                label_smoothing=0.1,
            )
        assert fused_losses._FUSED_CE_COMPILE_SUPPORTED is None
        assert fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN == set()
