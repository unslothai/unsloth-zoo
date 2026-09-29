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


def _reference_logits(
    hidden,
    weight,
    bias,
    *,
    logit_scale_multiply=None,
    logit_scale_divide=None,
    logit_softcapping=None,
):
    logits = torch.nn.functional.linear(
        hidden.to(dtype=weight.dtype, device=weight.device), weight, bias
    )
    if logit_scale_multiply not in (None, 0):
        logits = logits * logit_scale_multiply
    if logit_scale_divide not in (None, 0):
        logits = logits / logit_scale_divide
    if logit_softcapping not in (None, 0):
        logits = torch.tanh(logits / logit_softcapping) * logit_softcapping
    return logits.reshape(-1, logits.shape[-1]).float().contiguous()


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
    logits = _reference_logits(
        hidden,
        weight,
        bias,
        logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide,
        logit_softcapping=logit_softcapping,
    )

    flat_labels = labels.reshape(-1).to(device=weight.device)
    token_nll = torch.nn.functional.cross_entropy(
        logits,
        flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=0.0,
    )
    divisor = (flat_labels != ignore_index).to(dtype=token_nll.dtype).sum()
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    weight_nll = token_nll.detach() if detach_weight else token_nll
    return (torch.exp(-weight_nll) * token_nll).sum() / divisor


def _reference_ce(
    hidden,
    weight,
    bias,
    labels,
    *,
    n_items=None,
    shift_labels=True,
    ignore_index=-100,
    label_smoothing=0.0,
    logit_scale_multiply=None,
    logit_scale_divide=None,
    logit_softcapping=None,
):
    if shift_labels:
        labels = _shift_labels(labels, ignore_index)
    logits = _reference_logits(
        hidden,
        weight,
        bias,
        logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide,
        logit_softcapping=logit_softcapping,
    )

    reduction = "sum" if n_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        logits,
        labels.reshape(-1).to(device=weight.device).contiguous(),
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    return loss / n_items if n_items is not None else loss


def _clone_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _collect_result(loss, hidden, weight, bias, *, grad_output=1.0):
    loss.backward(torch.full_like(loss, grad_output))
    return (
        loss.detach().clone(),
        hidden.grad.detach().clone(),
        weight.grad.detach().clone(),
        bias.grad.detach().clone(),
    )


def _run_reference(hidden, weight, bias, labels, *, loss_name="dft", **kwargs):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    reference = _reference_ce if loss_name == "ce" else _reference_dft
    loss = reference(hidden, weight, bias, labels.detach().clone(), **kwargs)
    return _collect_result(loss, hidden, weight, bias)


def _run_direct_ce(
    fused_losses, hidden, weight, bias, labels, *, scaling=None, **kwargs
):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    scaled_loss, (unscaled_loss,) = fused_losses.compute_fused_ce_loss(
        hidden,
        weight,
        bias,
        labels.detach().clone(),
        scaling=scaling,
        **kwargs,
    )
    assert not unscaled_loss.requires_grad
    result = _collect_result(scaled_loss, hidden, weight, bias)
    return result, unscaled_loss.detach().clone()


def _run_direct(fused_losses, hidden, weight, bias, labels, *, loss_name="dft", **kwargs):
    hidden = _clone_leaf(hidden)
    weight = _clone_leaf(weight)
    bias = _clone_leaf(bias)
    loss_fn = getattr(fused_losses, f"compute_fused_{loss_name}_loss")
    loss, (auxiliary_loss,) = loss_fn(
        hidden,
        weight,
        bias,
        labels.detach().clone(),
        **kwargs,
    )
    assert not auxiliary_loss.requires_grad
    return _collect_result(loss, hidden, weight, bias)


def _run_wrapper(loss_fn, hidden, weight, bias, labels, *, grad_output=1.0, **kwargs):
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
    return _collect_result(loss, hidden, weight, bias, grad_output=grad_output)


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


@pytest.mark.parametrize("n_items", [None, 5.5], ids=["mean", "explicit-divisor"])
def test_compute_fused_ce_preserves_values_and_gradients(fused_losses, n_items):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float64)
    transforms = {
        "logit_scale_multiply": 1.7,
        "logit_scale_divide": 0.8,
        "logit_softcapping": 2.25,
        "label_smoothing": 0.15,
    }
    scaling = 7.0

    reference_hidden = _clone_leaf(hidden)
    reference_weight = _clone_leaf(weight)
    reference_bias = _clone_leaf(bias)
    reference_loss = _reference_ce(
        reference_hidden,
        reference_weight,
        reference_bias,
        labels.detach().clone(),
        n_items=n_items,
        **transforms,
    )
    expected = _collect_result(
        reference_loss * scaling,
        reference_hidden,
        reference_weight,
        reference_bias,
    )
    actual, auxiliary_loss = _run_direct_ce(
        fused_losses,
        hidden,
        weight,
        bias,
        labels,
        n_items=n_items,
        scaling=scaling,
        **transforms,
    )

    _assert_result_close(actual, expected, rtol=1e-6)
    torch.testing.assert_close(
        auxiliary_loss, reference_loss.detach(), rtol=1e-6, atol=1e-7
    )


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
@pytest.mark.parametrize("loss_name", ["ce", "dft"])
def test_fused_loss_ignored_nonfinite_row_is_safe(fused_losses, ignore_index, loss_name):
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
        loss_name=loss_name,
        shift_labels=False,
        ignore_index=ignore_index,
        logit_softcapping=1.0,
    )
    expected = _run_reference(
        hidden[:, :1],
        weight,
        bias,
        labels[:, :1],
        loss_name=loss_name,
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
    chunked = _run_wrapper(
        getattr(fused_losses, f"unsloth_fused_{loss_name}_loss"),
        hidden, weight, bias, labels,
        shift_labels=False, ignore_index=ignore_index, logit_softcapping=1.0,
        torch_compile=False, n_chunks=2,
    )
    _assert_result_close(chunked, actual)


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


@pytest.mark.parametrize("loss_name", ["ce", "dft"])
@pytest.mark.parametrize(
    "n_items", [None, 0, torch.tensor(0)], ids=["inferred", "zero", "tensor-zero"]
)
def test_fused_loss_all_ignored_returns_connected_zero(fused_losses, loss_name, n_items):
    torch.manual_seed(0)
    hidden = torch.randn(1, 4, 3, dtype=torch.float64)
    weight = torch.randn(6, 3, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)
    labels = torch.full((1, 4), -100)

    direct = _run_direct(
        fused_losses, hidden, weight, bias, labels,
        loss_name=loss_name, n_items=n_items, scaling=7.0,
    )
    result = _run_wrapper(
        getattr(fused_losses, f"unsloth_fused_{loss_name}_loss"),
        hidden,
        weight,
        bias,
        labels,
        torch_compile=False,
        n_chunks=2,
        n_items=n_items,
        scaling=7.0,
    )

    _assert_result_close(direct, result)
    assert torch.isfinite(result[0])
    assert float(result[0]) == 0.0
    assert all(torch.count_nonzero(gradient) == 0 for gradient in result[1:])


@pytest.mark.parametrize("loss_name", ["ce", "dft"])
@pytest.mark.parametrize(
    "n_items", [5.5, torch.tensor(5.5), torch.tensor([5.5, 99.0])],
    ids=["scalar", "tensor", "replicated"],
)
def test_fused_loss_explicit_divisor_and_scaling(fused_losses, loss_name, n_items):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float64)
    expected = _run_reference(hidden, weight, bias, labels, loss_name=loss_name)
    # Three valid shifted targets. An explicit divisor must override that count.
    expected = tuple(value * (3.0 / 5.5) for value in expected)
    direct = _run_direct(
        fused_losses, hidden, weight, bias, labels,
        loss_name=loss_name, n_items=n_items, scaling=7.0,
    )
    _assert_result_close(direct, tuple(value * 7.0 for value in expected))
    chunked = _run_wrapper(
        getattr(fused_losses, f"unsloth_fused_{loss_name}_loss"),
        hidden, weight, bias, labels,
        n_items=n_items, scaling=7.0, grad_output=2.5,
        torch_compile=False, n_chunks=20,
    )
    _assert_result_close(chunked, (expected[0], *(value * 2.5 for value in expected[1:])))


@pytest.mark.parametrize("loss_name", ["ce", "dft"])
@pytest.mark.parametrize("n_chunks", [1, 20])
def test_fused_loss_zero_divisor_with_targets_is_nonfinite(
    fused_losses, loss_name, n_chunks
):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float64)
    direct = _run_direct(
        fused_losses, hidden, weight, bias, labels, loss_name=loss_name, n_items=0,
    )
    chunked = _run_wrapper(
        getattr(fused_losses, f"unsloth_fused_{loss_name}_loss"),
        hidden, weight, bias, labels,
        n_items=torch.tensor(0), torch_compile=False, n_chunks=n_chunks,
    )
    assert not torch.isfinite(direct[0])
    assert not torch.isfinite(chunked[0])


@pytest.mark.parametrize("loss_name", ["ce", "dft"])
def test_fused_loss_zero_scaling_still_rejects_backward(fused_losses, loss_name):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float64)
    with pytest.raises(RuntimeError, match="scaling=0 with non-zero grad_output"):
        _run_wrapper(
            getattr(fused_losses, f"unsloth_fused_{loss_name}_loss"),
            hidden, weight, bias, torch.full_like(labels, -100),
            scaling=0.0, torch_compile=False, n_chunks=2,
        )


@pytest.mark.parametrize(
    "loss_names", [("dft", "ce"), ("ce", "dft")], ids=["dft-first", "ce-first"]
)
def test_fused_loss_compiled_matches_eager(fused_losses, loss_names):
    if os.environ.get("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "0") == "1":
        pytest.skip("UNSLOTH_FUSED_CE_COMPILE_DISABLE=1 disables fused-loss compile")
    _skip_if_compile_unavailable()

    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float32)
    with _compile_flag_guard(fused_losses):
        for loss_name in loss_names:
            loss_fn = getattr(fused_losses, f"unsloth_fused_{loss_name}_loss")
            eager = _run_wrapper(
                loss_fn,
                hidden,
                weight,
                bias,
                labels,
                torch_compile=False,
                n_chunks=2,
                shift_labels=False,
            )
            compiled = _run_wrapper(
                loss_fn,
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
                getattr(fused_losses, f"compute_fused_{loss_name}_loss")
                in fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN
            )
            _assert_result_close(compiled, eager)
            empty = _run_wrapper(
                loss_fn, hidden, weight, bias, torch.full_like(labels, -100),
                torch_compile=True, n_chunks=2, shift_labels=False,
            )
            assert all(torch.count_nonzero(value) == 0 for value in empty)
            assert fused_losses._FUSED_CE_COMPILE_SUPPORTED is True


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


@pytest.mark.parametrize(
    "loss_name", ["unsloth_fused_ce_loss", "unsloth_fused_dft_loss"]
)
def test_fused_loss_rejects_internal_weighting_hook_before_compile(
    fused_losses, loss_name
):
    hidden, weight, bias, labels = _make_inputs(torch.device("cpu"), torch.float32)

    with _compile_flag_guard(fused_losses):
        with pytest.raises(TypeError, match="per_token_weight"):
            getattr(fused_losses, loss_name)(
                trainer=None,
                hidden_states=hidden,
                lm_head_weight=weight,
                lm_head_bias=bias,
                labels=labels,
                torch_compile=True,
                n_chunks=2,
                shift_labels=False,
                per_token_weight=lambda token_nll: token_nll,
            )
        assert fused_losses._FUSED_CE_COMPILE_SUPPORTED is None
        assert fused_losses._FUSED_CE_COMPILE_FASTPATH_PROVEN == set()
