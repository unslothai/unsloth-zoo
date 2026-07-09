from __future__ import annotations

from contextlib import contextmanager
import importlib.machinery
import math
import os
from pathlib import Path
import sys
import types

import pytest


DIRECT_RTOL = 1e-6
DIRECT_ATOL = 1e-7
WRAPPER_RTOL = 1e-5
WRAPPER_ATOL = 1e-7
CUDA_RTOL = 1e-4
CUDA_ATOL = 1e-6


@contextmanager
def _lightweight_unsloth_zoo_import():
    """Import fused-loss modules without running unsloth_zoo.__init__.

    These tests only need the local fused-loss modules, so install a temporary
    package skeleton and restore any pre-existing modules afterwards.
    """
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


@contextmanager
def _compile_flag_guard(torch, cross_entropy_loss):
    previous = cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED
    torch._dynamo.reset()
    cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED = None
    try:
        yield
    finally:
        cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED = previous
        torch._dynamo.reset()


def _shift_labels(torch, labels, ignore_index=-100):
    shifted = torch.empty_like(labels)
    shifted[..., :-1] = labels[..., 1:]
    shifted[..., -1] = ignore_index
    return shifted


def _linear_logits(torch, hidden, weight, bias):
    return torch.nn.functional.linear(
        hidden.to(dtype=weight.dtype, device=weight.device),
        weight,
        bias,
    )


def _apply_logit_transforms(
    torch,
    logits,
    *,
    logit_scale_multiply=None,
    logit_scale_divide=None,
    logit_softcapping=None,
):
    if logit_scale_multiply not in (None, 0):
        logits = logits * logit_scale_multiply
    if logit_scale_divide not in (None, 0):
        logits = logits / logit_scale_divide
    if logit_softcapping not in (None, 0):
        logits = torch.tanh(logits / logit_softcapping) * logit_softcapping
    return logits


def _dft_loss_from_logits(
    torch,
    logits,
    labels,
    *,
    n_items=None,
    ignore_index=-100,
):
    flat_logits = logits.view(-1, logits.shape[-1]).float().contiguous()
    flat_labels = labels.view(-1).to(device=flat_logits.device).contiguous()
    valid = flat_labels != ignore_index
    safe_labels = flat_labels.masked_fill(~valid, 0)
    logprobs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
    token_nll = -logprobs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    valid_float = valid.to(dtype=token_nll.dtype)
    weights = torch.exp(-token_nll.detach()) * valid_float

    if n_items is None:
        divisor = valid_float.sum()
    elif torch.is_tensor(n_items):
        divisor = n_items.to(device=flat_logits.device, dtype=token_nll.dtype)
    else:
        divisor = torch.tensor(n_items, device=flat_logits.device, dtype=token_nll.dtype)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    return (token_nll * weights).sum() / divisor


def _reference_dft_loss(
    torch,
    hidden,
    weight,
    bias,
    labels,
    *,
    n_items=None,
    ignore_index=-100,
    shift_labels=False,
    logit_scale_multiply=None,
    logit_scale_divide=None,
    logit_softcapping=None,
):
    if shift_labels:
        labels = _shift_labels(torch, labels, ignore_index)
    logits = _linear_logits(torch, hidden, weight, bias)
    logits = _apply_logit_transforms(
        torch,
        logits,
        logit_scale_multiply=logit_scale_multiply,
        logit_scale_divide=logit_scale_divide,
        logit_softcapping=logit_softcapping,
    )
    return _dft_loss_from_logits(
        torch,
        logits,
        labels,
        n_items=n_items,
        ignore_index=ignore_index,
    )


def _closed_form_dft(torch, transformed_logits, labels, *, n_items=None, ignore_index=-100):
    with torch.no_grad():
        flat_logits = transformed_logits.detach().view(-1, transformed_logits.shape[-1]).float()
        flat_labels = labels.view(-1).to(device=flat_logits.device)
        valid = flat_labels != ignore_index
        safe_labels = flat_labels.masked_fill(~valid, 0)
        logprobs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
        probs = logprobs.exp()
        token_nll = -logprobs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        valid_float = valid.to(dtype=token_nll.dtype)
        token_values = token_nll * torch.exp(-token_nll) * valid_float

        if n_items is None:
            divisor = valid_float.sum()
        elif torch.is_tensor(n_items):
            divisor = n_items.to(device=flat_logits.device, dtype=token_nll.dtype)
        else:
            divisor = torch.tensor(n_items, device=flat_logits.device, dtype=token_nll.dtype)
        if divisor.numel() != 1:
            divisor = divisor.ravel()[0]
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, safe_labels.unsqueeze(1), 1.0)
        weights = torch.exp(-token_nll) * valid_float
        logit_grad = weights.unsqueeze(1) * (probs - one_hot) / divisor
        return {
            "loss": token_values.sum() / divisor,
            "token_values": token_values,
            "logit_grad": logit_grad,
            "probs": probs,
        }


def _expected_linear_grads(torch, hidden, weight, bias, logit_grad):
    grad = logit_grad.to(dtype=weight.dtype)
    flat_hidden = hidden.detach().view(-1, hidden.shape[-1]).to(dtype=weight.dtype)
    hidden_grad = grad.matmul(weight.detach()).view_as(hidden).to(dtype=hidden.dtype)
    weight_grad = grad.transpose(0, 1).matmul(flat_hidden)
    bias_grad = None if bias is None else grad.sum(dim=0).to(dtype=bias.dtype)
    return hidden_grad, weight_grad, bias_grad


def _make_wrapper_inputs(torch, *, dtype=None, device=None):
    dtype = torch.float64 if dtype is None else dtype
    device = torch.device("cpu") if device is None else device
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
    bias = torch.tensor([0.10, -0.20, 0.30, -0.10, 0.20], dtype=dtype, device=device)
    labels = torch.tensor([[0, 1, 2], [3, -100, 4]], device=device)
    return hidden, weight, bias, labels


def _clone_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _run_wrapper(torch, loss_fn, hidden, weight, bias, labels, **kwargs):
    hidden_leaf = _clone_leaf(hidden)
    weight_leaf = _clone_leaf(weight)
    bias_leaf = None if bias is None else _clone_leaf(bias)
    loss = loss_fn(
        trainer=None,
        hidden_states=hidden_leaf,
        lm_head_weight=weight_leaf,
        lm_head_bias=bias_leaf,
        labels=labels.detach().clone(),
        **kwargs,
    )
    loss.backward()
    return (
        loss.detach().clone(),
        hidden_leaf.grad.detach().clone(),
        weight_leaf.grad.detach().clone(),
        None if bias_leaf is None else bias_leaf.grad.detach().clone(),
    )


def _run_direct(torch, compute_fused_dft_loss, hidden, weight, bias, labels, **kwargs):
    hidden_leaf = _clone_leaf(hidden)
    weight_leaf = _clone_leaf(weight)
    bias_leaf = None if bias is None else _clone_leaf(bias)
    loss, aux = compute_fused_dft_loss(
        hidden_leaf,
        weight_leaf,
        bias_leaf,
        labels.detach().clone(),
        **kwargs,
    )
    loss.backward()
    return (
        aux[0].detach().clone(),
        hidden_leaf.grad.detach().clone(),
        weight_leaf.grad.detach().clone(),
        None if bias_leaf is None else bias_leaf.grad.detach().clone(),
    )


def _assert_wrapper_results_close(torch, actual, expected, *, message=""):
    for actual_tensor, expected_tensor, name in zip(
        actual,
        expected,
        ("loss", "hidden grad", "weight grad", "bias grad"),
    ):
        if actual_tensor is None or expected_tensor is None:
            assert actual_tensor is expected_tensor, f"{message} {name} None mismatch"
            continue
        assert torch.allclose(
            actual_tensor,
            expected_tensor,
            rtol=WRAPPER_RTOL,
            atol=WRAPPER_ATOL,
        ), f"{message} {name} mismatch"


def _two_class_logits_for_probability(torch, probability, *, dtype=None):
    dtype = torch.float64 if dtype is None else dtype
    logit = math.log(probability / (1.0 - probability))
    return torch.tensor([[[logit, 0.0]]], dtype=dtype)


def _skip_if_compile_disabled():
    if os.environ.get("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "0") == "1":
        pytest.skip("UNSLOTH_FUSED_CE_COMPILE_DISABLE=1 disables fused-loss compile")


def test_compute_dft_single_token_closed_form_loss_and_gradient():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[0.25, -0.50]]], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(
            [[0.20, -0.30], [-0.70, 0.40], [0.50, 0.10]],
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.tensor([0.05, -0.10, 0.20], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[2]])

        loss, aux = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
        )
        logits = _linear_logits(torch, hidden, weight, bias)
        expected = _closed_form_dft(torch, logits, labels)
        expected_hidden, expected_weight, expected_bias = _expected_linear_grads(
            torch,
            hidden,
            weight,
            bias,
            expected["logit_grad"],
        )

        assert torch.allclose(loss, expected["loss"], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(aux[0], expected["loss"], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)

        loss.backward()
        assert torch.allclose(hidden.grad, expected_hidden, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(weight.grad, expected_weight, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(bias.grad, expected_bias, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_gradient_uses_detached_probability_weight():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = _two_class_logits_for_probability(torch, 0.10).requires_grad_(True)
        weight = torch.eye(2, dtype=torch.float64)
        labels = torch.tensor([[0]])
        loss, _ = compute_fused_dft_loss(hidden, weight, None, labels, shift_labels=False)
        loss.backward()

        expected = _closed_form_dft(torch, hidden, labels)
        probability = expected["probs"][0, 0]
        nll = -torch.log(probability)
        one_hot = torch.tensor([[1.0, 0.0]], dtype=expected["probs"].dtype)
        # In the (q - one_hot) convention, differentiating -p log(p) adds
        # the factor (1 - NLL); DFT intentionally omits it via detach().
        nondetached_grad = probability * (1.0 - nll) * (expected["probs"] - one_hot)

        assert torch.allclose(
            hidden.grad,
            expected["logit_grad"].view_as(hidden).to(dtype=hidden.dtype),
            rtol=DIRECT_RTOL,
            atol=DIRECT_ATOL,
        )
        assert torch.max(torch.abs(hidden.grad.float().view(1, 2) - nondetached_grad)) > 1e-3


def test_compute_dft_minus_p_log_p_shape():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        weight = torch.eye(2, dtype=torch.float64)
        labels = torch.tensor([[0]])
        probabilities = [0.99, math.exp(-1.0), 1e-4]
        observed = []
        expected_values = []

        for probability in probabilities:
            hidden = _two_class_logits_for_probability(torch, probability).requires_grad_(True)
            loss, aux = compute_fused_dft_loss(hidden, weight, None, labels, shift_labels=False)
            expected = _closed_form_dft(torch, hidden, labels)
            observed.append(loss.detach())
            expected_values.append(expected["loss"])
            assert torch.allclose(loss, expected["loss"], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
            assert torch.allclose(aux[0], expected["loss"], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)

        assert observed[1] > observed[0]
        assert observed[1] > observed[2]
        assert expected_values[2] < torch.tensor(1e-3, dtype=expected_values[2].dtype)


def test_compute_dft_low_probability_gradient_vanishes():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        weight = torch.eye(2, dtype=torch.float64)
        labels = torch.tensor([[0]])

        low_hidden = _two_class_logits_for_probability(torch, 1e-5).requires_grad_(True)
        low_loss, _ = compute_fused_dft_loss(low_hidden, weight, None, labels, shift_labels=False)
        low_loss.backward()

        peak_hidden = _two_class_logits_for_probability(torch, math.exp(-1.0)).requires_grad_(True)
        peak_loss, _ = compute_fused_dft_loss(peak_hidden, weight, None, labels, shift_labels=False)
        peak_loss.backward()

        assert torch.linalg.vector_norm(low_hidden.grad) < torch.linalg.vector_norm(peak_hidden.grad) * 1e-3


def test_compute_dft_ignore_index_zeroes_loss_and_grad_rows():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[0.40, -0.20], [9.0, -7.0]]], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(
            [[0.30, -0.10], [0.20, 0.50], [-0.40, 0.70]],
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.tensor([0.10, -0.20, 0.30], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, -100]])

        loss, _ = compute_fused_dft_loss(hidden, weight, bias, labels, shift_labels=False)
        logits = _linear_logits(torch, hidden, weight, bias)
        expected = _closed_form_dft(torch, logits, labels)
        expected_hidden, expected_weight, expected_bias = _expected_linear_grads(
            torch,
            hidden,
            weight,
            bias,
            expected["logit_grad"],
        )

        assert expected["token_values"][1] == 0
        assert torch.allclose(loss, expected["loss"], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        loss.backward()
        assert torch.count_nonzero(hidden.grad[0, 1]) == 0
        assert torch.allclose(hidden.grad, expected_hidden, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(weight.grad, expected_weight, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(bias.grad, expected_bias, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_ignored_rows_sanitize_nonfinite_logits_before_log_softmax():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor(
            [[[0.20, -0.10], [1.0e40, -1.0e40]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        weight = torch.tensor(
            [[0.30, -0.10], [0.20, 0.50], [-0.40, 0.70]],
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.tensor([0.10, -0.20, 0.30], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, -100]])

        loss, aux = compute_fused_dft_loss(hidden, weight, bias, labels, shift_labels=False)
        expected = _reference_dft_loss(
            torch,
            hidden[:, :1],
            weight,
            bias,
            labels[:, :1],
            shift_labels=False,
        )

        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(aux[0], expected, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        loss.backward()
        assert torch.isfinite(hidden.grad).all()
        assert torch.isfinite(weight.grad).all()
        assert torch.isfinite(bias.grad).all()
        assert torch.count_nonzero(hidden.grad[0, 1]) == 0


def test_compute_dft_custom_ignore_index_is_safe_for_out_of_vocab_label():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10]]], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor([[0.30, 0.40], [-0.20, 0.10]], dtype=torch.float64, requires_grad=True)
        bias = torch.tensor([0.05, -0.15], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[999]])

        loss, aux = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
            ignore_index=999,
        )
        assert torch.isfinite(loss)
        assert float(loss.detach()) == 0.0
        assert float(aux[0].detach()) == 0.0
        loss.backward()
        assert torch.count_nonzero(hidden.grad) == 0
        assert torch.count_nonzero(weight.grad) == 0
        assert torch.count_nonzero(bias.grad) == 0


def test_compute_dft_shift_labels_false_accepts_noncontiguous_labels():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor(
            [
                [[0.20, -0.10], [0.50, 0.30]],
                [[-0.70, 0.40], [0.90, -0.20]],
                [[0.10, 0.80], [-0.30, 0.60]],
            ],
            dtype=torch.float64,
        )
        weight = torch.tensor([[0.40, -0.30], [-0.20, 0.60], [0.70, 0.10]], dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2], [2, 1, 0]]).t()
        assert not labels.is_contiguous()

        loss, aux = compute_fused_dft_loss(hidden, weight, None, labels, shift_labels=False)
        expected, expected_aux = compute_fused_dft_loss(
            hidden,
            weight,
            None,
            labels.contiguous(),
            shift_labels=False,
        )

        assert torch.allclose(loss, expected, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(aux[0], expected_aux[0], rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_n_items_int_and_tensor_override_valid_count_divisor():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[0.10, 0.30], [-0.40, 0.20]]], dtype=torch.float64)
        weight = torch.tensor([[0.20, -0.50], [0.60, 0.10], [-0.30, 0.40]], dtype=torch.float64)
        labels = torch.tensor([[0, 2]])

        default_loss, _ = compute_fused_dft_loss(hidden, weight, None, labels, shift_labels=False)
        int_loss, _ = compute_fused_dft_loss(hidden, weight, None, labels, n_items=4, shift_labels=False)
        tensor_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            None,
            labels,
            n_items=torch.tensor(4),
            shift_labels=False,
        )

        assert torch.allclose(int_loss, default_loss / 2, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(tensor_loss, int_loss, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_zero_n_items_uses_one_as_divisor():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10], [0.50, 0.30]]], dtype=torch.float64)
        weight = torch.tensor([[0.10, 0.40], [-0.30, 0.20], [0.50, -0.60]], dtype=torch.float64)
        labels = torch.tensor([[1, 2]])

        default_loss, _ = compute_fused_dft_loss(hidden, weight, None, labels, shift_labels=False)
        zero_divisor_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            None,
            labels,
            n_items=0,
            shift_labels=False,
        )

        assert torch.isfinite(zero_divisor_loss)
        assert torch.allclose(zero_divisor_loss, default_loss * 2, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_logit_transforms_apply_in_implemented_order():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.tensor([[[3.25, -1.75], [0.50, 2.20]]], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(
            [[1.20, -0.70], [-1.50, 0.90], [0.30, 1.10]],
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.tensor([0.20, -0.30, 0.40], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[0, 2]])
        kwargs = dict(
            logit_scale_multiply=1.7,
            logit_scale_divide=0.8,
            logit_softcapping=2.25,
        )

        ref_hidden = _clone_leaf(hidden)
        ref_weight = _clone_leaf(weight)
        ref_bias = _clone_leaf(bias)
        reference_loss = _reference_dft_loss(
            torch,
            ref_hidden,
            ref_weight,
            ref_bias,
            labels,
            **kwargs,
        )
        actual_loss, _ = compute_fused_dft_loss(hidden, weight, bias, labels, shift_labels=False, **kwargs)

        assert torch.allclose(actual_loss, reference_loss, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        actual_loss.backward()
        reference_loss.backward()
        assert torch.allclose(hidden.grad, ref_hidden.grad, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(weight.grad, ref_weight.grad, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)
        assert torch.allclose(bias.grad, ref_bias.grad, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)

        no_transform_loss, _ = compute_fused_dft_loss(
            hidden.detach(),
            weight.detach(),
            bias.detach(),
            labels,
            shift_labels=False,
        )
        zero_transform_loss, _ = compute_fused_dft_loss(
            hidden.detach(),
            weight.detach(),
            bias.detach(),
            labels,
            shift_labels=False,
            logit_scale_multiply=0,
            logit_scale_divide=0,
            logit_softcapping=0,
        )
        assert torch.allclose(zero_transform_loss, no_transform_loss, rtol=DIRECT_RTOL, atol=DIRECT_ATOL)


def test_compute_dft_rejects_label_smoothing():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss

        hidden = torch.randn(1, 1, 2, dtype=torch.float64)
        weight = torch.randn(3, 2, dtype=torch.float64)
        labels = torch.tensor([[0]])

        with pytest.raises(ValueError, match="label_smoothing"):
            compute_fused_dft_loss(
                hidden,
                weight,
                None,
                labels,
                shift_labels=False,
                label_smoothing=0.1,
            )


def test_unsloth_fused_dft_matches_compute_for_preshifted_one_chunk():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        wrapper = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=1,
            shift_labels=False,
        )
        direct = _run_direct(
            torch,
            compute_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
        )

        _assert_wrapper_results_close(torch, wrapper, direct)


def test_unsloth_fused_dft_chunking_invariant_value_and_gradients():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        baseline = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=1,
            shift_labels=False,
        )

        for n_chunks in (2, 20):
            chunked = _run_wrapper(
                torch,
                unsloth_fused_dft_loss,
                hidden,
                weight,
                bias,
                labels,
                torch_compile=False,
                n_chunks=n_chunks,
                shift_labels=False,
            )
            _assert_wrapper_results_close(torch, chunked, baseline, message=f"n_chunks={n_chunks}")


def test_unsloth_fused_dft_uses_shared_automatic_chunk_sizer(monkeypatch):
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        import unsloth_zoo.fused_losses.cross_entropy_loss as cross_entropy_loss

        calls = []

        def get_chunk_size(bsz, qlen, vocab_size, target_gb=None):
            calls.append((bsz, qlen, vocab_size, target_gb))
            return 2

        monkeypatch.setattr(cross_entropy_loss, "get_chunk_size", get_chunk_size)
        monkeypatch.setattr(cross_entropy_loss, "TARGET_GB", None)
        monkeypatch.setattr(cross_entropy_loss, "N_CHUNKS", None)

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        automatic = _run_wrapper(
            torch,
            cross_entropy_loss.unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            shift_labels=False,
        )
        assert calls == [(2, 3, 5, None)]

        calls.clear()
        explicit = _run_wrapper(
            torch,
            cross_entropy_loss.unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )
        assert calls == []
        _assert_wrapper_results_close(torch, automatic, explicit)


def test_unsloth_fused_dft_wrapper_uses_global_divisor_not_per_chunk():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10], [0.50, 0.30], [-0.70, 0.40], [0.90, -0.20]]], dtype=torch.float64)
        weight = torch.tensor([[0.40, -0.30], [-0.20, 0.60], [0.70, 0.10]], dtype=torch.float64)
        bias = torch.tensor([0.05, -0.10, 0.20], dtype=torch.float64)
        labels = torch.tensor([[0, -100, 2, -100]])

        loss, _, _, _ = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )

        logits = _linear_logits(torch, hidden, weight, bias)
        closed_form = _closed_form_dft(torch, logits, labels)
        token_values = closed_form["token_values"]
        global_loss = (token_values[0] + token_values[2]) / 2
        per_chunk_bug_loss = token_values[0] + token_values[2]

        assert torch.allclose(loss, global_loss, rtol=WRAPPER_RTOL, atol=WRAPPER_ATOL)
        assert torch.abs(loss - per_chunk_bug_loss) > 1e-3


def test_unsloth_fused_dft_overwrite_true_value_and_no_corruption():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        expected = _run_direct(
            torch,
            compute_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
        )
        overwritten = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
            overwrite=True,
        )

        _assert_wrapper_results_close(torch, overwritten, expected)


def test_unsloth_fused_dft_default_shift_masks_final_position():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10], [0.50, 0.30], [-0.70, 0.40], [0.90, -0.20]]], dtype=torch.float64)
        weight = torch.tensor(
            [[0.40, -0.30], [-0.20, 0.60], [0.70, 0.10], [-0.50, 0.20], [0.30, 0.80]],
            dtype=torch.float64,
        )
        bias = torch.tensor([0.05, -0.10, 0.20, -0.15, 0.25], dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2, 3]])
        labels_with_different_first_source = torch.tensor([[4, 1, 2, 3]])

        baseline = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=1,
        )
        changed_first_label = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels_with_different_first_source,
            torch_compile=False,
            n_chunks=1,
        )

        _assert_wrapper_results_close(torch, changed_first_label, baseline)
        assert torch.count_nonzero(baseline[1][0, -1]) == 0


def test_unsloth_fused_dft_mask_applies_to_shifted_targets():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10], [0.50, 0.30], [-0.70, 0.40], [0.90, -0.20]]], dtype=torch.float64)
        weight = torch.tensor([[0.40, -0.30], [-0.20, 0.60], [0.70, 0.10], [-0.50, 0.20]], dtype=torch.float64)
        bias = torch.tensor([0.05, -0.10, 0.20, -0.15], dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2, 3]])
        mask = torch.tensor([[1, 1, 0, 1]])
        expected_preshifted_labels = torch.tensor([[1, -100, 3, -100]])

        masked = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            mask=mask,
            torch_compile=False,
            n_chunks=1,
        )
        explicit = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            expected_preshifted_labels,
            torch_compile=False,
            n_chunks=1,
            shift_labels=False,
        )

        _assert_wrapper_results_close(torch, masked, explicit)
        assert torch.count_nonzero(masked[1][0, 1]) == 0


def test_unsloth_fused_dft_shift_labels_false_uses_labels_as_targets():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_dft_loss
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.tensor([[[0.10, -0.20], [0.30, 0.40], [-0.50, 0.60]]], dtype=torch.float64)
        weight = torch.tensor([[0.20, -0.10], [-0.40, 0.30], [0.50, 0.70]], dtype=torch.float64)
        bias = torch.tensor([0.05, -0.15, 0.25], dtype=torch.float64)
        labels = torch.tensor([[0, 1, 2]])

        wrapper = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=1,
            shift_labels=False,
        )
        direct = _run_direct(
            torch,
            compute_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            shift_labels=False,
        )

        _assert_wrapper_results_close(torch, wrapper, direct)
        assert torch.count_nonzero(wrapper[1][0, -1]) > 0


def test_unsloth_fused_dft_shift_labels_false_ignores_mask_by_contract():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        mask = torch.zeros_like(labels)

        without_mask = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )
        with_mask = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            mask=mask,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )

        _assert_wrapper_results_close(torch, with_mask, without_mask)


def test_unsloth_fused_dft_custom_ignore_index_through_wrapper():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.tensor([[[0.20, -0.10], [0.50, 0.30], [-0.70, 0.40]]], dtype=torch.float64)
        weight = torch.tensor([[0.40, -0.30], [-0.20, 0.60], [0.70, 0.10]], dtype=torch.float64)
        bias = torch.tensor([0.05, -0.10, 0.20], dtype=torch.float64)
        labels = torch.tensor([[0, 999, 2]])

        loss, hidden_grad, weight_grad, bias_grad = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=1,
            ignore_index=999,
        )

        assert torch.isfinite(loss)
        assert torch.count_nonzero(hidden_grad[0, 0]) == 0
        assert torch.count_nonzero(hidden_grad[0, 1]) > 0
        assert torch.isfinite(weight_grad).all()
        assert torch.isfinite(bias_grad).all()


def test_unsloth_fused_dft_scaling_nonzero_preserves_public_loss_and_gradients():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch)
        unscaled = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
            scaling=None,
        )
        scaled = _run_wrapper(
            torch,
            unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
            scaling=7.0,
        )

        _assert_wrapper_results_close(torch, scaled, unscaled)


def test_unsloth_fused_dft_all_ignored_returns_connected_zero():
    torch = pytest.importorskip("torch")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        hidden = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(6, dtype=torch.float64, requires_grad=True)
        labels = torch.full((1, 4), -100)

        loss = unsloth_fused_dft_loss(
            trainer=None,
            hidden_states=hidden,
            lm_head_weight=weight,
            lm_head_bias=bias,
            labels=labels,
            torch_compile=False,
            n_chunks=2,
        )
        assert torch.isfinite(loss)
        assert float(loss.detach()) == 0.0
        loss.backward()
        assert torch.count_nonzero(hidden.grad) == 0
        assert torch.count_nonzero(weight.grad) == 0
        assert torch.count_nonzero(bias.grad) == 0


def test_unsloth_fused_dft_compiled_parity_and_probe_success():
    torch = pytest.importorskip("torch")
    _skip_if_compile_disabled()

    with _lightweight_unsloth_zoo_import():
        import unsloth_zoo.fused_losses.cross_entropy_loss as cross_entropy_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch, dtype=torch.float32)
        eager = _run_wrapper(
            torch,
            cross_entropy_loss.unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )

        with _compile_flag_guard(torch, cross_entropy_loss):
            compiled = _run_wrapper(
                torch,
                cross_entropy_loss.unsloth_fused_dft_loss,
                hidden,
                weight,
                bias,
                labels,
                torch_compile=True,
                n_chunks=2,
                shift_labels=False,
            )
            assert cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED is True

        _assert_wrapper_results_close(torch, compiled, eager)


def test_unsloth_fused_dft_label_smoothing_raise_does_not_poison_compile_flag():
    torch = pytest.importorskip("torch")
    _skip_if_compile_disabled()

    with _lightweight_unsloth_zoo_import():
        import unsloth_zoo.fused_losses.cross_entropy_loss as cross_entropy_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch, dtype=torch.float32)

        with _compile_flag_guard(torch, cross_entropy_loss):
            with pytest.raises(ValueError, match="label_smoothing"):
                cross_entropy_loss.unsloth_fused_dft_loss(
                    trainer=None,
                    hidden_states=hidden.detach().clone().requires_grad_(True),
                    lm_head_weight=weight.detach().clone().requires_grad_(True),
                    lm_head_bias=bias.detach().clone().requires_grad_(True),
                    labels=labels.detach().clone(),
                    torch_compile=True,
                    n_chunks=2,
                    shift_labels=False,
                    label_smoothing=0.1,
                )
            assert cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED is None


def test_unsloth_fused_dft_compiles_when_flag_preset_true():
    torch = pytest.importorskip("torch")
    _skip_if_compile_disabled()

    with _lightweight_unsloth_zoo_import():
        import unsloth_zoo.fused_losses.cross_entropy_loss as cross_entropy_loss

        hidden, weight, bias, labels = _make_wrapper_inputs(torch, dtype=torch.float32)
        eager = _run_wrapper(
            torch,
            cross_entropy_loss.unsloth_fused_dft_loss,
            hidden,
            weight,
            bias,
            labels,
            torch_compile=False,
            n_chunks=2,
            shift_labels=False,
        )

        with _compile_flag_guard(torch, cross_entropy_loss):
            cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED = True
            compiled = _run_wrapper(
                torch,
                cross_entropy_loss.unsloth_fused_dft_loss,
                hidden,
                weight,
                bias,
                labels,
                torch_compile=True,
                n_chunks=2,
                shift_labels=False,
            )
            assert cross_entropy_loss._FUSED_CE_COMPILE_SUPPORTED is True

        _assert_wrapper_results_close(torch, compiled, eager)


def test_unsloth_fused_dft_exports_public_symbol():
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses import unsloth_fused_dft_loss
        from unsloth_zoo.loss_utils import unsloth_fused_dft_loss as exported

        assert exported is unsloth_fused_dft_loss


def test_unsloth_fused_dft_cuda_smoke():
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        pytest.skip("requires CUDA")

    with _lightweight_unsloth_zoo_import():
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_dft_loss

        torch.manual_seed(1)
        device = torch.device("cuda")
        hidden = torch.randn(1, 6, 8, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(11, 8, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(11, device=device, dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, 11, (1, 6), device=device)
        labels[0, 3] = -100

        ref_hidden = _clone_leaf(hidden)
        ref_weight = _clone_leaf(weight)
        ref_bias = _clone_leaf(bias)
        loss = unsloth_fused_dft_loss(
            trainer=None,
            hidden_states=hidden,
            lm_head_weight=weight,
            lm_head_bias=bias,
            labels=labels,
            torch_compile=False,
            n_chunks=2,
        )
        ref_loss = _reference_dft_loss(
            torch,
            ref_hidden,
            ref_weight,
            ref_bias,
            labels,
            shift_labels=True,
        )

        assert torch.allclose(loss, ref_loss, rtol=CUDA_RTOL, atol=CUDA_ATOL)
        loss.backward()
        ref_loss.backward()
        assert torch.allclose(hidden.grad, ref_hidden.grad, rtol=CUDA_RTOL, atol=CUDA_ATOL)
        assert torch.allclose(weight.grad, ref_weight.grad, rtol=CUDA_RTOL, atol=CUDA_ATOL)
        assert torch.allclose(bias.grad, ref_bias.grad, rtol=CUDA_RTOL, atol=CUDA_ATOL)
