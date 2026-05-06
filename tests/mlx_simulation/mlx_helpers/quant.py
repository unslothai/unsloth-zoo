"""mx.dequantize for the MLX 'affine' bit-packing layout.

MLX's affine quantization stores int{bits} values bit-packed into
uint32 words, with per-group scales and biases. The dequantization
math is:

    w_fp[g, i] = w_int[g, i] * scales[g] + biases[g]

Bit packing per group: ``elements_per_word = 32 // bits``
elements share a single uint32 word, with element 0 in the lowest
bits and element (elements_per_word-1) in the highest.

Other modes (mxfp4, nvfp4, mxfp8) raise NotImplementedError — their
bit layouts are mode-specific and PR-A's primary callsites all use
mode='affine'.
"""

from __future__ import annotations

import torch


def dequantize_affine(w: torch.Tensor,
                      scales: torch.Tensor,
                      biases: torch.Tensor | None,
                      group_size: int,
                      bits: int,
                      dtype: torch.dtype | None = None) -> torch.Tensor:
    """Affine dequantization.

    Args:
        w: int32/uint32 packed weight, last-dim = group_size // elements_per_word
        scales: per-group scale (last dim is the group dim)
        biases: per-group bias (same shape as scales). May be None for
            zero-mean affine.
        group_size: number of elements that share each scale/bias.
        bits: bits per element (2, 3, 4, 6, 8).
        dtype: output dtype; defaults to scales.dtype.
    """
    if bits not in (2, 3, 4, 6, 8):
        raise NotImplementedError(
            f"mlx-shim quant: bits={bits} not implemented (need 2, 3, 4, 6, or 8)."
        )
    elements_per_word = 32 // bits
    if elements_per_word * bits != 32:
        # 3-bit and 6-bit: do not divide 32 evenly; MLX uses a more
        # complex packing in those cases that we don't model yet.
        raise NotImplementedError(
            f"mlx-shim quant: bits={bits} packing layout not implemented "
            f"(packing requires bits dividing 32 evenly)."
        )
    mask = (1 << bits) - 1
    shifts = torch.arange(elements_per_word, device=w.device, dtype=torch.int64) * bits

    # Unpack: each word holds elements_per_word values.
    # w shape: (..., n_packed_per_row).  After unpack: (..., n_packed_per_row * elements_per_word).
    unpacked = (w.to(torch.int64).unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(*w.shape[:-1], -1)

    # Apply per-group scale and bias.
    *prefix, n_total = unpacked.shape
    n_groups = scales.shape[-1] if scales.ndim == w.ndim else n_total // group_size
    if n_total != n_groups * group_size:
        raise ValueError(
            f"mlx-shim quant: unpacked length {n_total} not divisible by "
            f"group_size {group_size} (n_groups={n_groups})."
        )
    grouped = unpacked.reshape(*prefix, n_groups, group_size).to(scales.dtype)
    out = grouped * scales.unsqueeze(-1)
    if biases is not None:
        out = out + biases.unsqueeze(-1)
    out = out.reshape(*prefix, n_total)
    if dtype is not None:
        out = out.to(dtype)
    return out
