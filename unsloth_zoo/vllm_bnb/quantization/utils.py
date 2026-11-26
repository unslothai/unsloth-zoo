# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored by Unsloth from vllm-project/vllm-bnb-plugin v0.0.3 (commit d4914a9), Apache-2.0 (LICENSE in this
# directory). Changes: relocated under unsloth_zoo.vllm_bnb; see unsloth_zoo/vllm_bnb/__init__.py for the list.

from collections.abc import Generator
from contextlib import contextmanager

import torch
from packaging import version
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


@contextmanager
def bnb_quantize_stream() -> Generator[None, None, None]:
    """Run bitsandbytes quantization on the default stream.

    `bitsandbytes::quantize_4bit` and `bitsandbytes::quantize_blockwise` launch
    their kernels without forwarding torch's current stream, so they always run
    on the default stream. vLLM runs on a dedicated non-default stream, so the
    torch ops feeding those kernels (e.g. the `absmax - offset` subtraction in
    the double quantization path) are unordered with respect to them and the
    quantization statistics come out corrupted. Running the torch side on the
    default stream too restores the ordering.
    """
    if not current_platform.is_cuda_alike():
        yield
        return

    device_module = torch.get_device_module(current_platform.device_type)
    default_stream = device_module.default_stream()
    stream = current_stream()
    if stream == default_stream:
        yield
        return

    default_stream.wait_stream(stream)
    try:
        with device_module.stream(default_stream):
            yield
    finally:
        stream.wait_stream(default_stream)


def _get_min_bitsandbytes_version() -> str:
    return "0.49.2" if current_platform.is_rocm() else "0.48.1"


def _check_bitsandbytes_version() -> None:
    min_version = _get_min_bitsandbytes_version()
    try:
        import bitsandbytes

        if version.parse(bitsandbytes.__version__) < version.parse(min_version):
            raise ImportError(
                "bitsandbytes version is wrong. Please "
                f"install bitsandbytes>={min_version}."
            )
    except ImportError as err:
        raise ImportError(
            f"Please install bitsandbytes>={min_version} via "
            f"`pip install bitsandbytes>={min_version}` to use "
            "bitsandbytes quantizer."
        ) from err


def _adjust_bitsandbytes_4bit_shard(
    param: torch.nn.Parameter,
    shard_offsets: dict[str, tuple[int, int]],
    loaded_shard_id: str,
) -> tuple[int, int]:
    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]
    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total
    return quantized_size, quantized_offset


def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]) -> bool:
    components = prefix.split(".")
    substr_check = any(
        module_name in components for module_name in llm_int8_skip_modules
    )
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))
    set_llm_int8_skip_modules = set(llm_int8_skip_modules)
    prefix_check = len(set_llm_int8_skip_modules & set_components) != 0

    return substr_check or prefix_check


def calculate_quant_ratio(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits
    return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits
