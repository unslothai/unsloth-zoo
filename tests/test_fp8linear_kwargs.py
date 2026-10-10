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


"""convert_vllm_to_huggingface builds FP8Linear with the kwargs this transformers accepts."""
import pytest

from unsloth_zoo.vllm_utils import _accepted_kwargs


KWARGS = dict(in_features = 0, out_features = 0, has_bias = False, dtype = "bf16", block_size = (128, 128), activation_scheme = "dynamic")


def test_real_fp8linear_builds_with_the_filtered_kwargs():
    # transformers 5.10+ FP8Linear.__init__ has no `dtype`; passing it raised TypeError.
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    layer = fp8.FP8Linear(**_accepted_kwargs(fp8.FP8Linear, KWARGS))
    assert layer.block_size == (128, 128)


def test_dtype_is_kept_where_the_signature_takes_it():
    class OldFP8Linear:  # transformers 5.0 to 5.5 layout
        def __init__(self, in_features, out_features, block_size = None, activation_scheme = "dynamic", has_bias = False, dtype = None):
            self.dtype = dtype

    class NewFP8Linear:  # transformers 5.10+ layout
        def __init__(self, in_features, out_features, block_size = None, activation_scheme = "dynamic", scale_fmt = "float", has_bias = False):
            pass

    class AnyKwargs:
        def __init__(self, **kwargs):
            pass

    assert _accepted_kwargs(OldFP8Linear, KWARGS) == KWARGS
    assert "dtype" not in _accepted_kwargs(NewFP8Linear, KWARGS)
    assert _accepted_kwargs(AnyKwargs, KWARGS) == KWARGS
    NewFP8Linear(**_accepted_kwargs(NewFP8Linear, KWARGS))


def test_real_fbgemm_fp8linear_builds_with_the_filtered_kwargs():
    # transformers 5.x FbgemmFp8Linear takes `dtype`; 4.x took `weight_dtype`.
    fbgemm = pytest.importorskip("transformers.integrations.fbgemm_fp8")
    import torch
    kwargs = dict(in_features = 0, out_features = 0, bias = False, weight_dtype = torch.bfloat16, dtype = torch.bfloat16)
    fbgemm.FbgemmFp8Linear(**_accepted_kwargs(fbgemm.FbgemmFp8Linear, kwargs))
