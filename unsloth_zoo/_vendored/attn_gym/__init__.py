# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Unsloth: narrowed. Upstream attn_gym/__init__.py imports every
# attention family; only the sparse gather-attention closure is vendored here
# (see MANIFEST). Import it from unsloth_zoo._vendored.attn_gym.sparse.gather_attn.
"""attention-gym (meta-pytorch/attention-gym), sparse gather attention only."""
