# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Embedded copies of the vendored third-party license and provenance text.

The vendored components under ``unsloth_zoo/_vendored`` keep their upstream
``LICENSE``, ``NOTICE`` and ``MANIFEST`` as plain files in the source tree, which
are the canonical copies. A built wheel, however, installs only importable
``.py`` modules (setuptools ``packages.find`` with ``include-package-data =
false``), so those plain files do not travel inside the wheel. Redistributing the
MIT-licensed ``fla`` sources requires the license and copyright notice to ship
with them, so this module mirrors that text as importable module-level strings.
It is discovered and packaged like any other module, so the attribution ships in
the wheel without a ``pyproject.toml`` package-data entry. A test asserts these
strings stay byte-identical to the canonical files so the two cannot drift.
"""

# fla-core 0.5.1 upstream license, verbatim from unsloth_zoo/_vendored/fla/LICENSE.
FLA_LICENSE = """MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Bundled-components summary, verbatim from unsloth_zoo/_vendored/NOTICE.
NOTICE = """Unsloth Zoo
Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.

This product bundles the following third-party components. Each is redistributed
under its own license, a copy of which is included alongside the vendored source.

--------------------------------------------------------------------------------
flash-linear-attention (fla-core)
--------------------------------------------------------------------------------
  Version:   0.5.1
  Tag:       v0.5.1
  Upstream:  https://github.com/fla-org/flash-linear-attention
  PyPI:      https://pypi.org/project/fla-core/0.5.1/
  License:   MIT
  Copyright: (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
  Location:  unsloth_zoo/_vendored/fla/
  License text + provenance: unsloth_zoo/_vendored/fla/LICENSE and
                             unsloth_zoo/_vendored/fla/MANIFEST
                             (also embedded verbatim in the shipped module
                             unsloth_zoo/_vendored/_licenses.py, so the license
                             travels inside wheel installs too).

  Only the minimal transitive file closure required by the gated-delta-rule fast
  path is vendored (FusedRMSNormGated, chunk_gated_delta_rule and
  fused_recurrent_gated_delta_rule), so the Qwen3.5 / Qwen3.6 / Qwen3-Next
  gated-deltanet models get the Triton kernels without an extra
  `pip install flash-linear-attention`. Package __init__ exports were narrowed;
  all other files retain their original upstream MIT headers verbatim.
"""

# fla-core snapshot provenance, verbatim from unsloth_zoo/_vendored/fla/MANIFEST.
FLA_MANIFEST = """name = flash-linear-attention (fla-core)
version = 0.5.1
tag = v0.5.1
upstream = https://github.com/fla-org/flash-linear-attention
pypi = https://pypi.org/project/fla-core/0.5.1/
license = MIT
license_file = LICENSE
copyright = 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

description =
    Minimal, pinned, pruned snapshot of fla-core 0.5.1. Only the transitive file
    closure needed by the gated-delta-rule fast path is vendored, so Unsloth can
    enable the Triton kernels for the Qwen3.5 / Qwen3.6 / Qwen3-Next
    gated-deltanet models without requiring `pip install flash-linear-attention`.

exported_api =
    fla.modules.FusedRMSNormGated
    fla.ops.gated_delta_rule.chunk_gated_delta_rule
    fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule

modifications =
    - Narrowed package __init__ exports for fla, fla.ops, fla.modules and
      fla.ops.gated_delta_rule so importing the package does not eagerly pull
      fla.layers / fla.models / "import every op" (see the "Modified by Unsloth"
      note atop each narrowed __init__.py). All other files are verbatim upstream
      with their original MIT headers preserved, except the backported fixes below.
    - Backported three post-v0.5.1 upstream correctness fixes (each marked with an
      inline "Unsloth: backported from fla PR #..." comment):
        * PR #953 (issue #945) in ops/common/chunk_delta_h.py: pin the Blackwell
          fwd-h kernel to num_warps=2 (the 4-warp config hits a Triton tl.dot
          recurrence race that silently corrupts h / v_new on B200).
        * PR #1000 (issue #999) in ops/gated_delta_rule/wy_fast.py: restrict the
          Blackwell prepare_wy_repr_bwd_kernel autotune to the B200-validated
          config (unstable configs can hang / misaligned-address the bwd).
        * PR #983 (issue #640) in ops/common/chunk_o.py (+ TRITON_ABOVE_3_7_1 in
          utils/_compat.py and utils/__init__.py): narrow the Hopper gated
          chunk_bwd_dqkwg guard to Triton [3.4.0, 3.7.1); 3.7.1 fixes the bug.
    - Dropped fla/ops/gated_delta_rule/naive.py (the only einops dependency; the
      reference implementation is unused on the fast path).
    - Dropped the three heavy tilelang kernel files
      fla/ops/common/backends/tilelang/{chunk_bwd,parallel_attn_fwd,
      parallel_attn_bwd}.py. The guarded tilelang/__init__.py wrapper is kept;
      its lazy imports of those files execute only when the optional `tilelang`
      package is installed and its backend is selected (not the default path).

file_count = 42 python modules + this MANIFEST + LICENSE
closure_measured = 43 python modules (includes naive.py); vendored set drops
    naive.py, leaving 42.

vendored_by = Unsloth (unsloth_zoo/_vendored/fla)
"""

__all__ = ["FLA_LICENSE", "NOTICE", "FLA_MANIFEST"]
