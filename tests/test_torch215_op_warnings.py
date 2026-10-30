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

"""torch 2.15 routes TORCH_WARN from operators called through torch.ops.* into Python's warnings module, attributed
to torch/_ops.py. Guard: importing unsloth_zoo neither silences those warnings process-wide (an ignore filter on
module "torch" would match torch._ops) nor escalates them to errors, so an expected op warning cannot fail training."""

import warnings

import torch


def _op_warning():
    a, b, out = torch.ones(2), torch.ones(2), torch.empty(3)
    torch.ops.aten.add.out(a, b, out = out)  # resizes a non-empty out: TORCH_WARN


def test_unsloth_zoo_installs_no_global_filter_for_torch_op_warnings():
    import unsloth_zoo  # noqa: F401
    for action, message, category, module, _ in warnings.filters:
        pattern = getattr(module, "pattern", module) or ""
        assert not (action == "error" and issubclass(UserWarning, category) and not pattern), warnings.filters
        if action == "ignore" and issubclass(UserWarning, category) and not getattr(message, "pattern", message):
            assert not pattern or not __import__("re").match(pattern, "torch._ops"), (action, category, pattern)


def test_op_warning_reaches_python_and_does_not_raise():
    import unsloth_zoo  # noqa: F401
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _op_warning()
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 15):
        assert any("resized" in str(w.message) for w in caught), [str(w.message) for w in caught]
