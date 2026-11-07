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

"""Per-thread scopes Unsloth enters while generate() decodes through CUDA graphs.

graph_generate(): the whole generate() call is planned for graph decode (masks are built outside the
model forward, so mask selection reads this one). graph_decode_step(): a one-token decode step of such
a call, eager or captured. Kernels picked under these scopes must be the same for the eager warm-up,
the capture and every eager step of the call, so the first replay validates bitwise; without them,
dispatch is exactly as before. Both are plain counters, read only outside torch.compile tracing.
"""

__all__ = [
    "graph_generate",
    "graph_decode_step",
    "in_graph_generate",
    "in_graph_decode_step",
]

import threading

_TLS = threading.local()


class _Scope:
    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        setattr(_TLS, self._name, getattr(_TLS, self._name, 0) + 1)
        return self

    def __exit__(self, *exc):
        setattr(_TLS, self._name, getattr(_TLS, self._name, 1) - 1)
        return False


def graph_generate():
    """Scope of one generate() call planned for CUDA-graph decode."""
    return _Scope("generate")


def graph_decode_step():
    """Scope of one one-token decode step inside graph_generate()."""
    return _Scope("step")


def in_graph_generate():
    return getattr(_TLS, "generate", 0) > 0


def in_graph_decode_step():
    return getattr(_TLS, "step", 0) > 0 and getattr(_TLS, "generate", 0) > 0
