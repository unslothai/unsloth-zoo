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

"""Per-thread scopes for generate() decoding through CUDA graphs. Kernels picked under them must match
across eager warm-up, capture and eager steps so the first replay validates bitwise.
"""

__all__ = [
    "graph_generate",
    "graph_decode_step",
    "in_graph_generate",
    "in_graph_decode_step",
]

import threading


class _State(threading.local):
    # Class defaults: a thread that never entered a scope reads 0 without a failed attribute lookup (per-layer path).
    generate = 0
    step = 0


_TLS = _State()


class _Scope:
    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        setattr(_TLS, self._name, getattr(_TLS, self._name) + 1)
        return self

    def __exit__(self, *exc):
        setattr(_TLS, self._name, getattr(_TLS, self._name) - 1)
        return False


def graph_generate():
    return _Scope("generate")


def graph_decode_step():
    return _Scope("step")


def in_graph_generate():
    return _TLS.generate > 0


def in_graph_decode_step():
    return _TLS.step > 0 and _TLS.generate > 0
