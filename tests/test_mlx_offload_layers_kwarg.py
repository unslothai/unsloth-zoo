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

import ast
import os

_LOADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth_zoo", "mlx", "loader.py")


def _warn():
    # loader.py imports mlx, so lift the one helper out of its source.
    src = open(_LOADER, encoding = "utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_warn_block_swap")
    ns = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), _LOADER, "exec"), ns)
    return ns["_warn_block_swap"]


def test_mlx_drops_offload_layers_and_its_old_name(capsys):
    warn = _warn()
    for kwargs in ({"offload_layers": 4}, {"block_swap_layers": "auto"}, {"offload_layers": 4, "block_swap_layers": 2}):
        warn(kwargs)
        assert kwargs == {}
        assert "offload_layers has no effect on Apple Silicon" in capsys.readouterr().out
    kwargs = {"max_seq_length": 2048}
    warn(kwargs)
    assert kwargs == {"max_seq_length": 2048} and capsys.readouterr().out == ""
