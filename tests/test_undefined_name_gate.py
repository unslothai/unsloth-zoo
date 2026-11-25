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

"""Two real undefined names the ruff gate hid under continue-on-error; 11 others exec defines."""


import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def test_the_float32_grpo_path_can_build_its_autocast_context():
    """nullcontext unbound on UNSLOTH_FORCE_FLOAT32=1 (NameError); trainer copy has no module
    imports, so test its source in a bare namespace; a module-level import would pass while the
    trainer still raises."""
    import inspect

    import unsloth_zoo.rl_replacements as rl

    source = inspect.getsource(rl.grpo_accumulated_loss)
    namespace = {"torch": rl.torch, "os": rl.os}
    exec(compile(source, "<generated-trainer>", "exec"), namespace)
    rebuilt = namespace["grpo_accumulated_loss"]

    assert "nullcontext" in rebuilt.__code__.co_varnames, (
        "nullcontext resolves as a global, so the generated UnslothGRPOTrainer raises "
        "NameError on the UNSLOTH_FORCE_FLOAT32 path; bind it inside the function body"
    )


def test_peft_utils_exports_only_names_it_has():
    """merge_and_overwrite_lora and merge_and_dequantize_lora stayed in __all__ after moving to saving_utils, so a star import raised AttributeError."""
    import unsloth_zoo.peft_utils as peft_utils

    missing = [name for name in peft_utils.__all__ if not hasattr(peft_utils, name)]
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_a_star_import_of_peft_utils_succeeds():
    """The failure mode itself, in a subprocess because a star import needs module scope."""
    result = subprocess.run(
        [sys.executable, "-c", "from unsloth_zoo.peft_utils import *"],
        capture_output = True,
        text = True,
        cwd = REPO,
        timeout = 600,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output = True
    ).returncode != 0,
    reason = "needs ruff",
)
def test_the_narrow_gate_is_clean_so_it_can_stay_gating():
    """Source lint now gates on this: a new undefined name must fail here, and an exec-defined one needs the inline noqa that says so."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ruff", "check",
            "--select", "E9,F63,F7,F82",
            "--output-format", "concise",
            "unsloth_zoo", "tests", "scripts",
        ],
        capture_output = True,
        text = True,
        cwd = REPO,
        timeout = 600,
    )
    assert result.returncode == 0, result.stdout[-4000:]
