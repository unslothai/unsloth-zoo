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

"""Regression guard for the chunked-prefill number load_vllm prints.

vLLM's chunked-prefill budget is the `max_num_batched_tokens` load_vllm passes to
EngineArgs. A separate `chunked_prefill_tokens` local used to be sized by its own
memory ladder, overwritten by `max_seq_length` on the next line, and then printed,
so the reported number described nothing that reached vLLM. The surrounding
function needs a live GPU, so this checks the source-level invariant the way
tests/test_vllm_utils_xpu_sm_cap.py does.
"""

import importlib.util
import pathlib
import re


def _module_source_text(module_name: str) -> str:
    # find_spec is metadata-only, so importing vllm_utils (and its import-time
    # torch.cuda probes) is avoided on a CPU-only box.
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin in (None, "built-in"):
        raise ImportError(f"could not locate source for {module_name!r}")
    return pathlib.Path(spec.origin).read_text(encoding="utf-8")


def test_chunked_prefill_line_reports_the_value_vllm_receives():
    src = _module_source_text("unsloth_zoo.vllm_utils")

    printed = re.search(r"Chunked prefill tokens = \{(\w+)\}", src)
    assert printed is not None, "the chunked prefill line vanished from vllm_utils.py"

    passed = re.search(r"max_num_batched_tokens\s*=\s*max_num_batched_tokens", src)
    assert passed is not None, "max_num_batched_tokens is no longer the EngineArgs value"

    assert printed.group(1) == "max_num_batched_tokens", (
        f"the line prints {printed.group(1)!r}, which is not what EngineArgs is given"
    )


def test_no_second_chunked_prefill_variable_is_computed_and_dropped():
    src = _module_source_text("unsloth_zoo.vllm_utils")

    # Every assignment to the old local was overwritten before anything read it, so the
    # eight-branch ladder that produced them could not change the run.
    assert "chunked_prefill_tokens" not in src
