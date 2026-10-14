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

"""Training speculative-decoding drafters (native MTP heads) against a frozen target.

Submodules import transformers lazily so `import unsloth_zoo` stays unchanged.
"""

__all__ = [
    "resolve_text_parts",
    "capture_hidden_states",
    "run_frozen_target",
    "chunked_lm_loss",
    "fastmtp_step_weights",
    "Qwen35MTPHead",
    "build_mtp_head",
    "load_mtp_state_dict",
    "find_mtp_tensor_names",
    "write_mtp_tensors",
    "save_mtp_head",
    "load_saved_mtp_head",
    "MTPTrainingArguments",
    "UnslothMTPTrainer",
    "MTPDataCollator",
    "Qwen3DSparkDrafter",
    "DSparkTrainingArguments",
    "UnslothDSparkTrainer",
]

_LOCATIONS = {
    "resolve_text_parts": "capture",
    "capture_hidden_states": "capture",
    "run_frozen_target": "capture",
    "chunked_lm_loss": "losses",
    "fastmtp_step_weights": "losses",
    "Qwen35MTPHead": "mtp",
    "build_mtp_head": "mtp",
    "load_mtp_state_dict": "mtp",
    "find_mtp_tensor_names": "mtp",
    "write_mtp_tensors": "stitch",
    "save_mtp_head": "stitch",
    "load_saved_mtp_head": "stitch",
    "MTPTrainingArguments": "trainer",
    "UnslothMTPTrainer": "trainer",
    "MTPDataCollator": "trainer",
    "Qwen3DSparkDrafter": "dspark",
    "DSparkTrainingArguments": "dspark",
    "UnslothDSparkTrainer": "dspark",
}


def __getattr__(name):
    module = _LOCATIONS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{module}", __name__), name)
