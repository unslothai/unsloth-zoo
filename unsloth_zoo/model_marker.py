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

"""Zoo patches replace transformers code process-wide. After require_model_marker() (Unsloth's import) they
apply only to models passed to mark_model; without it every model is patched, as before."""

__all__ = ["MODEL_MARKER", "mark_model", "require_model_marker", "model_marker_required", "is_unsloth_model"]

MODEL_MARKER = "_unsloth_owned_model"
_REQUIRE_MARKER = [False]


def require_model_marker(enabled: bool = True) -> None:
    _REQUIRE_MARKER[0] = bool(enabled)


def model_marker_required() -> bool:
    return _REQUIRE_MARKER[0]


def mark_model(model) -> None:
    try:
        from transformers import PreTrainedModel  # noqa: PLC0415
        modules = model.modules()
    except Exception:
        return
    for module in modules:
        if isinstance(module, PreTrainedModel):
            try:
                object.__setattr__(module, MODEL_MARKER, True)
            except Exception:
                pass


def is_unsloth_model(module) -> bool:
    return not _REQUIRE_MARKER[0] or getattr(module, MODEL_MARKER, False) is True
