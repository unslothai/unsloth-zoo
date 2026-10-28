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

"""CleanupHook weakref callbacks must not raise once torch._dynamo.utils globals
were set to None at interpreter shutdown ("'NoneType' object has no attribute 'pop'")."""
import pytest

dynamo_utils = pytest.importorskip("torch._dynamo.utils")
if not hasattr(dynamo_utils, "_cleanup_owners"):
    pytest.skip("torch without CleanupHook ownership tokens", allow_module_level = True)

from unsloth_zoo.patching_utils import patch_dynamo_cleanup_hook_shutdown


def _fire_after_module_wipe(hook):
    saved = {k: dynamo_utils.__dict__[k] for k in ("_cleanup_owners", "CleanupManager")}
    try:
        # What Python finalization does to module globals.
        for k in saved:
            dynamo_utils.__dict__[k] = None
        hook()
    finally:
        dynamo_utils.__dict__.update(saved)


def test_cleanup_hook_shutdown_safe():
    patch_dynamo_cleanup_hook_shutdown()
    patch_dynamo_cleanup_hook_shutdown()  # idempotent
    assert getattr(dynamo_utils.CleanupHook.__call__, "_unsloth_shutdown_safe", False)

    count = dynamo_utils.CleanupManager.count
    scope = {}
    hook = dynamo_utils.CleanupHook.create(scope, "__unsloth_test_name", 1)
    _fire_after_module_wipe(hook)  # must not raise

    # Normal (non shutdown) behaviour is unchanged: the owner removes the name.
    hook()
    assert "__unsloth_test_name" not in scope
    dynamo_utils.CleanupManager.count = count
