# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Patches a torch ConfigModule key for one test; not monkeypatch.setattr, since pytest >= 9.2 reads
the old value from __dict__, where such keys never live, so undo calls delattr, hiding the key
from torch for the rest of the process; later torch.compile fails with
'torch._dynamo.config.suppress_errors does not exist'."""


def patch_torch_config(request, config, name, value):
    patch = config.patch(name, value)
    patch.__enter__()
    request.addfinalizer(lambda: patch.__exit__(None, None, None))
