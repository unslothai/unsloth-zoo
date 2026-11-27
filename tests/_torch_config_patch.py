# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Patches one torch ConfigModule key for a test; not monkeypatch.setattr: pytest >= 9.2 reads old
values from __dict__, where such keys never live, so undo's delattr hides the key process-wide,
and later torch.compile fails ('torch._dynamo.config.suppress_errors does not exist')."""


def patch_torch_config(request, config, name, value):
    patch = config.patch(name, value)
    patch.__enter__()
    request.addfinalizer(lambda: patch.__exit__(None, None, None))
