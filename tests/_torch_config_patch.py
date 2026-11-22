# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Patch a torch ConfigModule key (torch._dynamo.config, torch._functorch.config, ...) for the rest of one test.

Not monkeypatch.setattr: pytest >= 9.2 reads the old value from the target's instance __dict__, where a ConfigModule
key never lives, so its undo calls delattr, which hides the key from torch itself for the rest of the process (every
later torch.compile in that xdist worker then dies with "torch._dynamo.config.suppress_errors does not exist")."""


def patch_torch_config(request, config, name, value):
    patch = config.patch(name, value)
    patch.__enter__()
    request.addfinalizer(lambda: patch.__exit__(None, None, None))
