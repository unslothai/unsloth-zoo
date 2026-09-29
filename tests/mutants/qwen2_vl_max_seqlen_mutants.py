# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""Mutation harness for
``test_compiler_custom_gradient_checkpointing_qwen2_vl_max_seqlen_is_recomputed``.

That guard pins the recompute fallback behind the transformers 5.x entry in
``unsloth_zoo/compiler.py:custom_gradient_checkpointing_replacements``. The
entry keeps ``max_seqlen`` (bound into the checkpointed callable), and the
recompute is what makes a lost or mis-bound keyword degrade into an extra
``.max()`` instead of into wrong attention. A guard that only greps for
``get_max_seqlen(`` stays green on upstreams where omitting the keyword is no
longer lossless, so this harness rebuilds
``VisionAttention.forward`` from its own source with one line mutated and
requires the shipped guard to fail on each mutant.

Each mutant is checked twice:

* ``full``       -- the whole guard (AST pin + executed forward).
* ``exec only``  -- the guard fed the *pristine* source for its AST stage, so
                    only the executed forward can catch the mutant.

Run it directly (CPU only, no model downloads)::

    python tests/mutants/qwen2_vl_max_seqlen_mutants.py
"""

from __future__ import annotations

import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_upstream_source_patterns import (  # noqa: E402
    _assert_qwen2_vl_max_seqlen_recomputed,
)


# Each entry: name, (old, new) source substitution, why it matters.
MUTATIONS = (
    (
        "different sequence tensor",
        "get_max_seqlen(cu_seqlens, self.config",
        "get_max_seqlen(cu_seqlens[:2], self.config",
        "recompute kept, but off a different tensor -> 4 instead of 5",
    ),
    (
        "non-None precomputed value",
        'kwargs={"max_seqlen": max_seqlen}',
        'kwargs={"max_seqlen": max_seqlen if max_seqlen is not None else 128}',
        "recompute kept, but a config-style fallback wins -> 128 instead of 5",
    ),
)


def _build_mutant(old: str, new: str):
    """Recompile ``VisionAttention.forward`` from source with one substitution
    and return ``(subclass, mutated_source)``. Compiled against a copy of the
    upstream module globals, so the real module is never touched."""
    from transformers.models.qwen2_vl import modeling_qwen2_vl as module

    src = textwrap.dedent(inspect.getsource(module.VisionAttention.forward))
    if old not in src:
        raise SystemExit(
            f"mutation anchor is stale, not found in VisionAttention.forward: {old!r}"
        )
    mutated_src = src.replace(old, new)
    namespace = dict(module.__dict__)
    exec(compile(mutated_src, "<mutant VisionAttention.forward>", "exec"),
         namespace)
    mutant = type(
        "MutantVisionAttention",
        (module.VisionAttention,),
        {"forward": namespace["forward"]},
    )
    return mutant, mutated_src, src


def _run_guard(attn_cls, forward_src):
    """Return ``None`` if the guard passed, else its failure message."""
    try:
        _assert_qwen2_vl_max_seqlen_recomputed(attn_cls, forward_src = forward_src)
    except BaseException as error:            # pytest.fail raises OutcomeException
        return getattr(error, "msg", None) or str(error)
    return None


def main() -> int:
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (  # noqa: F401
            get_max_seqlen, VisionAttention,
        )
    except ImportError:
        print("SKIP: get_max_seqlen / VisionAttention not in this transformers "
              "build (4.x) -- nothing to mutate")
        return 0

    failures = []

    control = _run_guard(VisionAttention, None)
    if control is None:
        print("PASS  control (unmutated upstream VisionAttention): guard green")
    else:
        failures.append("control")
        print(f"FAIL  control (unmutated upstream VisionAttention): {control}")

    for name, old, new, why in MUTATIONS:
        mutant, mutated_src, pristine_src = _build_mutant(old, new)
        print(f"\n-- mutant: {name} ({why})")
        for label, src in (("full     ", mutated_src),
                           ("exec only", pristine_src)):
            caught = _run_guard(mutant, src)
            if caught is None:
                failures.append(f"{name} [{label.strip()}]")
                print(f"   FAIL  {label}: guard stayed GREEN on the mutant")
            else:
                print(f"   PASS  {label}: caught -- {caught}")

    print()
    if failures:
        print(f"MUTATION HARNESS FAILED: {failures}")
        return 1
    print("MUTATION HARNESS OK: guard is green on real upstream and red on "
          "every mutant, under both the AST pin and the executed forward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
