# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A relative import transformers cannot resolve breaks every remote-code save.

`Paddle_OCR_(1B)_Vision` died at the trainer's first checkpoint save:

    FileNotFoundError: [Errno 2] No such file or directory:
    '/usr/local/lib/python3.12/dist-packages/unsloth_zoo/mlx.runtime.py'

Nothing is missing from the wheel. The real module is `unsloth_zoo/mlx/runtime.py`
and it imports fine. The path with a dot in it is built by TRANSFORMERS, in
`dynamic_module_utils`:

    relative_imports += re.findall(r"^\\s*from\\s+\\.(\\S+)\\s+import", content, ...)
    ...
    module_path / f"{imp}.py"

Whatever that regex captures is pasted after the directory and `.py`. It does
not translate dots into separators, does not strip further leading dots, and
does not consider that the target might be a package. So three shapes break:

    from .mlx.runtime import x   ->  unsloth_zoo/mlx.runtime.py
    from .temporary_patches ...  ->  unsloth_zoo/temporary_patches.py  (a dir)
    from .. import __version__   ->  unsloth_zoo/mlx/..py

and one shape does not: `from .sibling import x`, which resolves to a real
file. `from . import x` is also safe -- the regex needs at least one non-space
character after the dot, so it never matches.

It reaches unsloth_zoo because a remote-code model takes the `custom_object_save`
path on save, which walks the relative imports of the module its class lives in,
recursively. One unresolvable import anywhere in that graph is a crash at
checkpoint time, on a code path with nothing to do with the import.

The repair is to spell those imports absolutely. Both of transformers' patterns
require a leading `.`, so an absolute import is invisible to the walk, and
inside a package the two forms mean the same thing.

The first pass of this fix looked only for DOTTED imports in `unsloth_zoo/*.py`
and found 12. The end-to-end test below -- running the function that actually
crashed -- found 24 more, in subpackages and in the two other shapes. That is
why the regex guard here checks resolvability rather than dottedness, and why
the walk is tested directly rather than modelled.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"

# Transformers' own two patterns, copied rather than approximated, because what
# matters is exactly what IT matches -- not what looks like an import.
RELATIVE_FROM = re.compile(r"^\s*from\s+\.(\S+)\s+import", re.MULTILINE)
RELATIVE_IMPORT = re.compile(r"^\s*import\s+\.(\S+)\s*$", re.MULTILINE)


def _modules():
    return sorted(ROOT.rglob("*.py"))


def _rel(p):
    return str(p.relative_to(ROOT))


@pytest.mark.parametrize("path", _modules(), ids = _rel)
def test_every_relative_import_transformers_sees_resolves_to_a_file(path):
    source = path.read_text(encoding = "utf-8")
    seen = RELATIVE_FROM.findall(source) + RELATIVE_IMPORT.findall(source)
    broken = [imp for imp in seen if not (path.parent / f"{imp}.py").is_file()]
    assert not broken, (
        f"{_rel(path)}: transformers would try to open "
        f"{_rel(path.parent)}/{broken[0]}.py during custom_object_save and "
        f"fail. Spell these absolutely: {broken}"
    )


def test_resolvable_relative_imports_are_left_alone():
    """The guard must not have been read as 'no relative imports'. Sibling
    imports resolve to a real file, cost nothing, and are the normal spelling.
    A rewrite that removed them all would be a far larger change than the bug."""
    sources = "\n".join(p.read_text(encoding = "utf-8") for p in _modules())
    assert len(re.findall(r"^\s*from \.\w+ import", sources, re.MULTILINE)) > 100


def test_the_transformers_patterns_are_the_ones_we_copied():
    """If upstream changes its regexes, this guard is checking the wrong thing
    and should be updated rather than trusted."""
    dynamic = pytest.importorskip("transformers.dynamic_module_utils")
    import inspect

    source = inspect.getsource(dynamic.get_relative_imports)
    assert r"from\s+\.(\S+)\s+import" in source
    assert r"import\s+\.(\S+)\s*$" in source


def test_transformers_can_walk_the_package_without_raising():
    """The end-to-end version: run the function that crashed, on the files it
    crashes from. A regex guard can go green on a rewrite that resolves to a
    different nonexistent path; this cannot. It is also what found the 24 sites
    the first pass missed."""
    dynamic = pytest.importorskip("transformers.dynamic_module_utils")

    for name in ("__init__.py", "device_type.py", "saving_utils.py", "loss_utils.py"):
        files = dynamic.get_relative_import_files(str(ROOT / name))
        missing = [f for f in files if not Path(f).is_file()]
        assert not missing, f"walking {name} produced unopenable paths: {missing}"


def test_the_import_it_died_on_is_actually_gone():
    """The one the Paddle OCR run died on, named rather than implied."""
    for name in ("__init__.py", "device_type.py"):
        source = (ROOT / name).read_text(encoding = "utf-8")
        assert "from .mlx.runtime import" not in source
        assert "from unsloth_zoo.mlx.runtime import" in source


def test_the_module_it_could_not_find_exists_and_is_nested():
    """Confirms the diagnosis: nothing was missing from the wheel."""
    assert (ROOT / "mlx" / "runtime.py").is_file()
    assert not (ROOT / "mlx.runtime.py").exists()
