# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A relative import transformers cannot resolve breaks every remote-code save.

`Paddle_OCR_(1B)_Vision` died at its first checkpoint save with
`FileNotFoundError: .../unsloth_zoo/mlx.runtime.py`. Nothing was missing from
the wheel, and `from .mlx.runtime import x` is valid Python: the dotted path is
built by TRANSFORMERS, in `dynamic_module_utils`, which finds imports with a
regex and joins the raw capture onto the directory:

    re.findall(r"^\\s*from\\s+\\.(\\S+)\\s+import", content, ...)  # plus `import .xxx`
    f"{str(module_path / m)}.py"

`\\S+` swallows the whole dotted name, and nothing converts dots to separators,
strips further dots, or considers that the target may be a package. So:

    from .mlx.runtime import x   ->  unsloth_zoo/mlx.runtime.py
    from .temporary_patches ...  ->  unsloth_zoo/temporary_patches.py  (a dir)
    from .. import __version__   ->  unsloth_zoo/mlx/..py
    from .sibling import x       ->  a real file, so it is fine

(`from . import x` needs no fix either -- the regex wants a non-space after the
dot.) Saving a remote-code model runs `custom_object_save`, which walks this
graph recursively, so one unresolvable import anywhere crashes the save.

The repair is to spell those imports absolutely: both patterns require a
leading `.`, so an absolute import is invisible to the walk, while inside a
package the two forms mean the same thing. Simplifying them back to relative
form reintroduces the crash.

A first pass looked only for DOTTED imports in `unsloth_zoo/*.py` and found 12;
running the function that actually crashed found 24 more, in subpackages and in
the other two shapes. Hence a guard on resolvability rather than dottedness,
and a test of the real walk rather than a model of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"

# Transformers' own patterns, copied not approximated: what matters is exactly
# what IT matches, not what looks like an import.
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
    """The guard is not 'no relative imports': sibling imports resolve to a real
    file and are the normal spelling, so removing them all would be a far larger
    change than the bug."""
    sources = "\n".join(p.read_text(encoding = "utf-8") for p in _modules())
    assert len(re.findall(r"^\s*from \.\w+ import", sources, re.MULTILINE)) > 100


def test_the_transformers_patterns_are_the_ones_we_copied():
    """If upstream changes its regexes, this guard checks the wrong thing."""
    dynamic = pytest.importorskip("transformers.dynamic_module_utils")
    import inspect

    source = inspect.getsource(dynamic.get_relative_imports)
    assert r"from\s+\.(\S+)\s+import" in source
    assert r"import\s+\.(\S+)\s*$" in source


def test_transformers_can_walk_the_package_without_raising():
    """Runs the function that crashed. A regex guard can go green on a rewrite
    that resolves to a different nonexistent path; this cannot."""
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
