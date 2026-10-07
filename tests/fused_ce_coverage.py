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

"""Which model classes get the fused lm_head + CE loss on the installed transformers, and by which route.

Run as a script (a fresh process, so the import hook sees every modeling import):
    python tests/fused_ce_coverage.py              # print the census as JSON
    python tests/fused_ce_coverage.py --update     # record it in tests/fused_ce_coverage.json

Routes mirror production: "hook" (forward rewritten by the import hook), "regex" / "ast"
(compiler.fused_lm_head_forward would fuse it), None (trains with full logits).
"""
import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
import warnings

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_ce_coverage.json")


def _modeling_modules():
    import transformers.models as models
    names = []
    for info in pkgutil.walk_packages(models.__path__, models.__name__ + "."):
        leaf = info.name.rsplit(".", 1)[-1]
        if leaf.startswith("modeling_") and not leaf.startswith(("modeling_tf_", "modeling_flax_")):
            names.append(info.name)
    return sorted(names)


def _apply_temporary_patches():
    # What `import unsloth` runs at init: wrappers installed here must stay transparent to the compiler.
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    for patch in list(TEMPORARY_PATCHES):
        try:
            accepts_phase = "phase" in inspect.signature(patch).parameters
        except (TypeError, ValueError):
            accepts_phase = False
        try:
            patch(phase = "init") if accepts_phase else patch()
        except Exception:
            pass


def _computes_lm_loss(source):
    return "labels" in source and any(
        k in source for k in ("loss_function(", "CrossEntropyLoss", "cross_entropy(")
    )


def census():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import conftest  # CPU CI: the same device-type preload the test session uses
    if not conftest._has_real_accelerator():
        conftest._preload_real_device_type()
    import importlib.util
    if importlib.util.find_spec("unsloth") is None:
        # CI lanes install unsloth best-effort; unsloth_zoo only checks that the package can be found.
        import importlib.machinery, types
        stub = types.ModuleType("unsloth")
        stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", None)
        sys.modules["unsloth"] = stub
    import unsloth_zoo  # noqa: F401  installs the fused-forward import hook
    import transformers
    from transformers.generation import GenerationMixin
    from unsloth_zoo.compiler import fused_lm_head_forward

    modules = {}
    for name in _modeling_modules():
        try:
            modules[name] = importlib.import_module(name)
        except Exception:
            continue  # optional dependency missing; not a fusion question
    _apply_temporary_patches()

    routes, unfused_targets, candidates = {}, [], []
    for name, module in modules.items():
        for attr, cls in vars(module).items():
            # The compiler fuses GenerationMixin classes; the hook also takes every *ForCausalLM.
            if not (isinstance(cls, type) and cls.__module__ == name and "forward" in dir(cls)
                    and (issubclass(cls, GenerationMixin) or attr.endswith("ForCausalLM"))):
                continue
            try:
                # The compiler reads the same source, following __wrapped__.
                source = inspect.getsource(cls.forward)
            except (OSError, TypeError):
                continue
            candidates.append(attr)
            if "unsloth_fused_lm_head_loss" in source:
                route = "hook"
            elif ".deprecated." in name:
                route = None  # the compiler only loads transformers.models.<type>.modeling_<type>
            else:
                try:
                    route = fused_lm_head_forward(attr, cls, name, source)[1]
                except Exception:
                    route = None
                # Count aware but still full logits: not a fused lm_head + CE.
                if route == "count":
                    route = None
            if route is not None:
                routes[attr] = route
            elif _computes_lm_loss(source):
                unfused_targets.append(attr)
    return dict(transformers = transformers.__version__, fused = dict(sorted(routes.items())),
                unfused_targets = sorted(unfused_targets), candidates = sorted(candidates))


def load_baseline(path = BASELINE):
    """{version: {"fused": set, "unfused_targets": set}} from the range-compressed json."""
    data = json.load(open(path))
    versions = data["versions"]
    index = {v: i for i, v in enumerate(versions)}
    per_version = {v: {"fused": set(), "unfused_targets": set()} for v in versions}
    for key in ("fused", "unfused_targets"):
        for cls, spans in data[key].items():
            for span in spans:
                lo, _, hi = span.partition("..")
                for v in versions[index[lo]: index[hi or lo] + 1]:
                    per_version[v][key].add(cls)
    return per_version


def save_baseline(per_version, path = BASELINE):
    from packaging.version import Version
    versions = sorted(per_version, key = Version)
    data = {"versions": versions, "fused": {}, "unfused_targets": {}}
    for key in ("fused", "unfused_targets"):
        for cls in sorted({c for v in versions for c in per_version[v][key]}):
            spans, start = [], None
            for i, v in enumerate(versions + [None]):
                inside = v is not None and cls in per_version[v][key]
                if inside and start is None:
                    start = i
                elif not inside and start is not None:
                    lo, hi = versions[start], versions[i - 1]
                    spans.append(lo if lo == hi else f"{lo}..{hi}")
                    start = None
            data[key][cls] = spans
    with open(path, "w") as f:
        json.dump(data, f, indent = 1)
        f.write("\n")


def _record(per_version, result):
    per_version[result["transformers"]] = {"fused": set(result["fused"]),
                                           "unfused_targets": set(result["unfused_targets"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action = "store_true", help = "record this transformers in the baseline")
    parser.add_argument("--from-results", nargs = "*", default = None,
                        help = "rebuild the baseline from saved census outputs instead of running one")
    args = parser.parse_args()
    if args.from_results is not None:
        per_version = {}
        for path in args.from_results:
            _record(per_version, json.loads(open(path).read().strip().splitlines()[-1]))
        save_baseline(per_version)
        sys.exit(0)
    result = census()
    if args.update:
        per_version = load_baseline() if os.path.exists(BASELINE) else {}
        _record(per_version, result)
        save_baseline(per_version)
    print(json.dumps(result))
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown of ~400 imported modeling modules
