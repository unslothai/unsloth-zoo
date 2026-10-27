# SPDX-License-Identifier: AGPL-3.0-only

"""One Mega-cache writer process for test_compile_cache_concurrency.py.

argv: compile_cache.py path, cache root, comma-separated artifact keys, pause
point ("" | "after_read" | "before_manifest"), sync dir. Fakes torch's artifact
manager (bundle = sorted JSON list of [type, key]) so no torch is imported.
When paused it writes ``<sync>/paused`` and waits for ``<sync>/go``; every
writer writes ``<sync>/entered.<pid>`` as it enters megacache_save."""

import importlib.util
import json
import os
import sys
import time
import types
from collections import defaultdict

module_path, root, artifacts, pause, sync = sys.argv[1:6]


class CacheArtifactManager:
    _new_cache_artifacts = defaultdict(list)
    _seen_artifacts = set()

    @staticmethod
    def deserialize(data):
        result = defaultdict(list)
        for artifact_type, key in json.loads(data):
            result[artifact_type].append((artifact_type, key))
        return dict(result)


def save_cache_artifacts():
    seen = CacheArtifactManager._seen_artifacts
    return (json.dumps(sorted(seen)).encode(), {}) if seen else None


for key in artifacts.split(","):
    CacheArtifactManager._seen_artifacts.add(("inductor", key))

cache_module = types.ModuleType("torch.compiler._cache")
cache_module.CacheArtifactManager = CacheArtifactManager
compiler = types.ModuleType("torch.compiler")
compiler._cache = cache_module
compiler.save_cache_artifacts = save_cache_artifacts
compiler.load_cache_artifacts = lambda data: {}
torch = types.ModuleType("torch")
torch.compiler = compiler
sys.modules.update({"torch": torch, "torch.compiler": compiler, "torch.compiler._cache": cache_module})

spec = importlib.util.spec_from_file_location("zoo_compile_cache_writer", module_path)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
# Ownership/mode checks are covered by test_compile_cache_security.py.
cc._is_trusted_directory = lambda path, allow_missing = True: os.path.isdir(path) or allow_missing
os.environ["UNSLOTH_MEGA_CACHE"] = "1"
os.environ["UNSLOTH_MEGA_CACHE_DIR"] = root
cc._STATE["armed"] = True
cc._STATE["loaded_key"] = "deadbeef"


def _wait():
    open(os.path.join(sync, "paused"), "w").close()
    deadline = time.time() + 60
    while not os.path.exists(os.path.join(sync, "go")) and time.time() < deadline:
        time.sleep(0.02)


if pause == "after_read":
    real_read = cc._read_disk_bundle

    def _read_then_pause(*args):
        result = real_read(*args)
        _wait()
        return result

    cc._read_disk_bundle = _read_then_pause
elif pause == "before_manifest":
    real_dump = cc.json.dump

    def _pause_then_dump(*args, **kwargs):
        _wait()
        return real_dump(*args, **kwargs)

    cc.json = types.SimpleNamespace(**{**vars(json), "dump": _pause_then_dump})

open(os.path.join(sync, f"entered.{os.getpid()}"), "w").close()
print(cc.megacache_save())
