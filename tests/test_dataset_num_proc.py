# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for unsloth_zoo.dataset_num_proc.

The module is stdlib-only by design, so it is loaded straight off disk rather
than through ``import unsloth_zoo``: the worker-count assertions then stay
meaningful on a host whose torch cannot import, which is the same host most
likely to be tokenizing on CPU. A canary below pins the stdlib-only property,
and another checks the module is reachable the normal way too.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

try:
    # Before the dnp fixture spoofs sys.platform: multiprocess picks its concrete
    # contexts from sys.platform at import time, so a first import under a spoofed
    # one hands a Windows runner the POSIX fork contexts.
    import multiprocess  # noqa: F401
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "unsloth_zoo" / "dataset_num_proc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_zoo_dataset_num_proc_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dnp(monkeypatch):
    module = _load_module()
    module.reset_warning_state()
    monkeypatch.delenv(module.NUM_PROC_ENV_VAR, raising = False)
    # Pin the platform: macOS is refused by policy whatever the start method says,
    # so every assertion expecting a worker count is really about a forking
    # platform. Platform tests set their own value afterwards, which wins.
    monkeypatch.setattr(module.sys, "platform", "linux")
    return module


def _force_start_method(monkeypatch, dnp, method):
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: method)


# ---------- start-method veto ----------


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
def test_non_fork_start_method_disables_multiprocessing(monkeypatch, dnp, method):
    # Under spawn/forkserver the child must re-import the dynamically generated
    # trainer module, which has no importable name, so workers cannot run.
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(None) is None


def test_non_fork_start_method_warns_once(monkeypatch, dnp, capsys):
    # Regression for eeffa4c065: an explicit value used to sail through the
    # guard. It must be vetoed, and the veto must be visible.
    _force_start_method(monkeypatch, dnp, "spawn")
    dnp.get_dataset_num_proc(8)
    dnp.get_dataset_num_proc(8)
    out = capsys.readouterr().out
    assert out.count("uses the 'spawn' start method") == 1
    assert "dataset_num_proc = 8" in out


def test_fork_start_method_honours_explicit_value(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(6) == 6


# ---------- the 1 -> None normalisation ----------


@pytest.mark.parametrize("value", [1, 0, -4])
def test_non_positive_and_one_normalise_to_none(monkeypatch, dnp, value):
    # `1` is a trap: callers mean "serial", datasets >= 4.0 gives a Pool(1).
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(value) is None


def test_serial_as_none_false_preserves_an_explicit_one(monkeypatch, dnp):
    """The config layer must not collapse 1 to None.

    unsloth_zoo.sft_prepare_dataset reads a config ``None`` as "auto-size me",
    so writing None back for a user who asked for 1 would inflate it.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(1, serial_as_none = False) == 1
    # 0 and negatives are incoherent requests but still mean "not parallel", so
    # they land on the config serial sentinel (1), not on None.
    assert dnp.get_dataset_num_proc(0, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(-4, serial_as_none = False) == 1


def test_config_layer_never_returns_none_while_forking_is_available(monkeypatch, dnp):
    """On a fork host no path may write None back to a config.

    None means "auto-size me" downstream, so any route to it -- memory clamp,
    explicit serial -- would re-inflate.
    """
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)

    # memory clamp all the way down to serial
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 1 * 1024**3})()
    )
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) == 1


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
@pytest.mark.parametrize("desired", [None, 1, 16])
def test_config_layer_is_none_not_one_on_a_non_fork_start_method(monkeypatch, dnp, method, desired):
    """Regression: the config sentinel 1 reached unpatched TRL map() call sites.

    Only SFT gets its map site rewritten (rl_replacements.py); DPO, KTO, CPO,
    ORPO, Reward and PRM pass ``args.dataset_num_proc`` straight into
    ``Dataset.map``, where a ``1`` builds a ``Pool(1)`` whose spawned child
    re-imports the user's ``__main__`` (#3211 / #3397). Those configs carried
    ``None`` on spawn hosts before this module; they must still.

    None is safe here precisely because forking is unavailable: every auto-sizer
    reading the config vetoes on a non-fork start method too.
    """
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(desired, serial_as_none = False) is None


def test_config_layer_env_forced_serial_is_none_on_a_non_fork_start_method(monkeypatch, dnp):
    """UNSLOTH_DATASET_NUM_PROC=0 must not build a Pool(1) either."""
    _force_start_method(monkeypatch, dnp, "spawn")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) is None


def test_layering_config_then_map_site_is_correct(monkeypatch, dnp):
    """Composing the two layers must land on the right value for each intent."""
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 256 * 1024**3})(),
    )
    cfg = lambda v: dnp.get_dataset_num_proc(v, serial_as_none = False)  # noqa: E731
    site = dnp.get_dataset_num_proc

    # user asked for serial -> stays serial, never auto-inflated
    assert site(cfg(1)) is None
    # user asked for a specific count -> honoured end to end
    assert site(cfg(6)) == 6
    # user asked for nothing -> capped auto, and re-applying is idempotent
    assert cfg(None) == dnp.AUTO_NUM_PROC_CAP
    assert site(cfg(None)) == dnp.AUTO_NUM_PROC_CAP


def test_low_memory_auto_path_returns_none_not_one(monkeypatch, dnp):
    # The old heuristic returned 1 here, which still forked a Pool(1).
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 1 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(None) is None


# ---------- auto sizing ----------


def test_auto_value_is_capped(monkeypatch, dnp):
    # Was min(max(cpu_count + 4, 2), 64) -- up to 64 forked workers.
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 128)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 512 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(None) == dnp.AUTO_NUM_PROC_CAP
    assert dnp.AUTO_NUM_PROC_CAP < 64


def test_auto_value_clamped_by_available_memory(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 10 * 1024**3})(),
    )
    # 10 GB free, half of it budgeted, ~1 GB per worker -> 5.
    assert dnp.get_dataset_num_proc(None) == 5


def test_explicit_value_is_clamped_by_memory(monkeypatch, dnp, capsys):
    """The gap that caused issue #2693.

    Studio passes an explicit ``max(1, cpu_count // 4)``, dozens of workers at
    ~680 MB each on a big-core machine, and the old heuristic capped only the
    auto path -- so an explicit request sailed through however little RAM there
    was.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 16 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(48) == 8
    assert "reducing dataset_num_proc 48 -> 8" in capsys.readouterr().out


def test_explicit_value_is_not_capped_by_the_auto_cap(monkeypatch, dnp):
    # AUTO_NUM_PROC_CAP bounds auto-sizing only.
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 512 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(32) == 32
    assert 32 > dnp.AUTO_NUM_PROC_CAP


def test_memory_clamp_is_skipped_without_psutil(monkeypatch, dnp):
    # No psutil means no memory reading, so honour the request.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: None)
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(32) == 32


def test_bool_is_not_treated_as_an_int(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 8)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 64 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(True) == 4


# ---------- environment escape hatch ----------


def test_env_override_beats_start_method_veto(monkeypatch, dnp):
    # A user who knows their workload is fork-safe is never downgraded.
    _force_start_method(monkeypatch, dnp, "spawn")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "24")
    assert dnp.get_dataset_num_proc(None) == 24


@pytest.mark.parametrize("raw", ["0", "none", "None", "false", ""])
def test_env_override_can_force_in_process(monkeypatch, dnp, raw):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16) is None


@pytest.mark.parametrize("raw", ["0", "none", "None", "false", "", "1"])
def test_env_override_in_process_is_encoded_for_the_config_layer(monkeypatch, dnp, raw):
    # Regression: the env override used to return before _serial(), writing None
    # into the *config*, which unsloth_zoo.sft_prepare_dataset reads as
    # "auto-size me" -- so the hatch the dead-worker message recommends raised
    # the worker count instead of removing it. Config serial is 1, never None.
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1


def test_env_override_is_uncapped(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "100")
    assert dnp.get_dataset_num_proc(None) == 100


def test_invalid_env_override_is_ignored_with_a_warning(monkeypatch, dnp, capsys):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "banana")
    assert dnp.get_dataset_num_proc(4) == 4
    assert "is not an integer" in capsys.readouterr().out


# ---------- start-method probing must not mutate global state ----------


def test_start_method_probe_prefers_multiprocess_and_has_no_side_effects(dnp):
    """datasets does `from multiprocess import Pool`, so `multiprocess` -- not
    stdlib multiprocessing -- decides how map() spawns. Reading it must also not
    pin the context, which would make a later set_start_method() raise."""
    multiprocess = pytest.importorskip("multiprocess")
    import multiprocessing

    before_mp = multiprocess.get_start_method(allow_none = True)
    before_std = multiprocessing.get_start_method(allow_none = True)

    method = dnp.multiprocessing_start_method()

    # Must name a method this host offers. The private default-context chain has
    # answered "fork" on Windows, which offers only spawn; believing it would
    # read Windows as forkable and let workers through.
    assert method in multiprocess.get_all_start_methods()
    assert multiprocess.get_start_method(allow_none = True) == before_mp
    assert multiprocessing.get_start_method(allow_none = True) == before_std


def test_start_method_probe_reports_an_explicit_setting(monkeypatch, dnp):
    import sys as _sys
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: "forkserver"
    fake.get_all_start_methods = lambda: ["fork", "spawn", "forkserver"]
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "forkserver"


def _fake_multiprocess(listed, default_name):
    """A multiprocess stand-in with nothing pinned yet."""
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: None
    fake.get_all_start_methods = lambda: list(listed)
    if default_name is not None:
        context = types.ModuleType("multiprocess.context")
        context._default_context = types.SimpleNamespace(
            _default_context = types.SimpleNamespace(_name = default_name),
            _actual_context = None,
        )
        fake.context = context
    return fake


def test_start_method_probe_prefers_the_real_default_over_list_order(monkeypatch, dnp):
    """macOS: multiprocess lists 'spawn' first but its default context is fork.

    It copies stdlib ``get_all_start_methods()`` verbatim, darwin branch
    included, while keeping ``fork`` as its ``_default_context`` on every POSIX
    platform (``#FIXME: spawn`` in multiprocess/context.py). Since ``datasets``
    pools come from ``multiprocess``, trusting the list would say 'spawn' while
    ``Dataset.map`` forks -- vetoing every worker and misreporting the start
    method in the dead-worker diagnostics.
    """
    import sys as _sys

    darwin_order = ["spawn", "fork", "forkserver"]
    fake = _fake_multiprocess(darwin_order, "fork")
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "fork"


def test_macos_stays_in_process_even_though_multiprocess_forks(monkeypatch, dnp, capsys):
    """The probe reports fork on macOS; policy still refuses to use it.

    Two separate things on purpose. ``multiprocess`` really does fork on darwin,
    so the diagnostics must say so, but CPython moved the macOS default to spawn
    in 3.8 (bpo-33725) because forking there "can lead to crashes of the
    subprocess as macOS system libraries may start threads" -- and this parent
    holds Torch and a threaded BLAS. Fixing the probe without this guard would
    take macOS from always-serial to AUTO_NUM_PROC_CAP forked workers.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    # Pin memory so the contrast is about policy, not the runner's free RAM.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "darwin")

    # Serial at a map() call site, None -- not 1 -- at the config layer, so no
    # Pool is built on datasets >= 4.1 either way.
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(8, serial_as_none = False) is None
    assert dnp.get_dataset_num_proc(None) is None
    assert "macOS" in capsys.readouterr().out

    # The escape hatch still overrides it, and Linux is unaffected.
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "4")
    assert dnp.get_dataset_num_proc(8) == 4
    monkeypatch.delenv(dnp.NUM_PROC_ENV_VAR)
    monkeypatch.setattr(dnp.sys, "platform", "linux")
    assert dnp.get_dataset_num_proc(8) == 8


def test_start_method_probe_falls_back_to_list_order(monkeypatch, dnp):
    """The default context is private, so an unreadable one must not raise."""
    import sys as _sys

    fake = _fake_multiprocess(["spawn", "fork"], None)
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "spawn"


def test_start_method_probe_matches_the_pool_multiprocess_would_build(dnp):
    """The probe must agree with multiprocess's own default on this host."""
    multiprocess = pytest.importorskip("multiprocess")
    if multiprocess.get_start_method(allow_none = True) is not None:
        pytest.skip("a start method is already pinned in this process")
    assert (
        dnp.multiprocessing_start_method()
        == multiprocess.context._default_context._default_context._name
    )


# ---------- worker-death diagnostics ----------


_DATASETS_MESSAGE = (
    "One of the subprocesses has abruptly died during map operation."
    "To debug the error, disable multiprocessing."
)


def test_worker_death_is_reraised_with_context(dnp):
    # datasets discards the child's exit status, so the original message cannot
    # distinguish an OOM kill from anything else.
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError(_DATASETS_MESSAGE)

    message = str(caught.value)
    assert "dataset_num_proc = 8" in message
    assert "8 workers" in message
    assert "8GB" in message, "should estimate what those workers cost"
    assert dnp.NUM_PROC_ENV_VAR in message, "must name the escape hatch"
    assert "out-of-memory" in message
    # The child's traceback must survive for anyone who wants it.
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _DATASETS_MESSAGE in str(caught.value.__cause__)


def test_worker_death_diagnostics_handles_in_process_runs(dnp):
    # num_proc=None still reaches the wrapper; it must not divide by a None.
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(None):
            raise RuntimeError(_DATASETS_MESSAGE)
    assert "dataset_num_proc = None" in str(caught.value)
    assert "1 worker," in str(caught.value)


def test_unrelated_errors_pass_through_untouched(dnp):
    # Only the dead-worker message is rewritten, and non-RuntimeError types are
    # not caught at all.
    original = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(4):
            raise original
    assert caught.value is original

    key = KeyError("text")
    with pytest.raises(KeyError) as caught_key:
        with dnp.map_failure_diagnostics(4):
            raise key
    assert caught_key.value is key


def test_successful_map_is_not_disturbed(dnp):
    with dnp.map_failure_diagnostics(4):
        result = "tokenized"
    assert result == "tokenized"


def test_the_recovery_advice_does_not_promise_more_than_it_delivers(dnp):
    """UNSLOTH_DATASET_NUM_PROC=0 is not in-process on every path.

    The dead-worker message is the one place a user is told what to do next, so
    it has to be true. On fork, train_on_responses_only over the Zoo's threshold
    gets ``1`` rather than ``None`` -- deliberately, since a bare None reads as
    "size it for me" and would inflate to the Zoo's uncapped count -- which
    ``datasets`` >= 4.1 turns into a Pool(1). Saying "tokenize in-process"
    flatly was wrong for exactly the large-dataset runs that die.
    """
    with pytest.raises(RuntimeError) as excinfo:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError("One of the subprocesses has abruptly died during map operation.")
    message = str(excinfo.value)
    assert f"{dnp.NUM_PROC_ENV_VAR}=0" in message
    assert "one worker" in message, "the exception to in-process has to be stated"
    assert "train_on_responses_only" in message, "and which path it applies to"
    assert f"{dnp.ZOO_MIN_ROWS_FOR_MULTIPROC:,}" in message, "and above which size"


def test_the_advice_matches_what_the_resolver_actually_returns(dnp, monkeypatch):
    """Executed, not just read: drive both branches of the claim above.

    A message asserting a behaviour the resolver does not have would be worse
    than the vague one it replaced.
    """

    class _Split:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    class _Trainer:
        def __init__(self, split):
            self.train_dataset = split
            self.eval_dataset = None

    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "linux")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")

    over = dnp.resolve_responses_only_num_proc(
        _Trainer(_Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC + 1)), None
    )
    under = dnp.resolve_responses_only_num_proc(
        _Trainer(_Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC - 1)), None
    )
    assert over == 1, "over the threshold the best expressible request is one worker"
    assert under is None, "under it the Zoo's own guard already goes in-process"


def test_probe_rejects_a_start_method_the_host_does_not_offer(monkeypatch, dnp):
    """The private default-context chain is not trustworthy on its own.

    On a Windows runner it answered "fork" while get_all_start_methods() was
    ["spawn"]. Believing it read Windows as forkable, so
    _workers_unusable_reason() returned None and workers were let through --
    the spawn re-import loop of #3211 / #3397.
    """
    import sys as _sys
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: None
    fake.get_all_start_methods = lambda: ["spawn"]
    context = types.ModuleType("multiprocess.context")
    context._default_context = types.SimpleNamespace(
        _default_context = types.SimpleNamespace(_name = "fork"),
    )
    fake.context = context
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)

    assert dnp.multiprocessing_start_method() == "spawn"

    monkeypatch.setattr(dnp.sys, "platform", "win32")
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(8, serial_as_none = False) is None


# ---------- the module's own shape ----------


def test_the_module_imports_nothing_but_the_standard_library():
    """Its callers include generated trainer source and a torch-less installer.

    psutil and multiprocess are read inside functions, behind try/except, so a
    host without them degrades to a conservative answer instead of failing to
    import. A top-level dependency here would take that away, and would also
    make importing the policy cost whatever that dependency costs.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding = "utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top_level.append((node.module or "").split(".")[0])

    assert set(top_level) <= {"__future__", "contextlib", "os", "sys", "typing"}, (
        f"dataset_num_proc grew a non-stdlib top-level import: {sorted(set(top_level))}"
    )


def test_it_is_reachable_as_unsloth_zoo_dataset_num_proc():
    """The dotted path the generated trainer source imports.

    unsloth/models/rl.py bakes `from unsloth_zoo.dataset_num_proc import ...`
    into every trainer it generates, so a rename here breaks code that is
    already written to disk in users' unsloth_compiled_cache.
    """
    pytest.importorskip("torch")
    from unsloth_zoo.dataset_num_proc import get_dataset_num_proc

    assert callable(get_dataset_num_proc)
