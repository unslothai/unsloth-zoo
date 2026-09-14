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

# Auto-install missing notebook-only deps: requires_backends, check_imports and the bare
# ModuleNotFoundError from the IPython chain each get an allow-listed pip retry.

import ast
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import site
import subprocess
import tempfile
import threading
import time
import sys

# Absolute on purpose: `from ..log` makes transformers' custom_object_save write
# temporary_patches/.log.py and crash. See PR #1045.
from unsloth_zoo.log import logger
from .common import TEMPORARY_PATCHES

# pypi-name -> import-name (None means same).
_ALLOW_LIST = {
    "timm":          None,
    "addict":        None,
    "einops":        None,
    "easydict":      None,
    "snac":          None,
    "torchcodec":    None,
    "matplotlib":    None,
    "traitlets":     None,
    "soundfile":     None,
    "librosa":       None,
    "scipy":         None,
    "pyctcdecode":   None,
    "tiktoken":      None,
    "blobfile":      None,
    "pillow_heif":   "pillow_heif",
    "decord":        None,
    "av":            "av",
    "num2words":     None,
    "jieba":         None,
    "sentencepiece": None,
}

# Mirrors huggingface_hub's ENV_VARS_TRUE_VALUES: accepting only "1" would run pip against `=true`.
_TRUE_VALUES = frozenset({"1", "ON", "TRUE", "YES"})


def _env_is_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().upper() in _TRUE_VALUES


def _in_interactive_session() -> bool:
    """Whether a person is driving this interpreter through an IPython kernel.

    IPython being importable proves nothing: it is a transitive dependency of plenty of
    entirely non-interactive installs. Only a live shell object counts, and only one
    deriving from the two classes that mean Jupyter/Colab/Kaggle or a REPL."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        shell = ipython.get_ipython()
        # The whole MRO, not the exact class name. Colab's get_ipython() returns
        # google.colab._shell.Shell, declared as `class Shell(zmqshell.ZMQInteractiveShell)`,
        # so an exact-name test reports the single most important notebook platform as
        # non-interactive. Kaggle subclasses too.
        return any(
            klass.__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
            for klass in type(shell).__mro__
        )
    except Exception:
        return False


def _auto_install_enabled() -> bool:
    # Read at the attempt, not at import: the user sets this after seeing the _run_install
    # warning, and a notebook can enter an interactive shell after unsloth was imported.
    value = os.environ.get("UNSLOTH_AUTO_INSTALL", "").strip()
    if value:
        return value.upper() in _TRUE_VALUES
    # Unset means interactive-only. Running pip as a side effect of `import unsloth` is
    # defensible when a person is watching a notebook cell and can read the warning; it is
    # not defensible in CI, an inference server, or a scheduled script, where the process
    # would silently gain a package nobody asked for. Those all fail the check above.
    return _in_interactive_session()


def _no_network() -> bool:
    # Read at the attempt (a notebook can go offline later). The cross-sync in
    # unsloth_zoo/__init__.py establishes that any one of the three Hugging Face flags
    # implies all three, but it runs once at import, so a flag set afterwards is only
    # seen if it is read here. All four are therefore checked at the attempt.
    return (
        _env_is_true("UNSLOTH_OFFLINE")
        or _env_is_true("HF_HUB_OFFLINE")
        or _env_is_true("TRANSFORMERS_OFFLINE")
        or _env_is_true("HF_DATASETS_OFFLINE")
    )
# pkg -> Event, set once that package's one attempt has finished. An Event rather than
# a set so a second thread can WAIT for the first instead of reporting failure while the
# install is still running, which would re-raise the original ImportError for a package
# that is about to exist.
_attempted: dict = {}
# Guards the check-then-claim on _attempted. Two threads hitting the same missing backend
# could otherwise both pass the membership test and launch two pip processes against one
# prefix, which is a known way to leave a half-unpacked dist-info behind.
_attempt_lock = threading.Lock()
# Slightly over the 300s subprocess timeout plus the uv-then-pip fallback, so a waiter
# outlives the worst-case attempt instead of giving up early on a healthy install. It
# bounds IDLE time, not total time: see _note_install_activity.
_ATTEMPT_WAIT_SECONDS = 620
# monotonic() of the last time any attempt claimed the package or started an installer.
# The waiter's budget is measured against this rather than against its own start, because
# _install_lock serialises across packages: an owner queued behind two 300s installs would
# otherwise have only ~20s of the waiter's 620s left for its own command, and the waiter
# would give up and re-raise ImportError for a package about to be installed. Bounding
# idle time instead means the waiter only gives up when nothing is happening at all.
_last_install_activity = 0.0


def _note_install_activity():
    global _last_install_activity
    with _attempt_lock:
        _last_install_activity = time.monotonic()


def _install_is_idle() -> bool:
    with _attempt_lock:
        return (time.monotonic() - _last_install_activity) > _ATTEMPT_WAIT_SECONDS


# Serialises the installer subprocesses themselves, across package names. Distinct from
# _attempt_lock, which only guards the tiny check-then-claim on _attempted and is never
# held across a subprocess.
_install_lock = threading.Lock()

# Backends we installed mid-process. Once one is installed `requires_backends` starts
# SUCCEEDING, so every later call short-circuits past the repair path and its checks.
# These two sets are what survives that, and they are deliberately never cleared: only a
# restart can actually fix either condition.
#
# _installed_backends distinguishes "this object was already a placeholder before we did
# anything", which is upstream's business, from "our install is why the check now passes
# while the object is still frozen", which is ours to report.
_installed_backends: set = set()
# Backends whose guarded imports could not be replayed into already-imported modules.
# Those modules still have the guarded names unbound, so letting a later call through
# hands the consumer the bare NameError this whole path exists to prevent.
_replay_failed: set = set()
# Backends already replayed successfully, and the lock that makes the refresh-and-replay
# phase one-at-a-time. The per-package install event only serialises the pip subprocess:
# once it is set, every waiter returns True together and would replay concurrently.
# importlib.reload of a module another thread is already reloading returns immediately
# with the guarded name still unbound, so the loser would record a permanent
# _replay_failed for a backend that was in fact repaired, and demand a restart for nothing.
_replay_done: set = set()
_replay_lock = threading.Lock()

# Distributions whose REPLACEMENT would corrupt the running process: compiled
# extensions already loaded by torch, and torch itself. Nothing in _ALLOW_LIST is
# one of these, but pip resolves the whole transitive closure, so an allowed
# package can still drag one in. `timm` is the worked example: it requires
# torchvision, and torchvision pins an exact torch, so an unconstrained
# `pip install timm` can uninstall a CUDA torch mid-session.
_PINNED_DISTRIBUTIONS = (
    "torch", "torchvision", "torchaudio", "triton",
    "numpy", "pillow", "transformers",
    "bitsandbytes", "xformers", "flash-attn", "vllm",
)
# dist-name -> import-name, for the ones that differ. Used to tell "not installed, so
# nothing to protect" apart from "installed but unpinnable", which are opposite answers.
_PINNED_IMPORT_NAMES = {
    "pillow": "PIL", "flash-attn": "flash_attn",
}
_constraints_path = None
_constraints_lock = threading.Lock()


def _invalidate_constraints():
    """Drop the cached snapshot so the next install re-reads the environment.

    One install can introduce a critical distribution that did not exist when the file
    was written: `timm` pulls in `torchvision`. Reusing the old snapshot would leave that
    new torchvision unpinned, so the next allow-listed install could replace it or resolve
    an incompatible build against the running torch, which is the whole failure this file
    exists to prevent."""
    global _constraints_path
    with _constraints_lock:
        stale, _constraints_path = _constraints_path, None
    if stale:
        try:
            os.remove(stale)
        except OSError:
            pass


def _is_unsloth_stub(obj) -> bool:
    """Whether this module or spec is one of our own MLX shims rather than a real install.

    unsloth_zoo/__init__.py injects a synthetic `triton` (and often `bitsandbytes`) into
    sys.modules on an MLX host, along with a meta_path finder, so both sys.modules and
    find_spec report them present while they have no metadata at all. Counting those as
    unpinnable would refuse every install on Apple Silicon."""
    for candidate in (obj, getattr(obj, "loader", None)):
        if candidate is None:
            continue
        for attribute in ("__name__", "__module__"):
            name = getattr(candidate, attribute, "") or ""
            if name.startswith("unsloth_zoo.stubs"):
                return True
        if type(candidate).__module__.startswith("unsloth_zoo.stubs"):
            return True
    return False


def _unpinnable_critical():
    """Critical distributions that are importable but carry no readable version.

    A vendor build, a system package or a source tree torch has no dist-info, so
    importlib.metadata raises and the pin is silently dropped. Treating that as "not
    installed" is backwards: it is the case where an unconstrained resolver is most
    likely to replace a torch that is already loaded into a live CUDA process."""
    unpinnable = []
    for dist in _PINNED_DISTRIBUTIONS:
        try:
            importlib.metadata.version(dist)
            continue
        except Exception:
            pass
        import_name = _PINNED_IMPORT_NAMES.get(dist, dist.replace("-", "_"))
        module = sys.modules.get(import_name)
        if module is not None:
            if not _is_unsloth_stub(module):
                unpinnable.append(dist)
            continue
        try:
            spec = importlib.util.find_spec(import_name)
        except Exception:
            # A parent package that refuses to import is not evidence either way.
            continue
        if spec is not None and not _is_unsloth_stub(spec):
            unpinnable.append(dist)
    return unpinnable


def _constraints_file():
    """A pip constraints file pinning the critical distributions that are ALREADY
    installed to their exact current versions.

    Constraints only bind packages the resolver actually touches, so this does not
    force anything to be installed. It means an install that would have replaced
    torch now fails and leaves the session intact, and the caller re-raises the
    original ImportError, which is the honest outcome: we could not repair this
    without breaking something else."""
    global _constraints_path
    with _constraints_lock:
        if _constraints_path is not None:
            return _constraints_path
        lines = []
        for dist in _PINNED_DISTRIBUTIONS:
            try:
                lines.append(f"{dist}=={importlib.metadata.version(dist)}")
            except Exception:
                # Not installed, so there is nothing to protect.
                continue
        try:
            handle = tempfile.NamedTemporaryFile(
                mode = "w", suffix = ".txt", prefix = "unsloth-pins-", delete = False,
            )
            with handle:
                handle.write("\n".join(lines) + "\n")
            _constraints_path = handle.name
        except Exception:
            # "" means we could not build the file. _pip_install refuses on it rather than
            # falling back to an unconstrained install, which is the very thing this file
            # exists to prevent.
            _constraints_path = ""
        return _constraints_path


def _is_running_prefix(root: str) -> bool:
    # samefile, not string compare: handles symlinks, and a deleted environment raises.
    try:
        return os.path.samefile(root, sys.prefix)
    except Exception:
        return False


def _in_venv() -> bool:
    # From the RUNNING interpreter: a kernel can run A with VIRTUAL_ENV inherited from B.
    if hasattr(sys, "real_prefix"):
        return True
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    # conda reports `base_prefix == prefix`, so the variable is its only marker.
    return any(
        bool(root) and _is_running_prefix(root)
        for root in (os.environ.get("VIRTUAL_ENV"), os.environ.get("CONDA_PREFIX"))
    )


def _uv_command(pkg: str) -> list:
    # `--python` is required: uv otherwise targets VIRTUAL_ENV/CONDA_PREFIX/a .venv.
    cmd = ["uv", "pip", "install", "--quiet", "--python", sys.executable]
    constraints = _constraints_file()
    if constraints:
        cmd += ["-c", constraints]
    return cmd + [pkg]


def _pip_command(pkg: str) -> list:
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--disable-pip-version-check", "--no-input",
    ]
    constraints = _constraints_file()
    if constraints:
        cmd += ["-c", constraints]
    cmd.append(pkg)
    if not _in_venv() and hasattr(os, "geteuid") and os.geteuid() != 0:
        try:
            sp = site.getsitepackages()[0]
            probe = os.path.join(sp, ".unsloth_write_probe")
            open(probe, "w").close()
            os.remove(probe)
        except Exception:
            cmd.append("--user")
    return cmd


def _add_user_site() -> None:
    """Put a user site directory this install has just CREATED on sys.path.

    site.addusersitepackages adds it only ``if os.path.isdir(user_site)``, evaluated at
    interpreter start, so the FIRST --user install of a session leaves the package
    importable by nothing. addsitedir dedupes, so calling this twice is harmless."""
    try:
        if not site.ENABLE_USER_SITE:
            # -s, PYTHONNOUSERSITE or a venv: pip's --user would have refused too.
            return
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            site.addsitedir(user_site)
    except Exception:
        pass


def _run_install(pkg: str, cmd: list) -> tuple:
    """Run one installer command. Returns ``(succeeded, retry_with_pip)``."""
    logger.warning(
        f"Unsloth: auto-installing missing notebook dep `{pkg}` via "
        f"`{' '.join(cmd)}`. Set UNSLOTH_AUTO_INSTALL=0 to disable."
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        # Distinct from a launch failure: an air-gapped host with no offline flag
        # set burns the full timeout on pip's own connection retries.
        logger.warning(
            f"Unsloth: auto-install of `{pkg}` timed out after 300s. If this machine "
            f"has no network, set UNSLOTH_AUTO_INSTALL=0 or one of UNSLOTH_OFFLINE / "
            f"HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to skip this immediately."
        )
        return False, False
    except Exception as e:
        logger.warning(f"Unsloth: auto-install of `{pkg}` failed to launch: {e}")
        return False, False
    if r.returncode == 0:
        # The environment just changed, so the pinned snapshot is out of date.
        _invalidate_constraints()
        importlib.invalidate_caches()
        # Unconditionally, not just when we passed --user. pip decides on its own:
        # decide_user_install (pip/_internal/commands/install.py) returns True whenever
        # site_packages_writable() is False, logging "Defaulting to user installation
        # because normal site-packages is not writeable". We never pass --user on
        # Windows at all, because the probe above is gated on os.geteuid, which does not
        # exist there. So a non-admin Windows install into Program Files takes pip's
        # silent fallback, lands in %APPDATA%\Python, and stays off sys.path if the
        # directory did not exist at interpreter start, leaving the install "successful"
        # and the import still broken.
        _add_user_site()
        try:
            list(importlib.metadata.distributions())
        except Exception:
            pass
        return True, False
    stderr = r.stderr or ""
    logger.warning(f"Unsloth: auto-install of `{pkg}` failed:\n{stderr[-500:]}")
    # Retry via pip only when uv could not be aimed at this interpreter; a real build
    # failure must not, or every failure costs two installs.
    lowered = stderr.lower()
    retry_with_pip = (
        "unexpected argument" in lowered
        or "unrecognized" in lowered
        or "no virtual environment or system python installation found" in lowered
    )
    return False, retry_with_pip


def _pip_install(pkg: str) -> bool:
    # Re-checked here and not only in _try_install_and_import: this is the last
    # point before a package name reaches a command line, and the name can come
    # from a downloaded trust_remote_code modeling file.
    if pkg not in _ALLOW_LIST:
        return False
    # Fail closed. The constraints file is the only thing standing between an allowed
    # package and a resolver that replaces torch under a live CUDA process, and a
    # distribution we cannot read a version for is one we cannot pin. Refusing leaves the
    # original ImportError in place, which is the honest outcome: we could not repair this
    # without risking something worse.
    if not _constraints_file():
        logger.warning(
            f"Unsloth: not auto-installing `{pkg}`: the constraints file could not be "
            f"written, so an install could not be stopped from replacing torch. "
            f"Install `{pkg}` manually if you need it."
        )
        return False
    unpinnable = _unpinnable_critical()
    if unpinnable:
        logger.warning(
            f"Unsloth: not auto-installing `{pkg}`: {', '.join(unpinnable)} "
            f"{'is' if len(unpinnable) == 1 else 'are'} installed without readable "
            f"version metadata, so it cannot be protected from being replaced. "
            f"Install `{pkg}` manually if you need it."
        )
        return False
    with _attempt_lock:
        finished = _attempted.get(pkg)
        if finished is None:
            finished = _attempted[pkg] = threading.Event()
            ours = True
        else:
            ours = False
    if not ours:
        # Someone else owns this package's single attempt. Wait for it to finish and let
        # the caller re-probe importability, rather than returning a failure for a package
        # that another thread is in the middle of installing successfully.
        while not finished.wait(timeout = 5):
            if _install_is_idle():
                break
        return False
    _note_install_activity()
    try:
        # Per-package events stop a second thread duplicating or mis-reporting THIS
        # package. They do nothing for two threads installing DIFFERENT packages, which
        # is the common shape here: requires_backends can want several backends at once.
        # Two installers against one prefix can both rewrite a shared dependency and its
        # .dist-info, so the executions themselves are serialised. Held only around the
        # subprocesses, and never while waiting on _attempt_lock or on another package's
        # event, so there is no lock cycle.
        with _install_lock:
            _note_install_activity()
            if shutil.which("uv") and _in_venv():
                ok, retry_with_pip = _run_install(pkg, _uv_command(pkg))
                if ok:
                    return True
                if not retry_with_pip:
                    return False
            return _run_install(pkg, _pip_command(pkg))[0]
    finally:
        finished.set()


def _importable(import_name: str) -> bool:
    """Whether ``import_name`` actually imports: find_spec also passes a broken native
    extension, and calling that a success downgrades ImportError to NameError."""
    try:
        importlib.import_module(import_name)
    except Exception:
        return False
    return True


def _try_install_and_import(pkg: str) -> bool:
    if pkg not in _ALLOW_LIST:
        return False
    if not _auto_install_enabled() or _no_network():
        return False
    import_name = _ALLOW_LIST[pkg] or pkg.replace("-", "_")
    if importlib.util.find_spec(import_name) is not None and _importable(import_name):
        return True
    if not _pip_install(pkg):
        # Not necessarily a failure: another thread may have owned the attempt and
        # completed it while we waited, so ask the interpreter rather than assuming.
        return _importable(import_name)
    return _importable(import_name)


def _rebind_requires_backends(wrapper, original) -> None:
    """Point every alias of ``requires_backends`` at the wrapper: modeling files do
    ``from ...utils import requires_backends``, so patching only ``import_utils`` leaves
    the installer unreachable. ``vars()`` not ``getattr``, or lazy shims import submodules."""
    if original is None or wrapper is None:
        return
    for module in list(sys.modules.values()):
        try:
            namespace = vars(module)
            if namespace.get("requires_backends", None) is original:
                namespace["requires_backends"] = wrapper
        except Exception:
            continue


def _refresh_backend_availability(iu, backend) -> None:
    """Make transformers re-evaluate ``backend``: 5.x caches the probe behind ``lru_cache``,
    4.x used a ``_<backend>_available`` flag. Both handled, neither required to be present."""
    flag = f"_{backend.replace('-', '_')}_available"
    if hasattr(iu, flag):
        setattr(iu, flag, True)
    try:
        available = iu.BACKENDS_MAPPING[backend][0]
    except Exception:
        return
    cache_clear = getattr(available, "cache_clear", None)
    if cache_clear is None:
        return
    try:
        cache_clear()
    except Exception:
        pass


def _names_bound_by(statement) -> list:
    if isinstance(statement, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in statement.names]
    return [alias.asname or alias.name for alias in statement.names]


def _perform_import(statement, module) -> None:
    """Carry out one parsed import statement in ``module``'s namespace. importlib, not
    ``exec``, so the replay cannot run anything else in the file."""
    namespace = vars(module)
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            importlib.import_module(alias.name)
            if alias.asname:
                namespace[alias.asname] = sys.modules[alias.name]
            else:
                top = alias.name.split(".")[0]
                namespace[top] = sys.modules[top]
        return
    name = "." * statement.level + (statement.module or "")
    source = importlib.import_module(
        name, package = getattr(module, "__package__", None)
    )
    for alias in statement.names:
        try:
            value = getattr(source, alias.name)
        except AttributeError:
            # The attribute error names X and its module; the submodule one does not.
            try:
                value = importlib.import_module(f"{source.__name__}.{alias.name}")
            except ImportError:
                raise ImportError(
                    f"cannot import name {alias.name!r} from {source.__name__!r}"
                ) from None
        namespace[alias.asname or alias.name] = value


def _statement_imports(statement, import_name) -> bool:
    """Whether `statement` imports `import_name` itself, not a sibling."""
    if isinstance(statement, ast.Import):
        return any(
            alias.name == import_name or alias.name.startswith(import_name + ".")
            for alias in statement.names
        )
    return statement.level == 0 and (statement.module or "").split(".")[0] == import_name


def _is_backend_guard(node, guard) -> bool:
    """A bare ``if is_<backend>_available():`` at module scope. Negated and compound
    guards are skipped rather than guessed at."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == guard
        and not test.args
        and not test.keywords
    )


def _names_bound_without_importing(body) -> list:
    """Names a guard body binds by something other than an import.

    transformers/audio_utils.py guards an ASSIGNMENT: ``if is_torchcodec_available():
    TORCHCODEC_VERSION = version.parse(...)``. Replaying that would evaluate an arbitrary
    expression, so only the names are collected and the module is re-run instead."""
    names = []
    for statement in body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Assign):
            names.extend(
                target.id for target in statement.targets if isinstance(target, ast.Name)
            )
        elif isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name) and statement.value is not None:
                names.append(statement.target.id)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(statement.name)
    return names


def _skipped_import_statements(tree, guard, import_name) -> list:
    """Top-level import statements conditional on this backend. Two shapes: the ``if
    is_<backend>_available():`` guard leaves the name absent, while ``try: import x /
    except ImportError: x = None`` BINDS it to None (see ``_needs_rebinding``)."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.If):
            if not _is_backend_guard(node, guard):
                continue
            out.extend(
                statement for statement in node.body
                if isinstance(statement, (ast.Import, ast.ImportFrom))
            )
        elif isinstance(node, ast.Try):
            # Only an import-guard try; a body with real logic must not be re-run.
            if not node.body or not all(
                isinstance(statement, (ast.Import, ast.ImportFrom))
                for statement in node.body
            ):
                continue
            out.extend(
                statement for statement in node.body
                if _statement_imports(statement, import_name)
            )
    return out


def _needs_rebinding(module, names) -> bool:
    # None counts as missing: `except ImportError: spm = None` binds the name but is unusable.
    return any(getattr(module, each, None) is None for each in names)


def _replay_skipped_guarded_imports(iu, backend) -> bool:
    """Run the guarded import blocks skipped at first import; False if any raised.

    Refreshing the availability flag alone is not enough: the module scope import never ran,
    so ``requires_backends`` starts succeeding and the body dies on a bare ``NameError``."""
    guard = f"is_{backend.replace('-', '_')}_available"
    import_name = _ALLOW_LIST.get(backend) or backend.replace("-", "_")
    try:
        passes = bool(getattr(iu, guard)())
    except Exception:
        return True
    if not passes:
        return True
    ok = True
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name != "transformers" and not name.startswith("transformers."):
            continue
        source = _module_source(module)
        if source is None:
            # A module the import system really executed, whose source we cannot read, may
            # hold the guarded import we need to replay, and skipping it silently is what
            # produces the bare NameError this replay exists to prevent. So report it.
            #
            # Gated on having a spec or a loader, which is what separates that case from a
            # synthetic placeholder. transformers seeds sys.modules with bare module
            # objects carrying no __spec__, no __loader__ and no __file__; they never ran
            # any code, so they cannot hold a skipped import. Measured on a stock install:
            # 221 of 373 loaded transformers modules are placeholders and 0 are genuinely
            # sourceless, so failing closed on source alone would break every user.
            if getattr(module, "__spec__", None) is not None or \
               getattr(module, "__loader__", None) is not None:
                ok = _uninspectable(module, "its source is unavailable")
            continue
        if guard not in source and import_name not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # We CAN see the source and it does mention this backend, so this
            # module plausibly holds the guarded import we need to replay. A
            # silent skip here is what produced the bare NameError the replay
            # exists to prevent, so report it instead.
            ok = _uninspectable(module, "its source could not be parsed")
            continue
        for statement in _skipped_import_statements(tree, guard, import_name):
            if any(alias.name == "*" for alias in statement.names):
                # No name to check a star import against.
                continue
            bound = _names_bound_by(statement)
            if not bound or not _needs_rebinding(module, bound):
                continue
            try:
                _perform_import(statement, module)
            except Exception as exception:
                ok = False
                logger.warning(
                    f"Unsloth: {backend} installed, but replaying "
                    f"`{ast.unparse(statement)}` in {module.__name__} failed: "
                    f"{type(exception).__name__}: {exception}"
                )
        if not _rerun_for_guarded_state(module, tree, guard, backend):
            ok = False
    return ok


def _module_source(module):
    """Source for a loaded module, or None if it genuinely has none.

    Asks the module's own loader first. That is what makes a zipimported or
    otherwise non-plain-file transformers readable: reading `__file__` directly
    only ever worked for a loose `.py` on disk, and every other layout was
    skipped silently, which is indistinguishable from a successful replay and
    ends in the bare NameError this module exists to prevent."""
    loader = getattr(module, "__loader__", None)
    get_source = getattr(loader, "get_source", None)
    if get_source is not None:
        try:
            source = get_source(getattr(module, "__name__", ""))
            if source is not None:
                return source
        except Exception:
            pass
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return None
    try:
        with open(path, encoding = "utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _uninspectable(module, why) -> bool:
    """Report a module we could not replay into, and return False for `ok`.

    A skip used to be indistinguishable from a success, so the caller carried on
    into a module whose guarded name was never bound and the user got the bare
    NameError this replay exists to prevent. Saying so is better: the caller
    re-raises the original ImportError, which at least names the package."""
    logger.warning(
        f"Unsloth: installed the package, but could not restore "
        f"`{module.__name__}` because {why}. Please restart the runtime or kernel."
    )
    return False


def _rerun_for_guarded_state(module, tree, guard, backend) -> bool:
    """Re-import ``module`` when its guard body binds state an import replay cannot.

    ``importlib.reload`` re-runs the module in the SAME ``__dict__``, so a function another
    module already imported by name sees the new global too. The top-level ``transformers``
    package is never reloaded: it is the lazy-module entry point."""
    if module.__name__ == "transformers":
        return True
    missing = [
        name
        for node in tree.body
        if _is_backend_guard(node, guard)
        for name in _names_bound_without_importing(node.body)
        if not hasattr(module, name)
    ]
    if not missing:
        return True
    try:
        importlib.reload(module)
    except Exception as exception:
        logger.warning(
            f"Unsloth: {backend} installed, but re-running {module.__name__} to bind "
            f"{', '.join(missing)} failed: {type(exception).__name__}: {exception}"
        )
        return False
    still = [name for name in missing if not hasattr(module, name)]
    if still:
        logger.warning(
            f"Unsloth: {backend} installed, but {module.__name__} still does not "
            f"define {', '.join(still)}; restart the kernel to pick it up."
        )
        return False
    return True


def _is_dummy_export(obj) -> bool:
    """Whether ``obj`` is one of transformers' generated ``utils/dummy_*_objects.py`` stubs.

    A backend missing at ``import transformers`` binds the PUBLIC name to a stub whose whole
    body is ``requires_backends(...)``. Installing the backend later refreshes availability
    but does not rebuild that lazy export, so a retry that now SUCCEEDS runs the stub to
    completion and hands back ``None``. The wrapper says restart instead of returning."""
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str):
        module = getattr(type(obj), "__module__", None)
    return isinstance(module, str) and module.startswith("transformers.utils.dummy_")


def patch_requires_backends_autoinstall():
    """Wrap ``requires_backends`` so an allow-listed missing backend triggers a one-shot
    pip install and a retry. The original ImportError is preserved on failure, so error
    bytes stay identical to upstream under ``UNSLOTH_AUTO_INSTALL=0``."""
    try:
        from transformers.utils import import_utils as iu
    except Exception:
        return
    current = getattr(iu, "requires_backends", None)
    if current is None:
        return
    if getattr(current, "_unsloth_patched", False):
        # Re-broadcast: a module imported since the last pass still holds the original.
        _rebind_requires_backends(current, getattr(current, "_unsloth_original", None))
        return
    _orig = current

    def requires_backends(obj, backends):
        _names = [b for b in (backends if isinstance(backends, (list, tuple)) else [backends])
                  if isinstance(b, str)]
        _stuck = [b for b in _names if b in _replay_failed]
        if _stuck:
            raise ImportError(
                f"Unsloth: `{'`, `'.join(_stuck)}` is installed, but its guarded imports "
                f"could not be replayed into transformers modules that were already "
                f"imported without it. Please restart the runtime/kernel."
            )
        try:
            _result = _orig(obj, backends)
        except ImportError as original:
            if not _auto_install_enabled() or _no_network():
                raise
            wanted_iter = backends if isinstance(backends, (list, tuple)) else [backends]
            wanted = [b for b in wanted_iter if isinstance(b, str) and b in _ALLOW_LIST]
            if not wanted:
                raise
            # Only genuinely importable backends: on 4.x the refresh flips `_<backend>_available` blindly.
            # Deliberately NOT short-circuited on the first failure. A backend
            # that did install must still have its availability refreshed, or a
            # later retry sees a stale flag. `_attempted` already bounds this to
            # one attempt per package per process, and a timeout now says so.
            installed = [b for b in wanted if _try_install_and_import(b)]
            if not installed:
                raise
            _installed_backends.update(installed)
            with _replay_lock:
                for b in installed:
                    if b in _replay_done:
                        # Another thread already did this one; redoing it would reload the
                        # same modules underneath it for no gain.
                        continue
                    _refresh_backend_availability(iu, b)
                    # On replay failure the consumer is still unbound; the original error names the package.
                    if not _replay_skipped_guarded_imports(iu, b):
                        # Remembered, or the next call sails past _orig and hands the
                        # consumer a module whose guarded name was never bound.
                        _replay_failed.add(b)
                        raise
                    _replay_done.add(b)
            try:
                # A dummy can want backends this allow list does not carry, as
                # `["timm", "torchvision"]` does; a restart cannot supply those.
                result = _orig(obj, backends)
            except ImportError as remaining:
                raise remaining from None
            if _is_dummy_export(obj):
                # Returning would run the dummy body and hand back None; say what fixes it.
                raise ImportError(
                    f"{original}\n"
                    f"Unsloth: `{'`, `'.join(installed)}` is now installed, but "
                    f"transformers bound this object to a placeholder when it was "
                    f"first imported without it. Please restart the runtime/kernel."
                ) from None
            return result
        # _orig succeeding does not mean this object is usable. Once we have installed the
        # backend, a placeholder transformers froze at first import passes the check just
        # as happily, and running its body returns None. Gated on us having installed the
        # backend: a dummy that predates anything we did would have raised above.
        if _is_dummy_export(obj) and any(b in _installed_backends for b in _names):
            raise ImportError(
                f"Unsloth: `{'`, `'.join(b for b in _names if b in _installed_backends)}` "
                f"is installed, but transformers bound this object to a placeholder when "
                f"it was first imported without it. Please restart the runtime/kernel."
            )
        return _result

    requires_backends._unsloth_patched = True
    requires_backends._unsloth_original = _orig
    iu.requires_backends = requires_backends
    _rebind_requires_backends(requires_backends, _orig)


def patch_check_imports_autoinstall():
    """trust_remote_code modeling files raise via ``dynamic_module_utils.check_imports``,
    which never reaches ``requires_backends``, so wrap it too."""
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:
        return
    # getattr, mirroring the requires_backends guard above: a transformers that renames or
    # drops this helper must leave the patch a no-op. unsloth's TEMPORARY_PATCHES loop only
    # catches ValueError/TypeError, so a bare AttributeError here would skip every later patch.
    _orig = getattr(dmu, "check_imports", None)
    if _orig is None:
        return
    if getattr(_orig, "_unsloth_patched", False):
        return

    def check_imports(filename):
        try:
            return _orig(filename)
        except ImportError as e:
            if not _auto_install_enabled() or _no_network():
                raise
            msg = str(e)
            if "This modeling file requires" not in msg:
                raise
            # Message format: "... environment: pkg1, pkg2. Run `pip install...`"
            try:
                tail = msg.split("environment:", 1)[1]
                pkgs_str = tail.split(".", 1)[0]
            except Exception:
                raise
            pkgs = [p.strip() for p in pkgs_str.split(",") if p.strip() in _ALLOW_LIST]
            if not pkgs:
                raise
            ok = all(_try_install_and_import(p) for p in pkgs)
            if not ok:
                raise
            return _orig(filename)

    check_imports._unsloth_patched = True
    dmu.check_imports = check_imports


def _ipython_chain_is_broken() -> bool:
    """True only when IPython is installed but ``traitlets`` is missing: a plain ``import
    unsloth`` in a container without IPython must not reach the package manager.
    ``find_spec`` keeps the probe offline and avoids importing IPython."""
    return (
        importlib.util.find_spec("IPython") is not None
        and importlib.util.find_spec("traitlets") is None
    )


def _ensure_notebook_chain():
    """Repair deps that raise a bare ModuleNotFoundError outside transformers, where no
    wrapper hook can catch them."""
    if not _auto_install_enabled() or _no_network():
        return
    if not _ipython_chain_is_broken():
        return
    for pkg in ("traitlets",):
        if importlib.util.find_spec(pkg) is None:
            _try_install_and_import(pkg)


def patch_notebook_deps_autoinstall():
    patch_requires_backends_autoinstall()
    patch_check_imports_autoinstall()
    _ensure_notebook_chain()


def _patch_notebook_deps_autoinstall_safe():
    """The TEMPORARY_PATCHES entry, which must never raise.

    unsloth/models/_utils.py runs the patch list with `except (ValueError,
    TypeError)`, and that handler RE-INVOKES the patch rather than skipping it.
    So anything else we raise propagates out of a module-level call and aborts
    `import unsloth` outright, and every patch registered after us is skipped.
    That is a real hazard for a user who upgrades unsloth_zoo without upgrading
    unsloth, since there is no version pin between them. A dependency installer
    is never worth breaking the import over."""
    try:
        patch_notebook_deps_autoinstall()
    except Exception as exception:
        logger.warning(
            f"Unsloth: notebook dependency hooks unavailable "
            f"({type(exception).__name__}: {exception}). Continuing without them."
        )


TEMPORARY_PATCHES.append(_patch_notebook_deps_autoinstall_safe)

# Also run at import: a trust_remote_code modeling file can load before the TEMPORARY_PATCHES pass.
if not _env_is_true("UNSLOTH_NOTEBOOK_DEPS_NO_AUTORUN"):
    try:
        patch_notebook_deps_autoinstall()
    except Exception as _e:
        logger.warning(f"Unsloth: notebook dependency hooks deferred: {_e}")
