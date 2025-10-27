# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "check_python_modules",
    "create_locked_down_function",
    "execute_with_time_limit",
    "Benchmarker",
    "is_port_open",
    "launch_openenv",
]

import ast
import sys
import sysconfig
from pathlib import Path
import functools
import types
import __future__
import builtins as _py_builtins
import os, gc, time, statistics
import numpy as np
import signal
from contextlib import contextmanager
from functools import wraps
import threading
import errno
import time
from typing import Callable, TypeVar, Any, Tuple
T = TypeVar("T")


@functools.lru_cache
def _stdlib_names():
    """
    Build a set of canonical stdlib top-level module/package names.
    Uses sys.stdlib_module_names when available (3.10+), with a
    filesystem fallback for older versions/edge cases.
    """
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__") # special-case

    # Fallback/augmentation: scan the stdlib directory
    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        # conservative fallback; the names set above will still work well
        pass
    return names
pass

def check_python_modules(code: str):
    """
    Checks if function only calls Python standard library functions and nothing more
    Return (ok: bool, details: dict)

    ok == True  -> all absolute imports are from the stdlib.
    ok == False -> details['non_stdlib'] lists offending top-level modules.

    details includes:
      - stdlib: sorted list of stdlib imports found
      - non_stdlib: sorted list of non-stdlib imports found
      - relative_imports: count of relative imports (always allowed here)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0
    _STDLIB_SET = _stdlib_names()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])
        def visit_ImportFrom(self, node: ast.ImportFrom):
            nonlocal relative_count
            if (node.level or 0) > 0:
                # relative import
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }
pass


def _is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )
pass

def _is_safe_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all(_is_safe_literal(e) for e in node.elts)
    return False
pass

def _ensure_no_calls_in_signature(func: ast.AST, *, require_constant_defaults: bool = True):
    args = func.args
    sig_nodes = []

    for d in args.defaults:
        if require_constant_defaults and not _is_safe_literal(d):
            raise RuntimeError("Only simple literal default argument values are allowed.")
        sig_nodes.append(d)
    for d in args.kw_defaults:
        if d is not None:
            if require_constant_defaults and not _is_safe_literal(d):
                raise RuntimeError("Only simple literal default keyword argument values are allowed.")
            sig_nodes.append(d)

    for group in (
        getattr(args, "posonlyargs", []),
        args.args,
        args.kwonlyargs,
        [args.vararg] if args.vararg else [],
        [args.kwarg] if args.kwarg else [],
    ):
        for a in group:
            if getattr(a, "annotation", None) is not None:
                sig_nodes.append(a.annotation)
    if getattr(func, "returns", None) is not None:
        sig_nodes.append(func.returns)

    for node in sig_nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                raise RuntimeError("Function signature contains a call (disallowed).")
            if isinstance(n, (ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await)):
                raise RuntimeError("Function signature contains executable constructs (disallowed).")
pass


def validate_single_function_source(
    source: str,
    *,
    allow_async: bool = False,
    allow_decorators: bool = False,
    allow_nested_defs: bool = False,
    require_constant_defaults: bool = True,
):
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise RuntimeError(f"SyntaxError: {e}") from e

    body = list(tree.body)
    if body and _is_docstring_stmt(body[0]):
        body = body[1:]

    if not body:
        raise RuntimeError("No function found.")
    if len(body) != 1:
        raise RuntimeError("Only a single top-level function is allowed.")

    func = body[0]
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise RuntimeError("Top-level node must be a function definition.")
    if isinstance(func, ast.AsyncFunctionDef) and not allow_async:
        raise RuntimeError("Async functions are not allowed.")
    if func.decorator_list and not allow_decorators:
        raise RuntimeError("Decorators are not allowed (they execute at definition time).")

    if not allow_nested_defs:
        for n in ast.walk(func):
            if n is func:
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raise RuntimeError("Nested functions/classes are not allowed.")

    _ensure_no_calls_in_signature(func, require_constant_defaults=require_constant_defaults)
    return tree, func
pass


def load_single_function(
    source: str,
    *,
    allow_async: bool = True,
    allow_decorators: bool = True,
    allow_nested_defs: bool = True,
    require_constant_defaults: bool = True,
    builtins_policy: str = "full",  # "full" or "allowlist"
):
    """
    builtins_policy:
      - "full": expose the standard Python builtins (what you asked for).
      - "allowlist": expose a small safe subset (shown below).
    """
    tree, func_node = validate_single_function_source(
        source,
        allow_async = allow_async,
        allow_decorators = allow_decorators,
        allow_nested_defs = allow_nested_defs,
        require_constant_defaults = require_constant_defaults,
    )

    if builtins_policy == "full":
        exposed_builtins = _py_builtins.__dict__
    elif builtins_policy == "allowlist":
        exposed_builtins = {
            "abs": _py_builtins.abs,
            "all": _py_builtins.all,
            "any": _py_builtins.any,
            "bool": _py_builtins.bool,
            "dict": _py_builtins.dict,
            "enumerate": _py_builtins.enumerate,
            "float": _py_builtins.float,
            "int": _py_builtins.int,
            "len": _py_builtins.len,
            "list": _py_builtins.list,
            "max": _py_builtins.max,
            "min": _py_builtins.min,
            "pow": _py_builtins.pow,
            "range": _py_builtins.range,
            "reversed": _py_builtins.reversed,
            "round": _py_builtins.round,
            "str": _py_builtins.str,
            "sum": _py_builtins.sum,
            "tuple": _py_builtins.tuple,
            "zip": _py_builtins.zip,
        }
    else:
        raise ValueError("builtins_policy must be 'full' or 'allowlist'")

    glob = {"__builtins__": exposed_builtins}
    loc = {}

    try:
        code = compile(
            tree,
            filename = "<user_code>",
            mode = "exec",
            flags = __future__.annotations.compiler_flag, # annotations become strings
            dont_inherit = True,
        )
        exec(code, glob, loc)
    except Exception as e:
        raise RuntimeError(f"Failed while defining the function: {e}") from e

    fn = loc.get(func_node.name) or glob.get(func_node.name)
    if not isinstance(fn, types.FunctionType):
        raise RuntimeError("Compiled object is not a function.")
    return fn
pass

def create_locked_down_function(function):
    """
    Creates a singular Python function which disallows the following:
    1. No globals or
    """
    output_function = {}
    f = load_single_function(function)
    # Locks down function so it can see global variables of nothingness
    f = types.FunctionType(f.__code__, {})
    return f
pass


def _retry_eintr(func, *args):
    while True:
        try:
            return func(*args)
        except OSError as e:
            if getattr(e, "errno", None) == errno.EINTR:
                continue
            raise
pass

@contextmanager
def time_limit(seconds: float, *, strict: bool = True, leeway: float = 0.05):
    """
    Enforce a wall-clock time limit using SIGALRM/ITIMER_REAL.

    - Earliest deadline wins (respects any currently armed ITIMER_REAL).
    - EINTR-safe setup/teardown; resists Ctrl+C during cleanup.
    - strict=True: 'fail-closed' — if body returns after the deadline and the
      SIGALRM handler didn't get to run, raise TimeoutError on exit anyway.
    - Unix-like OS, main thread only. Process-wide SIGALRM: not composable with other users.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if not hasattr(signal, "setitimer"):
        raise NotImplementedError("time_limit requires Unix setitimer/SIGALRM support")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("time_limit must be used from the main thread")

    start = time.monotonic()
    deadline_at = start + seconds

    old_handler = signal.getsignal(signal.SIGALRM)
    prev_remaining, prev_interval = signal.getitimer(signal.ITIMER_REAL)

    # Always respect any already-armed timer: take the earlier deadline.
    deadline = seconds if prev_remaining <= 0.0 else min(seconds, prev_remaining)

    fired = False  # set by our handler

    def _handler(signum, frame):
        nonlocal fired
        fired = True
        # include the intended arming deadline for debugging
        raise TimeoutError(f"Timed out after {deadline:g}s")

    setattr(_handler, "__time_limit_handler__", True)

    _retry_eintr(signal.signal, signal.SIGALRM, _handler)
    try:
        # Ensure blocking syscalls are interrupted (avoid SA_RESTART)
        try:
            signal.siginterrupt(signal.SIGALRM, True)
        except (AttributeError, OSError):
            pass

        _retry_eintr(signal.setitimer, signal.ITIMER_REAL, deadline)
        yield
    finally:
        # Make teardown atomic wrt SIGINT and robust to EINTR
        old_sigint = signal.getsignal(signal.SIGINT)
        try:
            _retry_eintr(signal.signal, signal.SIGINT, signal.SIG_IGN)
            try:
                _retry_eintr(signal.setitimer, signal.ITIMER_REAL, 0.0)  # cancel ours
            finally:
                _retry_eintr(signal.signal, signal.SIGALRM, old_handler)

            # Restore prior timer with corrected remaining time.
            if prev_remaining != 0.0 or prev_interval != 0.0:
                elapsed = max(time.monotonic() - start, 0.0)
                remaining = max(prev_remaining - elapsed, 0.0)
                _retry_eintr(signal.setitimer, signal.ITIMER_REAL, remaining, prev_interval)
        finally:
            _retry_eintr(signal.signal, signal.SIGINT, old_sigint)

        # ---- Fail-closed check (only if no TimeoutError was raised inside) ----
        if strict and not fired:
            now = time.monotonic()
            if now > deadline_at + leeway:
                # We exceeded wall time but the handler didn't get a chance to run.
                # This typically means the body spent a long time in non-cooperative C code.
                raise TimeoutError(
                    f"Exceeded time limit ({seconds:g}s) without interrupt; "
                    f"elapsed ≈ {now - start:.3f}s. "
                    "The protected code likely blocked in a C extension or another SIGALRM user clobbered the timer."
                )
pass

import multiprocessing as mp
import traceback

class RemoteTracebackError(RuntimeError):
    pass

def _run_in_subprocess(func, seconds, args, kwargs, *, start_method="spawn", kill_grace=1.0):
    ctx = mp.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _child(entry_conn, f, a, kw):
        try:
            res = f(*a, **(kw or {}))
            entry_conn.send(("ok", res))
        except BaseException as e:
            entry_conn.send(("err", (e.__class__.__name__, str(e), traceback.format_exc())))
        finally:
            try:
                entry_conn.close()
            except Exception:
                pass

    proc = ctx.Process(target=_child, args=(child_conn, func, args, kwargs))
    proc.daemon = False
    proc.start()
    child_conn.close()

    try:
        # Wait for result up to 'seconds'
        if parent_conn.poll(seconds):
            kind, payload = parent_conn.recv()
            proc.join()
            if kind == "ok":
                return payload
            else:
                name, msg, tb = payload
                raise RemoteTracebackError(f"{name}: {msg}\nRemote traceback:\n{tb}")
        else:
            # Timeout: terminate, then kill if stubborn
            proc.terminate()
            proc.join(kill_grace)
            if proc.is_alive():
                try:
                    proc.kill()  # POSIX+Py3.7+, Windows uses TerminateProcess under the hood
                finally:
                    proc.join()
            raise TimeoutError(f"Timed out after {seconds:g}s")
    except KeyboardInterrupt:
        # Ensure no orphan on Ctrl+C
        try:
            proc.terminate()
            proc.join(kill_grace)
            if proc.is_alive():
                proc.kill()
                proc.join()
        finally:
            raise
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
pass

def execute_with_time_limit(
    seconds: float,
    *,
    backend: str = "signal",
    start_method: str = "process",
    kill_grace: float = 1.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that enforces a time limit.

    backend:
      - "signal": uses SIGALRM (fast, in-process; only cooperative C code).
      - "process": runs function in a child and kills it on timeout (robust).
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if backend == "signal":
                with time_limit(seconds):
                    return func(*args, **kwargs)
            elif backend == "process":
                return _run_in_subprocess(func, seconds, args, kwargs,
                                          start_method=start_method, kill_grace=kill_grace)
            else:
                raise ValueError("backend must be 'signal' or 'process'")
        return wrapper
    return decorator
pass

class Benchmarker:
    """
    Benchmarks functions correctly by wiping cache away
    Benchmarker(trials = 1, timeout = 10).benchmark(output_function["matmul"], [(A_list, B_list)])
    """
    def __init__(self, trials = 3, loops = 1, timeout = 30):
        self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype = np.uint8)
        self.trials = trials
        self.loops = loops
        assert timeout > 0 # Cannot be 0 since it won't work!
        self.timeout = timeout
    def thrash(self):
        # Edit the buffer to wipe cache lines
        self.buffer ^= 1
        return int(self.buffer[::4096].sum())

    def benchmark(self, function, arguments):
        assert len(arguments) == self.loops
        samples = []
        exceptions = []
        timed_out = 0
        for _ in range(self.trials):
            gc.collect()
            gc.disable()
            self.thrash()
            t_start = time.perf_counter_ns()
            for i in range(self.loops):
                try:
                    with time_limit(self.timeout):
                        function(*arguments[i])
                except TimeoutError as e:
                    timed_out += 1
                except Exception as e:
                    exceptions.append(str(e))
            t_end = time.perf_counter_ns()
            gc.enable()
            samples.append((t_end - t_start) // max(1, self.loops))
        return {
            "median_ns": int(statistics.median(samples)),
            "mean_ns": int(statistics.fmean(samples)),
            "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
            "exceptions" : exceptions,
            "timeouts" : timed_out,
        }
pass

####################
##### Open Env #####
####################
import socket
import requests
import time
import random
import sys
import subprocess

def is_port_open(host, port):
    """ Check if the port like localhost:8000 is open or closed """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)  # Set a timeout for the connection attempt
        result = sock.connect_ex((host, port))
        if result == 0:
            return True  # Port is open
        else:
            return False # Port is closed or connection failed
    except socket.error as e:
        print(f"Socket error: {e}")
        return False
    finally:
        sock.close()
pass


def launch_openenv(
    port : int = 8111,
    openenv_process = None,
    working_directory : str = None,
    server : str = "envs.openspiel_env.server.app:app",
    environment = {},
    openenv_class = None,
):
    """ Finds a new port or checks if the old open port actually works """
    # Check if OpenEnv is working first
    assert type(environment) is dict
    assert type(port) is int and port >= 0 and port <= (65535-1)
    assert type(working_directory) is str
    assert openenv_class is not None
    assert type(server) is str
    localhost = f"http://localhost:{port}"

    def check_openenv_works(process):
        if process is not None:
            try:
                request = requests.get(f"{localhost}/health", timeout = 0.1).content
                if b"healthy" not in request and hasattr(process, "close"):
                    try: process.close()
                    except: pass
                    process = None
                else:
                    # It should work, so simply return the old one!
                    return process
            except:
                process = None
        return process
    openenv_process = check_openenv_works(openenv_process)

    # Otherwise, find the next port which can be used
    trials = 0
    while openenv_process is None:
        # Port ID must be less than uint16_MAX
        port = random.randint(9000, 65535-1)
        localhost = f"http://localhost:{port}"
        print(f"Unsloth: Creating new OpenEnv process at port = {port}", end = "")
        openenv_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", server, "--host", "0.0.0.0", "--port", str(port)],
            env = environment,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            cwd = working_directory,
        )
        # Wait until port is open
        wait_trials = 0
        while not is_port_open("localhost", port):
            time.sleep(0.01)
            if wait_trials % 10 == 0:
                print(".", end = "")
            wait_trials += 1
            if wait_trials == 6000:
                raise TimeoutError("Unsloth: We tried launching a new OpenEnv Localhost for 60 seconds, but we still failed :(")
        print()
        openenv_process = openenv_class(base_url = localhost)
        openenv_process = check_openenv_works(openenv_process)
        if openenv_process is not None: break
        trials += 1
        if trials == 30:
            raise TimeoutError("Unsloth: We tried launching a new OpenEnv process 30 times, but we still failed :(")
    if openenv_process is not None:
        return port, openenv_process
pass
