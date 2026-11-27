# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Runs `python -c CODE` in children forked from a pre-warmed server so unsloth_zoo's import-time
code runs fresh; children exit via os._exit, so atexit hooks do not run; cases needing HF_HOME,
torch._dynamo, torch thread vars or torch unimported keep subprocess.run; falls back when fork is
unavailable, UNSLOTH_TESTS_FORKSERVER=0, or the server dies."""

from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import sys
import threading

STARTUP_TIMEOUT_S = 120

PRELOAD_THIRD_PARTY = (
    "torch", "torchvision", "transformers", "transformers.modeling_utils", "transformers.processing_utils",
    "transformers.trainer", "peft", "accelerate", "datasets", "triton", "torch._dynamo", "torch._inductor.compile_fx",
)

_SERVER = r"""
import io, json, os, sys, traceback, importlib, time, signal
# fd 1 carries the protocol; anything the preload prints goes to /dev/null instead.
_proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
_null = os.open(os.devnull, os.O_WRONLY); os.dup2(_null, 1); os.close(_null)
for _name in json.loads(os.environ.pop("_ZOO_FORKSERVER_PRELOAD")):
    try:
        importlib.import_module(_name)
    except Exception:
        pass
assert not any(m == "unsloth_zoo" or m.startswith("unsloth_zoo.") for m in sys.modules), "server must not import the zoo"
_proto_in = sys.stdin.buffer
_proto_out.write(b"READY\n")
for line in _proto_in:
    req = json.loads(line)
    out_path, err_path = req["out"], req["err"]
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            fd_out = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            fd_err = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(fd_out, 1); os.dup2(fd_err, 2)
            devnull = os.open(os.devnull, os.O_RDONLY); os.dup2(devnull, 0)
            os.chdir(req["cwd"])
            os.environ.clear(); os.environ.update(req["env"])
            for p in reversed([p for p in req["env"].get("PYTHONPATH", "").split(os.pathsep) if p]):
                if p not in sys.path:
                    sys.path.insert(1, p)
            sys.argv = ["-c"]
            main = type(sys)("__main__")
            sys.modules["__main__"] = main
            exec(compile(req["code"], "<string>", "exec"), main.__dict__)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, int) and e.code is not None:
                print(e.code, file=sys.stderr)
        except BaseException:
            traceback.print_exc()
            code = 1
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            os._exit(code)
    deadline = time.monotonic() + req["timeout"]
    rc = None
    while rc is None:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            rc = os.waitstatus_to_exitcode(status)
        elif time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); rc = "timeout"
        else:
            time.sleep(0.01)
    _proto_out.write((json.dumps({"id": req["id"], "rc": rc}) + "\n").encode())
"""


class _Server:
    def __init__(self, preload, cwd):
        env = dict(os.environ)
        env["_ZOO_FORKSERVER_PRELOAD"] = json.dumps(list(preload))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [cwd, env.get("PYTHONPATH")]))
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=cwd, env=env,
        )
        self.lock = threading.Lock()
        self.seq = 0
        # A hang at server startup: wait STARTUP_TIMEOUT_S for READY, then kill and fall back.
        ready, _, _ = select.select([self.proc.stdout], [], [], STARTUP_TIMEOUT_S)
        if not ready or self.proc.stdout.readline() != b"READY\n":
            self.proc.kill()
            raise RuntimeError("fork server did not start")

    def run(self, code, env, cwd, timeout, tmpdir):
        out, err = os.path.join(tmpdir, "child.out"), os.path.join(tmpdir, "child.err")
        self.seq += 1
        req = {"id": self.seq, "code": code, "env": env, "cwd": cwd, "timeout": timeout, "out": out, "err": err}
        with self.lock:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("fork server died")
        reply = json.loads(line)
        # An interrupted call leaves a reply unread; the id stops the next call taking it.
        if reply.get("id") != self.seq:
            raise RuntimeError("fork server reply out of sync")
        rc = reply["rc"]
        if rc == "timeout":
            raise subprocess.TimeoutExpired([sys.executable, "-c", code], timeout)
        with open(out, errors="replace") as f_out, open(err, errors="replace") as f_err:
            return subprocess.CompletedProcess([sys.executable, "-c", code], rc, f_out.read(), f_err.read())

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


_servers: dict = {}


def _enabled():
    return hasattr(os, "fork") and sys.platform != "win32" and os.environ.get("UNSLOTH_TESTS_FORKSERVER", "1") != "0"


def run_python(code, *, env, cwd, timeout, preload=PRELOAD_THIRD_PARTY):
    """Like subprocess.run, but may fork from a server that already imported `preload`."""
    if _enabled():
        import tempfile
        key = (tuple(preload), cwd)
        try:
            server = _servers.get(key)
            if server is None or server.proc.poll() is not None:
                server = _servers[key] = _Server(preload, cwd)
            with tempfile.TemporaryDirectory() as tmpdir:
                return server.run(code, env, cwd, timeout, tmpdir)
        except subprocess.TimeoutExpired:
            raise
        except BaseException as e:
            # Any other failure, an interrupt included, retires this server: its protocol state is unknown.
            dead = _servers.pop(key, None)
            if dead is not None:
                dead.proc.kill()
            if not isinstance(e, Exception):
                raise
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


@atexit.register
def _close_all():
    for server in _servers.values():
        server.close()
