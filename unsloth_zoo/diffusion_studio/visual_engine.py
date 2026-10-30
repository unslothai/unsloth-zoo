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
"""Drives llama-diffusion-gemma-visual-server, which runs the generation loop from the GGUF's embedded tokenizer, so no
tokenizer files are needed; request file path on stdin, READY/F/C/DONE/ERR lines on stdout, QUIT exits."""
import ctypes
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile

CANVAS = 256


class ContextOverflow(RuntimeError):
    """Raised by generate_visual when the server replies ERR toolong; needed is the token count the request requires, budget the server's resolved MAXTOK."""

    def __init__(self, needed, budget):
        self.needed = int(needed)
        self.budget = int(budget)
        super().__init__(
            f"conversation needs {self.needed} tokens but the context budget is {self.budget}"
        )


class VisualServerCrashed(RuntimeError):
    """The visual-server subprocess died mid-request (EOF or broken pipe); the caller respawns it and retries the turn once if nothing was streamed."""


def _set_pdeathsig():
    """Linux: ask the kernel to SIGTERM this child when the parent shim dies, so a hard-killed shim never orphans a GPU process."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _resolve_bin(server_bin=None):
    """Find the visual server binary: explicit arg, then DG_VISUAL_BIN, then PATH."""
    cand = server_bin or os.environ.get("DG_VISUAL_BIN") or shutil.which("llama-diffusion-gemma-visual-server")
    if not cand or not os.path.exists(cand):
        raise RuntimeError(
            "llama-diffusion-gemma-visual-server not found. Set DG_VISUAL_BIN to the built binary "
            "(from llama.cpp), or put it on PATH."
        )
    return os.path.abspath(cand)


def _torch_cuda_lib_dir():
    """Windows: torch/lib if it ships cudart64_*.dll, else None; LD_LIBRARY_PATH is ignored there, so without torch/lib
    on PATH the child errors 126 and falls back to CPU (unsloth#6273; mirrors llama-server fix unsloth#5491); glob is
    CUDA-major-agnostic."""
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    except Exception:
        return None
    if not os.path.isdir(torch_lib):
        return None
    try:
        has_cudart = any(
            f.lower().startswith("cudart64_") and f.lower().endswith(".dll")
            for f in os.listdir(torch_lib)
        )
    except OSError:
        return None
    return torch_lib if has_cudart else None


def _bundled_cuda_lib_dirs():
    """Linux/WSL nvidia-* wheel lib dirs, prepended ahead of any system path so the bundled CUDA is used, not a missing
    or WSL-crashing system one (unsloth#6303); torch/lib excluded as its libgomp/libstdc++ could shadow system libs;
    [] if none."""
    import sysconfig

    roots, dirs = [], []
    # Search roots: the dir where unsloth_zoo is installed, then the env's purelib; covers normal, non-standard-prefix and editable installs.
    roots.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        roots.append(sysconfig.get_paths()["purelib"])
    except Exception:
        pass
    for root in roots:
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for name in sorted(os.listdir(nvidia)):  # cu13 sorts ahead of cudnn/nccl/...
            lib = os.path.join(nvidia, name, "lib")
            if os.path.isdir(lib) and lib not in dirs:
                dirs.append(lib)
    return dirs


DEFAULT_NGL = 99


def _resolve_ngl(ngl = None, base_env = None):
    """GPU layers (llama.cpp n_gpu_layers): explicit ngl wins, then the NGL env var, else all layers, so partial offload
    works for GGUFs larger than VRAM (unsloth#7574)."""
    if ngl is not None:
        return max(0, int(ngl))
    raw = (os.environ if base_env is None else base_env).get("NGL")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULT_NGL


def _first_visible_gpu(mask):
    """First card of an inherited mask, so 0,1 means GPU 0 alone and 2,1 never reaches a hidden GPU 0; an all-hiding mask is kept; no mask means GPU 0."""
    if mask is None:
        return "0"
    # CUDA stops at the first invalid entry, so a leading empty entry (",2") hides every GPU.
    first = mask.split(",")[0].strip()
    if first in ("", "-1"):
        return mask
    return first


def _build_subprocess_env(server_bin, gpu = None, maxtok = 0, ngl = None, base_env = None, os_name = None):
    """Child env: binary dir, bundled CUDA runtime, then inherited path (torch/lib on PATH on Windows, unsloth#6273;
    nvidia-* dirs on LD_LIBRARY_PATH on Linux, unsloth#6303); gpu=None keeps the first inherited CUDA_VISIBLE_DEVICES
    entry else GPU 0, other values replace it, '' is CPU only."""
    name = os.name if os_name is None else os_name
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else _first_visible_gpu(env.get("CUDA_VISIBLE_DEVICES"))
    env["NGL"] = str(_resolve_ngl(ngl, base_env))
    env["MAXTOK"] = str(maxtok)

    bin_dir = os.path.dirname(server_bin)
    ld_dirs = [bin_dir]
    if name != "nt":
        ld_dirs += _bundled_cuda_lib_dirs()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_dirs.append(existing_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_dirs)

    if name == "nt":
        torch_lib = _torch_cuda_lib_dir()
        if torch_lib:
            env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
    return env


def _canvas_maxtok(maxtok):
    """Keeps MAXTOK only when 0 < value <= 8192, else 0 for auto-size: the non-causal canvas compute buffer grows ~quadratically, so oversized values must defer."""
    maxtok = int(maxtok)
    return maxtok if 0 < maxtok <= 8192 else 0


def _replace_request_file(req_path, req):
    """Writes via mkstemp then os.replace so a link planted at req_path is replaced, not written through; kept local
    since importing unsloth_zoo.compiler would pull in torch."""
    directory = os.path.dirname(req_path) or "."
    descriptor, temporary_path = tempfile.mkstemp(
        prefix = f".{os.path.basename(req_path)}.", suffix = ".tmp", dir = directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding = "utf-8") as f:
            descriptor = None
            json.dump(req, f, ensure_ascii = False)
        os.replace(temporary_path, req_path)
    except BaseException:
        if descriptor is not None:
            try: os.close(descriptor)
            except OSError: pass
        try: os.remove(temporary_path)
        except OSError: pass
        raise


class VisualServer:
    """Persistent optimized decoder: send chat messages, stream per-step canvas frames + committed text."""

    def __init__(self, gguf, gpu=None, maxtok=0, server_bin=None, req_path=None, ngl=None):
        self.gguf = gguf
        self.server_bin = _resolve_bin(server_bin)
        self.maxtok_req = int(maxtok)
        req_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        # /dev/shm and /tmp are 1777 with enumerable PIDs, so a PID-derived name is world readable and can be
        # pre-empted; each server gets its own 0700 dir with an unpredictable name.
        self._req_dir = None
        if req_path:
            self.req = req_path
        else:
            self._req_dir = tempfile.mkdtemp(prefix = "dg_visual_", dir = req_dir)
            self.req = os.path.join(self._req_dir, "request.json")
        try:
            self.env = _build_subprocess_env(self.server_bin, gpu=gpu, maxtok=_canvas_maxtok(maxtok), ngl=ngl)
            self.ngl = int(self.env["NGL"])
            self.p = None
            self._spawn()
        except BaseException:
            # close() never runs on an object whose __init__ failed, so the request dir is removed here or it leaks once per supervisor retry.
            if self._req_dir is not None:
                shutil.rmtree(self._req_dir, ignore_errors = True)
                self._req_dir = None
            raise

    def _spawn(self):
        """Launches the server and waits for the READY handshake; used at startup and by restart() after a crash, since it re-loads the model."""
        self.p = subprocess.Popen([self.server_bin, self.gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=self.env, bufsize=1, text=True,
                                  preexec_fn=_set_pdeathsig if os.name == "posix" else None)
        # Only stdout is UTF-8: the server fopen()s the stdin request path, which is ANSI on Windows.
        self.p.stdout = io.TextIOWrapper(self.p.stdout.detach(), encoding="utf-8")
        line = self.p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"visual server failed to start: {line!r}")
        # Server prints "READY <n_vocab> <maxtok>": maxtok is the resolved per-turn context budget, possibly auto-sized
        # to fit VRAM when MAXTOK=0; older servers print only "READY <n_vocab>".
        parts = line.split()
        self.n_vocab = int(parts[1]) if len(parts) > 1 else 0
        self.maxtok = int(parts[2]) if len(parts) > 2 else self.maxtok_req

    def restart(self):
        """Kills a dead or stuck server and spawns a fresh one, so one bad turn does not leave later turns failing with broken pipes."""
        try:
            if self.p is not None:
                self.p.kill()
                self.p.wait(timeout=10)
        except Exception:
            pass
        self._spawn()

    def _send(self, messages, n_blocks, seed, tools=None):
        # A previous turn may have crashed the decoder, so respawn before writing; a dead child would otherwise break every later turn with a broken pipe.
        if self.p is None or self.p.poll() is not None:
            self.restart()
        req = {"seed": int(seed), "n_blocks": int(n_blocks), "messages": messages}
        # Forward tools so the server renders them into the chat template (inputs.tools) for schema-correct <|tool_call> args.
        if tools:
            req["tools"] = tools
        # The request body carries the whole conversation: no link, not world
        # readable, and no FIFO. O_NONBLOCK refuses one with no reader; one whose read
        # end is already held open opens fine, so the fstat is what refuses that.
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if not no_follow:
            # No atomic no-follow open here, and an lstat first is only a time of check.
            _replace_request_file(self.req, req)
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            flags |= no_follow
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(self.req, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError(f"Unsloth: refusing to write the request at `{self.req}`: not a regular file.")
                with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                    descriptor = None
                    json.dump(req, f, ensure_ascii=False)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        try:
            self.p.stdin.write(self.req + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            # Died between the poll above and the write: respawn once and retry the send.
            self.restart()
            self.p.stdin.write(self.req + "\n")
            self.p.stdin.flush()

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
        finally:
            # The request file used to outlive the process, in a shared directory.
            if self._req_dir is not None:
                shutil.rmtree(self._req_dir, ignore_errors = True)
                self._req_dir = None


def _parse_stats(line):
    """Parse a 'STATS k=v k=v ...' summary line into a dict (ints/floats where possible)."""
    stats = {}
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            stats[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            stats[k] = v
    return stats


def generate_visual(server, messages, seed=3407, max_blocks=8, on_frame=None, on_commit=None, on_stats=None, tools=None):
    """Streams a turn via on_frame(block, step, total, text), on_commit(cumulative_text) and on_stats(dict), returns the
    committed reply, raises ContextOverflow past the budget; a crash before any frame is retried once."""
    progressed = {"emitted": False}

    def _frame(b, s, t, x):
        progressed["emitted"] = True
        if on_frame is not None:
            on_frame(b, s, t, x)

    def _commit(txt):
        progressed["emitted"] = True
        if on_commit is not None:
            on_commit(txt)

    for attempt in (0, 1):
        try:
            return _generate_visual_once(server, messages, seed, max_blocks, _frame, _commit, on_stats, tools)
        except VisualServerCrashed:
            server.restart()  # always bring a fresh server up so the next turn works regardless
            if attempt == 1 or progressed["emitted"]:
                raise
            # nothing emitted yet -> safe to transparently resend on the fresh server


def _generate_visual_once(server, messages, seed, max_blocks, on_frame, on_commit, on_stats, tools=None):
    server._send(messages, max_blocks, seed, tools)

    full_text = ""
    while True:
        line = server.p.stdout.readline()
        if not line:
            raise VisualServerCrashed("visual server closed the stream")
        line = line.rstrip("\n")
        if line == "DONE":
            break
        if line.startswith("ERR"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "toolong":
                needed = int(parts[2]) if len(parts) > 2 else 0
                budget = int(parts[3]) if len(parts) > 3 else 0
                raise ContextOverflow(needed, budget)
            raise RuntimeError(f"visual server error: {line}")
        if line.startswith("STATS"):
            if on_stats is not None:
                on_stats(_parse_stats(line))
            continue
        if line[:1] == "F":
            parts = line.split(" ", 4)            # F <block> <step> <total> <json-string>
            if len(parts) < 5:
                continue
            block, step, total = int(parts[1]), int(parts[2]), int(parts[3])
            if on_frame is not None:
                on_frame(block, step, total, json.loads(parts[4]))
        elif line[:1] == "C":
            parts = line.split(" ", 2)            # C <block> <json-string> (cumulative answer text)
            if len(parts) < 3:
                continue
            full_text = json.loads(parts[2])
            if on_commit is not None:
                on_commit(full_text)
    return full_text


if __name__ == "__main__":
    # standalone smoke: one prompt end to end, printing a couple of frames per block
    import sys, time
    gguf = sys.argv[1] if len(sys.argv) > 1 else "diffusion-gemma-Q8_0.gguf"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "In one sentence, what is a transformer in ML?"
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    srv = VisualServer(gguf, gpu=os.environ.get("DG_GPU"))
    print("visual server ready, n_vocab=", srv.n_vocab, flush=True)
    n_frames = [0]

    def on_frame(b, s, t, text):
        n_frames[0] += 1
        if s % 5 == 0 or s == 1:
            print(f"  [F b{b} {s}/{t}] {text[:90]!r}", flush=True)

    t0 = time.time()
    reply = generate_visual(srv, [{"role": "user", "content": prompt}], seed=3407,
                            max_blocks=blocks, on_frame=on_frame,
                            on_commit=lambda s: print(f"  [commit] ...{s[-80:]!r}", flush=True))
    print(f"\nframes={n_frames[0]}  REPLY ({time.time()-t0:.1f}s):\n{reply}", flush=True)
    srv.close()
