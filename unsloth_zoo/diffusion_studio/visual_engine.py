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
"""Optimized visual driver for DiffusionGemma.

Drives llama-diffusion-gemma-visual-server, which runs the same on-device entropy-bound decoder as the
CLI's --diffusion-visual (Stage 1 device sampling + Stage 2 device-resident self-conditioning) and owns
the whole generation loop. The server tokenizes, applies the chat template and detokenizes from the
GGUF's own embedded tokenizer, so nothing here needs tokenizer files: we hand it the chat messages and it
streams back the per-step canvas already decoded to text, plus the committed answer text.

Protocol (matches diffusion-gemma-visual-server.cpp):
  startup -> 'READY <n_vocab>'. Per request, write a request-file path on stdin; the file holds UTF-8 JSON
  {"seed": <int>, "n_blocks": <int>, "messages": [ {"role","content"}, ... ]}. The server streams lines:
    F <block> <step> <total> <json-string>   one per denoising step (current canvas, decoded)
    C <block> <json-string>                   cumulative committed answer text after this block
    DONE                                      end of request
    ERR <msg>                                 failure
  'QUIT' exits.
"""
import ctypes
import json
import os
import shutil
import signal
import subprocess
import tempfile

CANVAS = 256


class ContextOverflow(RuntimeError):
    """The templated conversation + canvas exceeds the visual server's per-turn context budget (MAXTOK).

    Raised from ``generate_visual`` when the server reports ``ERR toolong <needed> <budget>`` so the caller
    can surface a clean, user-facing message instead of a raw protocol string. ``needed`` is the token count
    the request would require; ``budget`` is the server's resolved MAXTOK.
    """

    def __init__(self, needed, budget):
        self.needed = int(needed)
        self.budget = int(budget)
        super().__init__(
            f"conversation needs {self.needed} tokens but the context budget is {self.budget}"
        )


class VisualServerCrashed(RuntimeError):
    """The visual-server subprocess died mid-request (EOF or broken pipe), e.g. a CUDA fault on a long
    context. The caller respawns it and, if nothing was streamed yet, retries the turn once."""


def _set_pdeathsig():
    """Linux: ask the kernel to SIGTERM this child when its parent (the shim) dies for any reason, so a
    hard-killed shim never orphans a GPU process."""
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
    """Windows: return torch's bundled CUDA-runtime dir (``torch/lib``) when it actually ships the
    ``cudart64_*.dll`` the GPU backend needs, else None.

    On Windows ``LD_LIBRARY_PATH`` is a no-op, so the visual-server child cannot load ``ggml-cuda.dll``
    unless torch's bundled ``cudart64_*.dll`` / ``cublas64_*.dll`` (shipped in ``torch/lib`` by the
    ``torch+cuXXX`` wheel) are on ``PATH``; otherwise the DLL fails to load with error 126 and the
    server silently falls back to CPU. Mirrors the llama-server fix (unsloth#5491) for the diffusion
    runner (unsloth#6273). The version glob keeps this CUDA-major-agnostic (cudart64_12/13/...)."""
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


class VisualServer:
    """Persistent optimized decoder: send chat messages, stream per-step canvas frames + committed text."""

    def __init__(self, gguf, gpu="0", maxtok=0, server_bin=None, req_path=None):
        self.gguf = gguf
        self.server_bin = _resolve_bin(server_bin)
        self.maxtok_req = int(maxtok)
        req_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        self.req = req_path or os.path.join(req_dir, f"dg_visual_{os.getpid()}.req")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NGL="99", MAXTOK=str(maxtok))
        env["LD_LIBRARY_PATH"] = os.path.dirname(self.server_bin) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        if os.name == "nt":
            torch_lib = _torch_cuda_lib_dir()
            if torch_lib:
                env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
        self.env = env
        self.p = None
        self._spawn()

    def _spawn(self):
        """Launch the subprocess and finish the READY handshake. Used at startup and by restart()
        after a crash (re-loads the model, so it costs tens of seconds)."""
        self.p = subprocess.Popen([self.server_bin, self.gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=self.env, bufsize=1, text=True,
                                  preexec_fn=_set_pdeathsig if os.name == "posix" else None)
        line = self.p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"visual server failed to start: {line!r}")
        # "READY <n_vocab> <maxtok>": maxtok is the server's resolved per-turn context budget (it may have
        # auto-sized it to fit VRAM when launched with MAXTOK=0). Older servers print only "READY <n_vocab>".
        parts = line.split()
        self.n_vocab = int(parts[1]) if len(parts) > 1 else 0
        self.maxtok = int(parts[2]) if len(parts) > 2 else self.maxtok_req

    def restart(self):
        """Tear down a dead/stuck server and spawn a fresh one, so one bad turn does not leave every
        later turn failing with a broken pipe."""
        try:
            if self.p is not None:
                self.p.kill()
                self.p.wait(timeout=10)
        except Exception:
            pass
        self._spawn()

    def _send(self, messages, n_blocks, seed):
        # A previous turn may have crashed the decoder; respawn before writing so a dead child does not
        # strand this and every later turn with a broken pipe.
        if self.p is None or self.p.poll() is not None:
            self.restart()
        with open(self.req, "w") as f:
            json.dump({"seed": int(seed), "n_blocks": int(n_blocks), "messages": messages}, f,
                      ensure_ascii=False)
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


def generate_visual(server, messages, seed=3407, max_blocks=8, on_frame=None, on_commit=None, on_stats=None):
    """Stream one turn through the optimized visual server.

    on_frame(block, step, total, text): a denoising frame (the current argmax canvas, already decoded).
    on_commit(cumulative_text): cumulative committed answer text after each block (for live OpenAI deltas).
    on_stats(stats_dict): the server's end-of-request summary (prompt_n, predicted_n,
        prompt_prepare_ms, wall_ms, decode_ms, blocks, steps, canvas, n_ctx) for generation stats.
    Returns the full committed reply text. Raises ContextOverflow if the request exceeds the context budget.

    Self-heals a crash: if the server dies mid-turn and nothing was streamed yet (the usual case, since
    the crash is in the per-block prefill before any frame), respawn and retry once; otherwise propagate.
    """
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
            return _generate_visual_once(server, messages, seed, max_blocks, _frame, _commit, on_stats)
        except VisualServerCrashed:
            server.restart()  # always bring a fresh server up so the next turn works regardless
            if attempt == 1 or progressed["emitted"]:
                raise
            # nothing emitted yet -> safe to transparently resend on the fresh server


def _generate_visual_once(server, messages, seed, max_blocks, on_frame, on_commit, on_stats):
    server._send(messages, max_blocks, seed)

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
    srv = VisualServer(gguf, gpu=os.environ.get("DG_GPU", "0"))
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
