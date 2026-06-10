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
"""DiffusionGemma generation engine.

A persistent llama.cpp logits server (the diffusiongemma build, model loaded once) driven by a pure
NumPy port of the C++ entropy-bound decoder (diffusion_generate_entropy_bound). No transformers / GPU
sampler: tokenization is `tokenizers` over tokenizer.json, chat formatting is the GGUF's own jinja
template. Stateless per request (full message history -> tokens -> blocks -> reply text).

The NumPy sampler mirrors the C++ algorithm (random canvas init, per-step argmax/entropy/multinomial,
accept the lowest-entropy positions within the MI bound, renoise the rest, stable+confident stop). It
is not bit-identical to the C++ RNG; a different RNG just yields an equally valid coherent sample.
"""
import os
import json
import subprocess

import numpy as np

__all__ = ["LlamaServer", "Tok", "generate", "VOCAB", "CANVAS", "EOS_IDS", "DEFAULTS"]

VOCAB = 262144
CANVAS = 256
EOS_IDS = {1, 106, 50}

# C++ diffusion_eb_params defaults (kept in sync with the llama.cpp diffusion-cli reference defaults).
DEFAULTS = dict(max_steps=48, t_min=0.4, t_max=0.8, entropy_bound=0.1,
                stability=1, confidence=0.005)


def _resolve(path, env_var, what):
    path = path or os.environ.get(env_var)
    if not path:
        raise RuntimeError(
            f"DiffusionGemma {what} not set. Pass it explicitly or set ${env_var}. "
            "Build the diffusiongemma llama.cpp branch (ggml-org/llama.cpp PR #24423) for the server "
            "binary, and export the model's tokenizer dir."
        )
    return path


class LlamaServer:
    """Persistent llama.cpp forward: send [prompt|canvas] ids, get canvas logits [C, VOCAB].

    Protocol (llama-diffusion-gemma-server): startup prints 'READY <n_vocab>'. Per request, write a
    request file path on stdin; the file holds int32 [P, C, use_sc, temp_bits, ids...]; the server
    writes C*VOCAB float32 to <path>.resp and prints 'OK <C>'. 'QUIT' exits.
    """

    def __init__(self, gguf, gpu="0", maxtok=8192, server_bin=None, req_path=None):
        server_bin = _resolve(server_bin, "DG_SERVER_BIN", "server binary")
        self.req = req_path or f"/dev/shm/dg_studio_{os.getpid()}.req"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NGL="99", MAXTOK=str(maxtok))
        # resolve the binary's own libs first, so a copied/snapshot dir is self-contained.
        bin_dir = os.path.dirname(os.path.abspath(server_bin))
        env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        self.p = subprocess.Popen([server_bin, gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, bufsize=1, text=True)
        line = self.p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"server failed to start: {line!r}")
        self.n_vocab = int(line.split()[1])

    def forward(self, ctx_ids, canvas_ids, use_sc=0, temp=1.0):
        P, C = len(ctx_ids), len(canvas_ids)
        payload = np.empty(4 + P + C, dtype=np.int32)
        payload[0] = P
        payload[1] = C
        payload[2] = int(use_sc)
        payload[3] = np.float32(temp).view(np.int32)
        payload[4:4 + P] = ctx_ids
        payload[4 + P:] = canvas_ids
        payload.tofile(self.req)
        self.p.stdin.write(self.req + "\n")
        self.p.stdin.flush()
        ack = self.p.stdout.readline().strip()
        if not ack.startswith("OK"):
            raise RuntimeError(f"server error: {ack!r}")
        return np.fromfile(self.req + ".resp", dtype=np.float32).reshape(C, self.n_vocab)

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


class Tok:
    """tokenizer.json encode/decode + the GGUF's own jinja chat template."""

    def __init__(self, tok_dir=None):
        import jinja2
        from tokenizers import Tokenizer
        tok_dir = _resolve(tok_dir, "DG_TOK_DIR", "tokenizer dir")
        self.t = Tokenizer.from_file(f"{tok_dir}/tokenizer.json")
        cfg = json.load(open(f"{tok_dir}/tokenizer_config.json"))
        self.bos = cfg.get("bos_token", "<bos>")
        self.eos = cfg.get("eos_token", "<eos>")
        src = open(f"{tok_dir}/chat_template.jinja").read()
        self.tmpl = jinja2.Environment().from_string(src)

    def encode_chat(self, messages):
        text = self.tmpl.render(messages=messages, add_generation_prompt=True,
                                bos_token=self.bos, eos_token=self.eos)
        # the template already emits <bos>; do not double-add specials.
        return np.asarray(self.t.encode(text, add_special_tokens=False).ids, dtype=np.int32)

    def decode(self, ids):
        return self.t.decode([int(x) for x in ids], skip_special_tokens=True)


def _generate_block(server, ctx_ids, rng, params):
    """One 256-token canvas: denoise to a settled, confident canvas; return its argmax ids [C]."""
    C, V, S = CANVAS, server.n_vocab, params["max_steps"]
    tmin, tmax = params["t_min"], params["t_max"]
    bound, stab, conf = params["entropy_bound"], params["stability"], params["confidence"]

    canvas = rng.integers(0, V, size=C).astype(np.int32)   # random init (not mask)
    out_argmax = canvas.copy()
    prev_argmax = None
    held = 0
    for cur_step in range(S, 0, -1):
        t = tmin + (tmax - tmin) * (cur_step / S)
        use_sc = 0 if cur_step == S else 1                 # first step of the block is zero-SC
        raw = server.forward(ctx_ids, canvas, use_sc=use_sc, temp=t)    # [C,V] raw logits
        z = raw.astype(np.float32) * (1.0 / t)             # processed = raw / t

        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        Z = e.sum(axis=1, keepdims=True)
        p = e / Z
        argmax = z.argmax(axis=1).astype(np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(p), 0.0)
        entropy = -(p * logp).sum(axis=1)                  # [C]

        # multinomial sample per position (inverse-CDF)
        u = rng.random(C).astype(np.float32)
        cdf = np.cumsum(p, axis=1)
        sampled = np.minimum((cdf < u[:, None]).sum(axis=1), V - 1).astype(np.int32)

        # accept the lowest-entropy positions s.t. sum of strictly-earlier entropies <= bound
        order = np.argsort(entropy, kind="stable")
        ent_sorted = entropy[order]
        earlier = np.cumsum(ent_sorted) - ent_sorted
        accepted = np.zeros(C, dtype=bool)
        accepted[order] = earlier <= bound

        renoise = rng.integers(0, V, size=C).astype(np.int32)
        canvas = np.where(accepted, sampled, renoise).astype(np.int32)
        out_argmax = argmax

        held = held + 1 if (prev_argmax is not None and np.array_equal(argmax, prev_argmax)) else 0
        confident = float(entropy.mean()) < conf
        if held >= stab and confident:
            break
        prev_argmax = argmax
    return out_argmax


def generate(server, tok, messages, rng, max_blocks=8, params=None, on_block=None):
    """Stateless turn: messages -> tokens -> up to max_blocks canvases -> reply text.

    Calls on_block(partial_text) after each committed block (for streaming). Returns the full reply.
    """
    params = {**DEFAULTS, **(params or {})}
    ctx = tok.encode_chat(messages)
    resp_ids = []
    for _ in range(max_blocks):
        committed = _generate_block(server, ctx, rng, params)          # [256]
        eos_pos = next((i for i, t in enumerate(committed) if int(t) in EOS_IDS), None)
        block_ids = committed[:eos_pos] if eos_pos is not None else committed
        resp_ids.extend(int(x) for x in block_ids)
        ctx = np.concatenate([ctx, committed])
        if on_block is not None:
            on_block(tok.decode(resp_ids))
        if eos_pos is not None:
            break
    return tok.decode(resp_ids)
