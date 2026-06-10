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
"""OpenAI-compatible HTTP shim for DiffusionGemma.

Lets Unsloth Studio (or any OpenAI-compatible client) chat with DiffusionGemma over the GGUF path:
add it in Studio under Settings -> Connections as a llama.cpp / OpenAI-compatible server pointing at
this shim's /v1 URL. Wraps engine.py (the persistent llama.cpp logits server + NumPy entropy-bound
decoder). Exposes /v1/models and /v1/chat/completions (per-block SSE streaming + non-streaming). The
model loads once; requests are serialized (the logits server is single-sequence / stateful SC).

Run:
    python -m unsloth_zoo.diffusion_studio.shim \
        --gguf diffusiongemma-26B-A4B-it-Q8_0.gguf \
        --server-bin /path/to/llama-diffusion-gemma-server \
        --tok-dir /path/to/diffusiongemma_tokenizer --port 8123
"""
import argparse
import asyncio
import json
import math
import os
import threading
import time
import uuid

import numpy as np

from . import engine as E

MODEL_ID = "diffusiongemma-26B-A4B-it"

_STATE = {}          # server, tok
_LOCK = threading.Lock()


def _chunk(cid, created, delta, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _max_blocks(body):
    mt = body.get("max_tokens") or body.get("max_completion_tokens") or 2048
    try:
        mt = int(mt)
    except (TypeError, ValueError):
        mt = 2048
    return max(1, math.ceil(mt / E.CANVAS))


def build_app():
    """Build the FastAPI app (imported lazily so unsloth_zoo has no hard fastapi dependency)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": MODEL_ID, "object": "model", "created": 1700000000, "owned_by": "unsloth"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        max_blocks = _max_blocks(body)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        srv, tok = _STATE["server"], _STATE["tok"]
        loop = asyncio.get_event_loop()

        if not stream:
            def work():
                with _LOCK:
                    return E.generate(srv, tok, messages, np.random.default_rng(), max_blocks=max_blocks)
            text = await loop.run_in_executor(None, work)
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        async def gen():
            q: asyncio.Queue = asyncio.Queue()

            def on_block(cumulative):
                loop.call_soon_threadsafe(q.put_nowait, ("delta", cumulative))

            def work():
                try:
                    with _LOCK:
                        full = E.generate(srv, tok, messages, np.random.default_rng(),
                                          max_blocks=max_blocks, on_block=on_block)
                    loop.call_soon_threadsafe(q.put_nowait, ("done", full))
                except Exception as exc:  # surface engine errors into the stream
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

            threading.Thread(target=work, daemon=True).start()
            yield _sse(_chunk(cid, created, {"role": "assistant"}))
            sent = ""
            while True:
                kind, payload = await q.get()
                if kind in ("delta", "done"):
                    new = payload[len(sent):]
                    if new:
                        yield _sse(_chunk(cid, created, {"content": new}))
                        sent = payload
                    if kind == "done":
                        yield _sse(_chunk(cid, created, {}, finish="stop"))
                        yield "data: [DONE]\n\n"
                        return
                else:  # error
                    yield _sse(_chunk(cid, created, {"content": f"\n[engine error: {payload}]"},
                                      finish="stop"))
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="OpenAI-compatible shim for DiffusionGemma (GGUF).")
    ap.add_argument("--gguf", required=True, help="Path to a DiffusionGemma GGUF.")
    ap.add_argument("--server-bin", default=None,
                    help="llama-diffusion-gemma-server binary (or set DG_SERVER_BIN).")
    ap.add_argument("--tok-dir", default=None,
                    help="Tokenizer dir with tokenizer.json + chat_template.jinja (or set DG_TOK_DIR).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--gpu", default=os.environ.get("DG_GPU", "0"))
    ap.add_argument("--maxtok", type=int, default=8192)
    args = ap.parse_args()

    print(f"loading {args.gguf} on GPU {args.gpu} ...", flush=True)
    _STATE["tok"] = E.Tok(tok_dir=args.tok_dir)
    _STATE["server"] = E.LlamaServer(args.gguf, gpu=args.gpu, maxtok=args.maxtok,
                                     server_bin=args.server_bin)
    print(f"DiffusionGemma OpenAI shim ready on http://{args.host}:{args.port}  (model={MODEL_ID})",
          flush=True)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
