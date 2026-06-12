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
"""OpenAI-compatible HTTP shim for DiffusionGemma, so Unsloth Studio can serve it as an ordinary
llama.cpp / OpenAI-compatible server.

Drives the OPTIMIZED visual decoder (visual_engine -> llama-diffusion-gemma-visual-server, the same
on-device entropy-bound path as the CLI's --diffusion-visual: Stage 1 device sampling + Stage 2
device-resident self-conditioning). Per streaming request it (a) streams the committed answer text as
normal OpenAI deltas and (b) streams per-step "diffusion_frame" events so the 256-token canvas resolves
in place in the chat bubble, like the terminal. Set DG_ARTIFACT=1 to also append a replay HTML artifact.

Exposes /v1/models and /v1/chat/completions (streaming SSE + non-streaming) plus /health. The model loads
once; requests are serialized (the decoder is single-sequence / stateful SC).

Run:  DG_VISUAL_BIN=.../llama-diffusion-gemma-visual-server \
          python -m unsloth_zoo.diffusion_studio.shim --gguf diffusion-gemma-Q8_0.gguf --gpu 0 --port 8123
"""
import argparse
import asyncio
import atexit
import json
import math
import os
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from . import visual_engine as V          # optimized visual decoder driver (self-tokenizing server)

MODEL_ID = os.environ.get("DG_MODEL_ID", "diffusiongemma")
_PLAYER_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_player.html")
# DG_ARTIFACT=1 also appends the legacy replay HTML artifact (export/debug); default is live frames only.
_WANT_ARTIFACT = os.environ.get("DG_ARTIFACT", "") not in ("", "0", "false", "False", "no", "off")

app = FastAPI()
_STATE = {}          # server (VisualServer), player (html template str)
_LOCK = threading.Lock()


def _close_server():
    """Terminate the visual-server child so a parent exit (e.g. Studio's teardown) never leaves an
    orphaned GPU process. Idempotent. The child is also launched with PR_SET_PDEATHSIG (see
    visual_engine) so a hard kill of this process still reaps it."""
    srv = _STATE.pop("server", None)
    if srv is not None:
        try:
            srv.close()
        except Exception:
            pass


atexit.register(_close_server)  # graceful interpreter exit (uvicorn SIGTERM -> here)


def _chunk(cid, created, delta, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _timings_from_stats(stats):
    """Build a `timings` block from the server STATS summary. Returns (timings, prompt_n, completion_n).

    No autoregressive prefill exists, so we emit no prompt tok/s (the old one divided tokens by host
    tokenize time and showed a meaningless ~14000). We report the diffusion CLI's two rates so Studio
    matches the terminal: effective = canvas*blocks/wall (end-to-end, ~hundreds) and in-step parallel =
    canvas*steps/wall (the model's real per-step rate, ~thousands)."""
    def rate(n, ms):
        return (n / ms * 1000.0) if ms > 0 else 0.0
    P = int(stats.get("prompt_n", 0))
    G = int(stats.get("predicted_n", 0))
    prep_ms   = float(stats.get("prompt_prepare_ms", 0.0))
    wall_ms   = float(stats.get("wall_ms", 0.0))
    decode_ms = float(stats.get("decode_ms", 0.0))
    steps     = int(stats.get("steps", 0))
    blocks    = int(stats.get("blocks", 0))
    canvas    = int(stats.get("canvas", 0))
    # CLI-matching rates; wall_ms is the CLI's end-to-end measure (incl. visualization).
    eff_tps   = rate(canvas * blocks, wall_ms)   # effective: total canvas tokens generated / wall
    par_tps   = rate(canvas * steps,  wall_ms)   # in-step parallel: canvas / per_step (per_step = wall/steps)
    out_tps   = rate(G, wall_ms)                 # committed answer tokens the user received / wall
    timings = {
        # Headline "Speed" = in-step parallel rate (the model's real per-step throughput, ~thousands tok/s,
        # matching the CLI), not the old bogus 14000.
        "predicted_n": G,
        "predicted_ms": wall_ms,
        "predicted_per_second": par_tps,
        "cache_n": 0,
        # Diffusion-specific breakdown (Studio shows these in a dedicated tooltip section).
        "diffusion": True,
        "diffusion_blocks": blocks,
        "diffusion_steps": steps,
        "diffusion_canvas": canvas,
        "diffusion_prompt_n": P,
        "diffusion_prompt_prepare_ms": prep_ms,
        "diffusion_decode_ms": decode_ms,
        "diffusion_wall_ms": wall_ms,
        "diffusion_effective_tok_s": eff_tps,   # canvas*blocks / wall  (CLI "throughput")
        "diffusion_parallel_tok_s": par_tps,    # canvas / per_step      (CLI "in-step parallel")
        "diffusion_output_tok_s": out_tps,      # committed answer tokens / wall
        "diffusion_steps_per_second": rate(steps, wall_ms),
    }
    return timings, P, G


def _overflow_message(exc):
    return (
        f"This conversation needs {exc.needed} tokens but DiffusionGemma is running with a "
        f"{exc.budget}-token context. Start a new chat or shorten your message."
    )


def _max_blocks(body):
    mt = body.get("max_tokens") or body.get("max_completion_tokens") or 2048
    try:
        mt = int(mt)
    except (TypeError, ValueError):
        mt = 2048
    return max(1, math.ceil(mt / V.CANVAS))


def _artifact(frames):
    """Build the ```html canvas-player artifact from the collected denoising frames."""
    if not frames:
        return ""
    # embed the frames as JSON in a non-executing <script>; escape </ so it cannot close the tag early
    payload = json.dumps(frames, ensure_ascii=False).replace("</", "<\\/")
    doc = _STATE["player"].replace("__FRAMES_JSON__", payload)
    return "\n\n```html\n" + doc + "\n```\n"


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
    seed = int(body.get("seed", 3407))
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    srv = _STATE["server"]
    loop = asyncio.get_event_loop()

    if not stream:
        stats_box = {}
        def work():
            with _LOCK:
                return V.generate_visual(srv, messages, seed=seed, max_blocks=max_blocks,
                                         on_stats=stats_box.update)
        try:
            text = await loop.run_in_executor(None, work)
        except V.ContextOverflow as exc:
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": _overflow_message(exc)},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": exc.needed, "completion_tokens": 0, "total_tokens": exc.needed},
            })
        _, P, G = _timings_from_stats(stats_box)
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": P, "completion_tokens": G, "total_tokens": P + G},
        })

    async def gen():
        q: asyncio.Queue = asyncio.Queue()
        frames = []        # retained only for the optional replay artifact (DG_ARTIFACT=1)
        stats_box = {}     # the server's end-of-request STATS summary

        def on_frame(block, step, total, text):
            # Stream each frame (transient, not committed text); the frontend renders the canvas in
            # place and drops it on commit.
            loop.call_soon_threadsafe(q.put_nowait, ("frame", (block, step, total, text)))
            if _WANT_ARTIFACT:
                frames.append({"b": block, "s": step, "t": total, "x": text})

        def on_commit(cumulative):
            loop.call_soon_threadsafe(q.put_nowait, ("delta", cumulative))

        def work():
            try:
                with _LOCK:
                    full = V.generate_visual(srv, messages, seed=seed, max_blocks=max_blocks,
                                             on_frame=on_frame, on_commit=on_commit,
                                             on_stats=stats_box.update)
                loop.call_soon_threadsafe(q.put_nowait, ("done", full))
            except V.ContextOverflow as exc:  # context budget exceeded -> clean user-facing message
                loop.call_soon_threadsafe(q.put_nowait, ("overflow", exc))
            except Exception as exc:  # surface decoder errors into the stream
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        yield _sse(_chunk(cid, created, {"role": "assistant"}))
        sent = ""
        while True:
            kind, payload = await q.get()
            if kind == "frame":
                # type-tagged event on the tool_status channel; no assistant content, so plain
                # OpenAI clients ignore it and still get the committed text.
                block, step, total, text = payload
                yield _sse({"type": "diffusion_frame", "block": block, "step": step,
                            "total": total, "text": text})
                continue
            if kind in ("delta", "done"):
                new = payload[len(sent):]
                if new:
                    # committed block: normal assistant text (what the transcript keeps).
                    yield _sse(_chunk(cid, created, {"content": new}))
                    sent = payload
                if kind == "done":
                    if _WANT_ARTIFACT:
                        art = _artifact(frames)
                        if art:
                            yield _sse(_chunk(cid, created, {"content": art}))
                    # llama-server-style usage+timings chunk (empty choices) so Studio shows the
                    # stat tooltip plus the diffusion rows (steps, blocks).
                    timings, P, G = _timings_from_stats(stats_box)
                    yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                                "model": MODEL_ID, "choices": [],
                                "usage": {"prompt_tokens": P, "completion_tokens": G,
                                          "total_tokens": P + G},
                                "timings": timings})
                    yield _sse(_chunk(cid, created, {}, finish="stop"))
                    yield "data: [DONE]\n\n"
                    return
            elif kind == "overflow":
                yield _sse(_chunk(cid, created, {"content": "\n[" + _overflow_message(payload) + "]"},
                                  finish="length"))
                yield "data: [DONE]\n\n"
                return
            else:  # error
                yield _sse(_chunk(cid, created, {"content": f"\n[engine error: {payload}]"}, finish="stop"))
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(gen(), media_type="text/event-stream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--gpu", default=os.environ.get("DG_GPU", "0"))
    ap.add_argument("--maxtok", type=int, default=0,
                    help="per-turn context budget; 0 = auto-size the largest that fits VRAM")
    args = ap.parse_args()

    _STATE["player"] = open(_PLAYER_TEMPLATE).read()
    print(f"loading {args.gguf} on GPU {args.gpu} (optimized visual decoder) ...", flush=True)
    _STATE["server"] = V.VisualServer(args.gguf, gpu=args.gpu, maxtok=args.maxtok)
    print(f"DiffusionGemma OpenAI shim ready on http://{args.host}:{args.port}  (model={MODEL_ID})",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
