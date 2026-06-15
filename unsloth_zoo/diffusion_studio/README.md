# DiffusionGemma in Unsloth Studio

DiffusionGemma is a block-diffusion model: it denoises a fixed 256-token canvas in parallel instead of
emitting tokens left to right, so llama.cpp serves it through a dedicated diffusion runner rather than
the standard autoregressive `llama-server`. This package drives the optimized on-device visual decoder
and exposes it as an OpenAI-compatible HTTP shim, so Unsloth Studio can serve DiffusionGemma as an
ordinary llama.cpp model. The shim streams the committed answer text and a self-contained ```html
artifact that replays the per-step denoising canvas; Studio auto-renders that for DiffusionGemma so you
watch the canvas resolve out of noise. It is an additive path - the autoregressive flow is untouched.

## What you need

- A DiffusionGemma GGUF (e.g. `unsloth/diffusiongemma-26B-A4B-it-GGUF`, `*Q8_0*`). The GGUF already
  embeds the tokenizer and chat template, so no separate tokenizer files are needed.
- The `llama-diffusion-gemma-visual-server` binary from the diffusiongemma llama.cpp branch
  (ggml-org/llama.cpp PR #24423). Build it:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  gh pr checkout 24423
  cmake -B build -DGGML_CUDA=ON
  cmake --build build -j --config Release --target llama-diffusion-gemma-visual-server
  ```

- `pip install fastapi uvicorn` for the shim.

## 1. Start the shim

```bash
DG_VISUAL_BIN=/path/to/llama.cpp/build/bin/llama-diffusion-gemma-visual-server \
python -m unsloth_zoo.diffusion_studio.shim \
    --gguf /path/to/diffusiongemma-26B-A4B-it-Q8_0.gguf \
    --gpu 0 --port 8123
```

The binary also resolves from `DG_VISUAL_BIN` or from `PATH`. The model loads once; requests are
serialized. It prints `... shim ready on http://127.0.0.1:8123`.

## 2. Use it from Studio

Unsloth Studio detects a DiffusionGemma GGUF (`general.architecture = diffusion-gemma`) and launches this
shim for it automatically, so selecting any quant of `unsloth/diffusiongemma-26B-A4B-it-GGUF` just works
and the denoising canvas renders inline. To wire it up by hand instead, open **Settings -> Connections**,
add a **llama.cpp** (OpenAI-compatible) connection with base URL `http://127.0.0.1:8123/v1`, then chat.

## Programmatic use

```python
from unsloth_zoo.diffusion_studio import VisualServer, generate_visual

srv = VisualServer("diffusiongemma-26B-A4B-it-Q8_0.gguf", gpu="0",
                   server_bin="/path/to/llama-diffusion-gemma-visual-server")
reply = generate_visual(srv, [{"role": "user", "content": "Hello!"}], seed=3407, max_blocks=2,
                        on_frame=lambda b, s, t, x: None)
print(reply)
srv.close()
```

## How it works

`visual_engine.py` keeps the visual server resident (model loaded once). The server runs the same
on-device entropy-bound decoder as the CLI's `--diffusion-visual` (Stage 1 device sampling + Stage 2
device-resident self-conditioning): random canvas init, then per step argmax/entropy/multinomial, accept
the lowest-entropy positions within the mutual-information bound, renoise the rest, stop when stable and
confident. Tokenization, chat templating and detokenization happen in the server from the GGUF's own
tokenizer, so it streams back the per-step canvas already decoded to text plus the committed answer.
`shim.py` wraps that as `/v1/chat/completions`, streaming committed text deltas and appending the
`canvas_player.html` artifact that animates the denoising in place.
