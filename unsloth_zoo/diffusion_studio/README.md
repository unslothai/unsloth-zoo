# DiffusionGemma in Unsloth Studio

DiffusionGemma is a block-diffusion model: it denoises a fixed 256-token canvas in parallel instead of
emitting tokens left to right, so llama.cpp serves it through a dedicated diffusion runner rather than
the standard autoregressive `llama-server`. This package exposes that runner as an OpenAI-compatible
HTTP shim, so Unsloth Studio can chat with DiffusionGemma as an ordinary llama.cpp Connection. It is an
additive path - nothing in the autoregressive flow changes.

## What you need

- A DiffusionGemma GGUF (e.g. `unsloth/diffusiongemma-26B-A4B-it-GGUF`, `*Q8_0*`).
- The `llama-diffusion-gemma-server` binary from the diffusiongemma llama.cpp branch
  (ggml-org/llama.cpp PR #24423). Build it:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  gh pr checkout 24423
  cmake -B build -DGGML_CUDA=ON
  cmake --build build -j --config Release --target llama-diffusion-gemma-server
  ```

- The model's tokenizer directory (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`).
- `pip install fastapi uvicorn jinja2 tokenizers numpy` for the shim.

## 1. Start the shim

```bash
python -m unsloth_zoo.diffusion_studio.shim \
    --gguf  /path/to/diffusiongemma-26B-A4B-it-Q8_0.gguf \
    --server-bin /path/to/llama.cpp/build/bin/llama-diffusion-gemma-server \
    --tok-dir /path/to/diffusiongemma_tokenizer \
    --gpu 0 --port 8123
```

`--server-bin` and `--tok-dir` can also come from `DG_SERVER_BIN` / `DG_TOK_DIR`. The model loads once;
requests are serialized. It prints `... shim ready on http://127.0.0.1:8123`.

## 2. Add it in Studio

In Unsloth Studio open **Settings -> Connections**, add a **llama.cpp** (OpenAI-compatible) connection
with base URL `http://127.0.0.1:8123/v1`, then pick `diffusiongemma-26B-A4B-it` in the model picker and
chat. Replies stream in per denoised block.

## Programmatic use

```python
import numpy as np
from unsloth_zoo.diffusion_studio import LlamaServer, Tok, generate

srv = LlamaServer("diffusiongemma-26B-A4B-it-Q8_0.gguf", gpu="0",
                  server_bin="/path/to/llama-diffusion-gemma-server")
tok = Tok(tok_dir="/path/to/diffusiongemma_tokenizer")
reply = generate(srv, tok, [{"role": "user", "content": "Hello!"}],
                 np.random.default_rng(), max_blocks=2)
print(reply)
srv.close()
```

## How it works

`engine.py` keeps the diffusion server resident (model loaded once) and drives it with a pure-NumPy port
of llama.cpp's entropy-bound decoder: random canvas init, then per step it reads the canvas logits,
computes argmax/entropy, samples, accepts the lowest-entropy positions within the mutual-information
bound, renoises the rest, and stops when the canvas is stable and confident. `shim.py` wraps that as
`/v1/chat/completions`, streaming each committed 256-token block as it settles.
