"""Export any trained MLX model folder to GGUF format.

Usage:
    python export_gguf.py trained_models/qlora
    python export_gguf.py trained_models/lora --quant q4_k_m
    python export_gguf.py trained_models/ft --quant q8_0
    python export_gguf.py trained_models/ft --quant not_quantized

Quantization options:
    not_quantized  - bf16, no quantization (largest, highest quality)
    fast_quantized - q8_0 (default, good balance)
    quantized      - q4_k_m (smallest, fast inference)
    Or any llama.cpp type: q2_k, q3_k_m, q4_k_m, q5_k_m, q6_k, q8_0, f16, bf16
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export a trained MLX model to GGUF")
    parser.add_argument("model_dir", help="Path to the trained model directory")
    parser.add_argument("--quant", default="fast_quantized",
                        help="Quantization method (default: fast_quantized)")
    parser.add_argument("--output", default=None,
                        help="Output directory for GGUF (default: <model_dir>_gguf)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: {model_dir} does not exist")
        sys.exit(1)

    # Check it's a valid model directory
    has_safetensors = list(model_dir.glob("*.safetensors"))
    has_config = (model_dir / "config.json").exists()
    if not has_safetensors or not has_config:
        print(f"Error: {model_dir} doesn't look like a valid model directory")
        print(f"  config.json: {'found' if has_config else 'MISSING'}")
        print(f"  safetensors: {len(has_safetensors)} file(s)")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(f"{model_dir}_gguf")

    print(f"\n{'='*50}")
    print(f"GGUF Export")
    print(f"  Model:  {model_dir}")
    print(f"  Quant:  {args.quant}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}\n")

    # Load the saved model
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(str(model_dir))

    # Stash metadata needed by save_pretrained_gguf
    import json
    with open(model_dir / "config.json") as f:
        model._config = json.load(f)
    model._hf_repo = None  # local model, no HF repo
    model._src_path = str(model_dir)

    # Export to GGUF
    from unsloth_zoo.mlx_utils import save_pretrained_gguf
    save_pretrained_gguf(model, tokenizer, str(output_dir), quantization_method=args.quant)

    # Show results
    print(f"\n{'='*50}")
    print(f"Done! GGUF files in {output_dir}/")
    for f in sorted(output_dir.glob("*.gguf")):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name}  ({size_mb:.1f} MB)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
