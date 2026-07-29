#!/usr/bin/env python
"""Test model loading, generation, and activation extraction for R2 models.

Usage:
  python scripts/00_test_models.py --model protgpt2
  python scripts/00_test_models.py --model zymctrl
  python scripts/00_test_models.py --model protgpt2 --gpu 1
  python scripts/00_test_models.py --all
"""

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from src.models.model_loader import load_model, MODEL_REGISTRY


def test_model(model_name: str, device: str = "cuda"):
    """Test a single model: load, generate, extract activations."""
    print(f"\n{'='*60}")
    print(f"  Testing: {model_name}")
    print(f"{'='*60}\n")

    # Check if model exists
    if model_name in MODEL_REGISTRY:
        model_path = MODEL_REGISTRY[model_name]["path"]
        if not Path(model_path).exists():
            print(f"  SKIP: Model not yet downloaded at {model_path}")
            return False

    # Load
    t0 = time.time()
    model = load_model(model_name, device=device)
    print(f"  Load time: {time.time()-t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Generate
    print("\n--- Generation Test ---")
    if model_name == "zymctrl":
        prompt = "3.2.1.17"  # EC number for lysozyme
    elif model_name == "instructprotein":
        prompt = "Generate a protein that binds to KRAS G12D mutant"
    else:
        prompt = "M"  # Start with methionine

    t0 = time.time()
    output = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    gen_time = time.time() - t0
    print(f"  Prompt: '{prompt}'")
    print(f"  Output ({gen_time:.2f}s): {output[:200]}...")

    # Activation extraction
    print("\n--- Activation Extraction Test ---")
    input_ids = model.tokenize(prompt)
    print(f"  Input shape: {input_ids.shape}")

    t0 = time.time()
    cache = model.get_activations(input_ids)
    act_time = time.time() - t0
    print(f"  Extraction time: {act_time:.2f}s")
    print(f"  resid_pre layers: {len(cache.resid_pre)}")
    print(f"  mlp_out layers:   {len(cache.mlp_out)}")
    print(f"  legacy algebraic resid_post layers: {len(cache.resid_post)}")

    if cache.resid_pre:
        print(f"  resid_pre[0] shape: {cache.resid_pre[0].shape}")
        print(f"  mlp_out[0] shape:   {cache.mlp_out[0].shape}")

        # Verify the legacy algebraic compatibility field.
        diff = (cache.resid_post[0] - cache.resid_pre[0] - cache.mlp_out[0]).abs().max()
        print(f"  legacy algebraic sum check max diff: {diff:.2e}")

        # Stats
        for l in [0, len(cache.resid_pre)//2, len(cache.resid_pre)-1]:
            rp_norm = cache.resid_pre[l].norm(dim=-1).mean().item()
            mo_norm = cache.mlp_out[l].norm(dim=-1).mean().item()
            print(f"  Layer {l:>2d}: resid_pre norm={rp_norm:.2f}, mlp_out norm={mo_norm:.2f}")

    # Memory after
    print(f"\n  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"\n  PASS: {model_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test R2 protein generation models")
    parser.add_argument("--model", type=str, default="protgpt2",
                        help=f"Model name: {', '.join(MODEL_REGISTRY.keys())}")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    if args.all:
        results = {}
        for name in MODEL_REGISTRY:
            try:
                results[name] = test_model(name, device)
            except Exception as e:
                print(f"\n  FAIL: {name} — {e}")
                results[name] = False
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        for name, ok in results.items():
            status = "PASS" if ok else "FAIL/SKIP"
            print(f"  {name:>20s}: {status}")
    else:
        test_model(args.model, device)


if __name__ == "__main__":
    main()
