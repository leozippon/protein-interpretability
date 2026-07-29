#!/usr/bin/env python
"""End-to-end pipeline test: model → CLT → attribution → steering.

Runs a quick validation of the full R2 pipeline on a single GPU.
Uses a small CLT (reduced d_clt) and few training steps for speed.

Usage:
  python scripts/02_test_pipeline.py --model protgpt2 --gpu 0
  python scripts/02_test_pipeline.py --model zymctrl --gpu 1
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


def main():
    parser = argparse.ArgumentParser(description="End-to-end R2 pipeline test")
    parser.add_argument("--model", type=str, default="protgpt2",
                        help="Model name from MODEL_REGISTRY")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--d-clt", type=int, default=4096,
                        help="CLT dict size (small for testing)")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--window", type=int, default=4, help="Decode window")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps (small for testing)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    from src.models.model_loader import load_model
    from src.training.clt_trainer import CLTForTraining
    from src.analysis.circuit_discovery import (
        compute_feature_attribution,
        compare_circuits,
        steer_generation,
    )

    # ==============================
    # Phase 0: Model Loading & Generation
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 0: Model Loading & Generation")
    print("="*60)

    model = load_model(args.model, device=device)
    print(f"\nModel loaded: {args.model}")
    print(f"  Layers: {model.n_layers}, d_model: {model.d_model}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate baseline sequences
    if args.model == "zymctrl":
        prompts = ["3.2.1.17", "1.1.1.1"]  # lysozyme, alcohol dehydrogenase
        prompt_names = ["lysozyme (EC 3.2.1.17)", "ADH (EC 1.1.1.1)"]
    else:
        prompts = ["M", "MKWVTFISLLFLFSSAYS"]
        prompt_names = ["minimal (M)", "signal peptide start"]

    print("\nBaseline generation:")
    for prompt, name in zip(prompts, prompt_names):
        output = model.generate(prompt, max_new_tokens=50, temperature=0.8)
        print(f"  {name}: {output[:100]}...")

    # ==============================
    # Phase 1: Activation Extraction
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 1: Activation Extraction")
    print("="*60)

    input_ids = model.tokenize(prompts[0])
    cache = model.get_activations(input_ids)
    print(f"\nActivations extracted for '{prompts[0]}':")
    print(f"  Layers: {len(cache.resid_pre)}")
    print(f"  resid_pre shape: {cache.resid_pre[0].shape}")
    print(f"  mlp_out shape: {cache.mlp_out[0].shape}")

    # Activation norms across layers
    print("\n  Layer activation norms:")
    for l in range(0, model.n_layers, model.n_layers // 6):
        rp = cache.resid_pre[l].norm(dim=-1).mean().item()
        mo = cache.mlp_out[l].norm(dim=-1).mean().item()
        print(f"    L{l:>2d}: resid_pre={rp:.1f}, mlp_out={mo:.1f}")

    # ==============================
    # Phase 2: CLT Training (mini)
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 2: CLT Training (mini)")
    print("="*60)

    clt = CLTForTraining(
        n_layers=model.n_layers,
        d_model=model.d_model,
        d_clt=args.d_clt,
        k=args.k,
        window=args.window,
    ).to(device).float()

    print(f"\nCLT created: d_clt={args.d_clt}, k={args.k}, window={args.window}")
    clt_params = sum(p.numel() for p in clt.parameters())
    print(f"  CLT parameters: {clt_params:,}")
    print(f"  VRAM after CLT: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    optimizer = torch.optim.Adam(clt.parameters(), lr=3e-4)

    # Load a small set of sequences for training
    fasta_path = os.environ.get(
        "R2_FASTA_PATH",
        "/Data/lzp/BioInterpretebility-CC/data/uniref50/uniref50.fasta",
    )
    sequences = []
    print(f"\nLoading sequences from {fasta_path}...")
    with open(fasta_path) as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    if 20 <= len(seq) <= 256:
                        sequences.append(seq)
                        if len(sequences) >= 1000:
                            break
                current_seq = []
            else:
                current_seq.append(line)
    print(f"  Loaded {len(sequences)} sequences")

    print(f"\nTraining CLT for {args.steps} steps...")
    t_start = time.time()

    for step in range(1, args.steps + 1):
        # Get a batch
        idx = (step * 2) % len(sequences)
        batch_seqs = sequences[idx:idx+2]
        tokens = model.tokenizer(
            batch_seqs, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        input_ids = tokens["input_ids"].to(device)

        # Extract activations
        with torch.no_grad():
            cache = model.get_activations(input_ids)

        resid_pre = [x.float() for x in cache.resid_pre]
        mlp_out = [x.float() for x in cache.mlp_out]

        # CLT forward + backward
        result = clt(resid_pre, mlp_out)
        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(clt.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            print(
                f"  Step {step:>4d}/{args.steps} | "
                f"loss={result['loss'].item():.4f} | "
                f"FVU={result['fvu_mean']:.4f} | "
                f"L0={result['l0_mean']:.1f} | "
                f"{elapsed/step:.2f}s/step"
            )

    elapsed = time.time() - t_start
    print(f"\nTraining complete: {args.steps} steps in {elapsed:.1f}s")
    print(f"  Final FVU: {result['fvu_mean']:.4f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ==============================
    # Phase 3: Feature Attribution
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 3: Feature Attribution")
    print("="*60)

    clt.eval()
    test_input = model.tokenize(prompts[0])
    attr_result = compute_feature_attribution(
        model, clt, test_input,
        target_positions=list(range(min(5, test_input.shape[1]))),
    )

    print(f"\nAttribution for '{prompts[0]}':")
    print(f"  Target positions: {attr_result['target_positions']}")
    for t, (pos, top_feats) in enumerate(zip(
        attr_result["target_positions"], attr_result["top_features"]
    )):
        print(f"  Position {pos} — top 5 features:")
        for layer, feat, score in top_feats[:5]:
            print(f"    Layer {layer:>2d}, Feature {feat:>5d}: score={score:.4f}")

    # ==============================
    # Phase 4: Circuit Comparison
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 4: Circuit Comparison")
    print("="*60)

    if len(prompts) >= 2:
        comp = compare_circuits(model, clt, prompts[0], prompts[1])
        print(f"\nComparing: '{prompts[0]}' vs '{prompts[1]}'")
        print(f"  Feature overlap per layer (fraction shared):")
        for l in range(0, model.n_layers, model.n_layers // 6):
            print(f"    Layer {l:>2d}: {comp['shared_fraction'][l]:.3f}")
        print(f"  Top 10 differential features:")
        for layer, feat, score in comp["top_differential_features"][:10]:
            print(f"    Layer {layer:>2d}, Feature {feat:>5d}: diff={score:.4f}")

    # ==============================
    # Phase 5: Steering
    # ==============================
    print("\n" + "="*60)
    print("  PHASE 5: Feature Steering")
    print("="*60)

    # Pick top feature from attribution and try amplifying/suppressing
    if attr_result["top_features"][0]:
        top_layer, top_feat, _ = attr_result["top_features"][0][0]

        print(f"\nSteering feature: Layer {top_layer}, Feature {top_feat}")

        # Baseline
        baseline = model.generate(prompts[0], max_new_tokens=50, temperature=0.8)
        print(f"  Baseline:    {baseline[:100]}...")

        # Amplify
        amplified = steer_generation(
            model, clt, prompts[0],
            interventions=[(top_layer, top_feat, 3.0)],
            max_new_tokens=50, temperature=0.8,
        )
        print(f"  Amplified:   {amplified[:100]}...")

        # Suppress
        suppressed = steer_generation(
            model, clt, prompts[0],
            interventions=[(top_layer, top_feat, 0.0)],
            max_new_tokens=50, temperature=0.8,
        )
        print(f"  Suppressed:  {suppressed[:100]}...")

    # ==============================
    # Summary
    # ==============================
    print("\n" + "="*60)
    print("  PIPELINE TEST COMPLETE")
    print("="*60)
    print(f"\n  Model: {args.model}")
    print(f"  d_clt={args.d_clt}, k={args.k}, window={args.window}, steps={args.steps}")
    print(f"  Final FVU: {result['fvu_mean']:.4f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  All phases passed!")


if __name__ == "__main__":
    main()
