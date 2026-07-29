"""Circuit discovery and attribution for protein generation models.

Uses trained CLTs to trace computational circuits from conditioning signals
(EC numbers, text prompts) to generated amino acid choices.

Two modes:
  1. Direct attribution using our CLTForTraining (fast, in-house)
  2. circuit-tracer library attribution (full graph visualization)
"""

import sys
from pathlib import Path

import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.model_loader import load_model, ProteinModel
from src.training.clt_trainer import CLTForTraining


def load_trained_clt(checkpoint_dir: str, device: str = "cuda") -> CLTForTraining:
    """Load a trained CLT from checkpoint."""
    ckpt_dir = Path(checkpoint_dir)
    with open(ckpt_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Infer n_layers and d_model from saved state dict
    state = torch.load(ckpt_dir / "clt.pt", map_location=device)
    n_layers = state["W_enc"].shape[0]
    d_model = state["W_enc"].shape[2]

    clt_cfg = config.get("clt", {})
    clt = CLTForTraining(
        n_layers=n_layers,
        d_model=d_model,
        d_clt=clt_cfg["d_clt"],
        k=clt_cfg["k"],
        window=clt_cfg.get("window", 8),
    )
    clt.load_state_dict(state)
    clt.to(device).eval()
    return clt


@torch.no_grad()
def compute_feature_attribution(
    protein_model: ProteinModel,
    clt: CLTForTraining,
    input_ids: torch.Tensor,
    target_positions: list[int] | None = None,
) -> dict:
    """Compute which CLT features most influence specific output positions.

    For each target position, computes the contribution of each feature at
    each layer to the logits at that position, by tracing through the CLT
    decode pathway.

    Args:
        protein_model: Frozen protein generation model
        clt: Trained CLT
        input_ids: Tokenized input (1, seq_len)
        target_positions: Which output positions to analyze (default: all)

    Returns:
        Dict with:
          - features: List of per-layer feature activations (n_layers, seq, d_clt)
          - attribution: (n_targets, n_layers, d_clt) contribution scores
          - top_features: List of (layer, feature_idx, score) for each target
    """
    cache = protein_model.get_activations(input_ids)
    resid_pre = [x.float() for x in cache.resid_pre]

    # Encode to get features
    features = clt.encode(resid_pre)

    if target_positions is None:
        seq_len = input_ids.shape[1]
        target_positions = list(range(seq_len))

    # For each target position, measure each feature's contribution
    # via the last layer's decoder
    n_layers = clt.n_layers
    last_layer = n_layers - 1
    attributions = []

    for pos in target_positions:
        pos_attr = torch.zeros(n_layers, clt.d_clt, device=input_ids.device)

        for l in range(n_layers):
            # Feature activations at this layer and position
            feat_vals = features[l][0, pos]  # (d_clt,)

            # Decoder vectors from layer l to last layer (within window)
            t_offset = last_layer - l
            if 0 <= t_offset < clt.W_dec[l].shape[1]:
                dec_vecs = clt.W_dec[l][:, t_offset, :]  # (d_clt, d_model)
                contribution = feat_vals * dec_vecs.norm(dim=-1)
                pos_attr[l] = contribution
            elif t_offset >= 0:
                # Beyond window — use same-layer decoder as proxy
                dec_vecs = clt.W_dec[l][:, 0, :]
                contribution = feat_vals * dec_vecs.norm(dim=-1) * 0.1
                pos_attr[l] = contribution

        attributions.append(pos_attr)

    attributions = torch.stack(attributions)  # (n_targets, n_layers, d_clt)

    # Top features per target position
    top_features = []
    for t in range(len(target_positions)):
        flat = attributions[t].flatten()
        topk_vals, topk_idx = flat.topk(min(20, flat.numel()))
        layer_idx = topk_idx // clt.d_clt
        feat_idx = topk_idx % clt.d_clt
        top_features.append([
            (layer_idx[i].item(), feat_idx[i].item(), topk_vals[i].item())
            for i in range(len(topk_vals))
        ])

    return {
        "features": features,
        "attribution": attributions,
        "top_features": top_features,
        "target_positions": target_positions,
    }


@torch.no_grad()
def compare_circuits(
    protein_model: ProteinModel,
    clt: CLTForTraining,
    prompt_a: str,
    prompt_b: str,
) -> dict:
    """Compare circuits activated by two different prompts.

    Useful for questions like:
      - How does the model treat EC 3.2.1.17 vs EC 1.1.1.1?
      - How does "inhibits KRAS" differ from "activates KRAS"?

    Returns:
        Dict with per-layer feature activation differences.
    """
    ids_a = protein_model.tokenize(prompt_a)
    ids_b = protein_model.tokenize(prompt_b)

    cache_a = protein_model.get_activations(ids_a)
    cache_b = protein_model.get_activations(ids_b)

    feats_a = clt.encode([x.float() for x in cache_a.resid_pre])
    feats_b = clt.encode([x.float() for x in cache_b.resid_pre])

    # Compare mean feature activations across all positions
    diff_per_layer = []
    shared_per_layer = []

    for l in range(clt.n_layers):
        mean_a = feats_a[l].mean(dim=(0, 1))  # (d_clt,)
        mean_b = feats_b[l].mean(dim=(0, 1))

        diff = (mean_a - mean_b).abs()
        shared = (mean_a > 0) & (mean_b > 0)

        diff_per_layer.append(diff)
        shared_per_layer.append(shared.float().mean().item())

    # Top differential features
    diff_all = torch.stack(diff_per_layer)  # (n_layers, d_clt)
    flat = diff_all.flatten()
    topk_vals, topk_idx = flat.topk(min(50, flat.numel()))
    layer_idx = topk_idx // clt.d_clt
    feat_idx = topk_idx % clt.d_clt

    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "diff_per_layer": diff_per_layer,
        "shared_fraction": shared_per_layer,
        "top_differential_features": [
            (layer_idx[i].item(), feat_idx[i].item(), topk_vals[i].item())
            for i in range(len(topk_vals))
        ],
    }


@torch.no_grad()
def steer_generation(
    protein_model: ProteinModel,
    clt: CLTForTraining,
    prompt: str,
    interventions: list[tuple[int, int, float]],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate with feature steering via activation patching.

    Modifies MLP outputs during generation by amplifying or suppressing
    specific CLT features.

    Args:
        protein_model: The protein generation model
        clt: Trained CLT
        prompt: Starting prompt/sequence
        interventions: List of (layer, feature_idx, multiplier) tuples.
            multiplier > 1 amplifies, < 1 suppresses, 0 ablates.
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated sequence with steering applied.
    """
    input_ids = protein_model.tokenize(prompt)
    hooks = []

    def make_steering_hook(layer_idx, feature_interventions):
        """Create a TopK-aware hook for on-manifold CLT feature steering."""
        def hook_fn(module, input, output):
            resid = input[0].float()
            pre_act = torch.einsum("bsd,fd->bsf", resid, clt.W_enc[layer_idx]) + clt.b_enc[layer_idx]
            pre_act = torch.relu(pre_act)

            def apply_topk(x: torch.Tensor) -> torch.Tensor:
                k = min(clt.k, x.shape[-1])
                vals, idx = x.topk(k, dim=-1)
                sparse = torch.zeros_like(x)
                sparse.scatter_(-1, idx, vals)
                return sparse

            sparse_unsteered = apply_topk(pre_act)
            steered_pre_act = pre_act.clone()
            for feat_idx, multiplier in feature_interventions:
                steered_pre_act[:, :, feat_idx] = steered_pre_act[:, :, feat_idx] * multiplier
            sparse_steered = apply_topk(steered_pre_act)

            # Replace the CLT-explained same-layer component instead of adding an
            # off-manifold single-feature delta.
            dec0 = clt.W_dec[layer_idx][:, 0, :]
            explained_unsteered = torch.einsum("bsf,fd->bsd", sparse_unsteered, dec0)
            explained_steered = torch.einsum("bsf,fd->bsd", sparse_steered, dec0)
            delta = explained_steered - explained_unsteered
            return output + delta.to(output.dtype)
        return hook_fn

    # Group interventions by layer
    layer_interventions = {}
    for layer, feat_idx, multiplier in interventions:
        if layer not in layer_interventions:
            layer_interventions[layer] = []
        layer_interventions[layer].append((feat_idx, multiplier))

    # Register hooks
    for layer_idx, feat_intervs in layer_interventions.items():
        block = protein_model._get_block(layer_idx)
        mlp = protein_model._get_mlp(block)
        h = mlp.register_forward_hook(make_steering_hook(layer_idx, feat_intervs))
        hooks.append(h)

    try:
        outputs = protein_model.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            pad_token_id=protein_model.tokenizer.eos_token_id,
        )
        result = protein_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return result
