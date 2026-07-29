"""Unified model loader for R2 protein generation models.

Loads decoder-only protein generators (ProtGPT2, ZymCTRL, InstructProtein, ProGen2)
and provides a common interface for:
  - Text/sequence generation
  - Architecture-specific CLT-input extraction (for CLT training)
  - MLP input/output hook extraction
"""

import os
import torch
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import modeling_utils as _transformers_modeling_utils

# Use HF mirror for trust_remote_code downloads (ProGen2 etc.)
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


_ORIG_CACHING_ALLOCATOR_WARMUP = _transformers_modeling_utils.caching_allocator_warmup

INFERENCE_DTYPE_VERIFICATION = (
    "all_floating_model_parameters_exactly_declared_before_first_activation"
)
_INFERENCE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _safe_caching_allocator_warmup(model, expanded_device_map, hf_quantizer):
    # Some local ProGen2 checkpoints expose _tp_plan = None in older Transformers builds.
    if getattr(model, "_tp_plan", None) is None:
        return
    return _ORIG_CACHING_ALLOCATOR_WARMUP(model, expanded_device_map, hf_quantizer)


_transformers_modeling_utils.caching_allocator_warmup = _safe_caching_allocator_warmup

# Allow overriding model base directory via env (for container deployments)
_MODEL_BASE = os.environ.get(
    "R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models"
)

MODEL_REGISTRY = {
    "protgpt2": {
        "path": f"{_MODEL_BASE}/ProtGPT2",
        "arch": "gpt2",
        "description": "Unconditional protein generator (GPT-2, 738M)",
    },
    "zymctrl": {
        "path": f"{_MODEL_BASE}/ZymCTRL",
        "arch": "gpt2",
        "description": "EC-conditioned enzyme generator (CTRL-based, 738M)",
    },
    "instructprotein": {
        "path": f"{_MODEL_BASE}/InstructProtein",
        "arch": "opt",
        "description": "Text-to-protein generator (OPT-1.3B)",
    },
    "progen2-xlarge": {
        "path": f"{_MODEL_BASE}/progen2-xlarge",
        "arch": "progen2",
        "description": "De novo protein generator (6.4B)",
    },
    "progen2-medium": {
        "path": f"{_MODEL_BASE}/progen2-medium",
        "arch": "progen2",
        "description": "De novo protein generator (764M, dev model)",
    },
}


@dataclass
class ActivationCache:
    """Holds extracted activations from a forward pass.

    ``resid_pre`` and ``resid_post`` are legacy API names. For the model panel
    used in Paper A, ``resid_pre`` is the architecture-specific normalized CLT
    input, and ``resid_post`` is only its algebraic sum with ``mlp_out``; the
    latter is not guaranteed to be an actual transformer residual state.
    """

    resid_pre: list[torch.Tensor]  # legacy name: architecture-specific CLT input
    mlp_out: list[torch.Tensor]  # MLP output at each layer
    resid_post: list[torch.Tensor]  # legacy algebraic sum; not a model-state guarantee

    @property
    def clt_input(self) -> list[torch.Tensor]:
        """Return the CLT encoder inputs under evidence-correct terminology."""
        return self.resid_pre


def inference_dtype(name: str) -> torch.dtype:
    """Resolve a declared frozen-model inference dtype fail-closed."""

    try:
        return _INFERENCE_DTYPES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported model inference dtype: {name!r}") from error


def verify_frozen_model_inference_dtype(
    protein_model: "ProteinModel",
    declared: str,
) -> dict[str, object]:
    """Verify every floating model parameter uses the declared inference dtype."""

    expected = inference_dtype(declared)
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in protein_model.model.parameters()
            if parameter.is_floating_point()
        }
    )
    if not observed:
        raise ValueError("frozen model exposes no floating parameters")
    if observed != [str(expected).removeprefix("torch.")]:
        raise ValueError(
            "frozen model parameter dtype disagrees with the declared inference "
            f"dtype: declared={declared}, observed={observed}"
        )
    return {
        "model_inference_dtype": declared,
        "observed_model_parameter_dtypes": observed,
        "model_inference_dtype_verification": INFERENCE_DTYPE_VERIFICATION,
        "model_inference_dtype_verified": True,
    }


def assert_finite_captured_activations(cache: ActivationCache) -> None:
    """Reject any non-finite captured CLT input or MLP output before casting."""

    for name, tensors in (("CLT-input", cache.clt_input), ("MLP-output", cache.mlp_out)):
        for layer, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} activation at layer {layer} is not a tensor")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(
                    f"non-finite frozen-model {name} activation at layer {layer}"
                )


class ProteinModel:
    """Wrapper around HuggingFace causal LM for protein generation with hooks."""

    def __init__(self, model, tokenizer, config, model_name: str, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_name = model_name
        self.device = device
        # Handle varying config attribute names across architectures
        for attr in ("n_layer", "num_hidden_layers"):
            if hasattr(config, attr):
                self.n_layers = getattr(config, attr)
                break
        for attr in ("n_embd", "hidden_size", "embed_dim"):
            if hasattr(config, attr):
                self.d_model = getattr(config, attr)
                break

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate protein sequence from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ActivationCache:
        """Extract architecture-specific CLT inputs and MLP outputs by layer.

        ``resid_pre`` is retained as a compatibility name. In ProtGPT2 and
        ZymCTRL it is the layer-normalized post-attention MLP input. In the
        local ProGen2 implementation it is the layer-normalized block input
        shared by attention and the MLP. ``resid_post`` is the algebraic value
        ``resid_pre + mlp_out`` and must not be interpreted as a captured raw
        residual state for these normalized hook points.

        Args:
            input_ids: Token IDs with shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional valid-token mask with the same leading shape.

        Returns:
            ActivationCache with per-layer tensors of shape (batch, seq_len, d_model)
        """
        resid_pre = [None] * self.n_layers
        mlp_out = [None] * self.n_layers
        hooks = []

        def capture(slots, layer_idx, value, name):
            if slots[layer_idx] is not None:
                raise RuntimeError(f"duplicate {name} capture for layer {layer_idx}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} hook for layer {layer_idx} returned non-tensor"
                )
            slots[layer_idx] = value.detach()

        arch = self._get_arch_type()

        if arch == "gpt2":
            # GPT-2: MLP is a single submodule; hook its input/output
            def make_mlp_hook(layer_idx):
                def hook_fn(module, input, output):
                    capture(resid_pre, layer_idx, input[0], "CLT-input")
                    capture(mlp_out, layer_idx, output, "MLP-output")

                return hook_fn

            for i in range(self.n_layers):
                block = self._get_block(i)
                mlp = block.mlp
                h = mlp.register_forward_hook(make_mlp_hook(i))
                hooks.append(h)

        elif arch == "opt":
            # OPT: MLP = final_layer_norm -> fc1 -> activation -> fc2
            def make_fc2_hook(layer_idx):
                def hook_fn(module, input, output):
                    # OPT is not part of the Paper A model panel. This legacy
                    # path captures the MLP-side tensors used by the CLT. OPT does:
                    #   residual = post_attn_hidden
                    #   hidden = final_layer_norm(residual)
                    #   hidden = fc1(hidden) -> act -> fc2(hidden)
                    #   output = residual + hidden
                    # So mlp_out = fc2 output; the compatibility field name is
                    # retained below.
                    capture(mlp_out, layer_idx, output, "MLP-output")

                return hook_fn

            # Use a different strategy for the MLP-side input and output.
            hooks.clear()

            def make_ln_hook(layer_idx):
                def hook_fn(module, input, output):
                    # final_layer_norm input (legacy compatibility path)
                    capture(resid_pre, layer_idx, input[0], "CLT-input")

                return hook_fn

            for i in range(self.n_layers):
                block = self._get_block(i)
                # Hook final_layer_norm to get pre-MLP residual
                h1 = block.final_layer_norm.register_forward_hook(make_ln_hook(i))
                # Hook fc2 to get MLP output
                h2 = block.fc2.register_forward_hook(make_fc2_hook(i))
                hooks.extend([h1, h2])

        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if attention_mask.shape != input_ids.shape:
                    raise ValueError(
                        "attention_mask must have the same shape as input_ids; "
                        f"got {attention_mask.shape} and {input_ids.shape}"
                    )
            model_inputs = {"input_ids": input_ids.to(self.device)}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask.to(self.device)
            self.model(**model_inputs)
        finally:
            for h in hooks:
                h.remove()

        expected_shape = tuple(input_ids.shape)
        for layer_idx, (inputs, outputs) in enumerate(zip(resid_pre, mlp_out)):
            if inputs is None or outputs is None:
                raise RuntimeError(f"missing activation capture for layer {layer_idx}")
            if (
                inputs.ndim != 3
                or outputs.ndim != 3
                or tuple(inputs.shape[:2]) != expected_shape
                or tuple(outputs.shape[:2]) != expected_shape
                or inputs.shape[-1] != self.d_model
                or outputs.shape[-1] != self.d_model
            ):
                raise ValueError(
                    f"activation shape mismatch at layer {layer_idx}: "
                    f"CLT-input={tuple(inputs.shape)}, MLP-output={tuple(outputs.shape)}, "
                    f"tokens={expected_shape}, d_model={self.d_model}"
                )

        # Legacy compatibility value only. Because resid_pre is normalized for
        # the Paper A models, this is not an actual captured residual state.
        resid_post = [resid_pre[i] + mlp_out[i] for i in range(self.n_layers)]

        return ActivationCache(
            resid_pre=resid_pre,
            mlp_out=mlp_out,
            resid_post=resid_post,
        )

    def _get_arch_type(self) -> str:
        """Detect model architecture type."""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return "gpt2"  # GPT-2, ProGen2, ZymCTRL all use this structure
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            return "opt"  # OPT, InstructProtein
        else:
            raise ValueError(f"Unknown architecture for {self.model_name}")

    def _get_block(self, layer_idx):
        """Get transformer block by index."""
        arch = self._get_arch_type()
        if arch == "gpt2":
            return self.model.transformer.h[layer_idx]
        elif arch == "opt":
            return self.model.model.decoder.layers[layer_idx]

    def _get_mlp(self, block):
        """Get MLP module from a transformer block."""
        if hasattr(block, "mlp"):
            return block.mlp
        elif hasattr(block, "fc2"):
            return block.fc2  # For OPT, return fc2 as the "MLP output" module
        else:
            raise ValueError(f"Cannot find MLP in block: {type(block)}")

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids tensor."""
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    device_map: str | None = None,
) -> ProteinModel:
    """Load a protein generation model by name.

    Args:
        model_name: Key from MODEL_REGISTRY or path to model directory
        device: Target device
        dtype: Weight dtype (float16 recommended for L20)
        device_map: Optional device_map for large models (e.g., "auto")

    Returns:
        ProteinModel wrapper with generation and activation extraction
    """
    if model_name in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model_name]
        model_path = info["path"]
    else:
        model_path = model_name

    model_path = str(Path(model_path).resolve())
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    def _get_cfg_val(cfg, *keys):
        for k in keys:
            if hasattr(cfg, k):
                return getattr(cfg, k)
        raise AttributeError(f"Config missing all of: {keys}")

    n_layers = _get_cfg_val(config, "n_layer", "num_hidden_layers")
    d_model = _get_cfg_val(config, "n_embd", "hidden_size", "embed_dim")

    print(f"Loading {model_name} from {model_path}...")
    print(f"  Architecture: {config.architectures}")
    print(f"  Layers: {n_layers}")
    print(f"  d_model: {d_model}")

    # Cluster transformers builds accept torch_dtype, not dtype.
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device_map:
        load_kwargs["device_map"] = device_map
    else:
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    # Patch configs that use n_layer instead of num_hidden_layers (ProGen2)
    if hasattr(config, "n_layer") and not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.n_layer
        model.config.num_hidden_layers = config.n_layer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ProteinModel(model, tokenizer, config, model_name, device)
