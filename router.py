"""
Unified Sparse Router.

A tiny learned network that predicts, per token and per layer, which FFN
neurons to activate. Bolted onto a frozen base model — only the router
trains. The training signal is reconstruction error: how close is the
sparse layer output to the full layer output?

Architecture:
  For each transformer layer's FFN:
    hidden_state → small MLP → scores over FFN neurons → top-k mask
    Only the top-k neurons are computed (gate_proj, up_proj, down_proj)
    Everything else is skipped → fewer memory reads → faster inference

The router is tiny (~1-5M params total) compared to the base model (2B+).
"""
import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM


class LayerRouter(nn.Module):
    """
    Per-layer router that predicts which FFN neurons to activate.

    Takes the hidden state before the FFN and outputs scores for each neuron.
    Top-k scoring neurons are kept, the rest are skipped.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, bottleneck: int = 128):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, ffn_dim),
        )

    def forward(self, hidden_state: Tensor, sparsity: float = 0.9) -> tuple[Tensor, Tensor]:
        """
        Predict which neurons to activate.

        Returns:
            scores: [batch, seq_len, ffn_dim] — raw neuron importance scores
            mask:   [batch, seq_len, ffn_dim] — binary mask, 1 = keep, 0 = skip
        """
        scores = self.proj(hidden_state)
        k = max(1, int(self.ffn_dim * (1 - sparsity)))
        # Top-k per token position — each token gets its own sparsity mask
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        return scores, mask


class SparseRouter(nn.Module):
    """
    Wraps a frozen base model with per-layer FFN routers.

    During training, both full and sparse FFN outputs are computed so the
    router can learn from reconstruction error. During inference, only
    the sparse path runs.
    """

    def __init__(self, base_model: AutoModelForCausalLM, bottleneck: int = 128):
        super().__init__()
        self.base = base_model
        self._freeze_base()
        text_config = getattr(base_model.config, "text_config", base_model.config)
        self.hidden_dim = text_config.hidden_size
        self.num_layers = text_config.num_hidden_layers
        # Discover FFN dimensions from the first layer's gate_proj
        first_layer = self._get_layers()[0]
        ffn = self._get_ffn(first_layer)
        self.ffn_dim = ffn.gate_proj.out_features
        # One router per layer
        self.routers = nn.ModuleList([
            LayerRouter(self.hidden_dim, self.ffn_dim, bottleneck)
            for _ in range(self.num_layers)
        ])

    def _freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def _get_layers(self):
        """Get the list of transformer layers from the base model."""
        # Gemma 4 multimodal (AutoModelForCausalLM loads Gemma4ForConditionalGeneration):
        #   base.model = Gemma4Model, base.model.language_model = Gemma4TextModel (has .layers)
        # Gemma 4 text-only (Gemma4ForCausalLM):
        #   base.model = Gemma4TextModel (has .layers directly)
        inner = self.base.model
        if hasattr(inner, "language_model"):
            return inner.language_model.layers
        return inner.layers

    def _get_ffn(self, layer):
        """Get the FFN/MLP module from a transformer layer."""
        if hasattr(layer, "mlp"):
            return layer.mlp
        return layer.feed_forward

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        params = []
        for router in self.routers:
            params.extend(router.parameters())
        return params

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

    def compute_full_ffn(self, ffn, hidden_state: Tensor) -> Tensor:
        """Run the full FFN — gate_proj * up_proj through activation, then down_proj."""
        gate = ffn.gate_proj(hidden_state)
        up = ffn.up_proj(hidden_state)
        activated = torch.nn.functional.silu(gate) * up
        return ffn.down_proj(activated)

    def compute_sparse_ffn(self, ffn, hidden_state: Tensor, mask: Tensor) -> Tensor:
        """
        Run FFN with only the active neurons.

        The mask determines which intermediate neurons to compute. We gather
        only the active columns from gate_proj and up_proj, compute those,
        then scatter back through the active rows of down_proj.

        This is the gather-scatter approach — in production you'd use custom
        CUDA kernels, but PyTorch indexing demonstrates the concept.
        """
        # Find which neurons are active (same mask across batch for simplicity)
        # Use the first token's mask as representative for weight gathering
        active_idx = mask[0, 0].nonzero(as_tuple=True)[0]
        # Gather only active columns from gate and up projections
        gate_w = ffn.gate_proj.weight[active_idx]  # [num_active, hidden_dim]
        up_w = ffn.up_proj.weight[active_idx]       # [num_active, hidden_dim]
        down_w = ffn.down_proj.weight[:, active_idx] # [hidden_dim, num_active]
        # Compute sparse FFN
        gate_out = torch.nn.functional.linear(hidden_state, gate_w)
        up_out = torch.nn.functional.linear(hidden_state, up_w)
        activated = torch.nn.functional.silu(gate_out) * up_out
        return torch.nn.functional.linear(activated, down_w)

    def forward_train(self, input_ids: Tensor, sparsity: float = 0.9) -> dict:
        """
        Training forward pass.

        Runs the full model normally, but at each layer also computes the
        sparse FFN output using the router's mask. Returns both outputs
        so the training loop can compute reconstruction error.
        """
        layers = self._get_layers()
        # Run the full model to get hidden states at each layer
        with torch.no_grad():
            outputs = self.base(input_ids, output_hidden_states=True)
        # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
        # We need the INPUT to each layer's FFN, which is after attention + residual.
        # We approximate this by using the hidden states between layers.
        hidden_states = outputs.hidden_states
        layer_losses = []
        layer_stats = []
        for i, (layer, router) in enumerate(zip(layers, self.routers)):
            ffn = self._get_ffn(layer)
            # Hidden state entering this layer (after attention + residual + norm)
            # This is approximate — the actual FFN input goes through attention first.
            # For training the router, this approximation works well enough.
            h = hidden_states[i + 1].float()
            # Router predicts which neurons matter (float32)
            scores, mask = router(h, sparsity=sparsity)
            # Cast to FFN weight dtype for the frozen FFN computations
            h_ffn = h.to(ffn.gate_proj.weight.dtype)
            # Full FFN output (teacher signal)
            with torch.no_grad():
                full_out = self.compute_full_ffn(ffn, h_ffn).float()
            # Sparse FFN output (student)
            sparse_out = self.compute_sparse_ffn(ffn, h_ffn, mask).float()
            # Reconstruction error
            loss = (sparse_out - full_out).pow(2).mean()
            layer_losses.append(loss)
            # Stats for logging
            with torch.no_grad():
                relative_error = (sparse_out - full_out).norm() / (full_out.norm() + 1e-8)
            layer_stats.append({
                "layer": i,
                "loss": loss.item(),
                "relative_error": relative_error.item(),
                "active_neurons": int(mask[0, 0].sum().item()),
                "total_neurons": self.ffn_dim,
            })
        total_loss = sum(layer_losses) / len(layer_losses)
        return {"loss": total_loss, "layer_losses": layer_losses, "layer_stats": layer_stats}


def load_sparse_router(model_name: str, device: str, dtype: str, bottleneck: int = 128) -> SparseRouter:
    torch_dtype = getattr(torch, dtype)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device)
    model = SparseRouter(base, bottleneck=bottleneck)
    for p in model.trainable_params:
        p.data = p.data.to(device=base.device, dtype=torch.float32)
    return model
