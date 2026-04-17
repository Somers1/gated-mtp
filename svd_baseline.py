"""
Stage 1: Training-free sparse FFN baselines.

Two approaches, both zero training cost:

1. Gate Threshold — compute full gate_proj, threshold output, skip inactive
   neurons in up_proj and down_proj. Saves ~2/3 of FFN compute.

2. SVD Predictor — approximate gate_proj via truncated SVD, threshold the
   approximation, skip all three matmuls for inactive neurons. Cheaper
   prediction but approximate.

Both can be combined with static layer skipping (skip layers with lowest
importance as measured by profile_sparsity.py).
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sparse_utils import (
    get_layers, get_ffn, get_text_config, get_ffn_dims,
    load_frozen_model, topk_mask, gather_sparse_ffn,
)


class _Restorer:
    """Restores a monkey-patched attribute on a module."""
    def __init__(self, module, attr_name, original):
        self.module = module
        self.attr_name = attr_name
        self.original = original

    def restore(self):
        setattr(self.module, self.attr_name, self.original)

    # Alias for compatibility with handle.remove() calling convention
    remove = restore


class GateThresholdPredictor:
    """
    Predicts active FFN neurons by running the full gate_proj and thresholding.

    This is the simplest baseline: compute gate, apply SiLU, keep top-k by
    magnitude. Skips up_proj and down_proj for inactive neurons.

    Cost: 1 full matmul (gate_proj). Saves: 2 full matmuls (up + down) for
    inactive neurons.
    """

    def __init__(self, ffn: nn.Module):
        self.ffn = ffn
        self.ffn_dim = ffn.gate_proj.out_features

    @torch.no_grad()
    def predict(self, hidden_state: Tensor, sparsity: float = 0.9) -> Tensor:
        """
        Returns active neuron indices.

        hidden_state: [B, T, D]
        returns: [num_active] index tensor (uniform across batch/seq for simplicity)
        """
        # Full gate computation
        gate_out = self.ffn.gate_proj(hidden_state)
        gate_activated = F.silu(gate_out)
        # Use mean absolute activation across batch/seq to rank neurons
        importance = gate_activated.abs().mean(dim=(0, 1))  # [ffn_dim]
        k = max(1, int(self.ffn_dim * (1 - sparsity)))
        _, active_idx = importance.topk(k)
        return active_idx

    @torch.no_grad()
    def sparse_ffn(self, hidden_state: Tensor, sparsity: float = 0.9) -> Tensor:
        """
        Run sparse FFN: full gate, then sparse up+down for active neurons only.
        """
        active_idx = self.predict(hidden_state, sparsity)
        # Full gate (already paid for in predict, recompute is fine for clarity)
        gate_out = self.ffn.gate_proj(hidden_state)
        gate_activated = F.silu(gate_out)
        # Sparse up and down
        up_w = self.ffn.up_proj.weight[active_idx]         # [k, D]
        down_w = self.ffn.down_proj.weight[:, active_idx]   # [D, k]
        up_out = F.linear(hidden_state, up_w)               # [B, T, k]
        # Gather only active gate channels
        active_gate = gate_activated[..., active_idx]        # [B, T, k]
        activated = active_gate * up_out
        return F.linear(activated, down_w)                   # [B, T, D]


class SVDPredictor:
    """
    Predicts active FFN neurons using a low-rank approximation of gate_proj.

    Takes truncated SVD of gate_proj.weight: W ≈ U_r @ S_r @ V_r^T
    where W is [ffn_dim, hidden_dim].

    Prediction: x @ V_r → [B, T, r] (cheap), then @ diag(S_r) @ U_r^T → [B, T, ffn_dim].
    Threshold the result to find active neurons.

    Cost: hidden_dim×r + r×ffn_dim (much cheaper than full gate if r << both).
    Saves: all three matmuls for inactive neurons.
    """

    def __init__(self, ffn: nn.Module, rank: int = 128):
        self.ffn = ffn
        self.ffn_dim = ffn.gate_proj.out_features
        self.rank = min(rank, min(ffn.gate_proj.weight.shape))

        # Compute truncated SVD of gate_proj weight
        W = ffn.gate_proj.weight.data.float()  # [ffn_dim, hidden_dim]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # Truncate to rank r
        U_r = U[:, :self.rank]      # [ffn_dim, r]
        S_r = S[:self.rank]          # [r]
        Vh_r = Vh[:self.rank, :]     # [r, hidden_dim]

        # Precompute projection matrices
        # Step 1: x @ Vh_r^T → [B, T, r]  (cheap projection)
        # Step 2: result @ diag(S_r) @ U_r^T → [B, T, ffn_dim]  (expand to scores)
        self.proj_down = Vh_r.T.to(W.device).to(ffn.gate_proj.weight.dtype)  # [hidden_dim, r]
        self.proj_up = (U_r * S_r.unsqueeze(0)).T.to(W.device).to(ffn.gate_proj.weight.dtype)  # [r, ffn_dim]

    @torch.no_grad()
    def predict(self, hidden_state: Tensor, sparsity: float = 0.9) -> Tensor:
        """
        Returns active neuron indices using SVD approximation.

        hidden_state: [B, T, D]
        returns: [num_active] index tensor
        """
        # Cheap low-rank gate approximation
        low_rank = hidden_state @ self.proj_down      # [B, T, r]
        approx_gate = low_rank @ self.proj_up          # [B, T, ffn_dim]
        approx_activated = F.silu(approx_gate)
        # Rank by mean activation magnitude
        importance = approx_activated.abs().mean(dim=(0, 1))  # [ffn_dim]
        k = max(1, int(self.ffn_dim * (1 - sparsity)))
        _, active_idx = importance.topk(k)
        return active_idx

    @torch.no_grad()
    def sparse_ffn(self, hidden_state: Tensor, sparsity: float = 0.9) -> Tensor:
        """
        Run sparse FFN: SVD-predicted mask, then sparse gate+up+down.
        """
        active_idx = self.predict(hidden_state, sparsity)
        return gather_sparse_ffn(self.ffn, hidden_state, active_idx)


class SVDSparseModel:
    """
    Wraps a frozen model with SVD or gate-threshold sparse FFN execution.

    Installs forward hooks on each layer's FFN to redirect through sparse path.
    Supports static layer skipping.
    """

    def __init__(
        self,
        model,
        predictor_type: str = "svd",
        rank: int = 128,
        sparsity: float = 0.9,
        skip_layers: list[int] | None = None,
    ):
        self.model = model
        self.sparsity = sparsity
        self.skip_layers = set(skip_layers or [])
        self.layers = get_layers(model)
        self.predictors = []

        for i, layer in enumerate(self.layers):
            ffn = get_ffn(layer)
            if predictor_type == "svd":
                self.predictors.append(SVDPredictor(ffn, rank=rank))
            else:
                self.predictors.append(GateThresholdPredictor(ffn))

    def install_hooks(self) -> list:
        """
        Install sparse FFN execution via monkey-patching.

        Replaces each FFN's forward method with a sparse version.
        Returns a list of restore callables (call each to undo).
        """
        restorers = []
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                # Layer skip: replace entire layer forward with identity
                original_forward = layer.forward
                def make_skip_forward(orig):
                    def skip_forward(*args, **kwargs):
                        # Return first positional arg (hidden_states) unchanged
                        h = args[0] if args else kwargs.get("hidden_states")
                        # Match original output format (hidden_states, ...)
                        return (h,)
                    return skip_forward
                layer.forward = make_skip_forward(original_forward)
                restorers.append(_Restorer(layer, "forward", original_forward))
            else:
                # Sparse FFN: replace FFN forward with sparse version
                ffn = get_ffn(layer)
                original_forward = ffn.forward
                predictor = self.predictors[i]
                sparsity = self.sparsity
                def make_sparse_forward(pred, sp, orig_fwd):
                    def sparse_forward(hidden_state):
                        return pred.sparse_ffn(hidden_state, sp)
                    return sparse_forward
                ffn.forward = make_sparse_forward(predictor, sparsity, original_forward)
                restorers.append(_Restorer(ffn, "forward", original_forward))
        return restorers

    def remove_hooks(self, handles: list):
        for h in handles:
            h.restore()
            h.remove()


def build_svd_sparse_model(
    model_name: str = "google/gemma-4-E2B",
    device: str = "auto",
    dtype: str = "bfloat16",
    predictor_type: str = "svd",
    rank: int = 128,
    sparsity: float = 0.9,
    skip_layers: list[int] | None = None,
) -> tuple:
    """Load model and build SVD sparse wrapper. Returns (sparse_model, base_model, tokenizer)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_frozen_model(model_name, device=device, dtype=dtype)
    sparse = SVDSparseModel(
        model,
        predictor_type=predictor_type,
        rank=rank,
        sparsity=sparsity,
        skip_layers=skip_layers,
    )
    return sparse, model, tokenizer
