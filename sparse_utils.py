"""
Shared utilities for sparse routing.

Model introspection, pooling, masking, and sparse FFN execution.
Used across all stages (SVD baseline, local routers, hierarchical router).
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model introspection — handle Gemma 4 multimodal nesting
# ---------------------------------------------------------------------------

def get_layers(model: AutoModelForCausalLM) -> nn.ModuleList:
    """Get transformer layers from a Gemma model, handling multimodal nesting."""
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    return inner.layers


def get_ffn(layer) -> nn.Module:
    """Get the FFN/MLP module from a transformer layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    return layer.feed_forward


def get_text_config(model: AutoModelForCausalLM):
    """Get the text config (handles Gemma 4 multimodal nesting)."""
    return getattr(model.config, "text_config", model.config)


def get_ffn_dims(model: AutoModelForCausalLM) -> list[int]:
    """Get FFN intermediate dimensions per layer (varies for Gemma 4 double-wide layers)."""
    return [get_ffn(layer).gate_proj.out_features for layer in get_layers(model)]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def block_pool_hidden(h: Tensor, block_size: int) -> Tensor:
    """
    Mean-pool hidden states over token blocks.

    h: [B, T, D] -> [B, T_blk, D]

    Cheaper and more structured than per-token routing.
    """
    B, T, D = h.shape
    pad = (block_size - (T % block_size)) % block_size
    if pad > 0:
        h = F.pad(h, (0, 0, 0, pad))
    T_padded = h.shape[1]
    T_blk = T_padded // block_size
    return h.view(B, T_blk, block_size, D).mean(dim=2)


def sequence_pool(h: Tensor) -> Tensor:
    """
    Mean-pool over the full sequence dimension.

    h: [B, T, D] -> [B, D]
    """
    return h.mean(dim=1)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def topk_mask(scores: Tensor, k: int) -> Tensor:
    """
    Hard top-k binary mask.

    scores: [..., N] -> mask [..., N] with 1s at top-k positions.
    """
    N = scores.shape[-1]
    k = max(0, min(k, N))
    if k == 0:
        return torch.zeros_like(scores)
    idx = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, idx, 1.0)
    return mask


def straight_through_topk(scores: Tensor, k: int, temperature: float = 1.0) -> Tensor:
    """
    STE mask: forward = hard top-k, backward = soft distribution.

    Allows gradients to flow through the routing decision during training.
    """
    soft = torch.softmax(scores / temperature, dim=-1)
    hard = topk_mask(scores, k)
    return hard + (soft - soft.detach())


def budget_to_k(budget: Tensor, total_units: int, min_keep: int = 1) -> Tensor:
    """
    Convert a [0,1] budget to an integer top-k count.

    budget: [...] in [0,1]
    returns: [...] integer tensor
    """
    k = (budget * total_units).long()
    return k.clamp(min=min_keep, max=total_units)


# ---------------------------------------------------------------------------
# Sparse FFN execution
# ---------------------------------------------------------------------------

def compute_full_ffn(ffn: nn.Module, hidden_state: Tensor) -> Tensor:
    """Run the full FFN: gate_proj * up_proj through SiLU, then down_proj."""
    gate = ffn.gate_proj(hidden_state)
    up = ffn.up_proj(hidden_state)
    activated = F.silu(gate) * up
    return ffn.down_proj(activated)


def gather_sparse_ffn(ffn: nn.Module, hidden_state: Tensor, active_idx: Tensor) -> Tensor:
    """
    Sparse FFN via gather-scatter.

    Only computes gate/up/down for neurons at active_idx positions.
    active_idx: [num_active] 1D tensor of neuron indices.
    """
    gate_w = ffn.gate_proj.weight[active_idx]    # [num_active, hidden_dim]
    up_w = ffn.up_proj.weight[active_idx]         # [num_active, hidden_dim]
    down_w = ffn.down_proj.weight[:, active_idx]  # [hidden_dim, num_active]
    gate_out = F.linear(hidden_state, gate_w)
    up_out = F.linear(hidden_state, up_w)
    activated = F.silu(gate_out) * up_out
    return F.linear(activated, down_w)


def masked_sparse_ffn(ffn: nn.Module, hidden_state: Tensor, mask: Tensor) -> Tensor:
    """
    Sparse FFN using a binary mask.

    mask: [B, T, ffn_dim] or broadcastable. 1=keep, 0=skip.
    Uses the first token's mask to determine active indices (uniform across batch).
    """
    # Flatten to find active neurons from first sample, first position
    if mask.dim() == 3:
        active_idx = mask[0, 0].nonzero(as_tuple=True)[0]
    elif mask.dim() == 2:
        active_idx = mask[0].nonzero(as_tuple=True)[0]
    else:
        active_idx = mask.nonzero(as_tuple=True)[0]
    return gather_sparse_ffn(ffn, hidden_state, active_idx)


# ---------------------------------------------------------------------------
# Differentiable sparse FFN for training
# ---------------------------------------------------------------------------

def differentiable_sparse_ffn(
    ffn: nn.Module,
    hidden_state: Tensor,
    ste_mask: Tensor,
) -> Tensor:
    """
    Training-mode sparse FFN using STE mask on intermediates.

    Computes full FFN intermediates (frozen), applies differentiable mask,
    then projects down. Gradients flow through ste_mask to the router.

    hidden_state: [B, T, D] in FFN weight dtype
    ste_mask: [B, T, ffn_dim] differentiable mask (from straight_through_topk)
    """
    with torch.no_grad():
        gate_out = ffn.gate_proj(hidden_state)
        up_out = ffn.up_proj(hidden_state)
        activated = F.silu(gate_out) * up_out  # [B, T, ffn_dim]
    # Apply mask in float32 for gradient flow
    masked = activated.float() * ste_mask
    return F.linear(
        masked.to(ffn.down_proj.weight.dtype),
        ffn.down_proj.weight,
        ffn.down_proj.bias,
    )


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared loss functions
# ---------------------------------------------------------------------------

def logit_kl_loss(dense_logits: Tensor, sparse_logits: Tensor, temperature: float = 1.0) -> Tensor:
    """KL divergence between dense teacher and sparse student logits."""
    teacher_probs = F.softmax(dense_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(sparse_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_frozen_model(model_name: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load a frozen pretrained model."""
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, device_map=device
    )
    for p in model.parameters():
        p.requires_grad = False
    return model
