# Unified Sparse Routing — Implementation Spec

## Overview

Three-stage implementation for inference-time sparse compute on frozen Gemma 4 E2B.

## Stages

### Stage 1: SVD Baseline (`svd_baseline.py`)

Training-free FFN sparsity using truncated SVD of gate_proj weights + static layer skipping.

**Components:**
- `SVDPredictor`: Per-layer predictor. Takes truncated SVD of `gate_proj.weight` (rank `r`, default 128). At inference, cheap matmul `x @ U_r` approximates gate output, threshold to get top-k active neurons.
- `SVDSparseModel`: Wraps frozen base model. Installs forward hooks on each layer's FFN to redirect through sparse path. Supports configurable layer skip list.
- `benchmark_svd()`: Measures tok/s at various sparsity levels vs dense baseline.

**Dimension notes:**
- gate_proj.weight shape: `[ffn_dim, hidden_dim]` — varies per layer (some double-wide)
- SVD: `U [ffn_dim, r]`, `S [r]`, `V [r, hidden_dim]`
- Cheap predictor: `x @ V^T @ diag(S)` → `[batch, seq, r]` then `@ U^T` → `[batch, seq, ffn_dim]`
- Actually simpler: just `x @ (V^T @ diag(S) @ U^T)` = `x @ W_approx` where `W_approx [hidden_dim, ffn_dim]`
- But that's the same cost as gate_proj! The point is we can threshold the *approximated* gate output to decide which neurons to skip, then only compute the *exact* gate/up/down for active neurons.
- Better approach: `x @ V^T` → `[batch, seq, r]`, then `@ (diag(S) @ U^T)` → `[batch, seq, ffn_dim]`. The first matmul is cheap (`hidden_dim × r`), but the second is full-size. We need to threshold in the low-rank space somehow, or just accept that the predictor costs `hidden_dim × r` and saves `2 × hidden_dim × ffn_dim` (up_proj + down_proj for inactive neurons).
- Simplest correct approach: compute `x @ gate_proj.weight^T` (full gate), apply SiLU, threshold to find active neurons, then only compute up_proj and down_proj for active neurons. This saves 2/3 of FFN compute with zero training. This is actually the "gate thresholding" baseline, not SVD.
- SVD variant: `x @ V_r^T` gives r-dimensional summary, `(V_r^T @ diag(S_r)) @ U_r^T` approximates gate output. Cost: `hidden_dim × r + r × ffn_dim`. If `r << min(hidden_dim, ffn_dim)`, this is cheaper than the full gate.

**Decision: implement both.**
1. Gate-threshold baseline: compute full gate_proj, threshold, sparse up+down. Saves 2/3.
2. SVD predictor: approximate gate via rank-r SVD, threshold, sparse gate+up+down. Saves more but approximate.

### Stage 2: Learned Local Routers (`local_router.py`)

Per-layer learned MLP routers for FFN sparsity + layer skip decisions.

**Components:**
- `LocalFFNRouter(nn.Module)`: `hidden_dim → bottleneck → ffn_dim` scores per neuron.
  - Input: block-pooled hidden state `[B, T_blk, hidden_dim]`
  - Output: FFN neuron scores `[B, T_blk, ffn_dim]`
  - Uses block pooling (mean over `block_size` tokens) for efficiency
  
- `LocalSkipPredictor(nn.Module)`: `hidden_dim → bottleneck → 1` skip score per layer.
  - Input: sequence-pooled hidden state `[B, hidden_dim]`
  - Output: skip probability `[B, 1]` (sigmoid)

- `LocallyRoutedModel(nn.Module)`: Wraps frozen base model.
  - One `LocalFFNRouter` per layer (sized to that layer's ffn_dim)
  - One `LocalSkipPredictor` per layer
  - Forward hooks for sparse execution
  - Fixed sparsity budget (not adaptive yet)

**Training (`train_local.py`):**
- Loss = `logit_kl(dense, sparse) + lambda_compute * compute_cost`
- `logit_kl`: KL divergence between dense and sparse model logits
- `compute_cost`: mean fraction of neurons kept + (1 - skip_rate), penalizes using too much compute
- Dense teacher: same frozen model run without sparsity
- Only router/skip params are trainable
- AdamW, lr=1e-3, cosine schedule

**Dimension notes:**
- ffn_dim varies per layer — router output dim must match
- block_size default 16 tokens
- bottleneck default 128

### Stage 3: Global Controller + Local Routers (`hierarchical_router.py`)

V3 hierarchical architecture.

**Components:**
- `GlobalController(nn.Module)`:
  - Input: summary vectors from checkpoint layers `[B, hidden_dim]` × num_summary_layers
  - `input_proj`: `hidden_dim → controller_dim` per summary
  - `summary_mlp`: `num_summary_layers * controller_dim → controller_dim`
  - `layer_queries`: learned `[num_layers, controller_dim]`
  - `planner`: `2 * controller_dim → controller_dim` (global + per-layer query)
  - Output heads per layer:
    - `ffn_budget_head`: `controller_dim → 1` (sigmoid, [0,1])
    - `skip_head`: `controller_dim → 1` (sigmoid, [0,1])
    - `control_head`: `controller_dim → control_dim`
    - NOTE: `attn_budget_head` deferred to Stage 4 (attention routing)

- `HierLocalRouter(nn.Module)`:
  - Input: local hidden summary `[B, T_blk, hidden_dim]` + control_vec `[B, control_dim]` + budgets `[B, 2]` (ffn_budget + skip_score)
  - `local_proj`: `hidden_dim → router_dim`
  - `control_proj`: `control_dim → router_dim`
  - `budget_proj`: `2 → router_dim`
  - `trunk`: `3 * router_dim → router_dim`
  - `ffn_head`: `router_dim → num_ffn_groups`
  - Output: FFN group scores `[B, T_blk, num_ffn_groups]`

- `HierarchicalSparseModel(nn.Module)`:
  - Frozen base model
  - One `GlobalController`
  - One `HierLocalRouter` per layer
  - Summary layer indices (configurable, default: layers 0, 8, 16)
  - Dense prefix: layers before first summary index run dense
  - Controller runs after summaries collected, routes remaining layers

**Training (`train_hierarchical.py`):**
- Stage 2→3 transfer: load pretrained local routers, add controller
- Loss = `logit_kl + lambda_compute * compute_cost + lambda_budget * budget_target_loss`
- Optional: `hidden_state_loss` if training is unstable
- `budget_target_loss`: `(mean_budget - target)^2` to hit desired compute level

**Dimension defaults:**
- controller_dim: 256
- control_dim: 64
- router_dim: 128
- num_ffn_groups: ffn_dim (1:1 mapping, can group later for efficiency)
- summary_layer_indices: [0, num_layers//4, num_layers//2]

## Shared Utilities (`sparse_utils.py`)

- `get_layers(model)`: handles Gemma 4 multimodal nesting
- `get_ffn(layer)`: handles mlp vs feed_forward
- `block_pool_hidden(h, block_size)`: mean pooling over token blocks
- `topk_mask(scores, k)`: hard top-k binary mask
- `straight_through_topk(scores, k, temperature)`: STE for training
- `gather_sparse_ffn(ffn, hidden_state, active_idx)`: gather-scatter sparse FFN
- `compute_full_ffn(ffn, hidden_state)`: full FFN for teacher signal

## Benchmark (`benchmark_all.py`)

Compare all stages:
1. Dense baseline (no sparsity)
2. Gate threshold (compute full gate, skip inactive up+down)
3. SVD predictor (approximate gate, skip all three for inactive)
4. Learned local router (Stage 2)
5. Hierarchical router (Stage 3)

Metrics per approach:
- Tokens per second
- Logit KL divergence vs dense (quality measure)
- Compute savings (fraction of FFN/attention skipped)
- Per-layer breakdown

## File Structure

```
gated-mtp/
  sparse_utils.py        # Shared helpers (model introspection, pooling, masking)
  svd_baseline.py        # Stage 1: training-free baselines
  local_router.py        # Stage 2: learned local routers
  train_local.py         # Stage 2 training script
  hierarchical_router.py # Stage 3: global controller + local routers
  train_hierarchical.py  # Stage 3 training script
  benchmark_all.py       # Unified benchmark for all stages
  profile_sparsity.py    # Stage 0: profiler (already built)
```
