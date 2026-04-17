"""
Stage 3: Hierarchical sparse routing — global controller + local routers.

V3 architecture:
  global_state → controller → per-layer plan (budgets + control vectors)
  per-layer plan + local hidden state → local router → sparse masks
  sparse masks → sparse transformer execution

The GlobalController reads summary hidden states from early checkpoint layers
and outputs per-layer routing plans (attention budget, FFN budget, skip score,
control vector). Each HierLocalRouter then uses its local hidden state plus
the controller's plan to make fine-grained neuron selection decisions.
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sparse_utils import (
    get_layers, get_ffn, get_text_config, get_ffn_dims,
    load_frozen_model, block_pool_hidden, sequence_pool,
    topk_mask, straight_through_topk, budget_to_k,
)


class GlobalController(nn.Module):
    """
    Global routing controller.

    Receives compact summaries from checkpoint layers and emits a routing
    plan for all layers: attention budget, FFN budget, skip score, and
    a control vector that communicates intent to local routers.
    """

    def __init__(
        self,
        model_dim: int,
        controller_dim: int,
        num_layers: int,
        control_dim: int,
        num_summary_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.control_dim = control_dim

        # Compress each summary input
        self.input_proj = nn.Linear(model_dim, controller_dim)

        # Aggregate multiple summary sources
        self.summary_mlp = nn.Sequential(
            nn.Linear(num_summary_layers * controller_dim, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, controller_dim),
        )

        # Learned per-layer query embeddings
        self.layer_queries = nn.Parameter(
            torch.randn(num_layers, controller_dim) * 0.02
        )

        # Shared planner trunk
        self.planner = nn.Sequential(
            nn.Linear(controller_dim * 2, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, controller_dim),
            nn.GELU(),
        )

        # Per-layer output heads
        self.ffn_budget_head = nn.Linear(controller_dim, 1)
        self.skip_head = nn.Linear(controller_dim, 1)
        self.control_head = nn.Linear(controller_dim, control_dim)

    def forward(self, summary_list: list[Tensor]) -> dict:
        """
        summary_list: list of [B, D] tensors from checkpoint layers.

        Returns dict with:
            ffn_budget:  [B, L] in [0, 1]
            skip_score:  [B, L] in [0, 1]
            control_vec: [B, L, C]
        """
        B = summary_list[0].shape[0]

        # Compress each summary
        projected = [self.input_proj(x) for x in summary_list]

        # Concatenate into global summary
        global_summary = torch.cat(projected, dim=-1)           # [B, n_summary * ctrl_dim]
        global_summary = self.summary_mlp(global_summary)       # [B, ctrl_dim]

        # Expand for each layer
        global_expand = global_summary.unsqueeze(1).expand(
            B, self.num_layers, -1
        )                                                        # [B, L, ctrl_dim]
        layer_q = self.layer_queries.unsqueeze(0).expand(
            B, -1, -1
        )                                                        # [B, L, ctrl_dim]

        planner_in = torch.cat([global_expand, layer_q], dim=-1)  # [B, L, 2*ctrl_dim]
        h = self.planner(planner_in)                               # [B, L, ctrl_dim]

        # Budgets in [0, 1]
        ffn_budget = torch.sigmoid(self.ffn_budget_head(h)).squeeze(-1)   # [B, L]
        skip_score = torch.sigmoid(self.skip_head(h)).squeeze(-1)         # [B, L]

        # Control vector (raw, no activation)
        control_vec = self.control_head(h)                                 # [B, L, C]

        return {
            "ffn_budget": ffn_budget,
            "skip_score": skip_score,
            "control_vec": control_vec,
        }


class HierLocalRouter(nn.Module):
    """
    Controller-conditioned local router for one transformer layer.

    Combines three sources of information:
    1. Local hidden state (what's happening at this layer)
    2. Control vector from controller (global intent)
    3. Budget scalars (how much compute is allocated)

    Outputs FFN neuron scores for top-k selection.
    """

    def __init__(
        self,
        model_dim: int,
        router_dim: int,
        control_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.ffn_dim = ffn_dim

        self.local_norm = nn.LayerNorm(model_dim)
        self.local_proj = nn.Linear(model_dim, router_dim)
        self.control_proj = nn.Linear(control_dim, router_dim)
        # ffn_budget + skip_score = 2 scalars
        self.budget_proj = nn.Linear(2, router_dim)

        self.trunk = nn.Sequential(
            nn.Linear(router_dim * 3, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, router_dim),
            nn.GELU(),
        )

        self.ffn_head = nn.Linear(router_dim, ffn_dim)

    def forward(
        self,
        local_hidden: Tensor,
        control_vec: Tensor,
        ffn_budget: Tensor,
        skip_score: Tensor,
    ) -> Tensor:
        """
        local_hidden: [B, T_blk, D] — block-pooled hidden state
        control_vec:  [B, C] — controller's intent vector for this layer
        ffn_budget:   [B] — fraction of FFN neurons to keep
        skip_score:   [B] — how likely this layer should be skipped

        returns: ffn_scores [B, T_blk, ffn_dim]
        """
        B, T_blk, D = local_hidden.shape

        x_local = self.local_proj(self.local_norm(local_hidden))  # [B, T_blk, R]

        x_control = self.control_proj(control_vec)                 # [B, R]
        x_control = x_control.unsqueeze(1).expand(B, T_blk, -1)   # [B, T_blk, R]

        budgets = torch.stack([ffn_budget, skip_score], dim=-1)    # [B, 2]
        x_budget = self.budget_proj(budgets)                        # [B, R]
        x_budget = x_budget.unsqueeze(1).expand(B, T_blk, -1)     # [B, T_blk, R]

        x = torch.cat([x_local, x_control, x_budget], dim=-1)     # [B, T_blk, 3*R]
        x = self.trunk(x)                                          # [B, T_blk, R]

        return self.ffn_head(x)                                    # [B, T_blk, ffn_dim]


class HierarchicalSparseModel(nn.Module):
    """
    Full hierarchical sparse model.

    Architecture:
    1. Early layers run dense to produce summary hidden states
    2. GlobalController reads summaries, outputs per-layer plan
    3. Remaining layers use HierLocalRouters conditioned on the plan
    4. Each layer does sparse FFN + optional skip
    """

    def __init__(
        self,
        base_model,
        controller_dim: int = 256,
        control_dim: int = 64,
        router_dim: int = 128,
        block_size: int = 16,
        skip_threshold: float = 0.95,
        summary_layer_indices: list[int] | None = None,
    ):
        super().__init__()
        self.base = base_model
        self.block_size = block_size
        self.skip_threshold = skip_threshold

        self._freeze_base()

        text_config = get_text_config(base_model)
        model_dim = text_config.hidden_size
        self.model_dim = model_dim
        self.num_layers = text_config.num_hidden_layers
        self.ffn_dims = get_ffn_dims(base_model)

        # Default summary layers: early, quarter, half
        if summary_layer_indices is None:
            summary_layer_indices = [
                0,
                self.num_layers // 4,
                self.num_layers // 2,
            ]
        self.summary_layer_indices = sorted(summary_layer_indices)
        self.dense_prefix = self.summary_layer_indices[-1] + 1
        num_summary = len(self.summary_layer_indices)

        # Global controller
        self.controller = GlobalController(
            model_dim=model_dim,
            controller_dim=controller_dim,
            num_layers=self.num_layers,
            control_dim=control_dim,
            num_summary_layers=num_summary,
        )

        # Per-layer local routers (only for layers after dense prefix)
        self.local_routers = nn.ModuleDict()
        for i in range(self.dense_prefix, self.num_layers):
            self.local_routers[str(i)] = HierLocalRouter(
                model_dim=model_dim,
                router_dim=router_dim,
                control_dim=control_dim,
                ffn_dim=self.ffn_dims[i],
            )

    def _freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        params = list(self.controller.parameters())
        for router in self.local_routers.values():
            params.extend(router.parameters())
        return params

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

    def forward_train(self, input_ids: Tensor) -> dict:
        """
        Training forward pass.

        1. Run full dense model to get all hidden states (teacher)
        2. Collect summaries from checkpoint layers
        3. Run controller to get per-layer plans
        4. For routed layers, compute sparse FFN with STE masks
        5. Accumulate sparse hidden states, compute sparse logits

        Returns dict with dense_logits, sparse_logits, controller_out, aux_stats.
        """
        layers = get_layers(self.base)

        # Dense teacher pass
        with torch.no_grad():
            dense_out = self.base(input_ids, output_hidden_states=True)
        dense_logits = dense_out.logits
        dense_hidden = dense_out.hidden_states
        B, T, D = dense_hidden[0].shape

        # Collect summaries from checkpoint layers
        summary_vectors = []
        for idx in self.summary_layer_indices:
            # hidden_states[idx+1] = output of layer idx
            summary = sequence_pool(dense_hidden[idx + 1].float())  # [B, D]
            summary_vectors.append(summary)

        # Run controller
        controller_out = self.controller(summary_vectors)

        # Process layers
        # Dense prefix: use dense hidden states directly
        h = dense_hidden[self.dense_prefix].float()  # output of last dense prefix layer

        aux_stats = []

        for i in range(self.dense_prefix, self.num_layers):
            layer = layers[i]
            ffn = get_ffn(layer)

            # Get controller plan for this layer
            ffn_budget = controller_out["ffn_budget"][:, i]      # [B]
            skip_score = controller_out["skip_score"][:, i]      # [B]
            control_vec = controller_out["control_vec"][:, i, :] # [B, C]

            # Local router
            h_pooled = block_pool_hidden(h, self.block_size)  # [B, T_blk, D]
            router = self.local_routers[str(i)]
            ffn_scores = router(h_pooled, control_vec, ffn_budget, skip_score)

            # Convert budget to k
            k_per_batch = budget_to_k(ffn_budget, self.ffn_dims[i], min_keep=1)
            # Use mean k across batch for uniform masking
            k = int(k_per_batch.float().mean().item())

            # Expand block scores to per-token
            T_blk = ffn_scores.shape[1]
            ffn_scores_expanded = ffn_scores.repeat_interleave(
                self.block_size, dim=1
            )[:, :T, :]

            # STE mask
            ste_mask = straight_through_topk(ffn_scores_expanded, k)

            # Dense layer output (full layer, post-attn + post-FFN).
            # Using this as FFN input is an approximation — see local_router.py note.
            dense_layer_out = dense_hidden[i + 1]

            # Sparse FFN computation on current sparse hidden state
            h_ffn = h.to(ffn.gate_proj.weight.dtype)
            with torch.no_grad():
                gate_out = ffn.gate_proj(h_ffn)
                up_out = ffn.up_proj(h_ffn)
                activated = F.silu(gate_out) * up_out
                full_ffn_out = ffn.down_proj(activated)

            masked = activated.float() * ste_mask
            sparse_ffn_out = F.linear(
                masked.to(ffn.down_proj.weight.dtype),
                ffn.down_proj.weight,
                ffn.down_proj.bias,
            )

            ffn_delta = (sparse_ffn_out - full_ffn_out).float()

            # Apply skip
            skip_weight = skip_score.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            h = dense_layer_out.float() + (1 - skip_weight) * ffn_delta

            aux_stats.append({
                "layer": i,
                "ffn_budget": ffn_budget.mean().item(),
                "skip_score": skip_score.mean().item(),
                "skip_score_tensor": skip_score.mean(),  # on-graph
                "ffn_budget_tensor": ffn_budget.mean(),   # on-graph
                "k": k,
                "total_neurons": self.ffn_dims[i],
            })

        # Sparse logits
        sparse_logits = self.base.lm_head(h.to(dense_logits.dtype))

        return {
            "dense_logits": dense_logits,
            "sparse_logits": sparse_logits,
            "controller_out": controller_out,
            "aux_stats": aux_stats,
        }


def load_hierarchical_model(
    model_name: str = "google/gemma-4-E2B",
    device: str = "auto",
    dtype: str = "bfloat16",
    controller_dim: int = 256,
    control_dim: int = 64,
    router_dim: int = 128,
    block_size: int = 16,
    summary_layer_indices: list[int] | None = None,
    local_router_checkpoint: str | None = None,
) -> "HierarchicalSparseModel":
    """
    Load a frozen model wrapped with hierarchical routing.

    If local_router_checkpoint is provided, loads pretrained local routers
    from Stage 2 to warm-start the hierarchical model.
    """
    base = load_frozen_model(model_name, device=device, dtype=dtype)
    model = HierarchicalSparseModel(
        base,
        controller_dim=controller_dim,
        control_dim=control_dim,
        router_dim=router_dim,
        block_size=block_size,
        summary_layer_indices=summary_layer_indices,
    )

    # Warm-start from Stage 2 local routers if available
    if local_router_checkpoint:
        ckpt = torch.load(local_router_checkpoint, map_location="cpu", weights_only=True)
        print(f"Loading local router checkpoint from {local_router_checkpoint}")
        # Stage 2 routers have a different structure (no control/budget inputs)
        # We can transfer the FFN head weights as initialization
        for i in range(model.dense_prefix, model.num_layers):
            key = f"router_{i}"
            if key in ckpt.get("ffn_router_states", {}):
                stage2_state = ckpt["ffn_router_states"][key]
                # Transfer the projection weights (norm + proj layers)
                local_router = model.local_routers[str(i)]
                # The local_norm and local_proj can be initialized from Stage 2's norm and proj
                if "norm.weight" in stage2_state and hasattr(local_router, "local_norm"):
                    local_router.local_norm.weight.data.copy_(stage2_state["norm.weight"])
                    local_router.local_norm.bias.data.copy_(stage2_state["norm.bias"])
                # The FFN head can be initialized from Stage 2's proj output layer
                if "proj.2.weight" in stage2_state:
                    # Stage 2 proj is Sequential(Linear, GELU, Linear)
                    # Index 2 is the output linear
                    local_router.ffn_head.weight.data.copy_(stage2_state["proj.2.weight"])
                    local_router.ffn_head.bias.data.copy_(stage2_state["proj.2.bias"])
        print("Transferred Stage 2 router weights to hierarchical model")

    # Move trainable params to float32
    base_device = next(base.parameters()).device
    for p in model.trainable_params:
        p.data = p.data.to(device=base_device, dtype=torch.float32)

    return model
