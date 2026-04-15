import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM


class GatedMTP(nn.Module):
    """
    Wraps a frozen pretrained causal LM with extra prediction heads and confidence gates.

    For a model that normally predicts token t+1, this adds N extra heads that predict
    tokens t+2, t+3, ... t+N+1. Each extra head has a gate — a small network that outputs
    a confidence score indicating whether that head's prediction is trustworthy.

    During inference, we accept consecutive extra predictions as long as their gates
    are confident, skipping forward passes for predictable tokens.
    """

    def __init__(self, base_model: AutoModelForCausalLM, num_extra_heads: int = 1):
        super().__init__()
        self.base = base_model
        self.num_extra_heads = num_extra_heads
        # Gemma 4 is multimodal — text config is nested under text_config.
        # Older/text-only models have hidden_size directly on config.
        text_config = getattr(base_model.config, "text_config", base_model.config)
        self.hidden_dim = text_config.hidden_size
        self.vocab_size = text_config.vocab_size
        self._freeze_base()
        self.extra_heads = nn.ModuleList([nn.Linear(self.hidden_dim, self.vocab_size, bias=False) for _ in range(num_extra_heads)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid()) for _ in range(num_extra_heads)])
        self._init_heads_from_base()

    def _freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def _init_heads_from_base(self):
        """
        Copy the base model's language model head weights into each extra head.
        The lm_head already knows how to project hidden states to vocabulary logits —
        our extra heads do the same job but for future token positions. Starting from
        these weights rather than random init dramatically speeds up convergence.
        """
        for head in self.extra_heads:
            head.weight.data.copy_(self.base.lm_head.weight.data)

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        params = []
        for head in self.extra_heads:
            params.extend(head.parameters())
        for gate in self.gates:
            params.extend(gate.parameters())
        return params

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Run a forward pass through the frozen base model, then compute extra head
        predictions and gate confidences from the final hidden state.

        Returns:
            base_logits:   [batch, seq_len, vocab] — the base model's standard t+1 predictions
            extra_logits:  list of [batch, seq_len, vocab] — each extra head's predictions (t+2, t+3, ...)
            confidences:   list of [batch, seq_len, 1] — each gate's confidence that its head is correct
        """
        with torch.no_grad():
            outputs = self.base(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        base_logits = outputs.logits
        # Cast hidden state to float32 to match the trainable heads/gates.
        # Base model outputs float16, but our trainable components need float32
        # for stable gradients and MPS compatibility.
        hidden_f32 = hidden.float()
        extra_logits = [head(hidden_f32) for head in self.extra_heads]
        confidences = [gate(hidden_f32) for gate in self.gates]
        return base_logits, extra_logits, confidences


class ChainedGatedMTP(nn.Module):
    """
    Chained variant — each prediction is conditioned on the previous one.

    Instead of predicting t+2 directly from the hidden state (which misses
    the context of what t+1 actually was), this chains predictions:

      hidden_state + embed(pred_t+1) → MLP → pred_t+2 + gate
      updated_state + embed(pred_t+2) → MLP → pred_t+3 + gate
      ...

    Each step's MLP takes the concatenation of the current state and the
    previous prediction's embedding, projects through a bottleneck, and
    produces both the next prediction logits and a gate confidence score.

    This should significantly outperform the linear variant because each
    prediction has access to the full chain of previous predictions.
    """

    def __init__(self, base_model: AutoModelForCausalLM, num_extra_heads: int = 1, hidden_mult: float = 0.25):
        super().__init__()
        self.base = base_model
        self.num_extra_heads = num_extra_heads
        text_config = getattr(base_model.config, "text_config", base_model.config)
        self.hidden_dim = text_config.hidden_size
        self.vocab_size = text_config.vocab_size
        self._freeze_base()
        # The MLP input is hidden_state + token_embedding concatenated.
        # We project the token embedding down to a manageable size first
        # to keep the MLP input reasonable.
        self.embed_dim = min(256, self.hidden_dim // 4)
        self.embed_proj = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
        mlp_input_dim = self.hidden_dim + self.embed_dim
        bottleneck_dim = int(self.hidden_dim * hidden_mult)
        # Shared MLP that transforms (state + prev_prediction_embed) → new_state.
        # Shared across all chain steps to keep parameter count low.
        # Each step reuses the same weights, similar to how RNN cells work.
        self.chain_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.hidden_dim),
        )
        # Single prediction head and gate, reused at each chain step
        self.pred_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.gate = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self._init_head_from_base()

    def _freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def _init_head_from_base(self):
        self.pred_head.weight.data.copy_(self.base.lm_head.weight.data)
        # Initialize embed_proj from the base embedding matrix (transposed slice).
        # The base model's token embeddings are [vocab, hidden_dim]. We want a
        # projection [hidden_dim, embed_dim] that compresses token representations.
        # Using a slice of the embedding weights gives a meaningful starting point.
        embed_weight = self.base.get_input_embeddings().weight.data
        # SVD-based init: get the top embed_dim components of the embedding space.
        # V is [min(vocab, hidden), hidden], we want [embed_dim, hidden] for the projection.
        if embed_weight.size(0) > self.embed_dim:
            U, S, V = torch.linalg.svd(embed_weight.float(), full_matrices=False)
            self.embed_proj.weight.data.copy_(V[:self.embed_dim].to(self.embed_proj.weight.dtype))

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        params = []
        params.extend(self.embed_proj.parameters())
        params.extend(self.chain_mlp.parameters())
        params.extend(self.pred_head.parameters())
        params.extend(self.gate.parameters())
        return params

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

    def _get_token_embedding(self, token_ids: Tensor) -> Tensor:
        """Look up token embeddings from the frozen base model's embedding table."""
        with torch.no_grad():
            return self.base.get_input_embeddings()(token_ids).float()

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Run base model, then chain predictions where each step conditions
        on the previous prediction.

        During training, we use teacher forcing — each step gets the ACTUAL
        next token embedding, not the predicted one. This gives clean gradients
        and avoids error accumulation during training.

        During inference (see generate.py), we use the predicted tokens instead.
        """
        with torch.no_grad():
            outputs = self.base(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        base_logits = outputs.logits
        hidden_f32 = hidden.float()
        extra_logits = []
        confidences = []
        current_state = hidden_f32
        for i in range(self.num_extra_heads):
            # Teacher forcing: use the actual token at position t+1+i
            # (shifted from input_ids) rather than our prediction.
            # This means during training the chain doesn't accumulate errors.
            offset = i + 1
            if offset < input_ids.size(1):
                actual_token_ids = input_ids[:, offset:]
                # Pad to match current_state sequence length
                pad_len = current_state.size(1) - actual_token_ids.size(1)
                if pad_len > 0:
                    actual_token_ids = torch.cat([actual_token_ids, actual_token_ids[:, -1:].expand(-1, pad_len)], dim=1)
                token_embed = self._get_token_embedding(actual_token_ids)
            else:
                token_embed = torch.zeros(current_state.size(0), current_state.size(1), self.hidden_dim, device=current_state.device, dtype=current_state.dtype)
            # Project token embedding down and concatenate with current state
            token_proj = self.embed_proj(token_embed)
            mlp_input = torch.cat([current_state, token_proj], dim=-1)
            # Transform to get new state for this chain step
            current_state = current_state + self.chain_mlp(mlp_input)
            # Predict and gate from the updated state
            step_logits = self.pred_head(current_state)
            step_conf = self.gate(current_state)
            extra_logits.append(step_logits)
            confidences.append(step_conf)
        return base_logits, extra_logits, confidences


def load_model(model_name: str, device: str, dtype: str, num_extra_heads: int, model_type: str = "linear", hidden_mult: float = 0.25):
    """
    Load a pretrained model and wrap it in GatedMTP or ChainedGatedMTP.

    model_type="linear"  — independent linear heads (fast, simple baseline)
    model_type="chained" — chained MLP where each prediction conditions on the previous
    """
    torch_dtype = getattr(torch, dtype)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device)
    if model_type == "chained":
        model = ChainedGatedMTP(base, num_extra_heads=num_extra_heads, hidden_mult=hidden_mult)
    else:
        model = GatedMTP(base, num_extra_heads=num_extra_heads)
    # Keep trainable components in float32 for stable training and MPS compatibility.
    for p in model.trainable_params:
        p.data = p.data.to(device=base.device, dtype=torch.float32)
    return model


# Backwards compatible alias
def load_gated_mtp(model_name: str, device: str, dtype: str, num_extra_heads: int) -> GatedMTP:
    return load_model(model_name, device, dtype, num_extra_heads, model_type="linear")
