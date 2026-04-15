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


def load_gated_mtp(model_name: str, device: str, dtype: str, num_extra_heads: int) -> GatedMTP:
    """
    Load a pretrained model and wrap it in GatedMTP.
    The base model loads in the specified dtype to save memory.
    Extra heads and gates are created in float32 for stable training,
    then moved to the same device.
    """
    torch_dtype = getattr(torch, dtype)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device)
    model = GatedMTP(base, num_extra_heads=num_extra_heads)
    # Keep trainable components in float32 for stable training and MPS compatibility.
    # The base model stays in the requested dtype (float16) to save memory —
    # the hidden states get cast to float32 when they hit the extra heads.
    for head in model.extra_heads:
        head.to(device=base.device, dtype=torch.float32)
    for gate in model.gates:
        gate.to(device=base.device, dtype=torch.float32)
    return model
