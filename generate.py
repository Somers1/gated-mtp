"""
Inference with Gated MTP.

Generates text using the base model + trained extra heads and gates.
At each step, the base model predicts token t+1 (always accepted).
Then each extra head's gate is checked in order — if confident,
that head's prediction is accepted too. We stop checking further
heads at the first gate that falls below the confidence threshold.

This means easy/predictable sequences generate multiple tokens per
forward pass, while hard/uncertain sequences fall back to standard
one-token-at-a-time generation. The model automatically adapts its
speed to the difficulty of what it's generating.
"""
import argparse
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
import config
from model import load_model


def load_checkpoint(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # Support both old format (separate extra_heads/gates) and new (model_state)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        for i, state_dict in enumerate(checkpoint["extra_heads"]):
            model.extra_heads[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint["gates"]):
            model.gates[i].load_state_dict(state_dict)
    return model


@torch.no_grad()
def _chained_inference(model, input_ids, gate_threshold):
    """
    Inference-time forward pass for ChainedGatedMTP.

    Unlike training (which uses teacher forcing with actual tokens),
    inference chains predictions using the model's own predicted tokens.
    We stop chaining at the first gate rejection to avoid error accumulation.
    """
    outputs = model.base(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    base_logits = outputs.logits
    hidden_f32 = hidden.float()
    extra_logits = []
    confidences = []
    current_state = hidden_f32
    # Start with the base model's prediction as the first "previous token"
    prev_token = base_logits[:, -1:].argmax(dim=-1)
    for i in range(model.num_extra_heads):
        token_embed = model._get_token_embedding(prev_token.expand(-1, current_state.size(1)))
        token_proj = model.embed_proj(token_embed)
        mlp_input = torch.cat([current_state, token_proj], dim=-1)
        current_state = current_state + model.chain_mlp(mlp_input)
        step_logits = model.pred_head(current_state)
        step_conf = model.gate(current_state)
        extra_logits.append(step_logits)
        confidences.append(step_conf)
        # For chaining: use this step's prediction as input to the next step
        prev_token = step_logits[:, -1:].argmax(dim=-1)
    return base_logits, extra_logits, confidences


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 256, gate_threshold: float = config.GATE_THRESHOLD) -> tuple[str, dict]:
    """
    Generate text with gated multi-token prediction.

    Returns the generated text and a stats dict showing how many forward
    passes were saved by the gate accepting extra tokens.
    """
    from model import ChainedGatedMTP
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
    tokens_generated = 0
    forward_passes = 0
    gate_accepts = 0
    is_chained = isinstance(model, ChainedGatedMTP)
    while tokens_generated < max_tokens:
        if is_chained:
            # For chained models, we need to do inference-time chaining manually
            # instead of using teacher forcing. Each step uses the PREDICTED
            # token embedding, not the ground truth.
            base_logits, extra_logits, confidences = _chained_inference(model, input_ids, gate_threshold)
        else:
            base_logits, extra_logits, confidences = model(input_ids)
        forward_passes += 1
        # Always accept the base model's next-token prediction
        next_token = base_logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        tokens_generated += 1
        if next_token.item() == tokenizer.eos_token_id:
            break
        # Check each extra head's gate in sequence.
        for logits_i, conf_i in zip(extra_logits, confidences):
            if conf_i[:, -1].item() > gate_threshold:
                extra_token = logits_i[:, -1].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, extra_token], dim=-1)
                tokens_generated += 1
                gate_accepts += 1
            else:
                break
    tokens_per_pass = tokens_generated / forward_passes
    speedup_pct = (1 - forward_passes / tokens_generated) * 100
    stats = {
        "tokens_generated": tokens_generated,
        "forward_passes": forward_passes,
        "gate_accepts": gate_accepts,
        "tokens_per_pass": round(tokens_per_pass, 2),
        "speedup": f"{speedup_pct:.1f}%",
    }
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=config.GATE_THRESHOLD)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    print(f"Loading model {config.BASE_MODEL}...")
    model = load_model(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE, num_extra_heads=config.NUM_EXTRA_HEADS, model_type=config.MODEL_TYPE, hidden_mult=config.CHAIN_HIDDEN_MULT)
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}...")
        load_checkpoint(model, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    print(f"Generating with threshold={args.threshold}...\n")
    t0 = time.time()
    text, stats = generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens, gate_threshold=args.threshold)
    elapsed = time.time() - t0
    print(text)
    print(f"\n--- Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  wall_time: {elapsed:.2f}s")
    print(f"  tokens_per_second: {stats['tokens_generated'] / elapsed:.1f}")


if __name__ == "__main__":
    main()
