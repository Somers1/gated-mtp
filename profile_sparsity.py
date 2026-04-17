"""
Profile per-layer, per-token compute importance in a frozen model.

Hooks into each transformer layer's attention and FFN to measure how much
each component changes the hidden state. High variance = opportunity for
adaptive sparsity or layer skipping.

Outputs:
  - Console summary (mean/std/min/max per layer)
  - JSON with full per-layer, per-token stats
  - Key takeaway: is there enough variance to justify a unified router?
"""
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config


PROMPTS = [
    "The history of artificial intelligence began in the 1950s when",
    "To bake a chocolate cake, you will need flour, sugar, cocoa powder,",
    "In quantum mechanics, the wave function describes the probability",
    "The mortgage broker reviewed the client's financial documents and",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(",
    "The Australian housing market experienced significant changes during",
    "Climate change poses an existential threat to many species because",
    "When implementing a distributed system, one must consider the CAP theorem,",
]


def get_layers(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    return inner.layers


def profile():
    print(f"Loading {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    torch_dtype = getattr(torch, config.DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=torch_dtype, device_map=config.DEVICE
    )
    model.eval()
    device = next(model.parameters()).device
    layers = get_layers(model)
    num_layers = len(layers)
    print(f"Model loaded on {device}, {num_layers} layers")

    # Storage for hook captures
    captures = {}

    def make_hooks(layer_idx, layer):
        """Install pre/post hooks on attention and FFN to capture norms."""
        ffn = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward
        attn = layer.self_attn

        def attn_hook(module, args, output):
            # output is typically (hidden_states, attn_weights, ...) or just hidden_states
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captures[f"attn_{layer_idx}"] = out.detach().float()

        def ffn_hook(module, args, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captures[f"ffn_{layer_idx}"] = out.detach().float()

        def layer_pre_hook(module, args, kwargs=None):
            # Capture input to this layer (the residual stream)
            inp = args[0] if args else kwargs.get("hidden_states")
            captures[f"input_{layer_idx}"] = inp.detach().float()

        def layer_post_hook(module, args, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captures[f"output_{layer_idx}"] = out.detach().float()

        handles = [
            attn.register_forward_hook(attn_hook),
            ffn.register_forward_hook(ffn_hook),
            layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True),
            layer.register_forward_hook(layer_post_hook),
        ]
        return handles

    # Install hooks on all layers
    all_handles = []
    for i, layer in enumerate(layers):
        all_handles.extend(make_hooks(i, layer))

    # Run prompts and collect stats
    all_layer_stats = []

    print(f"\nProfiling {len(PROMPTS)} prompts...\n")
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(PROMPTS):
            captures.clear()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            seq_len = input_ids.shape[1]
            model(input_ids)

            for layer_idx in range(num_layers):
                layer_input = captures.get(f"input_{layer_idx}")
                layer_output = captures.get(f"output_{layer_idx}")
                attn_out = captures.get(f"attn_{layer_idx}")
                ffn_out = captures.get(f"ffn_{layer_idx}")

                if layer_input is None or layer_output is None:
                    continue

                # Per-token norms (squeeze batch dim)
                input_norm = layer_input[0].norm(dim=-1)  # [seq_len]
                output_norm = layer_output[0].norm(dim=-1)

                # Layer contribution: how much does the residual change?
                layer_delta = (layer_output[0] - layer_input[0]).norm(dim=-1)
                layer_importance = layer_delta / (input_norm + 1e-8)

                # Attention contribution
                attn_importance = torch.zeros(seq_len, device=device)
                if attn_out is not None:
                    attn_norm = attn_out[0].norm(dim=-1)
                    attn_importance = attn_norm / (input_norm + 1e-8)

                # FFN contribution
                ffn_importance = torch.zeros(seq_len, device=device)
                if ffn_out is not None:
                    ffn_norm = ffn_out[0].norm(dim=-1)
                    ffn_importance = ffn_norm / (input_norm + 1e-8)

                all_layer_stats.append({
                    "prompt_idx": prompt_idx,
                    "layer": layer_idx,
                    "seq_len": seq_len,
                    "layer_importance": {
                        "mean": layer_importance.mean().item(),
                        "std": layer_importance.std().item(),
                        "min": layer_importance.min().item(),
                        "max": layer_importance.max().item(),
                    },
                    "attn_importance": {
                        "mean": attn_importance.mean().item(),
                        "std": attn_importance.std().item(),
                        "min": attn_importance.min().item(),
                        "max": attn_importance.max().item(),
                    },
                    "ffn_importance": {
                        "mean": ffn_importance.mean().item(),
                        "std": ffn_importance.std().item(),
                        "min": ffn_importance.min().item(),
                        "max": ffn_importance.max().item(),
                    },
                })

    # Remove hooks
    for h in all_handles:
        h.remove()

    # Aggregate per-layer across all prompts
    print(f"{'Layer':>5} | {'Layer Δ mean':>11} {'±std':>7} | {'Attn mean':>10} {'±std':>7} | {'FFN mean':>10} {'±std':>7} | {'Token var':>10}")
    print("-" * 95)

    per_layer_summary = []
    for layer_idx in range(num_layers):
        layer_stats = [s for s in all_layer_stats if s["layer"] == layer_idx]
        if not layer_stats:
            continue

        l_means = [s["layer_importance"]["mean"] for s in layer_stats]
        l_stds = [s["layer_importance"]["std"] for s in layer_stats]
        a_means = [s["attn_importance"]["mean"] for s in layer_stats]
        a_stds = [s["attn_importance"]["std"] for s in layer_stats]
        f_means = [s["ffn_importance"]["mean"] for s in layer_stats]
        f_stds = [s["ffn_importance"]["std"] for s in layer_stats]

        # Average of per-token std = how much variance across tokens within a layer
        avg_token_var = sum(l_stds) / len(l_stds)

        summary = {
            "layer": layer_idx,
            "layer_delta": {"mean": sum(l_means) / len(l_means), "std": sum(l_stds) / len(l_stds)},
            "attn": {"mean": sum(a_means) / len(a_means), "std": sum(a_stds) / len(a_stds)},
            "ffn": {"mean": sum(f_means) / len(f_means), "std": sum(f_stds) / len(f_stds)},
        }
        per_layer_summary.append(summary)

        print(
            f"{layer_idx:>5} | "
            f"{summary['layer_delta']['mean']:>11.4f} {summary['layer_delta']['std']:>7.4f} | "
            f"{summary['attn']['mean']:>10.4f} {summary['attn']['std']:>7.4f} | "
            f"{summary['ffn']['mean']:>10.4f} {summary['ffn']['std']:>7.4f} | "
            f"{avg_token_var:>10.4f}"
        )

    # Key findings
    layer_means = [s["layer_delta"]["mean"] for s in per_layer_summary]
    token_vars = [s["layer_delta"]["std"] for s in per_layer_summary]
    attn_means = [s["attn"]["mean"] for s in per_layer_summary]
    ffn_means = [s["ffn"]["mean"] for s in per_layer_summary]

    print(f"\n{'=' * 95}")
    print("KEY FINDINGS")
    print(f"{'=' * 95}")

    # Layer skipping potential
    mean_importance = sum(layer_means) / len(layer_means)
    least_important = sorted(enumerate(layer_means), key=lambda x: x[1])
    print(f"\nLayer skipping potential:")
    print(f"  Average layer importance: {mean_importance:.4f}")
    print(f"  Least important layers: {', '.join(f'L{i} ({v:.4f})' for i, v in least_important[:5])}")
    print(f"  Most important layers:  {', '.join(f'L{i} ({v:.4f})' for i, v in least_important[-5:])}")
    importance_range = max(layer_means) / (min(layer_means) + 1e-8)
    print(f"  Max/min ratio: {importance_range:.1f}x")
    if importance_range > 3:
        print(f"  → HIGH variance across layers — layer skipping looks promising")
    elif importance_range > 1.5:
        print(f"  → MODERATE variance — some layers skippable")
    else:
        print(f"  → LOW variance — layers contribute roughly equally, skipping risky")

    # Adaptive sparsity potential (per-token variance)
    avg_token_var_all = sum(token_vars) / len(token_vars)
    max_token_var = max(token_vars)
    print(f"\nAdaptive sparsity potential (per-token variance):")
    print(f"  Average per-token std: {avg_token_var_all:.4f}")
    print(f"  Max per-token std:     {max_token_var:.4f}")
    print(f"  Layers with highest token variance: {', '.join(f'L{i} ({v:.4f})' for i, v in sorted(enumerate(token_vars), key=lambda x: -x[1])[:5])}")
    if avg_token_var_all > 0.1 * mean_importance:
        print(f"  → HIGH per-token variance — adaptive sparsity has legs")
    else:
        print(f"  → LOW per-token variance — fixed sparsity might be fine")

    # Attention vs FFN balance
    total_attn = sum(attn_means)
    total_ffn = sum(ffn_means)
    print(f"\nAttention vs FFN contribution:")
    print(f"  Total attention contribution: {total_attn:.2f}")
    print(f"  Total FFN contribution:       {total_ffn:.2f}")
    print(f"  Ratio (attn/ffn):             {total_attn / (total_ffn + 1e-8):.2f}")

    # Save full results
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {"model": config.BASE_MODEL, "num_prompts": len(PROMPTS), "num_layers": num_layers},
        "per_layer_summary": per_layer_summary,
        "raw_stats": all_layer_stats,
    }
    output_path = results_dir / "sparsity_profile.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    profile()
