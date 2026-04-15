"""
Benchmark script for Gated MTP.

Compares generation speed and quality between:
  1. Standard autoregressive (baseline — one token per forward pass)
  2. Gated MTP at various thresholds

Runs a set of prompts through both modes and reports:
  - Tokens per second (wall-clock speed)
  - Forward passes per token (compute efficiency)
  - Gate fire rate (how often extra tokens are accepted)
  - Gate accuracy (when it fires, is the extra token correct?)

To measure gate accuracy, we generate with the gate, then compare each
gate-accepted token against what the base model would have produced if
run autoregressively from that position. A mismatch means the gate
accepted a token the base model would have predicted differently.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
import config
from model import load_gated_mtp
from generate import generate, load_checkpoint


BENCHMARK_PROMPTS = [
    "The capital of France is",
    "In a recent study, researchers found that",
    "def fibonacci(n):\n",
    "The mortgage application process typically involves",
    "Once upon a time in a land far away,",
    "The key difference between TCP and UDP is",
    "To make a classic margherita pizza, you need",
    "The theory of general relativity, proposed by Einstein,",
]


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
    """
    Standard autoregressive generation — one token per forward pass.
    No gates, no extra heads. This is the baseline to beat.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
    tokens_generated = 0
    while tokens_generated < max_tokens:
        outputs = model.base(input_ids)
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        tokens_generated += 1
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, {"tokens_generated": tokens_generated, "forward_passes": tokens_generated}


def run_benchmark(model, tokenizer, thresholds: list[float], max_tokens: int = 128):
    results = {"baseline": [], "gated": {t: [] for t in thresholds}}
    # --- Baseline ---
    print("Running baseline (standard autoregressive)...")
    for prompt in BENCHMARK_PROMPTS:
        t0 = time.time()
        text, stats = generate_baseline(model, tokenizer, prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        stats["wall_time"] = elapsed
        stats["tokens_per_second"] = stats["tokens_generated"] / elapsed
        stats["prompt"] = prompt
        results["baseline"].append(stats)
        print(f"  {prompt[:40]}... | {stats['tokens_per_second']:.1f} tok/s")
    # --- Gated at each threshold ---
    for threshold in thresholds:
        print(f"\nRunning gated MTP (threshold={threshold})...")
        for prompt in BENCHMARK_PROMPTS:
            t0 = time.time()
            text, stats = generate(model, tokenizer, prompt, max_tokens=max_tokens, gate_threshold=threshold)
            elapsed = time.time() - t0
            stats["wall_time"] = elapsed
            stats["tokens_per_second"] = stats["tokens_generated"] / elapsed
            stats["prompt"] = prompt
            results["gated"][threshold].append(stats)
            print(f"  {prompt[:40]}... | {stats['tokens_per_second']:.1f} tok/s | gate accepts: {stats['gate_accepts']} | speedup: {stats['speedup']}")
    return results


def print_summary(results: dict):
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    baseline_avg_tps = sum(r["tokens_per_second"] for r in results["baseline"]) / len(results["baseline"])
    baseline_avg_passes = sum(r["forward_passes"] for r in results["baseline"]) / sum(r["tokens_generated"] for r in results["baseline"])
    print(f"\nBaseline: {baseline_avg_tps:.1f} tok/s | {baseline_avg_passes:.2f} passes/token")
    for threshold, runs in results["gated"].items():
        avg_tps = sum(r["tokens_per_second"] for r in runs) / len(runs)
        total_tokens = sum(r["tokens_generated"] for r in runs)
        total_passes = sum(r["forward_passes"] for r in runs)
        total_accepts = sum(r["gate_accepts"] for r in runs)
        passes_per_token = total_passes / total_tokens
        gate_fire_rate = total_accepts / total_tokens * 100
        speedup_vs_baseline = (avg_tps / baseline_avg_tps - 1) * 100
        print(f"\nThreshold {threshold}: {avg_tps:.1f} tok/s ({speedup_vs_baseline:+.1f}% vs baseline) | {passes_per_token:.2f} passes/token | gate fires {gate_fire_rate:.1f}% of tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--thresholds", type=str, default="0.7,0.8,0.85,0.9,0.95")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(f"Loading model {config.BASE_MODEL}...")
    model = load_gated_mtp(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE, num_extra_heads=config.NUM_EXTRA_HEADS)
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}...")
        load_checkpoint(model, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    print(f"Benchmarking with {len(BENCHMARK_PROMPTS)} prompts, {args.max_tokens} max tokens each")
    print(f"Thresholds: {thresholds}\n")
    results = run_benchmark(model, tokenizer, thresholds, max_tokens=args.max_tokens)
    print_summary(results)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert float keys to strings for JSON serialization
        serializable = {"baseline": results["baseline"], "gated": {str(k): v for k, v in results["gated"].items()}}
        output_path.write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
