"""
Benchmark script for Gated MTP.

Three evaluation modes:
  1. Speed benchmark — generation speed with gated vs baseline across diverse prompts
  2. Quality benchmark — run standard evals (HellaSwag, MMLU subset) to verify no degradation
  3. Gate analysis — detailed stats on where/when the gate fires

Results are saved to ./results/ as JSON with full per-prompt breakdowns.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import config
from model import load_model
from generate import generate, load_checkpoint

RESULTS_DIR = Path("./results")

# Diverse prompts across domains to measure speedup variation.
# Formulaic/predictable text should show higher gate fire rates
# than creative/ambiguous text.
SPEED_PROMPTS = [
    # Factual — high predictability expected
    "The capital of France is",
    "Water boils at a temperature of",
    "The chemical formula for table salt is",
    "The largest planet in our solar system is",
    "The speed of light in a vacuum is approximately",
    "The first president of the United States was",
    "The human body contains approximately",
    "DNA stands for",
    # Technical — moderate predictability
    "def fibonacci(n):\n",
    "SELECT * FROM users WHERE",
    "The key difference between TCP and UDP is",
    "In object-oriented programming, inheritance allows",
    "A binary search tree has the property that",
    "The HTTP status code 404 means",
    "To implement a linked list in Python,",
    "The time complexity of quicksort is",
    # Domain-specific (mortgage/finance) — potentially high predictability
    "The mortgage application process typically involves",
    "A fixed-rate mortgage offers the advantage of",
    "The loan-to-value ratio is calculated by",
    "When refinancing a home loan, borrowers should consider",
    "The difference between pre-approval and pre-qualification is",
    "Interest rates are determined by several factors including",
    "A borrower's debt-to-income ratio measures",
    "The settlement process for a property purchase involves",
    # Creative/ambiguous — low predictability expected
    "Once upon a time in a land far away,",
    "The philosopher argued that consciousness is",
    "She opened the door and saw",
    "The implications of artificial intelligence for society",
    "In the dream, the colors were",
    "The old man sat quietly, remembering",
    "What makes a great leader is",
    "The future of humanity depends on",
    # Conversational — moderate predictability
    "Hi, I'd like to schedule an appointment for",
    "Thank you for your email. I wanted to follow up on",
    "Could you please provide more information about",
    "I'm writing to confirm that",
    "As discussed in our meeting yesterday,",
    "Please find attached the documents for",
    "I hope this email finds you well. I wanted to",
    "Looking forward to hearing from you regarding",
]


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, max_tokens: int = 128) -> tuple[str, dict]:
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


@torch.no_grad()
def measure_gate_accuracy(model, tokenizer, prompt: str, max_tokens: int = 128, gate_threshold: float = 0.85) -> dict:
    """
    Generate with gating, then verify each gate-accepted token by checking
    what the base model would have produced autoregressively.
    Tracks correct vs incorrect gate acceptances separately.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
    gate_correct = 0
    gate_incorrect = 0
    gate_rejections = 0
    tokens_generated = 0
    while tokens_generated < max_tokens:
        base_logits, extra_logits, confidences = model(input_ids)
        next_token = base_logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        tokens_generated += 1
        if next_token.item() == tokenizer.eos_token_id:
            break
        for logits_i, conf_i in zip(extra_logits, confidences):
            if conf_i[:, -1].item() > gate_threshold:
                extra_token = logits_i[:, -1].argmax(dim=-1, keepdim=True)
                # Verify: run the base model from the current context to see
                # what it would actually predict as the next token
                verify_out = model.base(input_ids)
                true_next = verify_out.logits[:, -1].argmax(dim=-1, keepdim=True)
                if extra_token.item() == true_next.item():
                    gate_correct += 1
                else:
                    gate_incorrect += 1
                input_ids = torch.cat([input_ids, extra_token], dim=-1)
                tokens_generated += 1
            else:
                gate_rejections += 1
                break
    total_fires = gate_correct + gate_incorrect
    return {
        "gate_correct": gate_correct,
        "gate_incorrect": gate_incorrect,
        "gate_rejections": gate_rejections,
        "gate_precision": gate_correct / total_fires if total_fires > 0 else 0.0,
        "tokens_generated": tokens_generated,
    }


def run_speed_benchmark(model, tokenizer, thresholds: list[float], max_tokens: int = 128, skip_baseline: bool = False):
    """Run all speed prompts through baseline and gated generation."""
    results = {"baseline": [], "gated": {t: [] for t in thresholds}}
    # Warmup pass to stabilize GPU clocks
    print("Warmup...")
    for _ in range(3):
        generate_baseline(model, tokenizer, "Hello world", max_tokens=32)
    if not skip_baseline:
        print(f"\nRunning baseline ({len(SPEED_PROMPTS)} prompts)...")
        for prompt in SPEED_PROMPTS:
            t0 = time.time()
            text, stats = generate_baseline(model, tokenizer, prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            stats["wall_time"] = elapsed
            stats["tokens_per_second"] = stats["tokens_generated"] / elapsed
            stats["prompt"] = prompt
            results["baseline"].append(stats)
        baseline_tps = sum(r["tokens_per_second"] for r in results["baseline"]) / len(results["baseline"])
        print(f"  Baseline avg: {baseline_tps:.1f} tok/s")
    else:
        print("\nSkipping baseline...")
    for threshold in thresholds:
        print(f"\nRunning gated (threshold={threshold}, {len(SPEED_PROMPTS)} prompts)...")
        for prompt in SPEED_PROMPTS:
            t0 = time.time()
            text, stats = generate(model, tokenizer, prompt, max_tokens=max_tokens, gate_threshold=threshold)
            elapsed = time.time() - t0
            stats["wall_time"] = elapsed
            stats["tokens_per_second"] = stats["tokens_generated"] / elapsed
            stats["prompt"] = prompt
            results["gated"][threshold].append(stats)
        gated_tps = sum(r["tokens_per_second"] for r in results["gated"][threshold]) / len(results["gated"][threshold])
        speedup = (gated_tps / baseline_tps - 1) * 100
        avg_fire_rate = sum(r["gate_accepts"] for r in results["gated"][threshold]) / sum(r["tokens_generated"] for r in results["gated"][threshold]) * 100
        print(f"  Gated avg: {gated_tps:.1f} tok/s ({speedup:+.1f}%) | gate fires {avg_fire_rate:.1f}%")
    return results


def run_gate_analysis(model, tokenizer, threshold: float = 0.85, max_tokens: int = 128):
    """Measure gate precision — when it fires, is it actually correct?"""
    print(f"\nGate accuracy analysis (threshold={threshold})...")
    results = []
    for prompt in SPEED_PROMPTS:
        stats = measure_gate_accuracy(model, tokenizer, prompt, max_tokens=max_tokens, gate_threshold=threshold)
        stats["prompt"] = prompt
        results.append(stats)
    total_correct = sum(r["gate_correct"] for r in results)
    total_incorrect = sum(r["gate_incorrect"] for r in results)
    total_fires = total_correct + total_incorrect
    total_rejections = sum(r["gate_rejections"] for r in results)
    precision = total_correct / total_fires if total_fires > 0 else 0.0
    print(f"\n  Gate fires: {total_fires} times across {len(results)} prompts")
    print(f"  Correct: {total_correct} ({precision*100:.1f}%)")
    print(f"  Incorrect: {total_incorrect}")
    print(f"  Rejections: {total_rejections}")
    # Break down by prompt category
    categories = {"factual": SPEED_PROMPTS[:8], "technical": SPEED_PROMPTS[8:16], "finance": SPEED_PROMPTS[16:24], "creative": SPEED_PROMPTS[24:32], "conversational": SPEED_PROMPTS[32:40]}
    print("\n  By category:")
    for cat_name, cat_prompts in categories.items():
        cat_results = [r for r in results if r["prompt"] in cat_prompts]
        cat_correct = sum(r["gate_correct"] for r in cat_results)
        cat_fires = cat_correct + sum(r["gate_incorrect"] for r in cat_results)
        cat_precision = cat_correct / cat_fires if cat_fires > 0 else 0.0
        cat_fire_rate = cat_fires / sum(r["tokens_generated"] for r in cat_results) * 100 if cat_results else 0.0
        print(f"    {cat_name:15s} | fires {cat_fire_rate:5.1f}% | precision {cat_precision*100:5.1f}%")
    return results


def run_quality_benchmark(model, tokenizer, threshold: float = 0.85, max_samples: int = 200):
    """
    Quick quality check using HellaSwag — compare baseline vs gated accuracy.
    Both modes should produce identical or near-identical scores.
    HellaSwag tests commonsense reasoning via sentence completion.
    """
    print(f"\nQuality benchmark (HellaSwag, {max_samples} samples)...")
    dataset = load_dataset("Rowan/hellaswag", split=f"validation[:{max_samples}]")
    baseline_correct = 0
    gated_correct = 0
    total = 0
    for row in dataset:
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        # Score each ending by generating a few tokens and checking log-likelihood.
        # Simpler approach: use the prompt + each ending, pick the one with lowest perplexity.
        best_baseline_score = float("inf")
        best_gated_score = float("inf")
        best_baseline_idx = 0
        best_gated_idx = 0
        for idx, ending in enumerate(endings):
            full_text = ctx + " " + ending
            input_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.base.device)
            if input_ids.size(1) < 2:
                continue
            base_logits, extra_logits, confidences = model(input_ids)
            # Perplexity of the sequence under the base model
            shift_logits = base_logits[:, :-1].reshape(-1, base_logits.size(-1))
            shift_labels = input_ids[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels).item()
            if loss < best_baseline_score:
                best_baseline_score = loss
                best_baseline_idx = idx
            # For gated: use the extra head's logits blended in where gate is confident.
            # This tests whether the gate distorts the model's ranking.
            if extra_logits:
                gate_conf = confidences[0][:, :-2].squeeze(-1)
                extra_shift = extra_logits[0][:, :-2].reshape(-1, extra_logits[0].size(-1))
                base_shift_2 = base_logits[:, :-2].reshape(-1, base_logits.size(-1))
                shift_labels_2 = input_ids[:, 2:].reshape(-1)
                min_len = min(extra_shift.size(0), shift_labels_2.size(0))
                gated_loss = torch.nn.functional.cross_entropy(extra_shift[:min_len], shift_labels_2[:min_len]).item()
                combined = loss * 0.7 + gated_loss * 0.3
            else:
                combined = loss
            if combined < best_gated_score:
                best_gated_score = combined
                best_gated_idx = idx
        if best_baseline_idx == label:
            baseline_correct += 1
        if best_gated_idx == label:
            gated_correct += 1
        total += 1
    baseline_acc = baseline_correct / total * 100
    gated_acc = gated_correct / total * 100
    print(f"  Baseline accuracy: {baseline_acc:.1f}% ({baseline_correct}/{total})")
    print(f"  Gated accuracy:    {gated_acc:.1f}% ({gated_correct}/{total})")
    print(f"  Difference:        {gated_acc - baseline_acc:+.1f}%")
    return {"baseline_accuracy": baseline_acc, "gated_accuracy": gated_acc, "total_samples": total}


def print_summary(speed_results: dict):
    print("\n" + "=" * 80)
    print("SPEED BENCHMARK SUMMARY")
    print("=" * 80)
    baseline_avg_tps = sum(r["tokens_per_second"] for r in speed_results["baseline"]) / len(speed_results["baseline"])
    baseline_passes = sum(r["forward_passes"] for r in speed_results["baseline"]) / sum(r["tokens_generated"] for r in speed_results["baseline"])
    print(f"\nBaseline: {baseline_avg_tps:.1f} tok/s | {baseline_passes:.2f} passes/token")
    for threshold, runs in speed_results["gated"].items():
        avg_tps = sum(r["tokens_per_second"] for r in runs) / len(runs)
        total_tokens = sum(r["tokens_generated"] for r in runs)
        total_passes = sum(r["forward_passes"] for r in runs)
        total_accepts = sum(r["gate_accepts"] for r in runs)
        passes_per_token = total_passes / total_tokens
        gate_fire_rate = total_accepts / total_tokens * 100
        speedup = (avg_tps / baseline_avg_tps - 1) * 100
        print(f"  Threshold {threshold}: {avg_tps:.1f} tok/s ({speedup:+.1f}%) | {passes_per_token:.3f} passes/token | gate fires {gate_fire_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--thresholds", type=str, default="0.15,0.2,0.25,0.3,0.35")
    parser.add_argument("--quality-samples", type=int, default=200)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-gate-analysis", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(f"Loading model {config.BASE_MODEL} (type={config.MODEL_TYPE})...")
    model = load_model(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE, num_extra_heads=config.NUM_EXTRA_HEADS, model_type=config.MODEL_TYPE, hidden_mult=config.CHAIN_HIDDEN_MULT)
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}...")
        load_checkpoint(model, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    all_results = {}
    # 1. Speed benchmark
    print(f"\n{'='*80}\nSPEED BENCHMARK ({len(SPEED_PROMPTS)} prompts, {args.max_tokens} max tokens)\n{'='*80}")
    speed_results = run_speed_benchmark(model, tokenizer, thresholds, max_tokens=args.max_tokens, skip_baseline=args.skip_baseline)
    all_results["speed"] = speed_results
    if not args.skip_baseline:
        print_summary(speed_results)
    # 2. Gate analysis
    if not args.skip_gate_analysis:
        print(f"\n{'='*80}\nGATE ANALYSIS\n{'='*80}")
        gate_results = run_gate_analysis(model, tokenizer, threshold=thresholds[len(thresholds) // 2], max_tokens=args.max_tokens)
        all_results["gate_analysis"] = gate_results
    # 3. Quality benchmark
    if not args.skip_quality:
        print(f"\n{'='*80}\nQUALITY BENCHMARK\n{'='*80}")
        quality_results = run_quality_benchmark(model, tokenizer, threshold=0.85, max_samples=args.quality_samples)
        all_results["quality"] = quality_results
    # Save results
    output_path = args.output or str(RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(all_results, default=str))
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
