"""
Unified benchmark for all sparse routing stages.

Compares:
1. Dense baseline (no sparsity)
2. Gate threshold (Stage 1a: compute full gate, skip inactive up+down)
3. SVD predictor (Stage 1b: approximate gate, skip all three for inactive)
4. Learned local routers (Stage 2, if checkpoint provided)
5. Hierarchical router (Stage 3, if checkpoint provided)

Metrics per approach:
- Tokens per second (wall-clock)
- Logit KL divergence vs dense (quality measure)
- Per-layer compute stats
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from sparse_utils import (
    get_layers, get_ffn, load_frozen_model, get_ffn_dims,
)
from svd_baseline import SVDSparseModel, GateThresholdPredictor, SVDPredictor

RESULTS_DIR = Path("./results")

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

SPARSITY_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]


@torch.no_grad()
def get_dense_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run dense model and return logits."""
    return model(input_ids).logits


@torch.no_grad()
def measure_logit_kl(dense_logits: torch.Tensor, sparse_logits: torch.Tensor) -> float:
    """Measure KL divergence between dense and sparse logits."""
    teacher = F.softmax(dense_logits, dim=-1)
    student = F.log_softmax(sparse_logits, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean").item()


@torch.no_grad()
def measure_speed(model, tokenizer, prompts: list[str], max_tokens: int = 64) -> dict:
    """Measure tok/s for autoregressive generation."""
    device = next(model.parameters()).device
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        for _ in range(max_tokens):
            outputs = model(input_ids)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            total_tokens += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.time() - t0
    return {"tokens": total_tokens, "elapsed": elapsed, "tok_per_s": total_tokens / elapsed}


@torch.no_grad()
def measure_speed_with_hooks(
    model, tokenizer, prompts: list[str],
    sparse_model: SVDSparseModel, max_tokens: int = 64,
) -> dict:
    """Measure tok/s with sparse hooks installed."""
    device = next(model.parameters()).device
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        for _ in range(max_tokens):
            handles = sparse_model.install_hooks()
            outputs = model(input_ids)
            sparse_model.remove_hooks(handles)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            total_tokens += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.time() - t0
    return {"tokens": total_tokens, "elapsed": elapsed, "tok_per_s": total_tokens / elapsed}


@torch.no_grad()
def measure_quality(
    model, tokenizer, prompts: list[str],
    sparse_model: SVDSparseModel | None = None,
) -> float:
    """Measure average logit KL divergence across prompts."""
    device = next(model.parameters()).device
    total_kl = 0.0
    n = 0
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        dense_logits = model(input_ids).logits

        if sparse_model is not None:
            handles = sparse_model.install_hooks()
            sparse_logits = model(input_ids).logits
            sparse_model.remove_hooks(handles)
        else:
            sparse_logits = dense_logits

        total_kl += measure_logit_kl(dense_logits, sparse_logits)
        n += 1
    return total_kl / n


def warmup(model, tokenizer, device, n: int = 3):
    """Warmup GPU caches."""
    for _ in range(n):
        ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
        model(ids)


def benchmark_dense(model, tokenizer, max_tokens: int) -> dict:
    """Benchmark dense baseline."""
    print("\n[1/6] Dense baseline...")
    speed = measure_speed(model, tokenizer, PROMPTS, max_tokens)
    print(f"  {speed['tok_per_s']:.1f} tok/s")
    return {"speed": speed, "kl": 0.0}


def benchmark_static_skip(model, tokenizer, max_tokens: int, skip_layers: list[int]) -> dict:
    """Benchmark static layer skipping using profiler-identified low-importance layers."""
    print(f"\n[2/6] Static layer skip (skipping layers {skip_layers})...")
    sparse = SVDSparseModel(model, predictor_type="gate", sparsity=0.0, skip_layers=skip_layers)
    speed = measure_speed_with_hooks(model, tokenizer, PROMPTS, sparse, max_tokens)
    kl = measure_quality(model, tokenizer, PROMPTS, sparse)
    print(f"  {speed['tok_per_s']:.1f} tok/s | KL={kl:.4f}")
    return {"speed": speed, "kl": kl, "skip_layers": skip_layers}


def benchmark_gate_threshold(model, tokenizer, max_tokens: int) -> dict:
    """Benchmark gate thresholding at multiple sparsity levels."""
    print("\n[3/6] Gate threshold baseline...")
    results = {}
    for sparsity in SPARSITY_LEVELS:
        sparse = SVDSparseModel(model, predictor_type="gate", sparsity=sparsity)
        speed = measure_speed_with_hooks(model, tokenizer, PROMPTS, sparse, max_tokens)
        kl = measure_quality(model, tokenizer, PROMPTS, sparse)
        print(f"  sparsity={sparsity:.0%}: {speed['tok_per_s']:.1f} tok/s | KL={kl:.4f}")
        results[str(sparsity)] = {"speed": speed, "kl": kl}
    return results


def benchmark_svd(model, tokenizer, max_tokens: int, rank: int = 128) -> dict:
    """Benchmark SVD predictor at multiple sparsity levels."""
    print(f"\n[4/6] SVD predictor (rank={rank})...")
    results = {}
    for sparsity in SPARSITY_LEVELS:
        sparse = SVDSparseModel(model, predictor_type="svd", rank=rank, sparsity=sparsity)
        speed = measure_speed_with_hooks(model, tokenizer, PROMPTS, sparse, max_tokens)
        kl = measure_quality(model, tokenizer, PROMPTS, sparse)
        print(f"  sparsity={sparsity:.0%}: {speed['tok_per_s']:.1f} tok/s | KL={kl:.4f}")
        results[str(sparsity)] = {"speed": speed, "kl": kl}
    return results


def benchmark_local_router(model, tokenizer, max_tokens: int, checkpoint_path: str) -> dict:
    """Benchmark Stage 2 learned local routers."""
    print(f"\n[5/6] Learned local routers ({checkpoint_path})...")
    from local_router import LocallyRoutedModel, LocalFFNRouter, LocalSkipPredictor
    from sparse_utils import get_text_config

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_config = ckpt["config"]
    text_config = get_text_config(model)
    hidden_dim = text_config.hidden_size

    # Build the locally routed model
    routed = LocallyRoutedModel(
        model,
        bottleneck=ckpt_config["bottleneck"],
        block_size=ckpt_config["block_size"],
        sparsity=ckpt_config["sparsity"],
    )

    # Load router weights
    for i, router in enumerate(routed.ffn_routers):
        key = f"router_{i}"
        if key in ckpt["ffn_router_states"]:
            router.load_state_dict(ckpt["ffn_router_states"][key])
    for i, skip_pred in enumerate(routed.skip_predictors):
        key = f"skip_{i}"
        if key in ckpt["skip_predictor_states"]:
            skip_pred.load_state_dict(ckpt["skip_predictor_states"][key])

    device = next(model.parameters()).device
    for p in routed.trainable_params:
        p.data = p.data.to(device=device, dtype=torch.float32)

    # Quality measurement via forward_train
    total_kl = 0.0
    n = 0
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        result = routed.forward_train(input_ids)
        kl = measure_logit_kl(result["dense_logits"], result["sparse_logits"])
        total_kl += kl
        n += 1
    avg_kl = total_kl / n
    print(f"  KL={avg_kl:.4f} | sparsity={ckpt_config['sparsity']:.0%}")

    # Speed would require inference hooks (not implemented in local_router yet)
    # For now report quality only
    return {"kl": avg_kl, "sparsity": ckpt_config["sparsity"]}


def benchmark_hierarchical(model, tokenizer, max_tokens: int, checkpoint_path: str) -> dict:
    """Benchmark Stage 3 hierarchical router."""
    print(f"\n[6/6] Hierarchical router ({checkpoint_path})...")
    from hierarchical_router import HierarchicalSparseModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_config = ckpt["config"]

    hier = HierarchicalSparseModel(
        model,
        controller_dim=ckpt_config["controller_dim"],
        control_dim=ckpt_config["control_dim"],
        router_dim=ckpt_config["router_dim"],
        block_size=ckpt_config["block_size"],
        summary_layer_indices=ckpt_config["summary_layer_indices"],
    )
    hier.controller.load_state_dict(ckpt["controller_state"])
    for k, state in ckpt["local_router_states"].items():
        hier.local_routers[k].load_state_dict(state)

    device = next(model.parameters()).device
    for p in hier.trainable_params:
        p.data = p.data.to(device=device, dtype=torch.float32)

    # Quality measurement
    total_kl = 0.0
    n = 0
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        result = hier.forward_train(input_ids)
        kl = measure_logit_kl(result["dense_logits"], result["sparse_logits"])
        total_kl += kl
        n += 1
    avg_kl = total_kl / n

    # Report mean budget from last run
    ctrl = result["controller_out"]
    mean_budget = ctrl["ffn_budget"].mean().item()
    mean_skip = ctrl["skip_score"].mean().item()
    print(f"  KL={avg_kl:.4f} | mean_budget={mean_budget:.3f} | mean_skip={mean_skip:.3f}")

    return {"kl": avg_kl, "mean_budget": mean_budget, "mean_skip": mean_skip}


def print_summary(results: dict, dense_tps: float):
    """Print a comparison table."""
    print(f"\n{'=' * 90}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Method':<30} {'Sparsity':>10} {'tok/s':>10} {'Speedup':>10} {'KL div':>10}")
    print("-" * 90)

    print(f"{'Dense (baseline)':<30} {'0%':>10} {dense_tps:>10.1f} {'1.00x':>10} {'0.0000':>10}")

    for method_name, method_results in results.items():
        if method_name == "dense":
            continue
        if isinstance(method_results, dict) and "speed" in method_results:
            # Single result
            tps = method_results["speed"]["tok_per_s"]
            speedup = tps / dense_tps
            kl = method_results.get("kl", 0)
            print(f"{method_name:<30} {'':>10} {tps:>10.1f} {speedup:>9.2f}x {kl:>10.4f}")
        elif isinstance(method_results, dict):
            # Multi-sparsity results
            for sparsity, data in sorted(method_results.items()):
                if isinstance(data, dict) and "speed" in data:
                    tps = data["speed"]["tok_per_s"]
                    speedup = tps / dense_tps
                    kl = data.get("kl", 0)
                    label = f"{method_name} ({sparsity})"
                    print(f"{label:<30} {sparsity:>10} {tps:>10.1f} {speedup:>9.2f}x {kl:>10.4f}")
                elif isinstance(data, (int, float)):
                    pass


def main():
    parser = argparse.ArgumentParser(description="Unified sparse routing benchmark")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--svd-rank", type=int, default=128)
    parser.add_argument("--local-checkpoint", type=str, default=None,
                        help="Stage 2 local router checkpoint")
    parser.add_argument("--hier-checkpoint", type=str, default=None,
                        help="Stage 3 hierarchical router checkpoint")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense baseline speed test (slow)")
    parser.add_argument("--skip-layers", type=str, default="28,29,30,31,32",
                        help="Comma-separated layer indices to skip (from profiler)")
    args = parser.parse_args()
    skip_layers = [int(x) for x in args.skip_layers.split(",") if x.strip()]

    print(f"Loading model {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = load_frozen_model(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE)
    device = next(model.parameters()).device
    print(f"Model on {device}")

    print("\nWarmup...")
    warmup(model, tokenizer, device)

    all_results = {}

    # 1. Dense baseline
    if not args.skip_dense:
        dense_result = benchmark_dense(model, tokenizer, args.max_tokens)
        all_results["dense"] = dense_result
        dense_tps = dense_result["speed"]["tok_per_s"]
    else:
        print("\n[1/6] Skipping dense baseline")
        dense_tps = 1.0  # placeholder

    # 2. Static layer skip (free win from profiler results)
    static_skip_result = benchmark_static_skip(
        model, tokenizer, args.max_tokens, skip_layers,
    )
    all_results["static_skip"] = static_skip_result

    # 3. Gate threshold
    gate_results = benchmark_gate_threshold(model, tokenizer, args.max_tokens)
    all_results["gate_threshold"] = gate_results

    # 4. SVD predictor
    svd_results = benchmark_svd(model, tokenizer, args.max_tokens, rank=args.svd_rank)
    all_results["svd_predictor"] = svd_results

    # 5. Learned local routers (if checkpoint exists)
    if args.local_checkpoint and Path(args.local_checkpoint).exists():
        local_results = benchmark_local_router(
            model, tokenizer, args.max_tokens, args.local_checkpoint,
        )
        all_results["local_router"] = local_results
    else:
        print("\n[5/6] Skipping local routers (no checkpoint)")

    # 6. Hierarchical router (if checkpoint exists)
    if args.hier_checkpoint and Path(args.hier_checkpoint).exists():
        hier_results = benchmark_hierarchical(
            model, tokenizer, args.max_tokens, args.hier_checkpoint,
        )
        all_results["hierarchical"] = hier_results
    else:
        print("\n[6/6] Skipping hierarchical router (no checkpoint)")

    # Summary
    print_summary(all_results, dense_tps)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"benchmark_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
