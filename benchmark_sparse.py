import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from router import SparseRouter, LayerRouter
from benchmark import SPEED_PROMPTS

RESULTS_DIR = Path("./results")
SPARSITY_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]


def load_model_with_router(checkpoint_path: str | None = None) -> tuple[SparseRouter, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    torch_dtype = getattr(torch, config.DTYPE)
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch_dtype, device_map=config.DEVICE)
    model = SparseRouter(base, bottleneck=config.ROUTER_BOTTLENECK)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=base.device, weights_only=True)
        for i, router in enumerate(model.routers):
            router.load_state_dict(ckpt["router_states"][f"router_{i}"])
        print(f"Loaded router checkpoint from {checkpoint_path}")
    for p in model.trainable_params:
        p.data = p.data.to(device=base.device, dtype=torch.float32)
    return model, tokenizer


@torch.no_grad()
def measure_full_speed(model: SparseRouter, tokenizer, prompts: list[str], max_tokens: int = 64) -> dict:
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
        for _ in range(max_tokens):
            outputs = model.base(input_ids)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            total_tokens += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.time() - t0
    return {"tokens": total_tokens, "elapsed": elapsed, "tokens_per_second": total_tokens / elapsed}


@torch.no_grad()
def measure_sparse_speed(model: SparseRouter, tokenizer, prompts: list[str], sparsity: float, max_tokens: int = 64) -> dict:
    layers = model._get_layers()
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
        for _ in range(max_tokens):
            hooks = []
            for i, layer_module in enumerate(layers):
                ffn = model._get_ffn(layer_module)
                router = model.routers[i]
                hooks.append(install_sparse_hook(ffn, router, sparsity))
            outputs = model.base(input_ids)
            for h in hooks:
                h.remove()
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            total_tokens += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.time() - t0
    return {"tokens": total_tokens, "elapsed": elapsed, "tokens_per_second": total_tokens / elapsed, "sparsity": sparsity}


def install_sparse_hook(ffn, router: LayerRouter, sparsity: float):
    original_forward = ffn.forward

    def sparse_forward(hidden_state):
        scores, mask = router(hidden_state.float(), sparsity=sparsity)
        active_idx = mask[0, 0].nonzero(as_tuple=True)[0]
        gate_w = ffn.gate_proj.weight[active_idx]
        up_w = ffn.up_proj.weight[active_idx]
        down_w = ffn.down_proj.weight[:, active_idx]
        gate_out = torch.nn.functional.linear(hidden_state, gate_w)
        up_out = torch.nn.functional.linear(hidden_state, up_w)
        activated = torch.nn.functional.silu(gate_out) * up_out
        return torch.nn.functional.linear(activated, down_w)

    ffn.forward = sparse_forward
    class HookHandle:
        def remove(self):
            ffn.forward = original_forward
    return HookHandle()


@torch.no_grad()
def measure_reconstruction_error(model: SparseRouter, tokenizer, prompts: list[str], sparsity: float) -> list[dict]:
    layer_errors = [{} for _ in range(model.num_layers)]
    n_samples = 0
    for prompt in prompts[:8]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.base.device)
        result = model.forward_train(input_ids, sparsity=sparsity)
        for stat in result["layer_stats"]:
            layer_i = stat["layer"]
            for key in ("relative_error", "loss"):
                layer_errors[layer_i][key] = layer_errors[layer_i].get(key, 0.0) + stat[key]
        n_samples += 1
    per_layer = []
    for i, errs in enumerate(layer_errors):
        per_layer.append({"layer": i, "avg_relative_error": errs.get("relative_error", 0) / n_samples, "avg_loss": errs.get("loss", 0) / n_samples})
    return sorted(per_layer, key=lambda x: x["avg_relative_error"])


def run_benchmark(checkpoint: str | None, max_tokens: int):
    print(f"Loading model {config.BASE_MODEL} with sparse router...")
    model, tokenizer = load_model_with_router(checkpoint)
    device = model.base.device
    print(f"Model on {device}, {len(SPEED_PROMPTS)} prompts, {max_tokens} max tokens")
    all_results = {"config": {"base_model": config.BASE_MODEL, "bottleneck": config.ROUTER_BOTTLENECK, "max_tokens": max_tokens, "checkpoint": checkpoint}, "full_inference": {}, "sparse_inference": {}, "reconstruction_error": {}}
    print("\nWarmup...")
    for _ in range(3):
        input_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
        model.base(input_ids)
    print("\nFull inference (baseline)...")
    full_result = measure_full_speed(model, tokenizer, SPEED_PROMPTS, max_tokens)
    all_results["full_inference"] = full_result
    print(f"  {full_result['tokens_per_second']:.1f} tok/s ({full_result['tokens']} tokens in {full_result['elapsed']:.1f}s)")
    for sparsity in SPARSITY_LEVELS:
        print(f"\nSparse inference (sparsity={sparsity})...")
        sparse_result = measure_sparse_speed(model, tokenizer, SPEED_PROMPTS, sparsity, max_tokens)
        speedup = (sparse_result["tokens_per_second"] / full_result["tokens_per_second"] - 1) * 100
        print(f"  {sparse_result['tokens_per_second']:.1f} tok/s ({speedup:+.1f}% vs full)")
        all_results["sparse_inference"][str(sparsity)] = sparse_result
        print(f"\n  Reconstruction error per layer (sparsity={sparsity}):")
        per_layer = measure_reconstruction_error(model, tokenizer, SPEED_PROMPTS, sparsity)
        all_results["reconstruction_error"][str(sparsity)] = per_layer
        best = per_layer[0]
        worst = per_layer[-1]
        avg_re = sum(l["avg_relative_error"] for l in per_layer) / len(per_layer)
        print(f"  avg_re: {avg_re:.4f} | best: layer {best['layer']} ({best['avg_relative_error']:.4f}) | worst: layer {worst['layer']} ({worst['avg_relative_error']:.4f})")
    print_summary(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"sparse_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


def print_summary(results: dict):
    print(f"\n{'=' * 80}")
    print("SPARSE BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    full_tps = results["full_inference"]["tokens_per_second"]
    print(f"\n  Full inference: {full_tps:.1f} tok/s")
    for sparsity_str, sparse in results["sparse_inference"].items():
        sparse_tps = sparse["tokens_per_second"]
        speedup = (sparse_tps / full_tps - 1) * 100
        re_data = results["reconstruction_error"][sparsity_str]
        avg_re = sum(l["avg_relative_error"] for l in re_data) / len(re_data)
        print(f"  Sparsity {sparsity_str}: {sparse_tps:.1f} tok/s ({speedup:+.1f}%) | avg_re: {avg_re:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.max_tokens)


if __name__ == "__main__":
    main()
