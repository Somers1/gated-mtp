"""
Stage 3 training script: global controller + local routers.

Loss = logit_kl + lambda_compute * compute_cost + lambda_budget * budget_target

Supports warm-starting from Stage 2 local router checkpoint.
"""
import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import config
from train import TokenizedDataset, CACHE_DIR
from hierarchical_router import load_hierarchical_model

LOG_DIR = Path("./logs")


from sparse_utils import logit_kl_loss


def compute_cost_loss(aux_stats: list[dict]) -> torch.Tensor:
    """Penalize active compute — differentiable via on-graph tensors."""
    costs = []
    for stat in aux_stats:
        if "skip_score_tensor" in stat:
            costs.append(1.0 - stat["skip_score_tensor"])
        if "ffn_budget_tensor" in stat:
            costs.append(stat["ffn_budget_tensor"])
    if not costs:
        return torch.tensor(0.0)
    return torch.stack(costs).mean()


def budget_target_loss(controller_out: dict, target: float = 0.1) -> torch.Tensor:
    """
    Encourage mean budget to hit a target operating point.

    target=0.1 means we want ~10% of compute on average.
    """
    mean_ffn = controller_out["ffn_budget"].mean()
    return (mean_ffn - target).pow(2)


def controller_smoothness_loss(controller_out: dict) -> torch.Tensor:
    """Discourage wildly oscillating budgets between adjacent layers."""
    ffn_budget = controller_out["ffn_budget"]  # [B, L]
    ffn_diff = (ffn_budget[:, 1:] - ffn_budget[:, :-1]).abs().mean()
    return ffn_diff


def build_dataset(tokenizer) -> TokenizedDataset:
    """Build or load cached tokenized dataset."""
    cache_key = hashlib.md5(
        f"{config.BASE_MODEL}:{config.DATASET}:{config.DATASET_SUBSET}:"
        f"{config.MAX_TRAIN_SAMPLES}:{config.SEQ_LEN}".encode()
    ).hexdigest()[:12]
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        return TokenizedDataset.from_texts([], tokenizer, config.SEQ_LEN, cache_key)
    print(f"Loading dataset {config.DATASET} (streaming, first {config.MAX_TRAIN_SAMPLES} samples)...")
    dataset = load_dataset(config.DATASET, config.DATASET_SUBSET, split="train", streaming=True)
    texts = []
    for row in dataset:
        if len(row["text"]) > 100:
            texts.append(row["text"])
        if len(texts) >= config.MAX_TRAIN_SAMPLES:
            break
    return TokenizedDataset.from_texts(texts, tokenizer, config.SEQ_LEN, cache_key)


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-checkpoint", type=str, default=None,
                        help="Stage 2 local router checkpoint for warm-start")
    parser.add_argument("--target-budget", type=float, default=0.1,
                        help="Target mean compute fraction (0.1 = 10%%)")
    parser.add_argument("--lambda-compute", type=float, default=0.1)
    parser.add_argument("--lambda-budget", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    lr = float(config.ROUTER_LR)

    print(f"Loading tokenizer from {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    print(f"Loading model with hierarchical routing...")
    model = load_hierarchical_model(
        model_name=config.BASE_MODEL,
        device=config.DEVICE,
        dtype=config.DTYPE,
        local_router_checkpoint=args.local_checkpoint,
    )
    device = next(model.base.parameters()).device
    print(f"Base model on {device}")
    print(f"Dense prefix: {model.dense_prefix} layers (run dense)")
    print(f"Routed layers: {model.dense_prefix}-{model.num_layers - 1}")
    print(f"Trainable params: {model.trainable_param_count:,} ({model.trainable_param_count / 1e6:.1f}M)")
    print(f"Target budget: {args.target_budget:.0%}")

    train_dataset = build_dataset(tokenizer)
    print(f"Training on {len(train_dataset)} chunks")
    dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.trainable_params, lr=lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_hierarchical_{run_id}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "step", "loss", "kl_loss", "compute_loss", "budget_loss",
        "smooth_loss", "mean_ffn_budget", "mean_skip_score", "lr", "elapsed_s",
    ])
    print(f"Logging to {log_path}")

    t0 = time.time()

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            result = model.forward_train(input_ids)

            # Losses
            kl = logit_kl_loss(
                result["dense_logits"], result["sparse_logits"], args.temperature,
            )
            comp = compute_cost_loss(result["aux_stats"]).to(kl.device)
            budget = budget_target_loss(result["controller_out"], args.target_budget).to(kl.device)
            smooth = controller_smoothness_loss(result["controller_out"]).to(kl.device)

            loss = (
                kl
                + args.lambda_compute * comp
                + args.lambda_budget * budget
                + args.lambda_smooth * smooth
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % 50 == 0:
                elapsed = time.time() - t0
                ctrl = result["controller_out"]
                mean_budget = ctrl["ffn_budget"].mean().item()
                mean_skip = ctrl["skip_score"].mean().item()
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch+1} step {step}/{len(dataloader)} | "
                    f"loss: {loss.item():.4f} | kl: {kl.item():.4f} | "
                    f"budget: {mean_budget:.3f} | skip: {mean_skip:.3f} | "
                    f"lr: {current_lr:.2e} | {elapsed:.0f}s"
                )
                log_writer.writerow([
                    epoch + 1, step, f"{loss.item():.6f}",
                    f"{kl.item():.6f}", f"{comp.item():.6f}",
                    f"{budget.item():.6f}", f"{smooth.item():.6f}",
                    f"{mean_budget:.6f}", f"{mean_skip:.6f}",
                    f"{current_lr:.2e}", f"{elapsed:.0f}",
                ])
                log_file.flush()

    log_file.close()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Save checkpoint
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "hierarchical_router.pt"
    save_dict = {
        "controller_state": model.controller.state_dict(),
        "local_router_states": {
            k: v.state_dict() for k, v in model.local_routers.items()
        },
        "config": {
            "base_model": config.BASE_MODEL,
            "controller_dim": model.controller.planner[0].in_features // 2,
            "control_dim": model.controller.control_dim,
            "router_dim": next(iter(model.local_routers.values())).trunk[0].in_features // 3,
            "block_size": model.block_size,
            "summary_layer_indices": model.summary_layer_indices,
            "num_layers": model.num_layers,
            "ffn_dims": model.ffn_dims,
        },
    }
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
