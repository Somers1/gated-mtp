"""
Stage 2 training script: learned local FFN routers + skip predictors.

Loss = logit_kl (quality) + lambda_compute * compute_cost (efficiency)

The dense teacher is the same frozen model run without sparsity.
Only router and skip predictor parameters are trainable.
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
from local_router import load_locally_routed_model

LOG_DIR = Path("./logs")


from sparse_utils import logit_kl_loss


def compute_cost_loss(aux_stats: list[dict], target_sparsity: float = 0.9) -> torch.Tensor:
    """
    Penalize compute usage — differentiable via skip_prob_tensor.

    FFN fraction is fixed by sparsity config (not differentiable), but skip_prob
    is on-graph so the optimizer can learn to skip layers.
    """
    skip_costs = []
    for stat in aux_stats:
        if "skip_prob_tensor" in stat:
            # Penalize not-skipping: (1 - skip_prob) encourages higher skip rates
            skip_costs.append(1.0 - stat["skip_prob_tensor"])
    if not skip_costs:
        return torch.tensor(0.0)
    return torch.stack(skip_costs).mean()


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
    # Hyperparams
    lr = float(config.ROUTER_LR)
    lambda_compute = 0.1
    temperature = 2.0
    num_epochs = 1  # Single pass is usually enough for router training

    print(f"Loading tokenizer from {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    print(f"Loading model with local routers (sparsity={config.SPARSITY})...")
    model = load_locally_routed_model(
        model_name=config.BASE_MODEL,
        device=config.DEVICE,
        dtype=config.DTYPE,
        bottleneck=config.ROUTER_BOTTLENECK,
        block_size=16,
        sparsity=config.SPARSITY,
    )
    device = next(model.base.parameters()).device
    print(f"Base model on {device}")
    print(f"Trainable params: {model.trainable_param_count:,} ({model.trainable_param_count / 1e6:.1f}M)")

    train_dataset = build_dataset(tokenizer)
    print(f"Training on {len(train_dataset)} chunks")
    dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.trainable_params, lr=lr)
    # Cosine schedule over total steps
    total_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_local_{run_id}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "step", "loss", "kl_loss", "compute_loss",
        "mean_skip_prob", "lr", "elapsed_s",
    ])
    print(f"Logging to {log_path}")

    t0 = time.time()
    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            result = model.forward_train(input_ids)

            kl = logit_kl_loss(
                result["dense_logits"], result["sparse_logits"], temperature
            )
            comp = compute_cost_loss(result["aux_stats"], config.SPARSITY)

            loss = kl + lambda_compute * comp.to(kl.device)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.trainable_params, max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                mean_skip = sum(
                    s["skip_prob"] for s in result["aux_stats"]
                ) / len(result["aux_stats"])
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch+1} step {step}/{len(dataloader)} | "
                    f"loss: {loss.item():.4f} | kl: {kl.item():.4f} | "
                    f"comp: {comp.item():.4f} | skip: {mean_skip:.3f} | "
                    f"lr: {current_lr:.2e} | {elapsed:.0f}s"
                )
                log_writer.writerow([
                    epoch + 1, step, f"{loss.item():.6f}",
                    f"{kl.item():.6f}", f"{comp.item():.6f}",
                    f"{mean_skip:.6f}", f"{current_lr:.2e}", f"{elapsed:.0f}",
                ])
                log_file.flush()

    log_file.close()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Save checkpoint
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "local_routers.pt"
    save_dict = {
        "ffn_router_states": {
            f"router_{i}": r.state_dict()
            for i, r in enumerate(model.ffn_routers)
        },
        "skip_predictor_states": {
            f"skip_{i}": s.state_dict()
            for i, s in enumerate(model.skip_predictors)
        },
        "config": {
            "base_model": config.BASE_MODEL,
            "bottleneck": config.ROUTER_BOTTLENECK,
            "sparsity": config.SPARSITY,
            "block_size": model.block_size,
            "num_layers": model.num_layers,
            "ffn_dims": model.ffn_dims,
        },
    }
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
