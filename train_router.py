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
from router import load_sparse_router
from train import CACHE_DIR, TokenizedDataset

LOG_DIR = Path("./logs")


def build_dataset(tokenizer) -> TokenizedDataset:
    cache_key = hashlib.md5(f"{config.BASE_MODEL}:{config.DATASET}:{config.DATASET_SUBSET}:{config.MAX_TRAIN_SAMPLES}:{config.SEQ_LEN}".encode()).hexdigest()[:12]
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
    print(f"Loading tokenizer from {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    print(f"Loading model {config.BASE_MODEL} with sparse router (bottleneck={config.ROUTER_BOTTLENECK})...")
    model = load_sparse_router(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE, bottleneck=config.ROUTER_BOTTLENECK)
    device = model.base.device
    print(f"Base model on {device}, router params: {model.trainable_param_count:,} ({model.trainable_param_count / 1e6:.1f}M)")
    train_dataset = build_dataset(tokenizer)
    print(f"Training on {len(train_dataset)} chunks, sparsity={config.SPARSITY}")
    dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.trainable_params, lr=config.ROUTER_LR)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_router_{run_id}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "avg_relative_error", "worst_layer", "worst_relative_error", "elapsed_s"])
    print(f"Logging to {log_path}")
    t0 = time.time()
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        result = model.forward_train(input_ids, sparsity=config.SPARSITY)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            stats = result["layer_stats"]
            avg_re = sum(s["relative_error"] for s in stats) / len(stats)
            worst = max(stats, key=lambda s: s["relative_error"])
            elapsed = time.time() - t0
            print(f"  step {step}/{len(dataloader)} | loss: {loss.item():.4f} | avg_re: {avg_re:.4f} | worst: layer {worst['layer']} ({worst['relative_error']:.4f}) | {elapsed:.0f}s")
            log_writer.writerow([step, f"{loss.item():.6f}", f"{avg_re:.6f}", worst["layer"], f"{worst['relative_error']:.6f}", f"{elapsed:.0f}"])
            log_file.flush()
    log_file.close()
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.0f}s")
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "sparse_router.pt"
    save_dict = {"router_states": {f"router_{i}": r.state_dict() for i, r in enumerate(model.routers)}, "config": {"base_model": config.BASE_MODEL, "bottleneck": config.ROUTER_BOTTLENECK, "sparsity": config.SPARSITY, "num_layers": model.num_layers}}
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
