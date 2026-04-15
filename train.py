"""
Training script for Gated MTP.

Trains only the extra prediction heads and confidence gates while keeping
the base model completely frozen. The training has two objectives per head:

1. Head loss — teach the head to predict token t+offset (cross-entropy)
2. Gate loss — teach the gate to output calibrated confidence scores
   (binary cross-entropy against whether the head was actually correct)

Further-ahead heads get decaying loss weights (0.8^i) since they're
inherently harder and less important than nearer predictions.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import config
from model import load_gated_mtp

CACHE_DIR = Path("./data_cache")


class TokenizedDataset(torch.utils.data.Dataset):
    """
    Pre-tokenizes a text dataset into fixed-length chunks for training.
    Caches the tokenized chunks to disk so subsequent runs load instantly.
    """

    def __init__(self, chunks: np.ndarray, seq_len: int):
        self.chunks = chunks
        self.seq_len = seq_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.chunks[idx], dtype=torch.long)}

    @classmethod
    def from_texts(cls, texts: list[str], tokenizer, seq_len: int, cache_key: str):
        """
        Tokenize texts and cache to disk. On subsequent runs with the same
        config, loads the cached numpy array instead of re-tokenizing.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{cache_key}.npy"
        if cache_path.exists():
            print(f"Loading cached tokenized data from {cache_path}...")
            chunks = np.load(cache_path)
            return cls(chunks, seq_len)
        print(f"Tokenizing {len(texts)} documents into {seq_len}-token chunks...")
        all_ids = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
            all_ids.extend(ids)
        chunk_list = [all_ids[i:i + seq_len] for i in range(0, len(all_ids) - seq_len, seq_len)]
        chunks = np.array(chunk_list, dtype=np.int32)
        np.save(cache_path, chunks)
        print(f"Cached {len(chunks)} chunks to {cache_path}")
        return cls(chunks, seq_len)


def compute_loss(extra_logits: list, confidences: list, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Compute combined head + gate loss for all extra prediction heads.

    For each extra head i:
      - The head predicts token at position t + (i+2) from hidden state at position t
      - We check if that prediction matches the actual token
      - The gate learns to predict whether the head will be correct

    The decay weight (0.8^i) means we care most about the nearest extra head
    and progressively less about further-ahead heads.

    Returns the total loss and a stats dict for logging.
    """
    total_loss = torch.tensor(0.0, device=input_ids.device)
    stats = {}
    for i, (logits_i, conf_i) in enumerate(zip(extra_logits, confidences)):
        offset = i + 2
        logits_trimmed = logits_i[:, :-offset]
        target = input_ids[:, offset:]
        conf_trimmed = conf_i[:, :-offset]
        min_len = min(logits_trimmed.size(1), target.size(1))
        logits_trimmed = logits_trimmed[:, :min_len]
        target = target[:, :min_len]
        conf_trimmed = conf_trimmed[:, :min_len]
        loss_head = nn.functional.cross_entropy(logits_trimmed.reshape(-1, logits_trimmed.size(-1)), target.reshape(-1))
        with torch.no_grad():
            preds = logits_trimmed.argmax(dim=-1)
            correct = (preds == target).float()
        loss_gate = nn.functional.binary_cross_entropy(conf_trimmed.squeeze(-1), correct)
        weight = 0.8 ** i
        total_loss = total_loss + weight * (loss_head + loss_gate)
        stats[f"head_{i+2}_loss"] = loss_head.item()
        stats[f"gate_{i+2}_loss"] = loss_gate.item()
        stats[f"head_{i+2}_accuracy"] = correct.mean().item()
    return total_loss, stats


def train():
    print(f"Loading tokenizer from {config.BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    print(f"Loading base model {config.BASE_MODEL}...")
    model = load_gated_mtp(config.BASE_MODEL, device=config.DEVICE, dtype=config.DTYPE, num_extra_heads=config.NUM_EXTRA_HEADS)
    device = model.base.device
    print(f"Base model on {device}, dtype {config.DTYPE}")
    print(f"Trainable parameters: {model.trainable_param_count:,} ({model.trainable_param_count / 1e6:.1f}M)")
    print(f"Extra heads: {config.NUM_EXTRA_HEADS}, predicting tokens t+2 through t+{config.NUM_EXTRA_HEADS + 1}")
    # Build a cache key from the dataset config so we only tokenize once.
    # Changing the model, dataset, subset, sample count, or seq length
    # produces a different key and triggers re-tokenization.
    cache_key = hashlib.md5(f"{config.BASE_MODEL}:{config.DATASET}:{config.DATASET_SUBSET}:{config.MAX_TRAIN_SAMPLES}:{config.SEQ_LEN}".encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        train_dataset = TokenizedDataset.from_texts([], tokenizer, config.SEQ_LEN, cache_key)
    else:
        print(f"\nLoading dataset {config.DATASET} (streaming, first {config.MAX_TRAIN_SAMPLES} samples)...")
        dataset = load_dataset(config.DATASET, config.DATASET_SUBSET, split="train", streaming=True)
        texts = []
        for row in dataset:
            if len(row["text"]) > 100:
                texts.append(row["text"])
            if len(texts) >= config.MAX_TRAIN_SAMPLES:
                break
        train_dataset = TokenizedDataset.from_texts(texts, tokenizer, config.SEQ_LEN, cache_key)
    print(f"Training on {len(train_dataset)} chunks")
    dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.trainable_params, lr=config.LEARNING_RATE)
    # --- Training loop ---
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            base_logits, extra_logits, confidences = model(input_ids)
            loss, stats = compute_loss(extra_logits, confidences, input_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_steps += 1
            if step % 100 == 0:
                elapsed = time.time() - t0
                stats_str = " | ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                print(f"  epoch {epoch+1} step {step}/{len(dataloader)} | loss: {loss.item():.4f} | {stats_str} | {elapsed:.0f}s")
        avg_loss = epoch_loss / epoch_steps
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{config.EPOCHS} complete | avg loss: {avg_loss:.4f} | {elapsed:.0f}s")
    # --- Save checkpoint ---
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"gated_mtp_{config.NUM_EXTRA_HEADS}heads.pt"
    torch.save({"extra_heads": [h.state_dict() for h in model.extra_heads], "gates": [g.state_dict() for g in model.gates], "config": {"num_extra_heads": config.NUM_EXTRA_HEADS, "base_model": config.BASE_MODEL}}, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
