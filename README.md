# Gated Multi-Token Prediction

Bolt-on module that adds learned confidence gates to multi-token prediction heads on a frozen pretrained model. The gate decides at each step how many tokens to emit — spending compute proportional to prediction difficulty.

## How it works

Standard autoregressive generation produces one token per forward pass regardless of how predictable the sequence is. Gated MTP adds extra prediction heads (for tokens t+2, t+3, etc.) with tiny confidence gates. When the gate is confident, we accept extra tokens without additional forward passes.

```
Forward pass → base predicts t+1 (always accept)
             → head_2 predicts t+2, gate says 0.95 → accept
             → head_3 predicts t+3, gate says 0.72 → reject, stop
Result: 2 tokens from 1 forward pass
```

## Quick start

```bash
pip install -r requirements.txt

# Train (defaults to Gemma 4 E2B on Wikipedia)
python train.py

# Generate with trained checkpoint
python generate.py --prompt "The capital of France is" --checkpoint ./checkpoints/gated_mtp_1heads.pt

# Benchmark against baseline
python benchmark.py --checkpoint ./checkpoints/gated_mtp_1heads.pt
```

## Configuration

All config is via environment variables (see `config.py`):

```bash
# Use a different model
BASE_MODEL=google/gemma-4-E4B python train.py

# Train with 3 extra heads instead of 1
NUM_EXTRA_HEADS=3 python train.py

# Adjust gate threshold for inference
GATE_THRESHOLD=0.9 python generate.py --prompt "Hello world"

# Run on Mac
DEVICE=mps DTYPE=float16 python train.py
```

## Files

- `config.py` — all hyperparameters and settings via env vars
- `model.py` — GatedMTP module (extra heads + gates wrapping frozen base)
- `train.py` — training loop for heads and gates
- `generate.py` — inference with gated token skipping
- `benchmark.py` — compare gated vs baseline at various thresholds

## Design doc

See the full architecture design, training plan, and paper writing guide at:
`/root/projects/gated-mtp-design.md`
