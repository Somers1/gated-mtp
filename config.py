import os


BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-E2B")
DEVICE = os.environ.get("DEVICE", "auto")
DTYPE = os.environ.get("DTYPE", "bfloat16")

NUM_EXTRA_HEADS = int(os.environ.get("NUM_EXTRA_HEADS", "1"))
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.85"))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
DATASET = os.environ.get("DATASET", "wikimedia/wikipedia")
DATASET_SUBSET = os.environ.get("DATASET_SUBSET", "20231101.en")
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "50000"))

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
