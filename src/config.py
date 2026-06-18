import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("APTOS_DATA_DIR", "data/aptos2019"))
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"

OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MASK_DIR = OUTPUT_DIR / "gradcam_masks"

STAGE1_CKPT = CHECKPOINT_DIR / "best_stage1_resvit.pth"
STAGE2_CKPT = CHECKPOINT_DIR / "best_stage2_vit.pth"

# ── Hyperparameters ─────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-4
NUM_CLASSES = 5
SEED = 42
PATIENCE = 10

# ── Fine-tuning ─────────────────────────────────────────────────────
STAGE1_UNFREEZE_VIT_BLOCKS = 10  # last N ViT encoder blocks
STAGE2_UNFREEZE_VIT_BLOCKS = 10

# ── Split ───────────────────────────────────────────────────────────
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
