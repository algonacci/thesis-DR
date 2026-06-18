# ResViT 2-Stage — Panduan Konversi ke Notebook

Panduan ini buat convert kode modular `src/` jadi notebook Jupyter (`.ipynb`), cell per cell.
Target: training di server GPU (CUDA), tapi bisa juga jalan di MPS/CPU.

---

## 0. Persiapan Environment

### Cell 0 — Install dependencies
```python
# !pip install torch torchvision numpy pandas scikit-learn pillow opencv-python matplotlib torchinfo tqdm
```
> Kalau pakai uv: `uv add torch torchvision numpy pandas scikit-learn pillow opencv-python matplotlib torchinfo tqdm`

### Cell 0b — Dataset
Pastikan dataset APTOS 2019 udah didownload dari Kaggle:
- `train.csv`
- Folder `train_images/` (isi gambar `.png`)

Atur path di config bawah nanti (Cell 1).

---

## 1. Config (Cell 1)

Copy seluruh isi `src/config.py` ke satu cell, **ganti path dataset** sesuai server.

```python
import os
from pathlib import Path

# ⚠️ GANTI PATH INI ⚠️
DATA_DIR = Path("/path/ke/dataset/aptos2019")   # ← SESUAIKAN
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"

OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MASK_DIR = OUTPUT_DIR / "gradcam_masks"

STAGE1_CKPT = CHECKPOINT_DIR / "best_stage1_resvit.pth"
STAGE2_CKPT = CHECKPOINT_DIR / "best_stage2_vit.pth"

# ── Hyperparameters ─────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32          # kecilkan kalau OOM (misal 16 atau 8)
EPOCHS = 100
LR = 5e-4
NUM_CLASSES = 5
SEED = 42
PATIENCE = 10

# ── Fine-tuning ─────────────────────────────────
STAGE1_UNFREEZE_VIT_BLOCKS = 10
STAGE2_UNFREEZE_VIT_BLOCKS = 10

# ── Split ───────────────────────────────────────
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
```

---

## 2. Utils (Cell 2)

Copy seluruh isi `src/utils.py` + ganti import path jadi lokal:

```python
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"[EarlyStopping] val_loss improved ({val_loss:.4f}) → saved")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model, device):
        model.load_state_dict(torch.load(self.path, map_location=device))
        model.to(device)

def plot_history(history, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="train_loss")
    ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_title(f"{title_prefix} Loss".strip())
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="train_acc")
    ax2.plot(history["val_acc"], label="val_acc")
    ax2.set_title(f"{title_prefix} Accuracy".strip())
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.show()
```

---

## 3. Dataset & Dataloader (Cell 3)

Copy seluruh `src/dataset.py`, ganti import config ke variabel lokal:

```python
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                    scale=(0.85, 1.15)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

class RetinaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row["id_code"]
        label = int(row["diagnosis"])
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class Stage2MaskedDataset(Dataset):
    """Loads original image, applies mask from stage-1 Grad-CAM heatmap.
    No augmentation — mask would misalign."""
    def __init__(self, dataframe, image_dir, mask_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row["id_code"]
        label = int(row["diagnosis"])
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        mask_path = os.path.join(self.mask_dir, f"{idx:06d}.npy")
        mask = np.load(mask_path)
        if self.transform:
            image = self.transform(image)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        masked_image = image * mask_tensor
        return masked_image, label

def create_dataloaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    train_dataset = RetinaDataset(train_df, str(TRAIN_IMAGES),
                                  transform=get_transform(train=True))
    val_dataset = RetinaDataset(val_df, str(TRAIN_IMAGES),
                                transform=get_transform(train=False))
    test_dataset = RetinaDataset(test_df, str(TRAIN_IMAGES),
                                 transform=get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def create_masked_dataloaders(train_df, val_df, test_df, mask_base_dir,
                              batch_size=BATCH_SIZE):
    no_aug = get_transform(train=False)
    train_ds = Stage2MaskedDataset(train_df, str(TRAIN_IMAGES),
                                   f"{mask_base_dir}/train", transform=no_aug)
    val_ds = Stage2MaskedDataset(val_df, str(TRAIN_IMAGES),
                                 f"{mask_base_dir}/val", transform=no_aug)
    test_ds = Stage2MaskedDataset(test_df, str(TRAIN_IMAGES),
                                  f"{mask_base_dir}/test", transform=no_aug)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader
```

> **Note**: `num_workers=2` aman buat server GPU. Kalau error, turunin ke `0`.

---

## 4. Models (Cell 4)

Copy seluruh `src/models.py`, ganti import config ke variabel lokal:

```python
import torch
import torch.nn as nn
from torchvision import models

RESNET_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1
VIT_WEIGHTS = models.ViT_B_32_Weights.IMAGENET1K_V1

class ResViTFusionNet(nn.Module):
    """Stage 1: ResNet50 + ViT-B/32 feature fusion."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.resnet50 = models.resnet50(weights=RESNET_WEIGHTS)
        resnet_dim = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()

        self.vit = models.vit_b_32(weights=VIT_WEIGHTS)
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        self.fc_combined = nn.Linear(resnet_dim + vit_dim, num_classes)

    def forward(self, x):
        r_feat = self.resnet50(x)
        v_feat = self.vit(x)
        fused = torch.cat((r_feat, v_feat), dim=1)
        return self.fc_combined(fused)

class ViTClassifier(nn.Module):
    """Stage 2: ViT-B/32 standalone classifier for masked input."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.vit = models.vit_b_32(weights=VIT_WEIGHTS)
        in_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.vit(x)

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_stage1(model):
    for p in model.resnet50.layer4.parameters():
        p.requires_grad = True
    for p in model.vit.encoder.layers[-STAGE1_UNFREEZE_VIT_BLOCKS:].parameters():
        p.requires_grad = True
    for p in model.fc_combined.parameters():
        p.requires_grad = True

def unfreeze_stage2(model):
    for p in model.vit.encoder.layers[-STAGE2_UNFREEZE_VIT_BLOCKS:].parameters():
        p.requires_grad = True
    for p in model.vit.heads.parameters():
        p.requires_grad = True

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

---

## 5. Grad-CAM (Cell 5)

Copy seluruh `src/gradcam.py`, ganti import config ke variabel lokal:

```python
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class GradCAM:
    """Grad-CAM supporting both CNN layers (4D feature maps) and
    ViT encoder blocks (3D token tensors — auto-reshape to spatial)."""
    def __init__(self, model, target_layer, is_vit=False):
        self.model = model
        self.target_layer = target_layer
        self.is_vit = is_vit
        self.activations = None
        self.gradients = None
        self.hook_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output.detach()
        output.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        if self.is_vit:
            act = self.activations[:, 1:, :]      # strip CLS token
            B, N, D = act.shape
            H = W = int(N**0.5)
            acts_4d = act.transpose(1, 2).reshape(B, D, H, W)
            grad = self.gradients[:, 1:, :]
            grads_4d = grad.transpose(1, 2).reshape(B, D, H, W)
        else:
            acts_4d = self.activations
            grads_4d = self.gradients

        weights = grads_4d.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts_4d).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        return cam, class_idx, probs

    def remove_hooks(self):
        self.hook_handle.remove()

def tensor_to_image(t):
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)

def overlay_heatmap(image, cam, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)

def plot_gradcam(image_tensor, cam, truth, pred, conf,
                 class_names=None, stage_label=""):
    image = tensor_to_image(image_tensor)
    overlay = overlay_heatmap(image, cam)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Original | True: {truth}")
    axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"{stage_label} Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    label_str = class_names[pred] if class_names else str(pred)
    axes[2].set_title(f"Overlay | Pred: {label_str} ({conf:.3f})")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def generate_masks(model, gradcam, dataset, mask_dir, device, batch_size=32):
    os.makedirs(mask_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    idx = 0
    for images, _ in loader:
        images = images.to(device)
        for i in range(images.size(0)):
            x = images[i:i+1]
            cam, _, _ = gradcam(x)
            np.save(os.path.join(mask_dir, f"{idx:06d}.npy"), cam)
            idx += 1
    print(f"Masks ({idx}) saved → {mask_dir}")
```

---

## 6. Train (Cell 6)

Copy seluruh `src/train.py`, ganti import config ke variabel lokal:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, device,
                stage_label="Stage", epochs=EPOCHS, lr=LR,
                checkpoint_path="checkpoint.pth", patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    early = EarlyStopping(patience=patience, path=checkpoint_path)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{stage_label}] Epoch {epoch+1:3d}/{epochs} │ "
              f"LR {lr_now:.2e} │ "
              f"T-Loss {t_loss:.4f} T-Acc {t_acc:.4f} │ "
              f"V-Loss {v_loss:.4f} V-Acc {v_acc:.4f}")

        early(v_loss, model)
        if early.early_stop:
            print(f"[{stage_label}] Early stopping @ epoch {epoch+1}")
            break

    early.load_best(model, device)
    plot_history(history, title_prefix=f"[{stage_label}]")
    return history

def predict(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )
```

---

## 7. Main Pipeline (Cell 7–11)

Ini breakdown pipeline jadi beberapa cell biar nggak terlalu panjang.

### Cell 7 — Setup & Load Data

```python
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
for d in [OUTPUT_DIR, MODEL_DIR, CHECKPOINT_DIR, MASK_DIR]:
    os.makedirs(d, exist_ok=True)

device = get_device()
print(f"Device: {device}")
set_seed()

# ── Load & Split ──────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df["id_code"] = df["id_code"].astype(str) + ".png"
df["diagnosis"] = df["diagnosis"].astype(int)

train_df, test_df = train_test_split(
    df, test_size=TEST_SPLIT, stratify=df["diagnosis"], random_state=SEED)
val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
train_df, val_df = train_test_split(
    train_df, test_size=val_ratio, stratify=train_df["diagnosis"],
    random_state=SEED)

for sub in (train_df, val_df, test_df):
    sub.reset_index(drop=True, inplace=True)

print(f"Split → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

# Dataloaders untuk Stage 1
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df)
```

### Cell 8 — Stage 1: Train ResViT

```python
print("\n" + "=" * 60)
print("STAGE 1 — ResViT FusionNet")
print("=" * 60)

model_s1 = ResViTFusionNet(NUM_CLASSES).to(device)
freeze_all(model_s1)
unfreeze_stage1(model_s1)
total, trainable = count_params(model_s1)
print(f"Params: {total:,} total / {trainable:,} trainable")

if STAGE1_CKPT.exists():
    print(f"Loading checkpoint → {STAGE1_CKPT}")
    model_s1.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
else:
    train_model(model_s1, train_loader, val_loader, device,
                stage_label="S1-ResViT", checkpoint_path=str(STAGE1_CKPT))

# Eval
model_s1.eval()
probs, preds, labels = predict(model_s1, test_loader, device)
acc = accuracy_score(labels, preds)
print(f"\n[S1 Test] Accuracy: {acc:.4f}")
print(classification_report(labels, preds,
        target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))
```

> **Note**: Training S1 download ~435MB pretrained weights. Di server GPU biasanya udah di-cache. Kalau belum, download dulu (butuh internet).

### Cell 9 — Generate Stage-1 Grad-CAM Masks

```python
if not (MASK_DIR / "train").exists():
    print("\nGenerating Stage-1 Grad-CAM masks …")
    noaug = get_transform(train=False)
    for split_name, df in [("train", train_df), ("val", val_df),
                            ("test", test_df)]:
        ds = RetinaDataset(df, str(TRAIN_IMAGES), transform=noaug)
        gradcam = GradCAM(model_s1, model_s1.resnet50.layer4[-1], is_vit=False)
        generate_masks(model_s1, gradcam, ds,
                       str(MASK_DIR / split_name), device)
        gradcam.remove_hooks()
else:
    print("Masks already exist, skipping generation.")
```

> **Note**: Cell ini lama (~1-3 menit per split), karena infer S1 model per gambar.

### Cell 10 — Stage 2: Train ViT on Masked Images

```python
print("\n" + "=" * 60)
print("STAGE 2 — ViT on Masked Images")
print("=" * 60)

# Dataloaders untuk Stage 2 (masked)
train_loader2, val_loader2, test_loader2 = create_masked_dataloaders(
    train_df, val_df, test_df, mask_base_dir=str(MASK_DIR), batch_size=BATCH_SIZE)

model_s2 = ViTClassifier(NUM_CLASSES).to(device)
freeze_all(model_s2)
unfreeze_stage2(model_s2)
total, trainable = count_params(model_s2)
print(f"Params: {total:,} total / {trainable:,} trainable")

if STAGE2_CKPT.exists():
    print(f"Loading checkpoint → {STAGE2_CKPT}")
    model_s2.load_state_dict(torch.load(STAGE2_CKPT, map_location=device))
else:
    train_model(model_s2, train_loader2, val_loader2, device,
                stage_label="S2-ViT", checkpoint_path=str(STAGE2_CKPT))

# Eval
model_s2.eval()
probs, preds, labels = predict(model_s2, test_loader2, device)
acc = accuracy_score(labels, preds)
print(f"\n[S2 Test] Accuracy: {acc:.4f}")
print(classification_report(labels, preds,
        target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))
```

### Cell 11 — Visualize Grad-CAM (Both Stages)

```python
import random as _random
noaug = get_transform(train=False)
test_ds = RetinaDataset(test_df, str(TRAIN_IMAGES), transform=noaug)

gradcam_s1 = GradCAM(model_s1, model_s1.resnet50.layer4[-1], is_vit=False)
gradcam_s2 = GradCAM(model_s2, model_s2.vit.encoder.layers[-1], is_vit=True)

N_SAMPLES = 4
indices = _random.sample(range(len(test_ds)), min(N_SAMPLES, len(test_ds)))
class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

for idx in indices:
    image_tensor, true_label = test_ds[idx]
    x = image_tensor.unsqueeze(0).to(device)

    # Stage 1 Grad-CAM
    cam1, pred1, prob1 = gradcam_s1(x)
    plot_gradcam(image_tensor, cam1, true_label, pred1, prob1[pred1],
                 class_names, stage_label="S1-ResViT")

    # Stage 2 Grad-CAM on masked image
    mask_t = torch.from_numpy(cam1).float().unsqueeze(0)
    image_masked = image_tensor * mask_t
    x_masked = image_masked.unsqueeze(0).to(device)
    cam2, pred2, prob2 = gradcam_s2(x_masked)
    plot_gradcam(image_masked, cam2, true_label, pred2, prob2[pred2],
                 class_names, stage_label="S2-ViT")

gradcam_s1.remove_hooks()
gradcam_s2.remove_hooks()
print("Done.")
```

---

## 8. Ringkasan Alur Cell

| Cell | Isi | Waktu Estimasi |
|------|-----|---------------|
| 0 | Install dependencies | 2 menit |
| 1 | Config (ganti path dataset!) | - |
| 2 | Utils (device, seed, early stop, plot) | - |
| 3 | Dataset & Dataloader | - |
| 4 | Models (ResViT + ViTClassifier) | download pretrained ~30s |
| 5 | Grad-CAM | - |
| 6 | Train loop | - |
| 7 | Setup path + Load & Split data | - |
| 8 | **Train Stage 1** | ⏳ 30-120 min (GPU) |
| 9 | Generate Grad-CAM masks | ⏳ 5-15 min |
| 10 | **Train Stage 2** | ⏳ 20-90 min (GPU) |
| 11 | Visualize Grad-CAM | 1 menit |

---

## 9. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| **CUDA out of memory** | Turunin `BATCH_SIZE` ke 16 atau 8 di Cell 1 |
| **Download pretrained gagal** | Pastikan server punya akses internet. Kalau offline, download manual `resnet50-0676ba61.pth` + `vit_b_32-d86f8d99.pth` ke `~/.cache/torch/hub/checkpoints/` |
| **num_workers error** | Ganti `num_workers=2` jadi `num_workers=0` di Cell 3 |
| **Grad-CAM di notebook lambat** | Cell 11 pilih `N_SAMPLES=2` aja |
| **Resume training** | Checkpoint auto-save ke `output/checkpoints/`. Tinggal re-run Cell 8 / Cell 10, bakal auto-load kalau file `.pth` exists |

---

## 10. Struktur Output

```
output/
├── checkpoints/
│   ├── best_stage1_resvit.pth    # (425 MB)
│   └── best_stage2_vit.pth       # (350 MB)
├── gradcam_masks/
│   ├── train/    # 000000.npy ... (N file × 224×224 float)
│   ├── val/
│   └── test/
└── models/
```
