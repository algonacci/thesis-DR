import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config import (
    TRAIN_CSV,
    SEED,
    NUM_CLASSES,
    BATCH_SIZE,
    OUTPUT_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    MASK_DIR,
    STAGE1_CKPT,
    STAGE2_CKPT,
    VAL_SPLIT,
    TEST_SPLIT,
)
from src.utils import get_device, set_seed
from src.dataset import (
    RetinaDataset,
    Stage2MaskedDataset,
    get_transform,
    create_dataloaders,
    create_masked_dataloaders,
    TRAIN_IMAGES,
)
from src.models import (
    ResViTFusionNet,
    ViTClassifier,
    freeze_all,
    unfreeze_stage1,
    unfreeze_stage2,
    count_params,
)
from src.gradcam import GradCAM, generate_masks, plot_gradcam
from src.train import train_model, predict


def load_and_split():
    df = pd.read_csv(TRAIN_CSV)
    df["id_code"] = df["id_code"].astype(str) + ".png"
    df["diagnosis"] = df["diagnosis"].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df["diagnosis"], random_state=SEED
    )
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, stratify=train_df["diagnosis"], random_state=SEED
    )

    for sub in (train_df, val_df, test_df):
        sub.reset_index(drop=True, inplace=True)

    print(f"Split → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")
    return train_df, val_df, test_df


def run_stage1(train_loader, val_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("STAGE 1 — ResViT FusionNet")
    print("=" * 60)

    model = ResViTFusionNet(NUM_CLASSES).to(device)
    freeze_all(model)
    unfreeze_stage1(model)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total / {trainable:,} trainable")

    if STAGE1_CKPT.exists():
        print(f"Loading checkpoint → {STAGE1_CKPT}")
        model.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    else:
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            stage_label="S1-ResViT",
            checkpoint_path=str(STAGE1_CKPT),
        )

    model.eval()
    probs, preds, labels = predict(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\n[S1 Test] Accuracy: {acc:.4f}")
    print(
        classification_report(
            labels, preds, target_names=[str(i) for i in range(NUM_CLASSES)]
        )
    )
    return model


def run_stage2(train_df, val_df, test_df, device):
    print("\n" + "=" * 60)
    print("STAGE 2 — ViT on Masked Images")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_masked_dataloaders(
        train_df,
        val_df,
        test_df,
        mask_base_dir=str(MASK_DIR),
        batch_size=BATCH_SIZE,
    )

    model = ViTClassifier(NUM_CLASSES).to(device)
    freeze_all(model)
    unfreeze_stage2(model)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total / {trainable:,} trainable")

    if STAGE2_CKPT.exists():
        print(f"Loading checkpoint → {STAGE2_CKPT}")
        model.load_state_dict(torch.load(STAGE2_CKPT, map_location=device))
    else:
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            stage_label="S2-ViT",
            checkpoint_path=str(STAGE2_CKPT),
        )

    model.eval()
    probs, preds, labels = predict(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\n[S2 Test] Accuracy: {acc:.4f}")
    print(
        classification_report(
            labels, preds, target_names=[str(i) for i in range(NUM_CLASSES)]
        )
    )
    return model


def visualize_stages(model_s1, model_s2, test_df, device, n: int = 6):
    import random as _random

    noaug = get_transform(train=False)
    test_ds = RetinaDataset(test_df, str(TRAIN_IMAGES), transform=noaug)

    gradcam_s1 = GradCAM(model_s1, model_s1.resnet50.layer4[-1], is_vit=False)
    gradcam_s2 = GradCAM(model_s2, model_s2.vit.encoder.layers[-1], is_vit=True)

    indices = _random.sample(range(len(test_ds)), min(n, len(test_ds)))
    class_names = [str(i) for i in range(NUM_CLASSES)]

    for idx in indices:
        image_tensor, true_label = test_ds[idx]
        x = image_tensor.unsqueeze(0).to(device)

        cam1, pred1, prob1 = gradcam_s1(x)
        plot_gradcam(
            image_tensor,
            cam1,
            true_label,
            pred1,
            prob1[pred1],
            class_names,
            stage_label="S1-ResViT",
        )

        mask_t = torch.from_numpy(cam1).float().unsqueeze(0)
        image_masked = image_tensor * mask_t
        x_masked = image_masked.unsqueeze(0).to(device)

        cam2, pred2, prob2 = gradcam_s2(x_masked)
        plot_gradcam(
            image_masked,
            cam2,
            true_label,
            pred2,
            prob2[pred2],
            class_names,
            stage_label="S2-ViT",
        )

    gradcam_s1.remove_hooks()
    gradcam_s2.remove_hooks()


def main():
    import os

    for d in [OUTPUT_DIR, MODEL_DIR, CHECKPOINT_DIR, MASK_DIR]:
        os.makedirs(d, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    set_seed()

    train_df, val_df, test_df = load_and_split()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df
    )

    # ── Stage 1 ──────────────────────────────────────────────
    model_s1 = run_stage1(train_loader, val_loader, test_loader, device)

    # Generate Grad-CAM masks (using val_transform, no augmentation)
    if not (MASK_DIR / "train").exists():
        print("\nGenerating Stage-1 Grad-CAM masks …")
        noaug = get_transform(train=False)
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            ds = RetinaDataset(df, str(TRAIN_IMAGES), transform=noaug)
            gradcam = GradCAM(model_s1, model_s1.resnet50.layer4[-1], is_vit=False)
            generate_masks(model_s1, gradcam, ds, str(MASK_DIR / split_name), device)
            gradcam.remove_hooks()

    # ── Stage 2 ──────────────────────────────────────────────
    model_s2 = run_stage2(train_df, val_df, test_df, device)

    # ── Visualize ────────────────────────────────────────────
    print("\nVisualizing Grad-CAM (both stages) …")
    visualize_stages(model_s1, model_s2, test_df, device, n=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
