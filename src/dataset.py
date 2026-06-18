import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_IMAGES


def get_transform(train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)
                ),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )


class RetinaDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
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

    def __init__(
        self, dataframe: pd.DataFrame, image_dir: str, mask_dir: str, transform=None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        img_name = row["id_code"]
        label = int(row["diagnosis"])

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.mask_dir, f"{idx:06d}.npy")
        mask = np.load(mask_path)  # (224, 224), values [0,1]

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        masked_image = image * mask_tensor
        return masked_image, label


def create_dataloaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    train_dataset = RetinaDataset(
        train_df, str(TRAIN_IMAGES), transform=get_transform(train=True)
    )
    val_dataset = RetinaDataset(
        val_df, str(TRAIN_IMAGES), transform=get_transform(train=False)
    )
    test_dataset = RetinaDataset(
        test_df, str(TRAIN_IMAGES), transform=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_masked_dataloaders(
    train_df, val_df, test_df, mask_base_dir, batch_size=BATCH_SIZE
):
    no_aug = get_transform(train=False)
    train_ds = Stage2MaskedDataset(
        train_df, str(TRAIN_IMAGES), f"{mask_base_dir}/train", transform=no_aug
    )
    val_ds = Stage2MaskedDataset(
        val_df, str(TRAIN_IMAGES), f"{mask_base_dir}/val", transform=no_aug
    )
    test_ds = Stage2MaskedDataset(
        test_df, str(TRAIN_IMAGES), f"{mask_base_dir}/test", transform=no_aug
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, val_loader, test_loader
