import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.utils import EarlyStopping, plot_history
from src.config import EPOCHS, LR


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


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    stage_label="Stage",
    epochs=EPOCHS,
    lr=LR,
    checkpoint_path="checkpoint.pth",
    patience=10,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
        print(
            f"[{stage_label}] Epoch {epoch + 1:3d}/{epochs} │ "
            f"LR {lr_now:.2e} │ "
            f"T-Loss {t_loss:.4f} T-Acc {t_acc:.4f} │ "
            f"V-Loss {v_loss:.4f} V-Acc {v_acc:.4f}"
        )

        early(v_loss, model)
        if early.early_stop:
            print(f"[{stage_label}] Early stopping @ epoch {epoch + 1}")
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
