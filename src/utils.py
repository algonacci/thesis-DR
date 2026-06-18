import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config import SEED


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(
        self, patience: int = 10, verbose: bool = True, path: str = "checkpoint.pth"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss: float, model: torch.nn.Module):
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

    def load_best(self, model: torch.nn.Module, device: torch.device):
        model.load_state_dict(torch.load(self.path, map_location=device))
        model.to(device)


def plot_history(history: dict, title_prefix: str = ""):
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
