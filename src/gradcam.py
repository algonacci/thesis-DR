import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import IMG_SIZE


class GradCAM:
    """Grad-CAM supporting both CNN layers (4D feature maps) and
    ViT encoder blocks (3D token tensors — auto-reshape to spatial)."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        is_vit: bool = False,
    ):
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

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        if self.is_vit:
            act = self.activations[:, 1:, :]  # strip CLS token
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
        cam = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        return cam, class_idx, probs

    def remove_hooks(self):
        self.hook_handle.remove()


def tensor_to_image(t: torch.Tensor):
    """Convert (3,H,W) tensor [0,1] → (H,W,3) numpy."""
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def plot_gradcam(
    image_tensor, cam, truth, pred, conf, class_names=None, stage_label=""
):
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


def generate_masks(
    model,
    gradcam,
    dataset,
    mask_dir: str,
    device: torch.device,
    batch_size: int = 32,
):
    """Run Grad-CAM on dataset in index-order, save heatmaps as {idx:06d}.npy."""
    os.makedirs(mask_dir, exist_ok=True)
    model.eval()

    from torch.utils.data import DataLoader as _DL

    loader = _DL(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    idx = 0
    for images, _ in loader:
        images = images.to(device)
        for i in range(images.size(0)):
            x = images[i : i + 1]
            cam, _, _ = gradcam(x)
            np.save(os.path.join(mask_dir, f"{idx:06d}.npy"), cam)
            idx += 1

    print(f"Masks ({idx}) saved → {mask_dir}")
