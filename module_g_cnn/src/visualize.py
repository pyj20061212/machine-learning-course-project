import math
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, save_path: Optional[str] = None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path: Optional[str] = None):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_sample_batch(loader, class_names=None, n: int = 16):
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]

    cols = 4
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        if class_names is None:
            title = str(labels[i].item())
        else:
            title = class_names[labels[i].item()]
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_conv1_kernels(model, save_path: Optional[str] = None):
    weights = model.conv1.weight.detach().cpu().numpy()  # [out_c, in_c, k, k]
    out_channels = weights.shape[0]

    plt.figure(figsize=(2 * out_channels, 2))
    for i in range(out_channels):
        plt.subplot(1, out_channels, i + 1)
        plt.imshow(weights[i, 0], cmap="gray")
        plt.title(f"K{i}")
        plt.axis("off")

    plt.suptitle("Conv1 Kernels", y=1.05)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def visualize_feature_maps(model, image_tensor, device, layer_name="conv1", max_channels: int = 8,
                           save_path: Optional[str] = None):
    model.eval()
    image_tensor = image_tensor.to(device).unsqueeze(0)

    feats = model.forward_features(image_tensor)
    fmap = feats[layer_name].detach().cpu()[0]  # [C,H,W]

    n_channels = min(max_channels, fmap.shape[0])

    plt.figure(figsize=(2 * (n_channels + 1), 2.5))

    plt.subplot(1, n_channels + 1, 1)
    plt.imshow(image_tensor.cpu().squeeze().numpy(), cmap="gray")
    plt.title("Input")
    plt.axis("off")

    for i in range(n_channels):
        plt.subplot(1, n_channels + 1, i + 2)
        plt.imshow(fmap[i], cmap="gray")
        plt.title(f"{layer_name}[{i}]")
        plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def show_misclassified_examples(model, loader, device, n: int = 12, save_path: Optional[str] = None):
    model.eval()

    mis_images = []
    mis_true = []
    mis_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        mask = preds != labels
        if mask.any():
            mis_images.extend(images[mask].cpu())
            mis_true.extend(labels[mask].cpu().tolist())
            mis_pred.extend(preds[mask].cpu().tolist())

        if len(mis_images) >= n:
            break

    n = min(n, len(mis_images))
    cols = 4
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(mis_images[i].squeeze(0), cmap="gray")
        plt.title(f"T:{mis_true[i]} / P:{mis_pred[i]}")
        plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



def show_image_grid(
    images,
    labels=None,
    preds=None,
    probs=None,
    ncols: int = 4,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    n = len(images)
    nrows = math.ceil(n / ncols)

    plt.figure(figsize=(2.2 * ncols, 2.4 * nrows))
    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        img = images[i]
        if img.ndim == 3:
            img = img.squeeze(0)
        plt.imshow(img, cmap="gray")
        parts = []
        if labels is not None:
            parts.append(f"T:{labels[i]}")
        if preds is not None:
            parts.append(f"P:{preds[i]}")
        if probs is not None:
            parts.append(f"{probs[i]:.3f}")
        plt.title(" | ".join(parts), fontsize=9)
        plt.axis("off")

    if title is not None:
        plt.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def visualize_class_prototypes(
    images,
    labels,
    probs,
    n_classes: int = 10,
    top_k: int = 8,
    save_dir: Optional[str] = None,
):
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        class_scores = probs[idx, c]
        order = np.argsort(-class_scores)[:top_k]
        chosen_idx = idx[order]

        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"class_{c}_top_confident.png")

        show_image_grid(
            images[chosen_idx],
            labels=labels[chosen_idx],
            probs=probs[chosen_idx, c],
            ncols=min(4, top_k),
            title=f"Top confident examples for class {c}",
            save_path=save_path,
        )


def plot_feature_embedding_2d(
    embedding_2d,
    labels,
    title: str = "Feature Embedding",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        s=12,
        alpha=0.7,
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_class_centroid_distance_heatmap(
    features,
    labels,
    n_classes: int = 10,
    save_path: Optional[str] = None,
):
    centroids = []
    for c in range(n_classes):
        centroids.append(features[labels == c].mean(axis=0))
    centroids = np.stack(centroids, axis=0)

    dist = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1))

    plt.figure(figsize=(7, 6))
    plt.imshow(dist, interpolation="nearest")
    plt.title("Class Centroid Distance Heatmap")
    plt.colorbar()
    plt.xlabel("Class")
    plt.ylabel("Class")

    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def show_confused_pair_examples(
    images,
    labels,
    preds,
    probs,
    true_class: int,
    pred_class: int,
    top_k: int = 8,
    save_path: Optional[str] = None,
):
    mask = (labels == true_class) & (preds == pred_class)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        print(f"No examples for true={true_class}, pred={pred_class}")
        return

    confidence = probs[idx, pred_class]
    order = np.argsort(-confidence)[:top_k]
    chosen_idx = idx[order]

    show_image_grid(
        images[chosen_idx],
        labels=labels[chosen_idx],
        preds=preds[chosen_idx],
        probs=probs[chosen_idx, pred_class],
        ncols=min(4, top_k),
        title=f"Confused pair: true={true_class}, pred={pred_class}",
        save_path=save_path,
    )