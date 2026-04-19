import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple,  Sequence
from sklearn.metrics import confusion_matrix

def plot_pca_scatter(
    Z: np.ndarray,
    labels: np.ndarray,
    title: str,
    figsize: Tuple[int, int] = (7, 6),
):
    plt.figure(figsize=figsize)
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=18, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def plot_log_likelihood(history: List[float], title: str = "GMM Log-Likelihood Curve"):
    plt.figure(figsize=(7, 5))
    plt.plot(history, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_entropy_hist(entropy_values: np.ndarray, title: str = "Assignment Entropy Histogram"):
    plt.figure(figsize=(7, 5))
    plt.hist(entropy_values, bins=30)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_entropy_histogram(
    entropy_values: np.ndarray,
    bins: int = 30,
    title: str = "Assignment Entropy Histogram",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(7, 5))
    plt.hist(entropy_values, bins=bins)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_classwise_entropy_bar(
    classwise_df,
    title: str = "Mean Assignment Entropy by Digit Class",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 5))
    plt.bar(classwise_df["digit"].astype(str), classwise_df["mean_entropy"])
    plt.xlabel("Digit")
    plt.ylabel("Mean Entropy")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_digit_grid(
    images: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    subtitles: Optional[Sequence[str]] = None,
    n_rows: int = 4,
    n_cols: int = 5,
    title: str = "Digit Grid",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(1.9 * n_cols, 2.1 * n_rows))
    total = min(len(images), n_rows * n_cols)
    for i in range(total):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

        text_lines = []
        if labels is not None:
            text_lines.append(f"y={labels[i]}")
        if subtitles is not None:
            text_lines.append(str(subtitles[i]))
        if text_lines:
            ax.set_title("\n".join(text_lines), fontsize=9)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_metric_vs_sigma2(
    results_df,
    metric: str,
    title: str,
    log_x: bool = True,
    save_path: Optional[str] = None,
):
    df = results_df.sort_values("sigma2")
    plt.figure(figsize=(7, 5))
    plt.plot(df["sigma2"], df[metric], marker="o")
    plt.xlabel("Fixed shared spherical variance sigma^2")
    plt.ylabel(metric)
    plt.title(title)
    if log_x:
        plt.xscale("log")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dual_metric_vs_sigma2(
    results_df,
    metric1: str,
    metric2: str,
    title: str,
    label1: str,
    label2: str,
    log_x: bool = True,
    save_path: Optional[str] = None,
):
    df = results_df.sort_values("sigma2")
    plt.figure(figsize=(7, 5))
    plt.plot(df["sigma2"], df[metric1], marker="o", label=label1)
    plt.plot(df["sigma2"], df[metric2], marker="s", label=label2)
    plt.xlabel("Fixed shared spherical variance sigma^2")
    plt.ylabel("Value")
    plt.title(title)
    if log_x:
        plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()