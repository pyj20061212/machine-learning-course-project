import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    matched = cm[row_ind, col_ind].sum()
    return float(matched / np.sum(cm))


def remap_cluster_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    mapping = {pred_label: true_label for true_label, pred_label in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred], dtype=np.int64)


def assignment_entropy(responsibilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-sample entropy of soft assignments.
    Shape of responsibilities: (n_samples, n_components)
    """
    R = np.clip(responsibilities, eps, 1.0)
    return -np.sum(R * np.log(R), axis=1)


def top2_probability_gap(responsibilities: np.ndarray) -> np.ndarray:
    """
    Gap between the largest and second-largest responsibility.
    Larger => more confident / harder assignment.
    """
    R = np.asarray(responsibilities, dtype=np.float64)
    top2 = np.partition(R, kth=-2, axis=1)[:, -2:]
    top1 = np.max(top2, axis=1)
    second = np.min(top2, axis=1)
    return top1 - second


def effective_hardness_from_resp(
    responsibilities: np.ndarray,
    entropy_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Summarize how 'hard' the soft assignments are.
    """
    ent = assignment_entropy(responsibilities)
    gap = top2_probability_gap(responsibilities)

    if entropy_threshold is None:
        # A practical threshold for "almost hard" assignment.
        entropy_threshold = 0.05

    return {
        "mean_entropy": float(np.mean(ent)),
        "std_entropy": float(np.std(ent)),
        "median_entropy": float(np.median(ent)),
        "q90_entropy": float(np.quantile(ent, 0.90)),
        "q99_entropy": float(np.quantile(ent, 0.99)),
        "mean_top2_gap": float(np.mean(gap)),
        "hard_fraction_entropy": float(np.mean(ent <= entropy_threshold)),
    }


def evaluate_clustering(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    responsibilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "ACC": clustering_accuracy(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }

    unique_pred = np.unique(y_pred)
    if 1 < len(unique_pred) < len(X):
        metrics["Silhouette"] = float(silhouette_score(X, y_pred))
    else:
        metrics["Silhouette"] = np.nan

    if responsibilities is not None:
        hardness = effective_hardness_from_resp(responsibilities)
        metrics.update({
            "MeanEntropy": hardness["mean_entropy"],
            "StdEntropy": hardness["std_entropy"],
            "MedianEntropy": hardness["median_entropy"],
            "Q90Entropy": hardness["q90_entropy"],
            "Q99Entropy": hardness["q99_entropy"],
            "MeanTop2Gap": hardness["mean_top2_gap"],
            "HardFraction": hardness["hard_fraction_entropy"],
        })

    return metrics


def classwise_entropy_summary(
    y_true: np.ndarray,
    responsibilities: np.ndarray,
) -> pd.DataFrame:
    ent = assignment_entropy(responsibilities)
    rows = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        rows.append({
            "digit": int(cls),
            "count": int(np.sum(mask)),
            "mean_entropy": float(np.mean(ent[mask])),
            "std_entropy": float(np.std(ent[mask])),
            "median_entropy": float(np.median(ent[mask])),
            "q90_entropy": float(np.quantile(ent[mask], 0.90)),
            "mean_top2_gap": float(np.mean(top2_probability_gap(responsibilities[mask]))),
        })
    return pd.DataFrame(rows).sort_values("digit").reset_index(drop=True)


def print_metrics_table(title: str, metrics: Dict[str, float]):
    print("=" * 80)
    print(title)
    for k, v in metrics.items():
        print(f"{k:>15s}: {v:.6f}" if np.isfinite(v) else f"{k:>15s}: nan")
    print("=" * 80)