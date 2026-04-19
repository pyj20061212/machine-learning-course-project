import os
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Tuple, Sequence, Iterable
from scipy.optimize import linear_sum_assignment

from .data_utils import load_digits_data
from .pca import PCAFromScratch
from .kmeans import KMeansFromScratch
from .gmm import GMMEMFromScratch
from .metrics import (
    evaluate_clustering,
    assignment_entropy,
    top2_probability_gap,
    classwise_entropy_summary,
    remap_cluster_labels,
)
from .plotting import (
    plot_entropy_histogram,
    plot_classwise_entropy_bar,
    show_digit_grid,
    plot_confusion_heatmap,
)


def covariance_floor_ratio(covariances: np.ndarray, reg_covar: float, atol: float = 1e-12) -> float:
    """
    Fraction of covariance entries that are effectively at the regularization floor.
    Higher ratio usually means stronger covariance collapse / degeneration.
    """
    return float(np.mean(np.isclose(covariances, reg_covar, atol=atol)))


def run_c2_soft_vs_hard_assignment(
    pca_dim: int = 10,
    random_state: int = 42,
    gmm_reg_covar: float = 1e-4,
    gmm_n_init: int = 20,
    top_k: int = 20,
    save_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    C2 experiment:
    Analyze soft vs hard assignment behavior of GMM on PCA representation.
    """
    data = load_digits_data(test_size=0.25, random_state=random_state, standardize=True)

    X_train_raw = data["X_train"]
    X_test_raw = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    images_test = data["images_test"]

    pca = PCAFromScratch(n_components=pca_dim)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    gmm = GMMEMFromScratch(
        n_components=10,
        max_iter=200,
        tol=1e-4,
        reg_covar=gmm_reg_covar,
        n_init=gmm_n_init,
        init_method="kmeans",
        random_state=random_state,
    )
    train_pred = gmm.fit_predict(X_train)
    test_pred = gmm.predict(X_test)
    test_resp = gmm.predict_proba(X_test)

    test_metrics = evaluate_clustering(
        X_test,
        y_test,
        test_pred,
        responsibilities=test_resp,
    )

    ent = assignment_entropy(test_resp)
    gap = top2_probability_gap(test_resp)
    remapped_test_pred = remap_cluster_labels(y_test, test_pred)

    classwise_df = classwise_entropy_summary(y_test, test_resp)

    high_idx = np.argsort(ent)[-top_k:][::-1]
    low_idx = np.argsort(ent)[:top_k]

    high_entropy_table = pd.DataFrame({
        "index": high_idx,
        "true_label": y_test[high_idx],
        "pred_label": remapped_test_pred[high_idx],
        "entropy": ent[high_idx],
        "top2_gap": gap[high_idx],
    })

    low_entropy_table = pd.DataFrame({
        "index": low_idx,
        "true_label": y_test[low_idx],
        "pred_label": remapped_test_pred[low_idx],
        "entropy": ent[low_idx],
        "top2_gap": gap[low_idx],
    })

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        plot_entropy_histogram(
            ent,
            title=f"C2 Entropy Histogram (PCA-{pca_dim} + GMM)",
            save_path=os.path.join(save_dir, "c2_entropy_histogram.png"),
        )

        plot_classwise_entropy_bar(
            classwise_df,
            title=f"C2 Mean Entropy by Class (PCA-{pca_dim} + GMM)",
            save_path=os.path.join(save_dir, "c2_classwise_entropy.png"),
        )

        show_digit_grid(
            images_test[high_idx],
            labels=y_test[high_idx],
            subtitles=[f"H={v:.3f}" for v in ent[high_idx]],
            n_rows=4,
            n_cols=5,
            title="Top High-Entropy Test Digits",
            save_path=os.path.join(save_dir, "c2_high_entropy_digits.png"),
        )

        show_digit_grid(
            images_test[low_idx],
            labels=y_test[low_idx],
            subtitles=[f"H={v:.3f}" for v in ent[low_idx]],
            n_rows=4,
            n_cols=5,
            title="Top Low-Entropy Test Digits",
            save_path=os.path.join(save_dir, "c2_low_entropy_digits.png"),
        )

        plot_confusion_heatmap(
            y_test,
            remapped_test_pred,
            title=f"C2 Confusion Matrix (PCA-{pca_dim} + GMM)",
            save_path=os.path.join(save_dir, "c2_confusion_matrix.png"),
        )

        classwise_df.to_csv(os.path.join(save_dir, "c2_classwise_entropy.csv"), index=False)
        high_entropy_table.to_csv(os.path.join(save_dir, "c2_high_entropy_samples.csv"), index=False)
        low_entropy_table.to_csv(os.path.join(save_dir, "c2_low_entropy_samples.csv"), index=False)
        pd.DataFrame([test_metrics]).to_csv(os.path.join(save_dir, "c2_test_metrics.csv"), index=False)

    return {
        "pca": pca,
        "gmm": gmm,
        "X_test": X_test,
        "y_test": y_test,
        "images_test": images_test,
        "test_pred": test_pred,
        "test_pred_remapped": remapped_test_pred,
        "test_resp": test_resp,
        "entropy": ent,
        "top2_gap": gap,
        "test_metrics": test_metrics,
        "classwise_entropy_df": classwise_df,
        "high_entropy_table": high_entropy_table,
        "low_entropy_table": low_entropy_table,
        "high_entropy_indices": high_idx,
        "low_entropy_indices": low_idx,
    }


def mean_assignment_entropy(responsibilities: np.ndarray) -> float:
    return float(np.mean(assignment_entropy(responsibilities)))


def center_matching_mean_distance(centers_a: np.ndarray, centers_b: np.ndarray) -> float:
    """
    Match two center sets by Hungarian algorithm on squared Euclidean distances.
    """
    diff = centers_a[:, None, :] - centers_b[None, :, :]
    cost = np.sum(diff ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_cost = cost[row_ind, col_ind]
    return float(np.mean(np.sqrt(np.maximum(matched_cost, 0.0))))


def hard_label_agreement(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """
    Best permutation agreement between two cluster labelings.
    """
    y_a = np.asarray(y_a, dtype=np.int64)
    y_b = np.asarray(y_b, dtype=np.int64)
    n_classes = max(y_a.max(), y_b.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, b in zip(y_a, y_b):
        cm[a, b] += 1
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    return float(cm[row_ind, col_ind].sum() / len(y_a))


def run_c3_gmm_to_kmeans(
    pca_dim: int = 10,
    sigma2_values: Iterable[float] = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
    random_state: int = 42,
    gmm_n_init: int = 10,
    save_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    C3 experiment:
    Fix a shared spherical variance sigma^2 and show GMM -> KMeans as sigma^2 shrinks.
    """
    data = load_digits_data(test_size=0.25, random_state=random_state, standardize=True)
    X_train_raw = data["X_train"]
    X_test_raw = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    pca = PCAFromScratch(n_components=pca_dim)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    km = KMeansFromScratch(
        n_clusters=10,
        max_iter=100,
        tol=1e-4,
        n_init=20,
        random_state=random_state,
    )
    km.fit(X_train)
    km_test_pred = km.predict(X_test)

    km_train_metrics = evaluate_clustering(X_train, y_train, km.labels_)
    km_test_metrics = evaluate_clustering(X_test, y_test, km_test_pred)

    rows = []
    artifacts = {}

    for sigma2 in sigma2_values:
        gmm = GMMEMFromScratch(
            n_components=10,
            max_iter=200,
            tol=1e-4,
            reg_covar=min(1e-6, sigma2 * 1e-3),
            n_init=gmm_n_init,
            init_method="kmeans",
            random_state=random_state,
            covariance_type="spherical",
            shared_covariance=True,
            fixed_spherical_variance=float(sigma2),
        )
        gmm.fit(X_train)

        gmm_test_pred = gmm.predict(X_test)
        gmm_test_resp = gmm.predict_proba(X_test)

        gmm_train_metrics = evaluate_clustering(
            X_train,
            y_train,
            gmm.labels_,
            responsibilities=gmm.responsibilities_,
        )
        gmm_test_metrics = evaluate_clustering(
            X_test,
            y_test,
            gmm_test_pred,
            responsibilities=gmm_test_resp,
        )

        row = {
            "sigma2": float(sigma2),
            "kmeans_train_acc": km_train_metrics["ACC"],
            "kmeans_test_acc": km_test_metrics["ACC"],
            "gmm_train_acc": gmm_train_metrics["ACC"],
            "gmm_test_acc": gmm_test_metrics["ACC"],
            "gmm_train_ari": gmm_train_metrics["ARI"],
            "gmm_test_ari": gmm_test_metrics["ARI"],
            "gmm_train_nmi": gmm_train_metrics["NMI"],
            "gmm_test_nmi": gmm_test_metrics["NMI"],
            "gmm_train_entropy": gmm_train_metrics["MeanEntropy"],
            "gmm_test_entropy": gmm_test_metrics["MeanEntropy"],
            "gmm_final_loglik": gmm.log_likelihood_history_[-1],
            "label_agreement_train": hard_label_agreement(km.labels_, gmm.labels_),
            "label_agreement_test": hard_label_agreement(km_test_pred, gmm_test_pred),
            "center_distance": center_matching_mean_distance(km.cluster_centers_, gmm.means_),
            "iterations": gmm.result_.n_iter,
            "converged": gmm.result_.converged,
        }
        rows.append(row)

        artifacts[float(sigma2)] = {
            "gmm": gmm,
            "gmm_test_pred": gmm_test_pred,
            "gmm_test_resp": gmm_test_resp,
        }

    results_df = pd.DataFrame(rows).sort_values("sigma2", ascending=False).reset_index(drop=True)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        results_df.to_csv(os.path.join(save_dir, "c3_gmm_to_kmeans_results.csv"), index=False)

    return {
        "pca": pca,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "kmeans": km,
        "kmeans_test_pred": km_test_pred,
        "results_df": results_df,
        "artifacts": artifacts,
    }