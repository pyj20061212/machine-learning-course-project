import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from typing import Optional

def pairwise_squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Efficient squared Euclidean distances between rows of X and rows of C.
    Returns shape (n_samples, n_centers).
    """
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    C_norm = np.sum(C ** 2, axis=1, keepdims=True).T
    dists = X_norm + C_norm - 2 * (X @ C.T)
    return np.maximum(dists, 0.0)


def kmeans_plus_plus_init(X: np.ndarray, n_clusters: int, random_state: Optional[int] = None):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    centers = []
    first_idx = rng.integers(0, n_samples)
    centers.append(X[first_idx].copy())

    for _ in range(1, n_clusters):
        current_centers = np.vstack(centers)
        dists = pairwise_squared_distances(X, current_centers)
        min_dists = np.min(dists, axis=1)

        total = np.sum(min_dists)
        if total <= 1e-12:
            idx = rng.integers(0, n_samples)
        else:
            probs = min_dists / total
            idx = rng.choice(n_samples, p=probs)
        centers.append(X[idx].copy())

    return np.vstack(centers)


@dataclass
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool


class KMeansFromScratch:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 20,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.result_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _single_run(self, X: np.ndarray, seed: int) -> KMeansResult:
        centers = kmeans_plus_plus_init(X, self.n_clusters, random_state=seed)
        prev_inertia = None

        for it in range(1, self.max_iter + 1):
            dists = pairwise_squared_distances(X, centers)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # empty cluster re-init
                    farthest_idx = np.argmax(np.min(dists, axis=1))
                    new_centers[k] = X[farthest_idx]

            new_dists = pairwise_squared_distances(X, new_centers)
            inertia = np.sum(np.min(new_dists, axis=1))

            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                return KMeansResult(
                    centers=centers,
                    labels=np.argmin(new_dists, axis=1),
                    inertia=float(inertia),
                    n_iter=it,
                    converged=True,
                )

            if center_shift < self.tol:
                return KMeansResult(
                    centers=centers,
                    labels=np.argmin(new_dists, axis=1),
                    inertia=float(inertia),
                    n_iter=it,
                    converged=True,
                )

            prev_inertia = inertia

        final_dists = pairwise_squared_distances(X, centers)
        final_labels = np.argmin(final_dists, axis=1)
        final_inertia = float(np.sum(np.min(final_dists, axis=1)))

        return KMeansResult(
            centers=centers,
            labels=final_labels,
            inertia=final_inertia,
            n_iter=self.max_iter,
            converged=False,
        )

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        best_result = None
        for _ in range(self.n_init):
            seed = int(rng.integers(0, 10**9))
            result = self._single_run(X, seed)
            if best_result is None or result.inertia < best_result.inertia:
                best_result = result

        self.result_ = best_result
        self.cluster_centers_ = best_result.centers
        self.labels_ = best_result.labels
        self.inertia_ = best_result.inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = pairwise_squared_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_