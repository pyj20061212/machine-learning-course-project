import numpy as np
from typing import Optional, List, Dict, Tuple

class PCAFromScratch:
    """
    PCA via eigendecomposition of covariance matrix.
    Suitable for digits (64-dim), simple and stable.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None              # shape: (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.eigenvalues_ = eigvals
        self.components_ = eigvecs[:, : self.n_components].T
        self.explained_variance_ = eigvals[: self.n_components]

        total_var = np.sum(np.clip(eigvals, a_min=0.0, a_max=None))
        if total_var <= 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def reconstruction_error(self, X: np.ndarray) -> float:
        Z = self.transform(X)
        X_hat = self.inverse_transform(Z)
        return np.mean(np.sum((X - X_hat) ** 2, axis=1))