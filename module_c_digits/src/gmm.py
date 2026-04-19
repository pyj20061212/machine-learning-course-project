import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.special import logsumexp

from .kmeans import KMeansFromScratch


@dataclass
class GMMResult:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray      # always shape (K, D)
    responsibilities: np.ndarray
    labels: np.ndarray
    log_likelihood_history: List[float]
    n_iter: int
    converged: bool


class GMMEMFromScratch:
    """
    Gaussian Mixture Model with EM.
    Backward-compatible:
      - default: diagonal covariance, same as C1/C2
    Extra modes for C3:
      - covariance_type = "spherical"
      - shared_covariance = True
      - fixed_spherical_variance = sigma^2
    Internally, covariances are always stored as shape (K, D).
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 10,
        init_method: str = "kmeans",
        random_state: Optional[int] = None,
        covariance_type: str = "diag",          # "diag" or "spherical"
        shared_covariance: bool = False,
        fixed_spherical_variance: Optional[float] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_method = init_method
        self.random_state = random_state

        self.covariance_type = covariance_type
        self.shared_covariance = shared_covariance
        self.fixed_spherical_variance = fixed_spherical_variance

        if self.covariance_type not in {"diag", "spherical"}:
            raise ValueError("covariance_type must be 'diag' or 'spherical'.")

        if self.fixed_spherical_variance is not None:
            if self.covariance_type != "spherical" or not self.shared_covariance:
                raise ValueError(
                    "fixed_spherical_variance requires covariance_type='spherical' "
                    "and shared_covariance=True."
                )

        self.result_ = None
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.labels_ = None
        self.log_likelihood_history_ = None

    def _expand_spherical_to_diag_matrix(self, scalar_vars: np.ndarray, n_features: int) -> np.ndarray:
        """
        scalar_vars: shape (K,)
        returns shape (K, D)
        """
        return np.repeat(scalar_vars[:, None], n_features, axis=1)

    def _initialize(self, X: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        n_samples, n_features = X.shape

        if self.init_method == "kmeans":
            km = KMeansFromScratch(
                n_clusters=self.n_components,
                max_iter=50,
                tol=1e-4,
                n_init=3,
                random_state=seed,
            ).fit(X)
            means = km.cluster_centers_.copy()
            labels = km.labels_

            weights = np.array(
                [(labels == k).mean() for k in range(self.n_components)],
                dtype=np.float64,
            )
            weights = np.clip(weights, 1e-8, None)
            weights /= weights.sum()
        else:
            idx = rng.choice(n_samples, size=self.n_components, replace=False)
            means = X[idx].copy()
            weights = np.ones(self.n_components, dtype=np.float64) / self.n_components

        global_var_diag = np.var(X, axis=0) + self.reg_covar

        if self.covariance_type == "diag":
            covariances = np.zeros((self.n_components, n_features), dtype=np.float64)

            if self.init_method == "kmeans":
                for k in range(self.n_components):
                    mask = labels == k
                    if np.sum(mask) >= 2:
                        covariances[k] = np.var(X[mask], axis=0) + self.reg_covar
                    else:
                        covariances[k] = global_var_diag.copy()
            else:
                covariances = np.tile(global_var_diag, (self.n_components, 1))

            if self.shared_covariance:
                shared = np.mean(covariances, axis=0, keepdims=True)
                covariances = np.repeat(shared, self.n_components, axis=0)

        else:  # spherical
            if self.fixed_spherical_variance is not None:
                scalar_vars = np.full(
                    self.n_components,
                    max(float(self.fixed_spherical_variance), self.reg_covar),
                    dtype=np.float64,
                )
            elif self.init_method == "kmeans":
                scalar_vars = np.zeros(self.n_components, dtype=np.float64)
                global_scalar = float(np.mean(global_var_diag))

                for k in range(self.n_components):
                    mask = labels == k
                    if np.sum(mask) >= 2:
                        cluster_var = np.var(X[mask], axis=0)
                        scalar_vars[k] = max(float(np.mean(cluster_var)), self.reg_covar)
                    else:
                        scalar_vars[k] = max(global_scalar, self.reg_covar)
            else:
                scalar_vars = np.full(
                    self.n_components,
                    max(float(np.mean(global_var_diag)), self.reg_covar),
                    dtype=np.float64,
                )

            if self.shared_covariance:
                shared_scalar = float(np.mean(scalar_vars))
                scalar_vars[:] = max(shared_scalar, self.reg_covar)

            covariances = self._expand_spherical_to_diag_matrix(scalar_vars, n_features)

        return weights, means, covariances

    def _estimate_log_gaussian_prob(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        n_samples, n_features = X.shape
        K = means.shape[0]
        log_prob = np.zeros((n_samples, K), dtype=np.float64)

        for k in range(K):
            diff = X - means[k]
            var = covariances[k]
            log_det = np.sum(np.log(var))
            quad = np.sum((diff ** 2) / var, axis=1)
            log_prob[:, k] = -0.5 * (
                n_features * np.log(2 * np.pi) + log_det + quad
            )

        return log_prob

    def _e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        log_gauss = self._estimate_log_gaussian_prob(X, means, covariances)
        log_weighted = log_gauss + np.log(weights + 1e-16)
        log_prob_norm = logsumexp(log_weighted, axis=1, keepdims=True)
        log_resp = log_weighted - log_prob_norm
        responsibilities = np.exp(log_resp)
        log_likelihood = float(np.sum(log_prob_norm))
        return responsibilities, log_likelihood

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0) + 1e-16

        weights = Nk / n_samples
        means = (responsibilities.T @ X) / Nk[:, None]

        if self.covariance_type == "diag":
            covariances = np.zeros((self.n_components, n_features), dtype=np.float64)

            for k in range(self.n_components):
                diff = X - means[k]
                weighted_sq = responsibilities[:, k][:, None] * (diff ** 2)
                covariances[k] = weighted_sq.sum(axis=0) / Nk[k]
                covariances[k] = np.clip(covariances[k], self.reg_covar, None)

            if self.shared_covariance:
                shared = np.mean(covariances, axis=0, keepdims=True)
                covariances = np.repeat(shared, self.n_components, axis=0)

        else:  # spherical
            if self.fixed_spherical_variance is not None:
                scalar = max(float(self.fixed_spherical_variance), self.reg_covar)
                scalar_vars = np.full(self.n_components, scalar, dtype=np.float64)
                covariances = self._expand_spherical_to_diag_matrix(scalar_vars, n_features)
                return weights, means, covariances

            scalar_vars = np.zeros(self.n_components, dtype=np.float64)

            for k in range(self.n_components):
                diff = X - means[k]
                weighted_norm2 = responsibilities[:, k] * np.sum(diff ** 2, axis=1)
                scalar_vars[k] = weighted_norm2.sum() / (Nk[k] * n_features)
                scalar_vars[k] = max(float(scalar_vars[k]), self.reg_covar)

            if self.shared_covariance:
                total_weighted_norm2 = 0.0
                total_weight = 0.0
                for k in range(self.n_components):
                    diff = X - means[k]
                    total_weighted_norm2 += np.sum(
                        responsibilities[:, k] * np.sum(diff ** 2, axis=1)
                    )
                    total_weight += Nk[k] * n_features

                shared_scalar = total_weighted_norm2 / max(total_weight, 1e-16)
                shared_scalar = max(float(shared_scalar), self.reg_covar)
                scalar_vars[:] = shared_scalar

            covariances = self._expand_spherical_to_diag_matrix(scalar_vars, n_features)

        return weights, means, covariances

    def _single_run(self, X: np.ndarray, seed: int) -> GMMResult:
        weights, means, covariances = self._initialize(X, seed)
        ll_history = []
        prev_ll = None
        converged = False

        for it in range(1, self.max_iter + 1):
            responsibilities, ll = self._e_step(X, weights, means, covariances)
            ll_history.append(ll)

            weights, means, covariances = self._m_step(X, responsibilities)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                converged = True
                break

            prev_ll = ll

        responsibilities, ll = self._e_step(X, weights, means, covariances)
        if len(ll_history) == 0 or ll_history[-1] != ll:
            ll_history.append(ll)

        labels = np.argmax(responsibilities, axis=1)

        return GMMResult(
            weights=weights,
            means=means,
            covariances=covariances,
            responsibilities=responsibilities,
            labels=labels,
            log_likelihood_history=ll_history,
            n_iter=it,
            converged=converged,
        )

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        best_result = None
        best_ll = -np.inf

        for _ in range(self.n_init):
            seed = int(rng.integers(0, 10**9))
            result = self._single_run(X, seed)
            final_ll = result.log_likelihood_history[-1]

            if final_ll > best_ll:
                best_ll = final_ll
                best_result = result

        self.result_ = best_result
        self.weights_ = best_result.weights
        self.means_ = best_result.means
        self.covariances_ = best_result.covariances
        self.responsibilities_ = best_result.responsibilities
        self.labels_ = best_result.labels
        self.log_likelihood_history_ = best_result.log_likelihood_history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        responsibilities, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        responsibilities, _ = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return responsibilities

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_