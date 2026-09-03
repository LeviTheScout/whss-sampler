
import numpy as np
import scipy.linalg
from numba import njit,prange
from ..base.whss import whss

class whss_gmm(whss):
    """
    This class is specifically for nsmc sampling using f_r density being a Gaussian Mixture Model (GMM) density.
    Parameters:
        d: dimension of the cube
        a: length of the cube.
        k: required number of accepted samples
        weights: (K,) mixture weights, should sum to 1
        mus: (K,d) means of each component
        sigmas: (K,d,d) covariance matrices of each component
    """
    def __init__(self, d, a, k, weights, mus, sigmas):
        super().__init__(d, a, k)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.mus = np.asarray(mus, dtype=np.float64)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.K = self.weights.shape[0]

    def f_r_gmm(self):
        """
        - using this nested function for first function to be used for substituting Parameters which are constant
          across all points (per-component Cholesky factors, log-norm-consts, log-weights, and the mu-only terms
          u_k = L_inv_k @ mu_k, C_k = ||u_k||^2) and inner function to be returned with actual density value.
        - Vectorised.
        This function returns the d-dimensional GMM density which takes input 'r' a length from origin and
        returns the GMM density at that point.
        - returns log(f(r))+(d-1)*log(r), where f(r) = sum_k weights[k] * N(r*theta; mu_k, sigma_k)
        - should be able to handle one theta - many r, many theta - many r, one theta - one r

        Edit: u_k and C_k depend only on mu_k and sigma_k (both fixed for the object's lifetime), so they're
        precomputed once here rather than recomputed inside _gmm_single on every call - same fix as the
        single-Gaussian version, applied per component.
        """
        K = self.K
        dimension = self.d

        L_inv_all = np.empty((K, dimension, dimension), dtype=np.float64)
        log_norm_const = np.empty(K, dtype=np.float64)

        for k in range(K):
            L = np.linalg.cholesky(self.sigmas[k])
            L_inv_all[k] = np.linalg.inv(L)
            log_det_sigma_k = 2.0 * np.sum(np.log(np.diag(L)))
            log_norm_const[k] = -0.5 * (dimension * np.log(2*np.pi) + log_det_sigma_k)

        mus = np.asarray(self.mus, dtype=np.float64)
        log_weights = np.log(self.weights)

        # precompute mu-only terms once (mirrors the single-Gaussian u/C hoist, per component)
        u_all = np.empty((K, dimension), dtype=np.float64)
        for k in range(K):
            u_all[k] = L_inv_all[k] @ mus[k]
        C_const_all = np.sum(u_all**2, axis=1)                  # (K,)
        log_weighted_norm_const = log_norm_const + log_weights  # (K,), also constant, hoisted

        @njit(fastmath=True)
        def _gmm_single(r_batch, theta):
            N = r_batch.shape[0]
            log_comp = np.empty((N, K))   # row-major layout: contiguous per-sample for the reduction below

            for k in range(K):
                v = L_inv_all[k] @ theta
                A = np.sum(v**2)
                B = -2.0 * np.sum(v * u_all[k])
                w = A * (r_batch**2) + B * r_batch + C_const_all[k]
                log_comp[:, k] = log_weighted_norm_const[k] - 0.5 * w

            out = np.empty(N)
            for i in range(N):
                m = log_comp[i, 0]
                for k in range(1, K):
                    if log_comp[i, k] > m:
                        m = log_comp[i, k]
                s = 0.0
                for k in range(K):
                    s += np.exp(log_comp[i, k] - m)
                out[i] = m + np.log(s)

            log_volume = (dimension - 1) * np.log(r_batch + 1e-10)
            return out + log_volume

        @njit(parallel=True)
        def _gmm_multi(r_batch, theta_batch):
            N = r_batch.shape[0]
            result = np.empty(N)

            for i in prange(N):
                r = r_batch[i]
                theta = theta_batch[i]

                log_comp_local = np.empty(K)
                for k in range(K):
                    diff = (r * theta) - mus[k]
                    y = L_inv_all[k] @ diff
                    w = np.sum(y**2)
                    log_comp_local[k] = log_weighted_norm_const[k] - 0.5 * w

                m = log_comp_local[0]
                for k in range(1, K):
                    if log_comp_local[k] > m:
                        m = log_comp_local[k]
                s = 0.0
                for k in range(K):
                    s += np.exp(log_comp_local[k] - m)
                log_density = m + np.log(s)

                log_volume = (dimension - 1) * np.log(r + 1e-10)
                result[i] = log_density + log_volume

            return result

        @njit
        def f_r_gmm(r_batch, theta_batch):
            if theta_batch.ndim == 1 or (theta_batch.ndim == 2 and theta_batch.shape[0] == 1):
                theta_1d = theta_batch.ravel()
                return _gmm_single(r_batch, theta_1d)
            else:
                return _gmm_multi(r_batch, theta_batch)

        return f_r_gmm

    def get_samples(self, batch_size=3256, fallback_proposer="vmf", switch_threshold=0.05, burn_in_samples=None, max_anchors=50):

        target_density = self.f_r_gmm()

        accepted, rejected = self._sampling_universal(
            density=target_density,
            batch_size=batch_size,
            fallback_proposer=fallback_proposer,
            switch_threshold=switch_threshold,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return accepted, rejected

    def ksd_distance(self, accepted):
        gmm_den = self.f_r_gmm()
        distance = self.ksd(accepted, gmm_den)
        return distance
