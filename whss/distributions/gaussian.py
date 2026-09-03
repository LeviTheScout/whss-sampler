import numpy as np
import scipy.linalg
from numba import njit
from ..base.whss import whss

class whss_gaussian(whss):
    """
    This class is specifically for Warped Hybrid Slice Sampling (WHSS)
    using a multivariate Gaussian density.
    Parameters:
        d: dimension of the space
        k: required number of MCMC samples
        sigma: covariance matrix
        mu: mean vector
    """
    def __init__(self, d, k, sigma, mu, a=None):
        # We keep 'a' (bounding box) in super init just for legacy/plotting compatibility
        super().__init__(d, a if a is not None else 100.0, k)
        self.sigma = sigma
        self.mu = mu

    def f_cartesian_gaussian(self):
        """
        Returns a Numba-compiled pure Cartesian log-density function.
        No volume corrections (Jacobian) are needed here!
        """
        L = np.linalg.cholesky(self.sigma)
        L_inv = np.linalg.inv(L) 

        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        log_norm_const = -0.5 * (self.d * np.log(2 * np.pi) + log_det_sigma)
        
        mu = np.asarray(self.mu, dtype=np.float64) 
        
        @njit
        def density_cartesian(x):
            # Pure Cartesian evaluation: log(N(x | mu, sigma))
            diff = x - mu
            y = diff @ L_inv.T 
            w = np.sum(y**2)
            
            return log_norm_const - 0.5 * w

        return density_cartesian

    def get_samples(self, batch_size=3256, burn_in_samples=10000, max_anchors=50):
        target_density = self.f_cartesian_gaussian()
        
        # MCMC only returns accepted valid samples in a single chain!
        mcmc_samples = self._sampling_universal(
            density_cartesian=target_density,
            batch_size=batch_size,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return mcmc_samples

    def ksd_distance(self, samples):
        # Useful for measuring the quality of the MCMC chain against the true target
        gauss_den = self.f_cartesian_gaussian()
        distance = self.ksd(samples, gauss_den)
        return distance