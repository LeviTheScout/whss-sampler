import numpy as np
import scipy.linalg
from numba import njit,prange
from ..base.nsmc_sampling import nsmc_sampling

class nsmc_sampling_neal_funnel(nsmc_sampling):
    """
    This class is for nsmc sampling using a Generalized Neal's Funnel density.
    Parameters:
        d: dimension of the space
        a: length of the cube
        k: required number of accepted samples
        sigma_v: Standard deviation of the first dimension (hyperparameter). Defaults to 3.0 (classic funnel).
        sigma_x: Base standard deviation for the remaining dimensions. Defaults to 1.0 (classic funnel).
    """
    def __init__(self, d, a, k, sigma_v=3.0, sigma_x=1.0):
        super().__init__(d, a, k)
        self.sigma_v = np.float64(sigma_v)
        self.sigma_x = np.float64(sigma_x)

    def f_r_neal_funnel(self):
        dimension = self.d
        var_v = self.sigma_v**2
        var_x = self.sigma_x**2
        
        # Precompute the normalizing constants using the user's parameters
        const_x0 = -0.5 * np.log(2 * np.pi * var_v)
        const_xi = -0.5 * (dimension - 1) * np.log(2 * np.pi * var_x)
        log_norm_const = const_x0 + const_xi

        @njit
        def _funnel_single(r_batch, theta):
            x0 = r_batch * theta[0]
            theta_rest_sq_sum = np.sum(theta[1:]**2)
            x_rest_sq_sum = (r_batch**2) * theta_rest_sq_sum
            
            # Incorporating var_v and var_x into the density
            log_density = log_norm_const \
                          - (x0**2) / (2.0 * var_v) \
                          - 0.5 * (dimension - 1) * x0 \
                          - 0.5 * (1.0 / var_x) * np.exp(-x0) * x_rest_sq_sum
                          
            log_volume = (dimension - 1) * np.log(r_batch + 1e-10)
            return log_density + log_volume

        @njit(parallel=True)
        def _funnel_multi(r_batch, theta_batch):
            N = r_batch.shape[0]
            result = np.empty(N)
        
            for i in prange(N):
                r = r_batch[i]
                theta = theta_batch[i]
                
                x0 = r * theta[0]
                theta_rest_sq_sum = np.sum(theta[1:]**2)
                x_rest_sq_sum = (r**2) * theta_rest_sq_sum
                
                # Incorporating var_v and var_x into the density
                log_density = log_norm_const \
                              - (x0**2) / (2.0 * var_v) \
                              - 0.5 * (dimension - 1) * x0 \
                              - 0.5 * (1.0 / var_x) * np.exp(-x0) * x_rest_sq_sum
                              
                log_volume = (dimension - 1) * np.log(r + 1e-10)
                result[i] = log_density + log_volume
                
            return result
        
        @njit 
        def f_r_funnel(r_batch, theta_batch):
            if theta_batch.ndim == 1 or (theta_batch.ndim == 2 and theta_batch.shape[0] == 1):
                theta_1d = theta_batch.ravel()
                return _funnel_single(r_batch, theta_1d)
            else:
                return _funnel_multi(r_batch, theta_batch)
                
        return f_r_funnel

    def get_samples(self, batch_size=3256, fallback_proposer="vmf", switch_threshold=0.05, burn_in_samples=None, max_anchors=50):
        funnel_den = self.f_r_neal_funnel()
        accepted, rejected = self._sampling_universal(
            density=funnel_den,
            batch_size=batch_size,
            fallback_proposer=fallback_proposer,
            switch_threshold=switch_threshold,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return accepted, rejected

