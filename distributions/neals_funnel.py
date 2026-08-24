import numpy as np
from numba import njit
from ..base.nsmc_sampling import nsmc_sampling

class nsmc_sampling_neal_funnel(nsmc_sampling):
    """
    This class is for WHSS using a Generalized Neal's Funnel density.
    Parameters:
        d: dimension of the space
        k: required number of MCMC samples
        sigma_v: Standard deviation of the v dimension. Defaults to 3.0.
        sigma_x: Base standard deviation for the remaining dimensions. Defaults to 1.0.
    """
    def __init__(self, d, k, sigma_v=3.0, sigma_x=1.0, a=None):
        super().__init__(d, a if a is not None else 100.0, k)
        self.sigma_v = np.float64(sigma_v)
        self.sigma_x = np.float64(sigma_x)

    def f_cartesian_neal_funnel(self):
        """
        Returns a Numba-compiled pure Cartesian log-density function.
        x[0] is the 'v' (variance-controlling) parameter.
        x[1:] are the highly correlated 'x' dimensions.
        """
        dimension = self.d
        var_v = self.sigma_v**2
        var_x = self.sigma_x**2
        
        const_x0 = -0.5 * np.log(2 * np.pi * var_v)
        const_xi = -0.5 * (dimension - 1) * np.log(2 * np.pi * var_x)
        log_norm_const = const_x0 + const_xi

        @njit
        def density_cartesian(x):
            # v is the first dimension
            v = x[0]
            
            # The rest of the space depends exponentially on v
            x_rest_sq_sum = np.sum(x[1:]**2)
            
            log_density = log_norm_const \
                          - (v**2) / (2.0 * var_v) \
                          - 0.5 * (dimension - 1) * v \
                          - 0.5 * (1.0 / var_x) * np.exp(-v) * x_rest_sq_sum
                          
            return log_density
                
        return density_cartesian

    def get_samples(self, batch_size=3256, burn_in_samples=10000, max_anchors=50):
        funnel_den = self.f_cartesian_neal_funnel()
        
        mcmc_samples = self._sampling_universal(
            density_cartesian=funnel_den,
            batch_size=batch_size,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return mcmc_samples