import numpy as np
from numba import njit, prange
from ..base.whss import whss

class whss_rosenbock(whss):
    """
    This class is for nsmc sampling using the Hybrid Rosenbrock density.
    Parameters:
        d: dimension of the space
        a: length of the cube
        k: required number of accepted samples
        mu: location parameter for the first dimension
        a_rosen: scaling parameter for the first dimension
        b_rosen: list of lists containing scaling parameters for the blocks
    """
    def __init__(self, d, a, k, mu, a_rosen, b_rosen):
        super().__init__(d, a, k)
        self.mu = np.float64(mu)
        self.a_rosen = np.float64(a_rosen)
        self.b_rosen = b_rosen

    def f_r_rosenbock(self):
            dimension = self.d
            mu = self.mu
            a_rosen = self.a_rosen
            
            # 1. Pre-parse the block structure into flat 1D arrays 
            flat_coeffs_list = []
            prev_indices_list = []
            curr_idx = 1
            
            for block in self.b_rosen:
                prev_idx = 0  # The first element in each block depends on x1 (index 0)
                for coeff in block:
                    flat_coeffs_list.append(coeff)
                    prev_indices_list.append(prev_idx)
                    prev_idx = curr_idx
                    curr_idx += 1
                    
            flat_coeffs = np.array(flat_coeffs_list, dtype=np.float64)
            prev_indices = np.array(prev_indices_list, dtype=np.int64)
            
            # 2. Compute the log normalizing constant
            log_norm_const = (0.5 * np.log(a_rosen) 
                            + 0.5 * np.sum(np.log(flat_coeffs)) 
                            - (dimension / 2.0) * np.log(np.pi))

            @njit
            def _rosen_single(r_batch, theta):
                # r_batch can be a scalar (from golden search) or a 1D array.
                # Vectorized arithmetic handles both automatically.
                x1 = r_batch * theta[0]
                
                # Base term for x1
                log_density = -a_rosen * (x1 - mu)**2
                
                # Dynamic terms for all blocks
                for j in range(len(flat_coeffs)):
                    curr_x = r_batch * theta[j + 1]
                    prev_x = r_batch * theta[prev_indices[j]]
                    log_density += -flat_coeffs[j] * (curr_x - prev_x**2)**2
                    
                log_density += log_norm_const
                log_volume = (dimension - 1) * np.log(r_batch + 1e-10)
                
                return log_density + log_volume

            @njit(parallel=True)
            def _rosen_multi(r_batch, theta_batch):
                N = r_batch.shape[0]
                result = np.empty(N)
                
                for i in prange(N):
                    r = r_batch[i]
                    theta = theta_batch[i]
                    x1 = r * theta[0]
                    
                    log_density = -a_rosen * (x1 - mu)**2
                    
                    for j in range(len(flat_coeffs)):
                        curr_x = r * theta[j + 1]
                        prev_x = r * theta[prev_indices[j]]
                        log_density += -flat_coeffs[j] * (curr_x - prev_x**2)**2
                    
                    log_density += log_norm_const
                    log_volume = (dimension - 1) * np.log(r + 1e-10)
                    
                    result[i] = log_density + log_volume
                    
                return result
            
            @njit 
            def f_r_rosen(r_batch, theta_batch):
                if theta_batch.ndim == 1 or (theta_batch.ndim == 2 and theta_batch.shape[0] == 1):
                    theta_1d = theta_batch.ravel()
                    return _rosen_single(r_batch, theta_1d)
                else:
                    return _rosen_multi(r_batch, theta_batch)
                    
            return f_r_rosen



    def get_samples(self, batch_size=3256, fallback_proposer="vmf", switch_threshold=0.05, burn_in_samples=None, max_anchors=50):
        rosen_den = self.f_r_rosenbock()
        accepted, rejected = self._sampling_universal(
            density=rosen_den,
            batch_size=batch_size,
            fallback_proposer=fallback_proposer,
            switch_threshold=switch_threshold,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return accepted, rejected