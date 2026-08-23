import numpy as np
import scipy.linalg
from numba import njit,prange
from ..base.nsmc_sampling import nsmc_sampling

class nsmc_sampling_gaussian(nsmc_sampling):
    """
    This class is specifically for nsmc sampling using f_r density being the gaussian density
    Parameters:
        d: dimsion of the cube
        a: length of the cube.
        k: required number of accepted samples
        sigma: for gaussian based density
        mu: required for the gaussian density
           """
    def __init__(self,d,a,k,sigma,mu):
        super().__init__(d,a,k)
        self.sigma=sigma
        self.mu=mu



    def f_r_gaussian(self):
            """
            - using this nested function for first function to be used for substituting Parameters which are constant
              across all points and inner function to be returend with actual density value.
            - Vectorised. 
            This function returns the d-dimensional multivariate Gaussian density which takes input 'r' a 
            length from origin and returns the gaussian at that function.
            - returns log(f(r))+(d-1)*log(r)
            - should be able to handle one theta - many r, many theta - many r, one theta - one r
            
            Edit: Now I tried to use cholesky to tackle inverse and uses log and then exponential to make it more numerically stable (not necessary but good addition maybe).
            """
            
            L = np.linalg.cholesky(self.sigma)

            # L_inv computation.
            L_inv=np.linalg.inv(L) # will this be fine in high d? could lead to bottleneck in high-d
            #can use scipy also to make it mroe feasible.
            #I=np.eye(self.d)
            #L_inv=scipy.linalg.solve_triangular(L,I,lower=True)

            log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
            log_norm_const = -0.5 * (self.d * np.log(2*np.pi) + log_det_sigma)
            mu=np.asarray(self.mu,dtype=np.float64) 
            dimension=self.d
            u=mu@L_inv.T
            C = np.sum(u**2)
            @njit
            def _gauss_single(r_batch,theta):
                v=theta@L_inv.T
                
                A = np.sum(v**2)
                B = -2.0 * np.sum(v * u)
                
                # 2. Evaluate the grid using fast 1D arrays (no d-dimensional matrices)
                # Numba vectorizes this brilliantly without massive allocations.
                w = A * (r_batch**2) + B * r_batch + C
                
                log_density = log_norm_const - 0.5 * w
                log_volume = (dimension - 1) * np.log(r_batch + 1e-10)
                
                return log_density + log_volume


            @njit(parallel=True)
            def _gauss_multi(r_batch,theta_batch):
                N = r_batch.shape[0]
                result = np.empty(N)
            
                for i in prange(N):
                    r = r_batch[i]
                    theta = theta_batch[i]
                    
                    diff = (r * theta) - mu
                    y = diff @ L_inv.T 
                    w = np.sum(y**2)
                        
                    log_density = log_norm_const - 0.5 * w
                    log_volume = (dimension - 1) * np.log(r + 1e-10)
                    
                    result[i] = log_density  +log_volume
                    
                return result
            
            @njit 
            def f_r_gauss(r_batch,theta_batch):
                if theta_batch.ndim==1 or (theta_batch.ndim==2 and theta_batch.shape[0]==1):
                    theta_1d=theta_batch.ravel()
                    return _gauss_single(r_batch,theta_1d)
                else:
                    return _gauss_multi(r_batch,theta_batch)

            return f_r_gauss

    def get_samples(self, batch_size=3256, fallback_proposer="vmf", switch_threshold=0.05, burn_in_samples=None, max_anchors=50):
        
        target_density = self.f_r_gaussian()
        
        accepted, rejected = self._sampling_universal(
            density=target_density,
            batch_size=batch_size,
            fallback_proposer=fallback_proposer,
            switch_threshold=switch_threshold,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return accepted, rejected

    def ksd_distance(self,accepted):
        gauss_den=self.f_r_gaussian()
        distance=self.ksd(accepted,gauss_den)
        return distance

