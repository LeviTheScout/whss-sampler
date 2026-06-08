import numpy as np
import scipy.linalg
from numba import njit
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
            mu=self.mu 
            dimension=self.d
             
            @njit
            def f_r_gauss(r_batch,theta_batch):
                x_pos=r_batch.reshape(-1,1)*theta_batch # (N,d)
                # works fine for both many theta-one r case and one theta-many r case.

                diff = x_pos - mu
                y = diff @ L_inv.T
                # this actual computation of inverse once and then use multiplication is feasible because of njit.
                # otherwise we would have used solve.
                w = np.sum(y**2, axis=1)   # = diff^T Sigma^{-1} diff

                log_density = log_norm_const - 0.5 * w
                
                log_volume = (dimension - 1) * np.log(r_batch+1e-10)

                # result=np.exp(log_volume + log_density)
                return log_volume+log_density
            #
            # def f_r_gauss(r, theta_batch): 
            #     # here r is distance from origin not directional vector.
            #     # r will be a (batch_size,) dim vector and theta_batch will be (batch_size,d) dim matrix
            #     # hence each row of theta_batch will be one sample
            #     r = np.atleast_1d(r)
            #
            #     theta_batch = np.asarray(theta_batch)
            #     if theta_batch.ndim == 1:
            #         theta_batch = theta_batch[None, :]   # (1, d)
            #
            #
            #     x_pos = r[:,None]*theta_batch
            #     if result.shape[0]==1:
            #         return result[0]
            #     return result            

            return f_r_gauss

    def get_samples(self,batch_size=None):
        gauss_den=self.f_r_gaussian()
        if batch_size is None:
            accepted,rejected=self.sampling_f_r_new(gauss_den)
        else:

            accepted,rejected=self.sampling_f_r_new(gauss_den,batch_size)
        return accepted,rejected
