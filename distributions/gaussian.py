import numpy as np
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
            - Vectorised. 
            This function returns the d-dimensional multivariate Gaussian density which takes input 'r' a 
            length from origin and returns the gaussian at that function.
            Also, it provides 'f_max' a mode value which is to be utilised for the purpose
            of rejection sampling.
            
            Edit: Now I tried to use cholesky to tackle inverse and uses log and then exponential to make it more numerically stable (not necessary but good addition maybe).
            """
            
            L = np.linalg.cholesky(self.sigma)
            log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
            log_norm_const = -0.5 * (self.d * np.log(2*np.pi) + log_det_sigma)
            
            def f_r_gauss(r, theta_batch): 
                # here r is distance from origin not directional vector.
                # r will be a (batch_size,) dim vector and theta_batch will be (batch_size,d) dim matrix
                # hence each row of theta_batch will be one sample
                r = np.atleast_1d(r)

                theta_batch = np.asarray(theta_batch)
                if theta_batch.ndim == 1:
                    theta_batch = theta_batch[None, :]   # (1, d)

                    
                x_pos = r[:,None]*theta_batch
                diff = x_pos - self.mu
                # Solve L y = diff
                y = np.linalg.solve(L, diff.T).T

                w = np.sum(y**2, axis=1)   # = diff^T Sigma^{-1} diff

                log_density = log_norm_const - 0.5 * w
                
                log_volume = (self.d - 1) * np.log(r)

                result=np.exp(log_volume + log_density)
                if result.shape[0]==1:
                    return result[0]
                return result            

            # 1. Get the largest eigenvalue (variance along the major axis)
            #eigvals = np.linalg.eigvalsh(self.sigma)
            #max_var = eigvals[-1] 

            # 2. Get the norm of the mean
            #mu_norm = np.linalg.norm(self.mu)

            # 3. Solve the radial mode equation using the maximum variance
            #x_mode = (mu_norm + np.sqrt(mu_norm**2 + 4 * (self.d - 1) * max_var)) / 2.0
            # mu_norm = np.linalg.norm(self.mu)
            # x_mode = np.sqrt(self.d - 1) + mu_norm
            #
            #
            # if mu_norm > 0:
            #     theta_star = self.mu / mu_norm
            # else:
            #     theta_star = np.zeros(self.d)
            #     theta_star[0] = 1
            # f_max = f_r_gauss(np.array([x_mode]), theta_star[None,:])
            
            #
            # x_mode = np.sqrt(max(0, self.d - 1 + np.linalg.norm(self.mu)**2))
            # mu_norm = np.linalg.norm(self.mu)
            # if mu_norm > 0:
            #     theta_star = self.mu / mu_norm
            # else:
            #     theta_star = np.zeros(self.d)
            #     theta_star[0] = 1
            #
            # f_max = f_r_gauss(x_mode, theta_star) 
            r_mode = np.sqrt(max(0, self.d - 1))
            theta_star = np.zeros(self.d)
            theta_star[0] = 1.0
            f_max = f_r_gauss(r_mode, theta_star) * 1.05
            return f_r_gauss

    def get_samples(self,batch_size=None):
        gauss_den=self.f_r_gaussian()
        if batch_size is None:
            accepted,rejected=self.sampling_f_r_new(gauss_den)
        else:

            accepted,rejected=self.sampling_f_r_new(gauss_den,batch_size)
        return accepted,rejected
