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
            This function returns the d-dimensional multivariate Gaussian density which takes input 'r' a 
            length from origin and returns the gaussian at that function.
            Also, it provides 'f_max' a mode value which is to be utilised for the purpose
            of rejection sampling.
            
            Edit: Now I tried to use cholesky to tackle inverse and uses log and then exponential to make it more numerically stable (not necessary but good addition maybe).
            """
            
            L = np.linalg.cholesky(self.sigma)
            log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
            log_norm_const = -0.5 * (self.d * np.log(2*np.pi) + log_det_sigma)
            
            def f_r_gauss(r, theta): # here r is distance from origin not directional vector.
                if r <= 0:
                    return 0.0 # Prevent log(0) error
                    
                r_vec, _ = self.R(theta) 
                x_pos = r*r_vec
                diff = x_pos - self.mu
                # Solve L y = diff
                y = np.linalg.solve(L, diff)

                w = np.dot(y, y)   # = diff^T Sigma^{-1} diff

                log_density = log_norm_const - 0.5 * w
                
                log_volume = (self.d - 1) * np.log(r)
                return np.exp(log_volume + log_density)
            # 1. Get the largest eigenvalue (variance along the major axis)
            #eigvals = np.linalg.eigvalsh(self.sigma)
            #max_var = eigvals[-1] 

            # 2. Get the norm of the mean
            #mu_norm = np.linalg.norm(self.mu)

            # 3. Solve the radial mode equation using the maximum variance
            #x_mode = (mu_norm + np.sqrt(mu_norm**2 + 4 * (self.d - 1) * max_var)) / 2.0
            
            x_mode = np.sqrt(max(0, self.d - 1 + np.linalg.norm(self.mu)**2))
            f_max = f_r_gauss(x_mode, self.theta_generation())
            return f_r_gauss, f_max

    def get_samples(self):
        gauss_den=self.f_r_gaussian()
        accepted,rejected=self.sampling_f_r(gauss_den)
        return accepted,rejected
