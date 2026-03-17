import numpy as np
from ..base.nsmc_sampling import nsmc_sampling


class nsmc_sampling_rosenbock(nsmc_sampling):

    def __init__(self, d, a, k,mu,a_rosen,b_rosen):
        super().__init__(d, a, k)
        self.mu=mu
        self.a_rosen=a_rosen
        self.b_rosen=b_rosen

    def f_r_rosenbock(self):

        all_b_coeffs = [coeff for block in self.b_rosen for coeff in block]
        prod_sqrt_b = np.prod(np.sqrt(all_b_coeffs))
        normaling_const = (np.pi**(self.d / 2.0)) / (np.sqrt(self.a_rosen) * prod_sqrt_b)
        
        def rosenbrock_density(r,theta): # here r is distance from origin not directional vector.
        
            x=r*theta
            x1 = x[0]
            log_numerator = -self.a_rosen * (x1 - self.mu)**2
    
            current_x_idx = 1
            
            for block_coeffs in self.b_rosen:
                x_prev = x1 
                # x_j,1=x1
                for coeff in block_coeffs:
                    x_curr = x[current_x_idx]
                    term = -coeff * (x_curr - x_prev**2)**2
                    log_numerator += term
                    
                    x_prev = x_curr
                    current_x_idx += 1
            numerator = np.exp(log_numerator)
            g_r=(r**(self.d-1)) * (numerator/normaling_const)
            return g_r

        return rosenbrock_density

    def get_samples(self):
        rosen_density=self.f_r_rosenbock()
        accepted,rejected=self.sampling_f_r(rosen_density)
        return accepted,rejected

