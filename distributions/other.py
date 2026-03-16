import numpy as np
from scipy.special import beta as beta_func

from ..base.nsmc_sampling import nsmc_sampling


class nsmc_sampling_beta(nsmc_sampling):
    """
    This class is to sample from beta distribution along each sampled theta. This is just to verify our 
    code. With low beta and high-alpha, it should sample near edges.
    Parameters:
        alpha:
        beta:
        d: dimsion of the cube
        a: length of the cube.
        k: required number of accepted samples
    """
    def __init__(self,d,a,k,alpha,beta):
        super().__init__(d,a,k)
        self.alpha=alpha
        self.beta=beta

    def f_r_beta(self,r,R_):
        # if np.any(r < 0) or np.any(r > R_):
        #     return 0
        normalization = 1 / (R_ * beta_func(self.alpha, self.beta))
        kernel = (r / R_)**(self.alpha - 1) * (1 - r / R_)**(self.beta - 1)
        return normalization * kernel   
    
    def get_samples(self):
        """
        """
        accepted=[]
        rejected=[]
        R_diag=(self.a/2)*np.sqrt(self.d)
        x_max=(R_diag)*(self.d+self.alpha-2)/(self.d+self.alpha+self.beta-3)
        f_max=((R_diag)**(self.d-1))*self.f_r_beta(x_max,R_diag)

        while len(accepted)<self.k:
            theta=self.theta_generation()
            _,R_=self.R(theta)
            sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)
            if sampled_f<=(sampled_r**(self.d-1))*self.f_r_beta(sampled_r,R_):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        return accepted,rejected



class nsmc_sampling_rosenbock(nsmc_sampling):

    def __init__(self, d, a, k,mu,a_rosen,b_array):
        super().__init__(d, a, k)
        self.mu=mu
        self.a_rosen=a_rosen
        self.b=b_array

    def f_r_rosenbock(self,r):
        return
    
# class nsmc_sampling_x_integrand(nsmc_sampling):
#     """
#     Parameters:
#         d: dimsion of the cube
#         a: length of the cube.
#         k: required number of accepted samples
#     """
#     def __init__(self,d,a,k,alpha,beta):
#         super().__init__(d,a,k)
#
#
#     def f(self,r):
#         theta=self.theta_generation()
#         r_vec,R_=self.R(theta)
#         x_1=np.zeros(r_vec.shape)
#         x_1[0]=1
#
#
# class gaussian_mixture(nsmc_sampling):
#
#     def __init__(self,d,a,k):
#         super().__init__(d,a,k)
#         self.mu=mu
#
#     def f_r_gaussian_mixture(self):
#         """
#         """        
#
#         def f_r_gauss(theta,r):
#             r_vec, _ = self.R(theta) 
#             x_pos = r * r_vec
#             diff_1=x_pos-self.mu
#             diff_2=x_pos+self.mu
#             normalization=1/(2*((2*np.pi)**(self.d/2)))
#             expo=(np.exp(-(np.linalg.norm(diff_1)**2)/2) + np.exp(-(np.linalg.norm(diff_2)**2)/2))
#             return expo/normalization
#
#         #Mode of chi asymtotically at root(d+lambda^2) where lambda=norm(mu)
#         x_mode=np.sqrt(self.d+np.linalg.norm(self.mu)**2)
#         f_max=((x_mode)**(self.d-1))*f_r_gauss(self.theta_generation(),x_mode)
#     return f_r_gauss,f_max
#
# class laplace(nsmc_sampling):
#     def __init__(self,d,a,k):
#         super().__init__(d,a,k,mu,b)
#         self.mu=np.array(mu)
#         self.b=b
#
#     def laplace(self):
#         norm_const = 1.0 / ((2.0 * self.b) ** self.d)
#         # The peak radius for the axis direction
#         r_peak_global = self.b * (self.d - 1)
#
#         # Calculate the height at this peak
#         # Note: We assume ||u||_1 = 1 here
#         peak_density = norm_const * np.exp(-r_peak_global / self.b)
#
#         # The Max value of the profile (Jacobian included)
#         global_f_max = (r_peak_global ** (self.d - 1)) * peak_density
#         def laplace_den(r):
#             r_vec, _ = self.R(theta) 
#             x_pos = r * r_vec
#             diff=x_pos-self.mu
#             val=(1 / ((2 * self.b) ** self.d)) * np.exp(-np.linalg.norm(diff, 1) / self.b)
#
#         f_max
#         return laplace_den(r)