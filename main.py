import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_func
from scipy.integrate import quad
from scipy.optimize import brentq

class nsmc_sampling:
    """
    Parent class to generalise the nsmc_sampling. It has the general functions used for any sampling density.
    Parameters:
        d: dimsion of the cube
        a: length of the cube.
        k: required number of accepted samples
    """
    def __init__(self,d,a,k):
        self.d=d
        self.a=a
        self.k=k
    
    def theta_generation(self):
        """
        This function generates the vector of angles in d-dimsion length 
        being d-1. 
        - First (d-2) angles are uniformly sampled in [0,pi]
        - (d-1)th angle is sampeld uniformly in [0,2*pi]
        """
        sample=[]
        for i in range(self.d-2):
            sample.append(np.random.uniform(0,np.pi))
        sample.append(np.random.uniform(0,2*np.pi))
        return np.array(sample)

    def R(self,theta):
        """
        This function genertes unit vecotor r along the randomly generated
        angle theta in dimisional space. 
        Also, returns the maximum length 'R' along that direcion in the cube.
        """
        x=np.zeros(theta.shape[0]+1)
        sins=np.sin(theta)
        cosines=np.cos(theta)
        for i in range(len(theta)):
            x[i]=np.prod(sins[:i])*cosines[i]
        x[-1]=np.prod(sins)
        # is this correct? 
        inf_norm = np.max(np.abs(x))
        return x, self.a / (2 * inf_norm)
    
    def importance_r(self, g_r, R_,theta,percentage_mass=0.99):
        total_mass = percentage_mass * quad(g_r, 0, R_,args=(theta,))[0]
        def temp(guess_a):
            current, _ = quad(g_r, guess_a, R_,args=(theta,))
            return current - total_mass
        a_result = brentq(temp, a=0, b=R_) 
        return a_result, total_mass    
    
    def x_y_view(self,accepted):
        """
        this function will help us see the 2d projection of the output
        of d-dimesional sampling
        Parameters:
            accepted: accepted samples generted by get_samples() function.
        """
        x_accepted,y_accepted=[],[]
        for i in accepted:
            try: #for dimsion more than 2, only first two angles matter
                angles=i[0]
                phi=angles[0]
                theta=angles[1]
                r=i[1]
                y_accepted.append(r*np.sin(phi) * np.cos(theta))
                x_accepted.append(r*np.cos(phi))
            except: # for 2d, since we only have one angle
                angle = i[0] 
                r = i[1]                      
                x_accepted.append(r * np.cos(angle))
                y_accepted.append(r * np.sin(angle))
        # is this correct projection corrdinates?
        plt.figure(figsize=(8, 8))
        plt.plot([-(self.a/2), (self.a/2), (self.a/2), -(self.a/2), -(self.a/2)], [-(self.a/2), -(self.a/2), (self.a/2), (self.a/2), -(self.a/2)], color='black', lw=2)
        r_boundary = self.a 


        plt.scatter(x_accepted, y_accepted, color='green', s=10 )
        plt.gca().set_aspect('equal')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Projection of d-dimsional Samples onto 2D Plane")
        plt.show()
        return



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
        This function returns the d-dimsional multivariate Gaussin density which takes input 'r' a 
        length from origin and returns the gaussina at that function.
        Also, it provides 'f_max' a mode value which is to be utilised for the purpose
        of rejection sampling.
        
        Edit: Now I tried to use cholesky to tackle inverse and uses log and then exponetial to make it more numerically stable (not necessary but good addition maybe).
        """
        
        L = np.linalg.cholesky(self.sigma)
        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        log_norm_const = -0.5 * (self.d * np.log(2*np.pi) + log_det_sigma)
        def f_r_gauss(r,theta):
            r_vec, _ = self.R(theta) 
            x_pos = r * r_vec
            diff=x_pos-self.mu
            # Solve L y = diff
            y = np.linalg.solve(L, diff)

            w = np.dot(y, y)   # = diff^T Sigma^{-1} diff

            log_density = log_norm_const - 0.5 * w
            return np.exp(log_density)
       
        #Mode of chi asymtotically at root(d+lambda^2) where lambda=norm(mu)
        x_mode=np.sqrt(self.d+np.linalg.norm(self.mu)**2)
        f_max=((x_mode)**(self.d-1))*f_r_gauss(x_mode,self.theta_generation())
        return f_r_gauss,f_max



    def get_samples(self):
        """
        The main sampling function utilising the concept of n-sphere Monte
        Carlo technique and rejection sampling.
        Returns:
            Two arrays, accepted samples and rejected samples.
        """
        
        temp=self.f_r_gaussian()
        gauss_den=temp[0]
        f_max=temp[1]+0.0
        #f_max give along with f_r , also doing +0,01 in f_max, might want to remove that later
        
        accepted=[]
        rejected=[]
        while len(accepted)<self.k:

            theta=self.theta_generation()
            _,R_=self.R(theta)
            
            # a,total_mass=self.importance_r(gauss_den, R_,theta,0.95)
            # sampled_r=np.random.uniform(a,R_)
            #
            sampled_r=np.random.uniform(np.sqrt(self.d)-(2.57/np.sqrt(2)),np.sqrt(self.d)+(2.57/np.sqrt(2)))
            #sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)

            if sampled_f<=(sampled_r**((self.d)-1))*gauss_den(sampled_r,theta):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        
        return accepted,rejected


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
        if np.any(r < 0) or np.any(r > R_):
            return 0
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
                         
