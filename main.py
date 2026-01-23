import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_func

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


        plt.scatter(x_accepted, y_accepted, color='green', s=10, label='Accepted Samples at this Angle')
        plt.gca().set_aspect('equal')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Projection of d-dimsional Samples onto 2D Plane")
        plt.legend()
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
        def f_r_gauss(theta,r):
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
        f_max=((x_mode)**(self.d-1))*f_r_gauss(self.theta_generation(),x_mode)
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
            
            sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)

            if sampled_f<=(sampled_r**((self.d)-1))*gauss_den(theta,sampled_r):
                accepted.append((theta,sampled_r))
            rejected.append((theta,sampled_r))
        
        return accepted,rejected


class nsmc_sampling_beta(nsmc_sampling):
    """
    This class is to sample from beta distribution along each sampled theta.
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


    def f_r_beta(self,r,theta):
        
        _, R = self.R(theta) 
        if np.any(r < 0) or np.any(r > R):
            return 0
    
        # After multiplying by r^(d-1), the effective alpha is alpha + d - 1
        alpha_eff = self.alpha + self.d - 1
        
        # Normalization constant for d-dimensions on support [0, R]
        # This accounts for the r^(d-1) geometric factor
        constant = 1 / (R**self.d * beta_func(alpha_eff, self.beta))
        
        # Core kernel of the radial distribution
        density = constant * (r**(alpha_eff - 1)) * (1 - r/R)**(self.beta - 1)
        return density        
    
    def get_samples(self):
        """
        """
        accepted=[]
        rejected=[]
        x_max=(self.a*np.sqrt(self.d)/2)*(self.d+self.alpha-2)/(self.d+self.alpha+self.beta-3)
        f_max=self.f_r_beta(x_max,self.theta_generation())

        while len(accepted)<self.k:
            theta=self.theta_generation()
            _,R_=self.R(theta)
        
           

            sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)

            if sampled_f<=self.f_r_beta(sampled_r,theta):
                accepted.append((theta,sampled_r))
            rejected.append((theta,sampled_r))
        
        return accepted,rejected 
