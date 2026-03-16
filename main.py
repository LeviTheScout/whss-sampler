import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_func
from scipy.integrate import quad
from scipy.optimize import brentq
from sambal import random_on_cap

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
        sample=np.random.normal(0, 1, self.d)
        r = np.linalg.norm(sample)
        return sample/r
    
    def caretisan_to_spherical(self,cart):
        cart=np.array(cart)
        d=len(cart)
         
        return

    def R(self,theta):
        """
        This function genertes unit vecotor r along the randomly generated
        angle theta in dimisional space. 
        Also, returns the maximum length 'R' along that direcion in the cube.
        """
        x=np.array(theta)
        inf_norm = np.max(np.abs(x))
        return x, self.a / (2 * inf_norm)
    

    def importance_r(self, g_r, R_, theta, tol, percentage_mass=0.99):
        total_mass = quad(g_r, 0, R_, args=(theta,))[0]
        target = percentage_mass * total_mass

        a, b = 0, R_
         
        def second_search(ini, final):
            cuts = np.linspace(ini, final, 11)
            bin_areas = np.array([quad(g_r, cuts[i], cuts[i+1], args=(theta,))[0] 
                                  for i in range(10)])
            
            best_a, best_b = ini, final
            min_width = final - ini
            found = False

            # 3. Sliding Window: Check all consecutive combinations (i to j)
            for i in range(10):
                for j in range(i, 10):
                    current_window_mass = np.sum(bin_areas[i : j+1])
                    
                    # If this window captures the target
                    if current_window_mass >= target:
                        current_width = cuts[j+1] - cuts[i]
                        # If it's the narrowest window found so far, keep it
                        if current_width < min_width:
                            min_width = current_width
                            best_a, best_b = cuts[i], cuts[j+1]
                            found = True
            
            #print(current_width)
            return best_a, best_b
        while True:
            mid = (a + b) / 2
            vol_a = quad(g_r, a, mid, args=(theta,))[0]
            vol_b = quad(g_r, mid, b, args=(theta,))[0]

            if vol_a >= target:
                b = mid
            elif vol_b >= target:
                a = mid
            else:
                res_a, res_b = second_search(a, b)
                return res_a, res_b, total_mass
                #break
        return a, b, total_mass




        # ------solution to recursive approach but max iter gets hitted and gives garbage results------
        # def second_search(ini, final, max_iter=900):
        #     curr_ini, curr_final = ini, final
        #     for _ in range(max_iter):
        #         mid = (curr_ini + curr_final) / 2
        #         new_ini = (curr_ini + mid) / 2
        #         new_final = (mid + curr_final) / 2
        #
        #         new_vol = quad(g_r, new_ini, new_final, args=(theta,))[0]
        #
        #         if abs(new_vol - target)/total_mass < tol:
        #             return new_ini, new_final
        #
        #         # Update bounds for next iteration
        #         curr_ini, curr_final = new_ini, new_final
        #
        #     # If it finishes the loop without returning, it failed to converge
        #     return curr_ini, curr_final

        # -----reucurisve approach, leading to recusrion limit------
        # def second_search(ini, final):
        #     mid = (ini + final) / 2
        #     new_ini = (ini + mid) / 2
        #     new_final = (mid + final) / 2
        #     new_vol = quad(g_r, new_ini, new_final, args=(theta,))[0]
        #
        #     if abs(new_vol - target)/total_mass < tol:
        #         return new_ini, new_final
        #     else:
        #         return second_search(new_ini, new_final)
        

        # def temp(guess_a):
        #     current, _ = quad(g_r, guess_a, R_,args=(theta,))
        #     return current - total_mass
        # a_result = brentq(temp, a=0, b=R_) 
    
    
    def importance_theta(self,theta,angle=np.pi/4):
        """
        Gives random uniformly generted direction around given direction (theta) at about given angle.
        """
        #convert theta to caretisan
        #theta_cartesian=
        cartesian_direction=random_on_cap(theta_cartesian,angle)
        #convert caretisan direction to spherical direction  
        return cartesian_direction,self.a/(2*np.max(np.abs(cartesian_direction)))

    def get_samples(self,f_r,alpha=0.1):
        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        """
        def f_max_along_theta(r_vec,f_r):
            return
        
        possible_samples=[]
        maximums=[]
    
        for i in range(self.k+round(alpha*self.k)):
            theta=theta_generation(self.d)
            r_vec,R_=R(self.a,theta) 
            
            local_f_max=f_max_along_theta(r_vec,f_r)
            sampled_r=np.random.uniform(0,R_) #use importance sampling in R later
            f_value=(sampled_r**(self.d-1))*gaussian(sampled_r)
            u=np.random.uniform(0,1)
            possible_samples.append((r_vec,u,sampled_r,f_value))
            maximums.append(local_f_max)
        emperical_f_max=np.max(np.array(maximums))
        accepted=[]
        rejected=[]
        for i in range(len(possible_samples)):
            if possible_samples[i][1]*emperical_f_max<=possible_samples[i][3]:
                accepted.append((possible_samples[i][0],possible_samples[2]))
            else:
                rejected.append((possible_samples[i][0],possible_samples[2]))

        # if i am get less than k samples, then i will do normal rejection sampling with emperical f_max for remaining samples, i will deploy the importance sampling in r, hence it should not come to this case more often.
        while len(accepted)<k:
            theta=theta_generation(self.d)
            r_vec,R_=R(self.a,theta)
            sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,emperical_f_max)
            if sampled_f<=(sampled_r**(self.d-1))*gaussian(theta,sampled_r):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        return accepted,rejected
            


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
            This function returns the d-dimensional multivariate Gaussian density which takes input 'r' a 
            length from origin and returns the gaussian at that function.
            Also, it provides 'f_max' a mode value which is to be utilised for the purpose
            of rejection sampling.
            
            Edit: Now I tried to use cholesky to tackle inverse and uses log and then exponential to make it more numerically stable (not necessary but good addition maybe).
            """
            
            L = np.linalg.cholesky(self.sigma)
            log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
            log_norm_const = -0.5 * (self.d * np.log(2*np.pi) + log_det_sigma)
            
            def f_r_gauss(r, theta):
                if r <= 0:
                    return 0.0 # Prevent log(0) error
                    
                r_vec, _ = self.R(theta) 
                x_pos = r * r_vec
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
            
            a,b,total_mass=self.importance_r(gauss_den, R_,theta,0.01,0.98)
            #print(a,b,R_)
            sampled_r=np.random.uniform(a,b)
            #sampled_r=np.random.uniform(np.sqrt(self.d)-(5/np.sqrt(2)),np.sqrt(self.d)+(5/np.sqrt(2)))
            #sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)

            if sampled_f<=gauss_den(sampled_r,theta):
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
