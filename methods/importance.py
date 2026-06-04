
from math import exp
import numpy as np
from numpy.core.fromnumeric import argmax
from scipy.integrate import quad
from numba import njit,prange
from sambal import random_on_cap
from joblib import Parallel,delayed
from scipy.optimize import direct, minimize_scalar

@njit(parallel=True)
def importance_r_numba(log_g_r, R_batch, theta_batch,grid_size=100000, percentage_mass=0.99):
    '''
    This finds interval [a,b] for given directions thetas such that this smaller region
    has percentage_mass*total area under the curve of g_r function from [0,R_] support.
    '''
    batch_size=len(R_batch)
    a_vals,b_vals,log_f_max_batch,total_mass_batch=np.zeros(batch_size),np.zeros(batch_size),np.zeros(batch_size),np.zeros(batch_size)
    
    for i in prange(batch_size):
        R_,theta=R_batch[i],theta_batch[i]
        r_grid= np.linspace(0,R_,grid_size)
        log_g_r_grid=log_g_r(r_grid,theta) # will have to fix this function.
        dr=R_/(grid_size-1)  #no_gaps = points-1

        arg_max=np.argmax(log_g_r_grid)
        local_log_f_max=log_g_r_grid[arg_max]
        
        prob_dens=np.exp(log_g_r_grid - local_log_f_max) #subtracting max makes sure peak sits at 0 --exp(0)=1, eveything else less than 1.
        cdf=np.cumsum(prob_dens)*dr  #area=length*bredth , making it sum to 1 by multiplying it with dr.

        log_total_mass=np.log(cdf[-1]+1e-100)+local_log_f_max # we subtracted f_max, getting it back in log space.
        target_mass=percentage_mass*cdf[-1]

        a,b=0.0,R_
        min_width=R_
        left=0

        for right in range(grid_size):
            while cdf[right]-cdf[left]>=target_mass:
                current_width=r_grid[right]-r_grid[left]
                if current_width<=min_width:
                    min_width=current_width
                    a,b=r_grid[left],r_grid[right]
                left+=1
        a_vals[i]=a
        b_vals[i]=b
        log_f_max_batch[i]=local_log_f_max
        total_mass_batch[i]=np.exp(log_total_mass)
        
    return a_vals,b_vals,log_f_max_batch,total_mass_batch

class importance_sampling:


    def away_thetas_batch(self,theta_batch,mass_batch,thresh_angle,tau,batch,orthants_batch=None,global_mean=None):
        """
        Returns directions from theta_batch that are aprat enough and are top-m based on the mass.
        maybe can be made such that takes threshold angle as param. and 
        check that across whole batch and decides which to keep and which not to.
        theta_batch: [[theta1],[theta2].....]
        mass_batch: [m1,m2,.....]
        tau: minimum fraction of m1 (maximum mass) each direction need to have.
        thresh_angle: decides the threshold of how big angle between two selected directions should be.
        """

        sorted_mass_indices=np.argsort(mass_batch)[::-1]
        if batch:
            orthant_ids=self.get_orthant(theta_batch)
            sorted_orthant_ids=orthant_ids[sorted_mass_indices]
            _, unique_sorted_orthant_indices=np.unique(sorted_orthant_ids, return_index=True,axis=0)
            sorted_orthants=orthant_ids[unique_sorted_orthant_indices]
            masses_new=mass_batch[unique_sorted_orthant_indices]
            thetas_new=theta_batch[unique_sorted_orthant_indices]
            #makes sure that higher mass is kept when there is clash of two directions in same orthant.
            return sorted_orthants,thetas_new,masses_new
        else:
            # input: top masees batch wise for orthants.
            # select top mass based on orthant. and assign it as mass of that orthatnt.
            # return: all orthatns in desceneding order of mass along with its maximum mass.
            orderd=orthants_batch[sorted_mass_indices]
            _,idx=np.unique(orderd,return_index=True)
            idx=np.sort(idx)
            orthants_descending=orderd[idx]
            masses_descending=mass_batch[sorted_mass_indices][idx]
            theta_descending=theta_batch[sorted_mass_indices][idx]
            print(np.unpackbits(orthants_descending,axis=1),masses_descending)
            # print(mass_batch[sorted_mass_indices])
            return orthants_descending,theta_descending,masses_descending
         
    
    def importance_r(self,density,R_batch,theta_batch):
        return importance_r_numba(density,R_batch,theta_batch)

        
    #
    #
    # def importance_r(self,log_g_r, R_batch, theta_batch, percentage_mass=0.99):
    #     '''
    #     This finds interval [a,b] for given directions thetas such that this smaller region
    #     has percentage_mass*total area under the curve of g_r function from [0,R_] support.
    #     '''
    #
    #     # had to do this since, shifted to log output in the original density functions.
    #     def process_single(R_,theta):
    #         def g_r(r,theta):
    #             return np.exp(log_g_r(r,theta))
    #         total_mass = quad(g_r, 0, R_, args=(theta,))[0]
    #
    #         target = percentage_mass * total_mass
    #
    #         a, b = 0, R_
    #
    #         def local_f_max(R_i, theta_i):
    #             # DIRECT coarsely - just find the right basin
    #             res_coarse = direct(
    #                 lambda r: -g_r(r[0], theta_i),
    #                 bounds=[(1e-10, R_i)],
    #                 eps=1e-2,          # coarse - just find the peak region
    #                 maxiter=200,       # limit evaluations hard
    #                 locally_biased=True  # faster, accepts some risk
    #             )
    #             peak_loc = res_coarse.x[0]
    #
    #             # Brent precisely within a tight window around DIRECT's answer
    #             lo = max(1e-10, peak_loc - 0.05 * R_i)
    #             hi = min(R_i, peak_loc + 0.05 * R_i)
    #             res_fine = minimize_scalar(
    #                 lambda r: -g_r(r, theta_i),
    #                 bounds=(lo, hi),
    #                 method='bounded'
    #             )
    #             return -res_fine.fun
    #         f_max_theta=local_f_max(R_,theta)
    #         # average_val=total_mass/R_
    #         #
    #         # r_points=np.linspace(0,R_,300)
    #         # values=g_r(r_points,theta.copy_300) #figure out how to do this.
    #         # mask=values>=average_val
    #         # r_masked=r_points[mask]
    #         # # check if there is discontinutiy (maybe)
    #         # min_r,max_r=r_masked[0],r_masked[-1]
    #         # if quad(g_r,g_r,min_r,max_r,args=(theta,))< percentage_mass*total_mass):
    #             # increse from both sides to accompany percentage_mass 
    #         #shorten if more than percentage_mass if possible
    #
    #         def second_search(ini, final):
    #             cuts = np.linspace(ini, final, 11)
    #             bin_areas = np.array([quad(g_r, cuts[i], cuts[i+1], args=(theta,))[0] 
    #                                   for i in range(10)])
    #
    #             best_a, best_b = ini, final
    #             min_width = final - ini
    #             found = False
    #
    #             # 3. Sliding Window: Check all consecutive combinations (i to j)
    #             for i in range(10):
    #                 for j in range(i, 10):
    #                     current_window_mass = np.sum(bin_areas[i : j+1])
    #
    #                     # If this window captures the target
    #                     if current_window_mass >= target:
    #                         current_width = cuts[j+1] - cuts[i]
    #                         # If it's the narrowest window found so far, keep it
    #                         if current_width < min_width:
    #                             min_width = current_width
    #                             best_a, best_b = cuts[i], cuts[j+1]
    #                             found = True
    #
    #             #print(current_width)
    #             return best_a, best_b
    #
    #         while True:
    #             mid = (a + b) / 2
    #             vol_a = quad(g_r, a, mid, args=(theta,))[0]
    #             vol_b = quad(g_r, mid, b, args=(theta,))[0]
    #
    #             if vol_a >= target:
    #                 b = mid
    #             elif vol_b >= target:
    #                 a = mid
    #             else:
    #                 res_a, res_b = second_search(a, b)
    #                 return res_a, res_b, f_max_theta,total_mass
    #                 #break
    #         return a, b,f_max_theta, total_mass
    #
    #     results=Parallel(n_jobs=-1)(
    #             delayed(process_single)(R_batch[i],theta_batch[i])
    #             for i in range(len(R_batch)) )       
    #     a_vals,b_vals,f_max_batch,total_mass_batch=zip(*results)
    #
    #     # Looping approach
    #     # a_vals,b_vals,total_masses=[],[],[]
    #     # for i in range(len(R_batch)):
    #     #     a_temp,b_temp,mass_temp=process_single(R_batch[i],theta_batch[i])
    #     #     a_vals.append(a_temp)
    #     #     b_vals.append(b_temp)
    #     #     total_masses.append(mass_temp)
    #
    #     return np.array(a_vals),np.array(b_vals),np.array(f_max_batch),np.array(total_mass_batch)



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
    
    

