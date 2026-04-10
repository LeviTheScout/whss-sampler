
import numpy as np
from scipy.integrate import quad
from sambal import random_on_cap
from joblib import Parallel,delayed
from scipy.optimize import direct, minimize_scalar


class importance_sampling:


    def importance_theta(self,theta,angle_importance):
        """
        Gives random uniformly generted direction around given direction (theta) at about given angle.
        """
        
        cartesian_direction=random_on_cap(theta,angle_importance)
        return cartesian_direction

    def away_thetas_batch(self,theta_batch,mass_batch,thresh_angle,tau,batch,global_mean=None):
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
        cos_thres=np.cos(thresh_angle)
        if batch:
            # will do matrix multiplication based check here since we have no involvment of mass anymore.
            cosine_matrix=theta_batch@theta_batch.T
            mask = cosine_matrix > cos_thres # ones signify that those angles are close.
            #discard one of those where we have 1 based on whichever has less mass.
            np.fill_diagonal(mask,False)
            discard_idx=set() 
            for i in sorted_mass_indices:
                if i in discard_idx:
                    continue

                close_to_i=np.where(mask[i])[0]
                discard_idx.update(close_to_i)
                #because i is heaviest surviving theta,all the neighbours must be lighter.

            discard_idx=list(discard_idx)
            far_apart_directions=np.delete(theta_batch,discard_idx,axis=0)
            far_apart_masses=np.delete(mass_batch,discard_idx,axis=0)
            return far_apart_directions,far_apart_masses
        else:
            sorted_mass_indices=np.argsort(mass_batch)[::-1]
            selected_directions_indices=[]
            j=0
            m1=mass_batch[sorted_mass_indices[0]] 
            # change m1---> m1/avg. problem
            # print(np.sort(mass_batch))
            while j < len(sorted_mass_indices) and mass_batch[sorted_mass_indices[j]] >= tau * (m1/global_mean):
                # either I have enough directions or I stop if i dont have enough far aprat directions.
                for k in selected_directions_indices:
                    if (theta_batch[k] @ theta_batch[sorted_mass_indices[j]]) > cos_thres:
                        j+=1
                        break
                else:
                    selected_directions_indices.append(sorted_mass_indices[j])
                    j+=1

                #print(mass_batch[sorted_mass_indices[j]],tau*(m1/global_mean)) 
                # print(theta_batch[sorted_mass_indices[j]])
            print(len(selected_directions_indices))
            print(theta_batch[selected_directions_indices])
            return theta_batch[selected_directions_indices],mass_batch[selected_directions_indices]


    def importance_r(self, g_r, R_batch, theta_batch, percentage_mass=0.99):
        '''
        This finds interval [a,b] for given directions thetas such that this smaller region
        has percentage_mass*total area under the curve of g_r function from [0,R_] support.
        '''
        def process_single(R_,theta):
            total_mass = quad(g_r, 0, R_, args=(theta,))[0]
            target = percentage_mass * total_mass

            a, b = 0, R_

            def local_f_max(R_i, theta_i):
                # DIRECT coarsely - just find the right basin
                res_coarse = direct(
                    lambda r: -g_r(r[0], theta_i),
                    bounds=[(1e-10, R_i)],
                    eps=1e-2,          # coarse - just find the peak region
                    maxiter=200,       # limit evaluations hard
                    locally_biased=True  # faster, accepts some risk
                )
                peak_loc = res_coarse.x[0]
                
                # Brent precisely within a tight window around DIRECT's answer
                lo = max(1e-10, peak_loc - 0.05 * R_i)
                hi = min(R_i, peak_loc + 0.05 * R_i)
                res_fine = minimize_scalar(
                    lambda r: -g_r(r, theta_i),
                    bounds=(lo, hi),
                    method='bounded'
                )
                return -res_fine.fun
            f_max_theta=local_f_max(R_,theta)

             
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
                    return res_a, res_b, f_max_theta,total_mass
                    #break
            return a, b,f_max_theta, total_mass


        results=Parallel(n_jobs=-1)(
                delayed(process_single)(R_batch[i],theta_batch[i])
                for i in range(len(R_batch)) )       

        a_vals,b_vals,f_max_batch,total_mass_batch=zip(*results)

        # Looping approach
        # a_vals,b_vals,total_masses=[],[],[]
        # for i in range(len(R_batch)):
        #     a_temp,b_temp,mass_temp=process_single(R_batch[i],theta_batch[i])
        #     a_vals.append(a_temp)
        #     b_vals.append(b_temp)
        #     total_masses.append(mass_temp)

        return np.array(a_vals),np.array(b_vals),np.array(f_max_batch),np.array(total_mass_batch)



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
    
    

