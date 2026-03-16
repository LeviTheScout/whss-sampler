
import numpy as np
from scipy.integrate import quad
from sambal import random_on_cap
#from scipy.optimize import brentq


class importance_sampling:

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
        cartesian_direction=random_on_cap(self.theta_cartesian,angle)
        #convert caretisan direction to spherical direction  
        return cartesian_direction,self.a/(2*np.max(np.abs(cartesian_direction)))