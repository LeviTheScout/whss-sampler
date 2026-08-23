from math import exp
import numpy as np
from numpy.core.fromnumeric import argmax
from numba import njit, prange
# (Removed joblib, scipy.integrate, etc., as Numba handles all parallelization/math natively now)

@njit
def find_peak_golden_section(user_log_g_r, theta, R_max, tol=1e-6):
    """
    Finds the exact r that maximizes log_g_r along a specific theta ray.
    Uses Golden-Section search for O(log N) speed and zero memory allocation.
    """
    invphi = (np.sqrt(5) - 1) / 2  
    invphi2 = (3 - np.sqrt(5)) / 2 

    a, b = 1e-10, R_max
    h = b - a
    if h <= tol: 
        return (a + b) / 2.0

    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = user_log_g_r(c, theta)
    yd = user_log_g_r(d, theta)

    for _ in range(n):
        if yc >= yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = user_log_g_r(c, theta)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = user_log_g_r(d, theta)

    return (a + b) / 2.0


# =====================================================================
# 2. THE MAIN ENGINE (Parallelized, Memory-Safe, High-Precision)
# =====================================================================

@njit(parallel=True)
def importance_r_numba(user_log_g_r, R_batch, theta_batch, grid_size_fine=5000, percentage_mass=0.99):
    batch_size = len(R_batch)
    a_vals = np.zeros(batch_size)
    b_vals = np.zeros(batch_size)
    log_f_max_batch = np.zeros(batch_size)
    
    # CRITICAL CHANGE: Returning Log Mass to prevent float64 overflow in high dimensions
    log_total_mass_batch = np.zeros(batch_size) 

    eps = 1e-10
    drop_threshold = 20.0

    for i in prange(batch_size):
        R_ = R_batch[i]
        theta = theta_batch[i]

        # --- PHASE 1: Pinpoint the Peak ---
        exact_peak_r = find_peak_golden_section(user_log_g_r, theta, R_)
        peak_log_val = user_log_g_r(exact_peak_r, theta)

        # --- PHASE 2: Dynamic Bounding (Bisection) ---
        # 2a. Left boundary
        if exact_peak_r <= eps or user_log_g_r(eps, theta) >= peak_log_val - drop_threshold:
            r_start = eps
        else:
            low, high = eps, exact_peak_r
            for _ in range(30):
                mid = (low + high) / 2.0
                if user_log_g_r(mid, theta) >= peak_log_val - drop_threshold:
                    high = mid
                else:
                    low = mid
            r_start = low

        # 2b. Right boundary
        if exact_peak_r >= R_ or user_log_g_r(R_, theta) >= peak_log_val - drop_threshold:
            r_end = R_
        else:
            low, high = exact_peak_r, R_
            for _ in range(30):
                mid = (low + high) / 2.0
                if user_log_g_r(mid, theta) >= peak_log_val - drop_threshold:
                    low = mid
                else:
                    high = mid
            r_end = high

# --- PHASE 3: Integration & CDF ---
        fine_grid = np.linspace(r_start, r_end, grid_size_fine)
        
        # REVERTED: Pass the entire grid at once to utilize the user's vectorized Numba function!
        log_g_r_grid = user_log_g_r(fine_grid, theta)
            
        local_log_f_max = np.max(log_g_r_grid)
        dr = (r_end - r_start) / (grid_size_fine - 1)
        cdf = np.empty(grid_size_fine)
        current_sum = 0.0
        
        for j in range(grid_size_fine):
            current_sum += np.exp(log_g_r_grid[j] - local_log_f_max) * dr
            cdf[j] = current_sum

        log_total_mass = np.log(current_sum + 1e-100) + local_log_f_max
        target_mass = percentage_mass * current_sum

# --- PHASE 4: The Sliding Window ---
        a, b = r_start, r_end
        min_width = r_end - r_start
        left = 0
        for right in range(grid_size_fine):
            while cdf[right] - cdf[left] >= target_mass:
                current_width = fine_grid[right] - fine_grid[left]
                if current_width <= min_width:
                    min_width = current_width
                    a, b = fine_grid[left], fine_grid[right]
                left += 1

        a_vals[i] = a
        b_vals[i] = b
        log_f_max_batch[i] = local_log_f_max
        
        # ==========================================
        # CRITICAL FIX: Use Box Mass for Rejection Math
        # ==========================================
        log_total_mass_batch[i] = local_log_f_max + np.log(b - a + 1e-100)

    return a_vals, b_vals, log_f_max_batch, log_total_mass_batch


class importance_sampling:

    def away_thetas_batch(self, theta_batch, weights, tau, batch, orthants_batch=None):
        """
        (Retained for backwards compatibility with pure Orthant sampling).
        """
        sorted_weight_indices=np.argsort(weights)[::-1]
        if batch:
            # Assuming get_orthant is imported or accessible from utilities
            # Note: For the new Universal architecture, we use Cosine Filtering instead.
            from utility import utilities
            utils = utilities()
            utils.d = theta_batch.shape[1]
            orthant_ids = utils.get_orthant(theta_batch)
            
            sorted_orthant_ids=orthant_ids[sorted_weight_indices]
            _, first_ocurrances=np.unique(sorted_orthant_ids, return_index=True,axis=0)
            first_ocurrances=np.sort(first_ocurrances)
            selected_indices=sorted_weight_indices[first_ocurrances]
            sorted_orthants=orthant_ids[selected_indices]
            weights_new=weights[selected_indices]
            thetas_new=theta_batch[selected_indices]
            return sorted_orthants,thetas_new,weights_new
        else:
            orderd=orthants_batch[sorted_weight_indices]
            _,idx=np.unique(orderd,return_index=True,axis=0)
            idx=np.sort(idx)
            orthants_descending=orderd[idx]
            weights_descending=weights[sorted_weight_indices][idx]
            theta_descending=theta_batch[sorted_weight_indices][idx]
            return orthants_descending,theta_descending,weights_descending
         
    def importance_r(self, density, R_batch, theta_batch):
        # We now expect log_mass_batch returned!
        return importance_r_numba(density, R_batch, theta_batch)
