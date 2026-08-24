import numpy as np
from numba import njit, prange

# =====================================================================
# TOOL 1: PHASE 2 WARM-UP (THE SCOUT)
# Used by utility.py to find peaks and build the L-Matrix
# =====================================================================
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
# TOOL 2: PHASE 3 MCMC ENGINE (THE SLICE WALKER)
# Replaces importance_r_numba and the Rejection logic
# =====================================================================
@njit
def slice_step_numba(target_log_density, x_curr, v, y_log, w=1.0, max_steps=1000):
    """
    Executes a mathematically reversible 1D Slice Sampling step along direction 'v'.
    Implements Neal (2003) "Step-out" and "Shrinkage" procedures.
    """
    # 1. RANDOM BRACKET PLACEMENT (Required for Detailed Balance)
    u = np.random.uniform(0.0, 1.0)
    t_min = -w * u
    t_max = t_min + w
    
    # 2. THE STEP-OUT PROCEDURE
    step_count = 0
    while step_count < max_steps:
        x_left = x_curr + t_min * v
        if target_log_density(x_left) <= y_log:
            break
        t_min -= w
        step_count += 1
        
    step_count = 0
    while step_count < max_steps:
        x_right = x_curr + t_max * v
        if target_log_density(x_right) <= y_log:
            break
        t_max += w
        step_count += 1

    # 3. THE SHRINKAGE PROCEDURE (Propose & Accept)
    while True:
        t_prop = np.random.uniform(t_min, t_max)
        x_prop = x_curr + t_prop * v
        
        if target_log_density(x_prop) > y_log:
            return t_prop  # Accepted!
            
        if t_prop < 0:
            t_min = t_prop
        else:
            t_max = t_prop