import numpy as np
from numba import njit, prange


@njit
def get_box_bounds(x_curr, v, a):
    """
    Analytically calculates exact line-intersection boundaries for a hypercube [-a/2, a/2]^d.
    This guarantees the MCMC chain NEVER proposes a point outside the Rejection Sampling bounds.
    """
    t_min = -1e20
    t_max = 1e20
    half_a = a / 2.0
    
    d = len(v)
    for i in range(d):
        if np.abs(v[i]) > 1e-14:
            t1 = (-half_a - x_curr[i]) / v[i]
            t2 = (half_a - x_curr[i]) / v[i]
            
            if t1 > t2:
                # Swap so t1 is always the minimum intersection for this axis
                temp = t1
                t1 = t2
                t2 = temp
                
            if t1 > t_min: t_min = t1
            if t2 < t_max: t_max = t2
        else:
            # If moving parallel to a wall, check if we are already outside
            if x_curr[i] < -half_a or x_curr[i] > half_a:
                return 0.0, 0.0
                
    return t_min, t_max

@njit
def slice_step_numba(density_func, x_curr, v, y_log, w, a):
    """
    1D Slice Sampler constrained analytically to the hypercube bounds.
    """
    # 1. Calculate hard geometric boundaries of the cube
    t_min_bound, t_max_bound = get_box_bounds(x_curr, v, a)
    
    # Failsafe if completely trapped
    if t_min_bound >= t_max_bound:
        return 0.0
        
    # 2. Randomly position the initial bracket
    u = np.random.uniform(0.0, 1.0)
    L = -u * w
    R = L + w
    
    # Clamp initial bracket to the exact hypercube walls
    L = max(L, t_min_bound)
    R = min(R, t_max_bound)
    
    # 3. Step-Out Phase (Restricted by Cube Bounds)
    while L > t_min_bound and density_func(x_curr + L * v) > y_log:
        L = max(L - w, t_min_bound)
    while R < t_max_bound and density_func(x_curr + R * v) > y_log:
        R = min(R + w, t_max_bound)
        
    # 4. Shrinkage Phase
    while True:
        t_cand = np.random.uniform(L, R)
        if density_func(x_curr + t_cand * v) > y_log:
            return t_cand  # Accept!
            
        # Shrink bracket
        if t_cand < 0:
            L = t_cand
        else:
            R = t_cand
            
        # Floating point failsafe
        if R - L < 1e-10:
            return 0.0
        
        
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