import numpy as np
from numba import njit

# =====================================================================
# TOOL 1: PHASE 2 WARM-UP (THE SCOUT)
# =====================================================================
@njit
def find_peak_golden_section(user_log_g_r, theta, R_max, tol=1e-6):
    """
    Finds the exact r that maximizes log_g_r along a specific theta ray.
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

@njit
def slice_step_polytope(density_func, x_curr, v, y_log, w, A, b):
    """
    1D Slice Sampler perfectly constrained by analytic polytope boundaries.
    """
    # 1. Calculate absolute geometric boundaries of the polytope
    t_bound_min, t_bound_max = get_polytope_bounds(A, b, x_curr, v)
    
    if t_bound_min >= t_bound_max:
        return 0.0
        
    # 2. Randomly position the initial bracket and CLAMP to walls
    u = np.random.uniform(0.0, 1.0)
    L = -u * w
    R = L + w
    
    L = max(L, t_bound_min)
    R = min(R, t_bound_max)
    
    # 3. Step-Out Phase (Restricted by Polytope Bounds)
    while L > t_bound_min and density_func(x_curr + L * v) > y_log:
        L = max(L - w, t_bound_min)
    while R < t_bound_max and density_func(x_curr + R * v) > y_log:
        R = min(R + w, t_bound_max)
        
    # 4. Shrinkage Phase
    while True:
        t_cand = np.random.uniform(L, R)
        if density_func(x_curr + t_cand * v) > y_log:
            return t_cand  # Accept!
            
        if t_cand < 0:
            L = t_cand
        else:
            R = t_cand
            
        if R - L < 1e-10:
            return 0.0
            


@njit
def get_polytope_bounds(A, b, x_curr, v):
    """
    Analytically calculates exact line-intersection boundaries for Ax <= b.
    Includes floating-point forgiveness for boundary-hugging chains.
    """
    t_min = -1e20
    t_max = 1e20
    
    m = A.shape[0]
    for i in range(m):
        denom = np.dot(A[i], v)
        slack = b[i] - np.dot(A[i], x_curr)
        
        # Floating point failsafe: If slightly outside due to rounding, push to wall
        if -1e-9 < slack < 0:
            slack = 0.0
            
        if denom > 1e-14:
            t = slack / denom
            if t < t_max: 
                t_max = t
        elif denom < -1e-14:
            t = slack / denom
            if t > t_min: 
                t_min = t
        else:
            if slack < 0:
                return 0.0, 0.0
                
    return t_min, t_max