import numpy as np
import math
from numba import njit, prange
from scipy.special import ive, loggamma
from .importance import find_peak_golden_section

# =====================================================================
# 1. THE HYBRID RAY GENERATOR
# =====================================================================
@njit
def generate_hybrid_ray_numba(d, L, anchors, num_anchors):
    """
    Scale-Aware Hybrid Proposer. 
    Guarantees no raw-coordinate proposals that crash into diagonal walls.
    """
    u = np.random.uniform(0.0, 1.0)
    
    if u < 0.33:
        # SKELETON MOVE: Slide along Phase 2 anchors
        idx1 = np.random.randint(0, num_anchors)
        idx2 = np.random.randint(0, num_anchors)
        while idx1 == idx2:
            idx2 = np.random.randint(0, num_anchors)
        v = anchors[idx1] - anchors[idx2]
        
    elif u < 0.66:
        # WARPED MOVE: Global stretch
        z = np.random.randn(d)
        v = np.dot(L, z)
        
    else:
        # PURE COORDINATE MOVE: Strict axis-aligned step.
        # This is strictly required to slide out of microscopic axis-aligned traps!
        v = np.zeros(d)
        axis = np.random.randint(0, d)
        
        # Scale the 1D slice bracket using the diagonal of L
        scale = np.abs(L[axis, axis]) + 1e-6
        v[axis] = scale if np.random.uniform(0.0, 1.0) > 0.5 else -scale
        
    if np.linalg.norm(v) < 1e-15:
        v[0] = 1.0
        
    return v

# =====================================================================
# 2. PARALLEL SCOUT EVALUATOR
# =====================================================================
@njit(parallel=True)
def parallel_scout_eval(rays, R_max_array, target_func):
    """
    Massively parallel 1D peak finder for the Phase 1.5 Scout.
    Evaluates thousands of bounded rays simultaneously using all CPU cores.
    """
    N, d = rays.shape
    x_coords = np.empty((N, d))
    peaks = np.empty(N)
    
    for i in prange(N):
        direction = rays[i]
        R_max = R_max_array[i]
        
        # Numba executes this directly in C
        r_max = find_peak_golden_section(target_func, direction, R_max)
        peak_val = target_func(r_max, direction)
        
        x_coords[i] = r_max * direction
        peaks[i] = peak_val
        
    return x_coords, peaks

# =====================================================================
# 3. THE L-MATRIX BUILDER
# =====================================================================
def build_warp_matrix(x_coords, d):
    """
    Statistically stable Unweighted Covariance from pre-calculated Cartesian peaks.
    """
    print("\n[WARP ENGINE] Booting Phase 1.5: Calculating global space transformation...")
    
    N = len(x_coords)
    mu = np.mean(x_coords, axis=0)
    centered = x_coords - mu
    cov = (centered.T @ centered) / (N - 1)
    
    reg = 1e-3 * np.eye(d)
    cov += reg
    
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        
    L_inv = np.linalg.inv(L)
    
    cond = np.linalg.cond(L)
    eigenvalues = np.linalg.eigvalsh(cov)
    
    print("\n=== [L MATRIX DIAGNOSTIC] ===")
    print(f"Condition Number : {cond:.2f}")
    print(f"Max Eigenvalue   : {np.max(eigenvalues):.4f}")
    print(f"Min Eigenvalue   : {np.min(eigenvalues):.4f}")
    print("=============================\n")
    
    return L, L_inv

# =====================================================================
# 4. PHASE 2 MATH TOOLS (vMF Distribution)
# =====================================================================
def log_ive(v, z):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(np.maximum(ive(v, z), 1e-300)) + np.abs(z)
    return res

@njit
def sample_vmf_numba(mu, kappa, d):
    b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (d - 1)**2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    
    while True:
        z = np.random.beta((d - 1) / 2.0, (d - 1) / 2.0)
        W = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = np.random.uniform(0.0, 1.0)
        if kappa * W + (d - 1) * np.log(1 - x0 * W) - c >= np.log(u):
            break
            
    v = np.random.randn(d - 1)
    v /= np.linalg.norm(v)
    
    ray = np.empty(d)
    ray[0] = W
    ray[1:] = np.sqrt(1 - W**2) * v
    
    e1 = np.zeros(d)
    e1[0] = 1.0
    u_h = e1 - mu
    u_norm_sq = np.sum(u_h**2)
    
    if u_norm_sq > 1e-10:
        ray = ray - 2 * (np.dot(u_h, ray) / u_norm_sq) * u_h
        
    return ray

@njit
def compute_log_q_mixture(theta, anchors_mu, log_alpha, kappas, log_C, log_w_vmf, log_w_unif, log_C_unif):
    K = len(log_alpha)
    log_components = np.empty(K + 1)
    
    for i in range(K):
        dot_prod = np.dot(theta, anchors_mu[i])
        log_components[i] = log_w_vmf + log_alpha[i] + log_C[i] + (kappas[i] * dot_prod)
        
    log_components[K] = log_w_unif + log_C_unif
        
    max_val = np.max(log_components)
    sum_exp = 0.0
    for i in range(K + 1):
        sum_exp += np.exp(log_components[i] - max_val)
        
    return max_val + np.log(sum_exp)

@njit(parallel=True)
def parallel_generate_and_evaluate(batch_size, d, parent_indices, anchors_mu, log_alpha, kappas, log_C, num_vmf, log_w_vmf, log_w_unif, log_C_unif):
    theta_batch = np.empty((batch_size, d))
    log_q_batch = np.empty(batch_size)
    
    for j in prange(batch_size):
        if j < num_vmf:
            idx = parent_indices[j]
            theta_batch[j] = sample_vmf_numba(anchors_mu[idx], kappas[idx], d)
        else:
            vec = np.random.randn(d)
            theta_batch[j] = vec / np.linalg.norm(vec)
        
        log_q_batch[j] = compute_log_q_mixture(
            theta_batch[j], anchors_mu, log_alpha, kappas, log_C, 
            log_w_vmf, log_w_unif, log_C_unif
        )
        
    return theta_batch, log_q_batch

def update_vmf_parameters(anchors_mass, d, kappas_array):
    K = len(anchors_mass)
    log_alpha = np.full(K, -np.log(float(K)))
    kappas = kappas_array 
    v = (d / 2.0) - 1.0
    log_C = (v * np.log(kappas)) - ((d / 2.0) * np.log(2.0 * np.pi)) - log_ive(v, kappas)
    log_C_unif = loggamma(d / 2.0) - np.log(2.0) - (d / 2.0) * np.log(math.pi)
    
    return log_alpha, kappas, log_C, log_C_unif

class utilities:

    def theta_generation(self, batch_size):
        """
        Generates random uniform directions on the d-dimensional sphere. 
        Used by: `whss_sampling` legacy functions and testing.
        """
        samples = np.random.normal(0, 1, (batch_size, self.d))
        r = np.linalg.norm(samples, axis=1)
        return samples / r[:, None]
    
    def orthant_theta_generator(self, orthant_id, batch_size):
        """
        Generates random uniform directions, changing signs to match specific orthants exactly.
        Used by: Legacy manual orthant sampling (now largely superseded by proposal.py).
        """
        batch_size = np.ceil(batch_size).astype(int)
        original_orthant_id = np.unpackbits(orthant_id)
        theta_batch = self.theta_generation(batch_size)
        target_bits = original_orthant_id[:self.d]
        target_bits = target_bits.astype(int)
        target_signs = (target_bits * 2) - 1 
        orthant_thetas = np.abs(theta_batch) * target_signs
        return orthant_thetas 

    def R(self, thetas):
        """
        Returns the maximum length 'R' along given ray directions for a bounding cube of side length a.
        Used by: `sampling.py` universal loop to bound the 1D search space.
        """
        inf_norm = np.max(np.abs(thetas), axis=1)
        R_vec = self.a / (2 * inf_norm)
        return R_vec

    def get_orthant(self, thetas):
        """
        Extracts the bit-packed integer ID of the orthant for given rays.
        Used by: `utilities` for debugging, or orthant tracker mechanisms.
        """
        mask = thetas >= 0
        orthant_ids = np.packbits(mask, axis=1)
        return orthant_ids
        
    def x_y_view(self, accepted):
        """
        Visualizes the 2D projection of samples generated in d-dimensional Cartesian directional form.
        Used by: The end user via Jupyter Notebooks or run scripts to verify sample distribution.
        """
        if not accepted:
            print("No accepted samples to plot.")
            return

        directions = np.array([item[0] for item in accepted]) 
        radii = np.array([item[1] for item in accepted])

        a_2d = directions[:, :2] 
        coordinates = radii[:, None] * a_2d 
        x_accepted = coordinates[:, 0]
        y_accepted = coordinates[:, 1]

        plt.figure(figsize=(8, 8))
        plt.plot([-(self.a/2), (self.a/2), (self.a/2), -(self.a/2), -(self.a/2)], 
                [-(self.a/2), -(self.a/2), (self.a/2), (self.a/2), -(self.a/2)], 
                color='black', lw=2)

        plt.scatter(x_accepted, y_accepted, color='green', s=10)
        
        plt.gca().set_aspect('equal')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Projection of d-dimensional Samples onto 2D Plane")
        
        plt.show()

    def acc_reject_view(self, accepted, rejected):
        """
        Visualizes the 2D projection of accepted and rejected samples 
        generated in d-dimensional Cartesian directional form side-by-side.
        Used by: The end user via Jupyter Notebooks or run scripts to verify 
        sample distribution and analyze rejection patterns.
        """
        
        # Helper function to extract 2D coordinates efficiently
        def get_2d_coordinates(samples):
            if not samples:
                return [], []
            directions = np.array([item[0] for item in samples]) 
            radii = np.array([item[1] for item in samples])
            
            # Slice first two dimensions and multiply by radius
            a_2d = directions[:, :2] 
            coordinates = radii[:, None] * a_2d 
            return coordinates[:, 0], coordinates[:, 1]

        # Extract coordinates for both sets
        x_acc, y_acc = get_2d_coordinates(accepted)
        x_rej, y_rej = get_2d_coordinates(rejected)

        # Create a figure with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bounding box coordinates using the class attribute 'self.a'
        box_x = [-(self.a/2), (self.a/2), (self.a/2), -(self.a/2), -(self.a/2)]
        box_y = [-(self.a/2), -(self.a/2), (self.a/2), (self.a/2), -(self.a/2)]

        # ------------------ Plot 1: Accepted ------------------
        ax1.plot(box_x, box_y, color='black', lw=2)
        if len(x_acc) > 0:
            ax1.scatter(x_acc, y_acc, color='green', s=10, alpha=0.7)
        
        ax1.set_aspect('equal')
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_title(f"Accepted Samples (N={len(accepted)})")

        # ------------------ Plot 2: Rejected ------------------
        ax2.plot(box_x, box_y, color='black', lw=2)
        if len(x_rej) > 0:
            ax2.scatter(x_rej, y_rej, color='red', s=10, alpha=0.5)
            
        ax2.set_aspect('equal')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.set_title(f"Rejected Samples (N={len(rejected)})")

        # Safely determine dimension 'd' from the direction array
        if accepted:
            d = len(accepted[0][0])
        elif rejected:
            d = len(rejected[0][0])
        else:
            d = "Unknown"

        # Display the plots
        plt.suptitle(f"Projection of d={d} Samples onto 2D Plane", fontsize=14, y=0.95)
        plt.show()
