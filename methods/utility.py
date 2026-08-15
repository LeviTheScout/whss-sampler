import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit, prange
from scipy.special import logsumexp, ive, loggamma

from scipy.linalg import cholesky
# Import your exact 1D searcher so Numba can link them at C-level
from .importance import find_peak_golden_section 
# ==============================================================================
# PHASE 2 MATH TOOLS (High-Performance Numba & SciPy Functions)
# Placed outside the class so Numba can compile them to raw C-code.
# ==============================================================================

def log_ive(v, z):
    """
    Computes log(I_v(z)) safely using scipy.special.ive to prevent underflow crashes.
    Used by: `update_vmf_parameters` (below) to calculate the vMF normalizing constant.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Enforce a floor of 1e-300 so np.log never evaluates exactly 0.0
        res = np.log(np.maximum(ive(v, z), 1e-300)) + np.abs(z)
    return res


@njit
def sample_vmf_numba(mu, kappa, d):
    """
    Generates a single vMF ray around center `mu` with tightness `kappa`.
    Used by: `parallel_generate_and_evaluate` (below) to generate individual rays.
    """
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
    """
    Evaluates Mask 2 proposal density (q) efficiently in log-space.
    Bakes in the 10% uniform safety net to mathematically guarantee stability.
    Used by: `parallel_generate_and_evaluate` (below) to determine ray probabilities.
    """
    K = len(log_alpha)
    # K vMF components + 1 Uniform component
    log_components = np.empty(K + 1)
    
    for i in range(K):
        dot_prod = np.dot(theta, anchors_mu[i])
        log_components[i] = log_w_vmf + log_alpha[i] + log_C[i] + (kappas[i] * dot_prod)
        
    # The Uniform Safety Net Component
    log_components[K] = log_w_unif + log_C_unif
        
    # LogSumExp inline (Numba compatible)
    max_val = np.max(log_components)
    sum_exp = 0.0
    for i in range(K + 1):
        sum_exp += np.exp(log_components[i] - max_val)
        
    return max_val + np.log(sum_exp)


@njit(parallel=True)
def parallel_generate_and_evaluate(batch_size, d, parent_indices, anchors_mu, log_alpha, kappas, log_C, num_vmf, log_w_vmf, log_w_unif, log_C_unif):
    """
    Massively parallel generation of vMF/Uniform rays and evaluation of q(theta).
    Distributes the workload evenly across all CPU cores.
    Used by: `vMFProposer.generate_batch` in `proposal.py`.
    """
    theta_batch = np.empty((batch_size, d))
    log_q_batch = np.empty(batch_size)
    
    # prange forces parallel multi-threading!
    for j in prange(batch_size):
        # 1. Generate either a vMF ray or a Uniform ray based on the 90/10 split
        if j < num_vmf:
            idx = parent_indices[j]
            # It already brilliantly uses kappas[idx]!
            theta_batch[j] = sample_vmf_numba(anchors_mu[idx], kappas[idx], d)
        else:
            vec = np.random.randn(d)
            theta_batch[j] = vec / np.linalg.norm(vec)
        
        # 2. Evaluate Mask 2 density (runs concurrently on different cores)
        log_q_batch[j] = compute_log_q_mixture(
            theta_batch[j], anchors_mu, log_alpha, kappas, log_C, 
            log_w_vmf, log_w_unif, log_C_unif
        )
        
    return theta_batch, log_q_batch


def update_vmf_parameters(anchors_mass, d, kappas_array):

    """
    SciPy Helper: Turns ray masses into mixture weights, variances, and Bessel constants.
    Also calculates the exact mathematical constant for the uniform sphere.
    """
    log_mass = np.log(anchors_mass + 1e-15)
    
    # 1. Selection Probabilities (alpha)
    log_alpha = log_mass - logsumexp(log_mass)
    
    # 2. Dynamic Tightness (kappa) is now PASSED IN from Phase 1.5
    kappas = kappas_array 
    
    # 3. Log Normalizing Constants for vMF (USING THE NEW SAFETY WRAPPER)
    v = (d / 2.0) - 1.0
    # Because kappas is an array, log_C naturally vectorizes into an array!
    log_C = (v * np.log(kappas)) - ((d / 2.0) * np.log(2.0 * np.pi)) - log_ive(v, kappas)
    
    # 4. Log Normalizing Constant for the d-dimensional Uniform Sphere
    log_C_unif = loggamma(d / 2.0) - np.log(2.0) - (d / 2.0) * np.log(math.pi)
    
    return log_alpha, kappas, log_C, log_C_unif


# ==============================================================================
# EXISTING UTILITIES CLASS
# Inherited directly by the `nsmc_sampling` base class for general geometric tasks.
# ==============================================================================

class utilities:

    def theta_generation(self, batch_size):
        """
        Generates random uniform directions on the d-dimensional sphere. 
        Used by: `nsmc_sampling` legacy functions and testing.
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
