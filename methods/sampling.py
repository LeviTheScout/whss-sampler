import numpy as np
from tqdm import tqdm
import time
from numba import njit
from .utility import build_warp_matrix, generate_hybrid_ray_numba
from .importance import find_peak_golden_section, slice_step_numba

class sampling:
    def _sampling_universal(self, density_cartesian, batch_size=3256, burn_in_samples=10000, max_anchors=50):
        
        # =====================================================================
        # 1. AUTOMATIC SPHERICAL WRAPPER (User only writes Cartesian!)
        # =====================================================================
        d = self.d
        @njit
        def density_spherical_single(r, theta):
            x_cartesian = r * theta
            # Jacobian is mathematically required for Phase 1 & 2 Ray-Casting
            jacobian = (d - 1.0) * np.log(r + 1e-15)
            return density_cartesian(x_cartesian) + jacobian
            
        @njit
        def density_spherical_batch(r_batch, theta_batch):
            out = np.empty(len(r_batch))
            for i in range(len(r_batch)):
                out[i] = density_spherical_single(r_batch[i], theta_batch[i])
            return out
        # =====================================================================

        # =====================================================================
        # 2. PHASE 1 & 2: THE WARM-UP (Ray Casting & L-Matrix)
        # =====================================================================
        from .proposal import PhaseManager
        proposer = PhaseManager(
            d=self.d, burn_in_samples=burn_in_samples, 
            max_anchors=max_anchors, exploration_batches=20
        )
        
        t_main = time.perf_counter()
        
        print("\n=== STARTING WHSS (Warped Hybrid Slice Sampler) ===")
        print("[STAGE 1] Spherical Warm-Up & Preconditioning...")
        
        # Keep proposing rays until PhaseManager hits Phase 3
        while proposer.phase < 3:
            theta_batch, log_q_batch = proposer.generate_batch(batch_size)
            R_batch = self.R(theta_batch)  
            
            # Find peaks using Golden Section (we do this in a fast loop)
            log_mass_batch = np.empty(batch_size)
            for i in range(batch_size):
                peak_r = find_peak_golden_section(density_spherical_single, theta_batch[i], R_batch[i])
                log_mass_batch[i] = density_spherical_single(peak_r, theta_batch[i])
                
            proposer.update_knowledge(theta_batch, log_mass_batch, log_q_batch)
            proposer.register_acceptances(batch_size) # Dummy increment for Phase 1 threshold
            
        # Build the L-Matrix
        best_anchors = np.array(proposer.best_thetas)
        self.L, self.L_inv = build_warp_matrix(best_anchors, density_spherical_single, self.R, self.d)
        
        # =====================================================================
        # 3. INITIALIZE ZERO-BURN-IN MCMC
        # =====================================================================
        # Find the absolute best anchor from Warm-Up to start the chain
        best_idx = np.argmax(proposer.best_masses)
        best_theta = best_anchors[best_idx]
        best_r = find_peak_golden_section(density_spherical_single, best_theta, self.R(np.array([best_theta]))[0])
        
        x_curr = best_r * best_theta 
        print(f"[STAGE 2] Initiating MCMC Chain from peak (Log Density: {density_cartesian(x_curr):.2f})")
        
        # =====================================================================
        # 4. PHASE 3: THE HYBRID SLICE WALK
        # =====================================================================
        mcmc_samples = np.empty((self.k, self.d))
        
        with tqdm(total=self.k, unit=' samples') as pbar:
            for i in range(self.k):
                # 1. Get current cartesian density and draw slice threshold
                current_log_prob = density_cartesian(x_curr)
                y_log = current_log_prob - np.random.exponential(1.0)
                
                # 2. Draw Hybrid Direction (80% Warped, 20% Coordinate)
                v = generate_hybrid_ray_numba(self.d, self.L, prob_warp=0.1)
                
                # 3. Exact Reversible Slice Step
                t_jump = slice_step_numba(density_cartesian, x_curr, v, y_log, w=1.0)
                
                # 4. Jump and Save
                x_curr = x_curr + t_jump * v
                mcmc_samples[i] = x_curr
                
                pbar.update(1)
                
        t_main_end = time.perf_counter()
        print(f'Done! Total MCMC time: {t_main_end - t_main:.2f}s')
        
        # Return samples (We no longer have rejected samples in MCMC!)
        return mcmc_samples