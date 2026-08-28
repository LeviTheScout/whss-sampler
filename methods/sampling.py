import numpy as np
from tqdm import tqdm
import time
from numba import njit
from .utility import build_warp_matrix, generate_hybrid_ray_numba, parallel_scout_eval
from .importance import slice_step_polytope, find_peak_golden_section

class sampling:
    def _sampling_universal(self, density_cartesian, A=None, b=None, batch_size=3256, burn_in_samples=10000, max_anchors=50):        
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
        
# ... (keep everything above this identical)
        # Keep proposing rays until PhaseManager hits Phase 3
        while proposer.phase < 3:
            theta_batch, log_q_batch = proposer.generate_batch(batch_size)
            R_batch = self.R(theta_batch)  
            
            # Find peaks using Golden Section
            log_mass_batch = np.empty(batch_size)
            for i in range(batch_size):
                peak_r = find_peak_golden_section(density_spherical_single, theta_batch[i], R_batch[i])
                log_mass_batch[i] = density_spherical_single(peak_r, theta_batch[i])
                
            proposer.update_knowledge(theta_batch, log_mass_batch, log_q_batch)
            proposer.register_acceptances(batch_size) 
            
# =====================================================================
        # FINAL PHASE 1.5: THE HYBRID COVARIANCE MERGE (STRETCH + FULL RANK)
        # =====================================================================
        N_warp = 10000 
        warp_rays = np.random.normal(0, 1, size=(N_warp, d))
        warp_rays /= np.linalg.norm(warp_rays, axis=1, keepdims=True)
        R_max_warp = self.R(warp_rays)

        @njit
        def fast_peak_extract(thetas, r_bounds):
            N = len(thetas)
            peaks = np.empty((N, d))
            for i in range(N):
                r_peak = find_peak_golden_section(density_spherical_single, thetas[i], r_bounds[i])
                peaks[i] = r_peak * thetas[i]
            return peaks
            
        cartesian_uniform = fast_peak_extract(warp_rays, R_max_warp)
        
        best_thetas = np.array(proposer.best_thetas)
        R_max_anchors = self.R(best_thetas)
        cartesian_anchors = fast_peak_extract(best_thetas, R_max_anchors)

        # THE CURE: Combine uniform (full-rank) with vMF anchors (the 100x stretch)
        cartesian_combined = np.vstack([cartesian_uniform, np.repeat(cartesian_anchors, 200, axis=0)])
        
        self.L, self.L_inv = build_warp_matrix(cartesian_combined, self.d)

        # =====================================================================
        # 3. INITIALIZE & PHASE 3: MCMC WALK
        # =====================================================================
        x_curr = np.mean(cartesian_anchors, axis=0)
        print(f"[STAGE 2] Initiating MCMC Chain from typical set (Log Density: {density_cartesian(x_curr):.2f})")
        
        mcmc_samples = np.empty((self.k, self.d))
        num_anchors = len(cartesian_anchors)
        use_polytope = (A is not None and b is not None)
        from .importance import slice_step_unconstrained 
        
        with tqdm(total=self.k, unit=' samples') as pbar:
            for i in range(self.k):
                current_log_prob = density_cartesian(x_curr)
                y_log = current_log_prob - np.random.exponential(1.0)
                v = generate_hybrid_ray_numba(self.d, self.L, cartesian_anchors, num_anchors)
                
                if use_polytope:
                    # Notice the '1.0' is gone!
                    t_jump = slice_step_polytope(density_cartesian, x_curr, v, y_log, A, b)
                else:
                    t_jump = slice_step_unconstrained(density_cartesian, x_curr, v, y_log, 1.0)
                
                x_curr = x_curr + t_jump * v
                mcmc_samples[i] = x_curr
                
                # =============================================================
                # NEW: THE ADAPTIVE ENGINE (Overcomes the Solid Angle Curse)
                # Updates the global L-matrix using the chain's exploration.
                # =============================================================
                if i > 1000 and i % 1000 == 0:
                    cov_emp = np.cov(mcmc_samples[:i], rowvar=False) + 1e-4 * np.eye(self.d)
                    try:
                        self.L = np.linalg.cholesky(cov_emp)
                    except np.linalg.LinAlgError:
                        pass
                pbar.update(1)
                
        t_main_end = time.perf_counter()
        print(f'Done! Total MCMC time: {t_main_end - t_main:.2f}s')
        return mcmc_samples