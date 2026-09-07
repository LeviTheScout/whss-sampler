import numpy as np
from tqdm import tqdm
import time
from numba import njit
from .utility import build_warp_matrix, generate_hybrid_ray_numba, parallel_scout_eval
from .importance import slice_step_polytope, find_peak_golden_section

class sampling:
    def _sampling_universal(self, density_cartesian, A=None, b=None, batch_size=3256, burn_in_samples=10000, max_anchors=50, bypass_safeguards=False):        
        # =====================================================================
        # 1. AUTOMATIC SPHERICAL WRAPPER (User only writes Cartesian!)
        # =====================================================================
        d = self.d
        if not bypass_safeguards:
            max_anchors = int(max(max_anchors, 3 * d))
            burn_in_samples=int(max(burn_in_samples, 1000 * d))
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

        # --- THE FIX: DETERMINISTIC AXIS-ALIGNED SCOUTS ---
        # Guarantees we discover extreme corridors before random exploration begins
        coord_rays = np.vstack([np.eye(d), -np.eye(d)])
        R_coord = np.empty(2*d)
        self.safe_shift = np.zeros(d)
        
        if A is not None and b is not None:
            for i in range(2*d):
                A_theta = A @ coord_rays[i]
                valid_idx = A_theta > 1e-12
                if np.any(valid_idx):
                    R_coord[i] = np.min(b[valid_idx] / A_theta[valid_idx])
                else:
                    R_coord[i] = 1e3
                    
            if np.min(R_coord) < 1e-3:
                print(f"[STAGE 0] Origin trapped in microscopic corner. Geometric re-centering...")
                best_idx = np.argmax(R_coord)
                self.safe_shift = coord_rays[best_idx] * (R_coord[best_idx] * 0.5)
                b = b - A @ self.safe_shift
                
                # Re-calculate R_coord from the new safe center
                for i in range(2*d):
                    A_theta = A @ coord_rays[i]
                    valid_idx = A_theta > 1e-12
                    if np.any(valid_idx):
                        R_coord[i] = np.min(b[valid_idx] / A_theta[valid_idx])
                    else:
                        R_coord[i] = 1e3
        else:
            R_coord = self.R(coord_rays)
            
        log_mass_coord = np.empty(2*d)
        for i in range(2*d):
            peak_r = find_peak_golden_section(density_spherical_single, coord_rays[i], R_coord[i])
            log_mass_coord[i] = density_spherical_single(peak_r, coord_rays[i])
            
        # Prime the PhaseManager with these perfect structural bounds
        proposer.update_knowledge(coord_rays, log_mass_coord, np.zeros(2*d))
        # --------------------------------------------------
        
        # Keep proposing random rays until PhaseManager hits Phase 3
        while proposer.phase < 3:
            theta_batch, log_q_batch = proposer.generate_batch(batch_size)
            
            # --- THE POLYTOPE RAY-BOUNDING FIX ---
            if A is not None and b is not None:
                R_batch = np.empty(batch_size)
                for i in range(batch_size):
                    A_theta = A @ theta_batch[i]
                    valid_idx = A_theta > 1e-12
                    if np.any(valid_idx):
                        R_batch[i] = np.min(b[valid_idx] / A_theta[valid_idx])
                    else:
                        R_batch[i] = 1e3
            else:
                R_batch = self.R(theta_batch)
            # -------------------------------------
            
            # Find peaks using Golden Section
            log_mass_batch = np.empty(batch_size)
            for i in range(batch_size):
                peak_r = find_peak_golden_section(density_spherical_single, theta_batch[i], R_batch[i])
                log_mass_batch[i] = density_spherical_single(peak_r, theta_batch[i])
                
            proposer.update_knowledge(theta_batch, log_mass_batch, log_q_batch)
            proposer.register_acceptances(batch_size) 
            
        # =====================================================================
        # FINAL PHASE 1.5: SCALE-AWARE FULL-RANK WARP MATRIX ESTIMATION
        # =====================================================================
        print("[WARP ENGINE] Booting Phase 1.5: Calculating global space transformation...")
        
        # 1. Extract peaks only from the Phase 2 vMF anchors
        best_thetas = np.array(proposer.best_thetas)
        
        # --- THE POLYTOPE RAY-BOUNDING FIX (Phase 1.5) ---
        if A is not None and b is not None:
            R_max_anchors = np.empty(len(best_thetas))
            for i in range(len(best_thetas)):
                A_theta = A @ best_thetas[i]
                valid_idx = A_theta > 1e-12
                if np.any(valid_idx):
                    R_max_anchors[i] = np.min(b[valid_idx] / A_theta[valid_idx])
                else:
                    R_max_anchors[i] = 1e3
        else:
            R_max_anchors = self.R(best_thetas)
        # -------------------------------------------------
        
        @njit
        def fast_peak_extract(thetas, r_bounds):
            N = len(thetas)
            peaks = np.empty((N, d))
            for i in range(N):
                r_peak = find_peak_golden_section(density_spherical_single, thetas[i], r_bounds[i])
                peaks[i] = r_peak * thetas[i]
            return peaks

        cartesian_anchors = fast_peak_extract(best_thetas, R_max_anchors)

        # 2. Empirical Covariance from structural anchors
        cov_emp = np.cov(cartesian_anchors, rowvar=False)
        
        # Handle 1D edge case if d=1
        if d == 1:
            cov_emp = np.array([[cov_emp]])

        # 3. Dynamic Regularization & Full-Rank Guarantee (Convex Shrinkage)
        max_var = np.max(np.diag(cov_emp))
        avg_var = np.mean(np.diag(cov_emp))
        
        # Adaptive shrinkage: stronger regularisation if anchor pool is close to dimension d
        n_anchors = len(cartesian_anchors)
        shrinkage = max(1e-4, float(d) / float(n_anchors + d))
        
        # Regularized target covariance: 
        # If the target density is so steep that all anchors collapsed to the origin (avg_var ~ 0)
        # we fall back to a geometric variance based on the actual physical boundaries of the polytope.
        if avg_var < 1e-8:
            geom_var = np.mean(R_max_anchors)**2 / float(d)
            target_diag = max(geom_var, 1e-4)
            shrinkage = max(shrinkage, 0.5) # Force strong mixing with the geometric prior
        else:
            target_diag = max(avg_var, 1e-6)
            
        safe_cov = (1.0 - shrinkage) * cov_emp + (shrinkage * target_diag) * np.eye(d)

        # 4. Safe Cholesky Decomposition
        try:
            self.L = np.linalg.cholesky(safe_cov)
        except np.linalg.LinAlgError:
            # Fallback for extreme collinearity
            safe_cov += (1e-3 * max_var + 1e-6) * np.eye(d)
            self.L = np.linalg.cholesky(safe_cov)
            
        self.L_inv = np.linalg.inv(self.L)

        # Allow external override of L-matrix.
        # Use case: high-dimensional peaked targets where ray-peak warm-up collapses.
        # The caller can pre-compute a geometric L from uniform LP samples and inject it.
        if hasattr(self, 'L_override') and self.L_override is not None:
            print("[WARP ENGINE] External L-matrix override detected. Replacing collapsed warm-up L.")
            self.L = self.L_override
            self.L_inv = np.linalg.inv(self.L)

        # Matrix condition diagnostic log
        eigvals = np.linalg.eigvalsh(self.L @ self.L.T)
        cond_num = np.sqrt(np.max(eigvals) / max(np.min(eigvals), 1e-12))
        print(f"=== [L MATRIX DIAGNOSTIC] ===")
        print(f"Condition Number : {cond_num:.2f}")
        print(f"Max Eigenvalue   : {np.max(eigvals):.4f}")
        print(f"Min Eigenvalue   : {np.min(eigvals):.4f}")
        print(f"=============================")

# =====================================================================
        # 3. INITIALIZE PHASE 3: ROBUST CENTERING (PATT UPGRADE)
        # =====================================================================
        print("[STAGE 2] Initiating Adaptive Multi-Chain MCMC...")

        # 1. The mean of feasible anchors is mathematically guaranteed to be in the strict interior of a convex set
        x_center = np.mean(cartesian_anchors, axis=0)
        
        if A is not None and b is not None:
            # 2. Measure strict feasibility (slack distance to the nearest boundary)
            slacks = b - (A @ x_center)
            
            # 3. If numerical precision puts the mean dangerously close to a boundary
            if np.any(slacks <= 1e-8):
                print("[WARNING] Center near boundary. Re-centering via max-slack anchor.")
                # Find the single anchor point furthest from any boundary
                safest_idx = np.argmax(np.min(b[:, None] - (A @ cartesian_anchors.T), axis=0))
                safest_anchor = cartesian_anchors[safest_idx]
                
                # Take a convex combination to pull the center safely inward
                x_center = 0.5 * x_center + 0.5 * safest_anchor
                
        # Override if user provides a specific feasible start point
        if hasattr(self, 'x_init_override') and self.x_init_override is not None:
            x_center = self.x_init_override
        
        # Spawn entangled parallel chains
        p_chains = 10 
        states = [x_center.copy() for _ in range(p_chains)]
        
        # Save the initial geometric covariance to act as a stabilizer
        cov_anchors = np.cov(cartesian_anchors, rowvar=False)
        if d == 1:
            cov_anchors = np.array([[cov_anchors[0, 0]]])

# =====================================================================
        # 4. PHASE 4: ADAPTIVE PARALLEL REFINEMENT
        # =====================================================================
        # Finite update limit to guarantee mathematical ergodicity
        k_max = 10  
        # Adaptive schedule interval scaled by dimension and chain count
        s_interval = max(d, 25) * p_chains  
        
        # Welford's Recursion Variables for memory-free covariance
        m_k = np.zeros(d)
        Q_k = np.zeros((d, d))
        n_welford = 0
        
        all_samples = [[] for _ in range(p_chains)]
        samples_generated = 0
        schedule_tick = 1
        
        # FIX 1: Ensure total_budget references the class variable or argument
        total_budget = self.k  
        
        while samples_generated < total_budget:
            # Determine how many steps each chain takes before the next sync
            if schedule_tick <= k_max:
                chunk_steps = s_interval // p_chains
            else:
                chunk_steps = (total_budget - samples_generated) // p_chains
                if chunk_steps <= 0: break
                
            chunk_samples_flat = []
            
            # -------------------------------------------------------------
            # A. INDEPENDENT PARALLEL WALK
            # -------------------------------------------------------------
            for j in range(p_chains):
                x_curr = states[j]
                
                for _ in range(chunk_steps):
                    # 1. Propose Hybrid Direction
                    move_type = np.random.rand()
                    if move_type < 0.6:  # Warped Move
                        z = np.random.randn(d)
                        u = self.L @ z
                    elif move_type < 0.9 and len(cartesian_anchors) >= 2: # Skeleton Move
                        idx = np.random.choice(len(cartesian_anchors), 2, replace=False)
                        u = cartesian_anchors[idx[0]] - cartesian_anchors[idx[1]]
                    else: # Coordinate Move
                        u = np.zeros(d)
                        u[np.random.randint(d)] = 1.0
                        
                    norm_u = np.linalg.norm(u)
                    if norm_u > 1e-15: 
                        u = u / norm_u
                    else: 
                        u = np.random.randn(d)
                        u = u / np.linalg.norm(u)

                    # 2. Analytic Chord Clipping (Polytope Bounds)
                    t_min, t_max = -np.inf, np.inf
                    if A is not None and b is not None:
                        Au = A @ u
                        Ax = A @ x_curr
                        for idx_b in range(len(b)):
                            if Au[idx_b] > 1e-12: 
                                t_max = min(t_max, (b[idx_b] - Ax[idx_b]) / Au[idx_b])
                            elif Au[idx_b] < -1e-12: 
                                t_min = max(t_min, (b[idx_b] - Ax[idx_b]) / Au[idx_b])
                    
                    if t_min == -np.inf: t_min = -1e3
                    if t_max == np.inf: t_max = 1e3
                    
                    # 3. 1D Shrinkage
                    if t_min < t_max:
                        y_slice = density_cartesian(x_curr) - np.random.exponential(1.0)
                        t = np.random.uniform(t_min, t_max)
                        x_prop = x_curr + t * u
                        
                        while density_cartesian(x_prop) < y_slice:
                            if t > 0: t_max = t
                            else: t_min = t
                            if t_max - t_min < 1e-10: 
                                break
                            t = np.random.uniform(t_min, t_max)
                            x_prop = x_curr + t * u
                            
                        x_curr = x_prop
                    
                    all_samples[j].append(x_curr.copy())
                    # FIX 2: Added .copy() to prevent reference mutation in Welford calculation
                    chunk_samples_flat.append(x_curr.copy())
                    
                states[j] = x_curr
                
            samples_generated += (chunk_steps * p_chains)
            
            # -------------------------------------------------------------
            # B. ADAPTIVE L-MATRIX REFINEMENT (Only runs up to k_max)
            # -------------------------------------------------------------
            if schedule_tick <= k_max:
                print(f"[PATT UPDATE] Tick {schedule_tick}/{k_max}: Synchronizing chains & refining L-Matrix...")
                
                # Welford's Recursive Covariance Update
                for x_val in chunk_samples_flat:
                    n_welford += 1
                    delta = x_val - m_k
                    m_k += delta / n_welford
                    Q_k += np.outer(delta, x_val - m_k)
                    
                # Compute statistical MCMC covariance
                cov_mcmc = Q_k / max(1, n_welford - 1)
                
                # SAFEGUARD: Wait until chains spread before checking for funnel collapse
                trace_mcmc = np.trace(cov_mcmc)
                trace_anchors = np.trace(cov_anchors)
                
                # THE FIX: Only trigger funnel safeguard on UNCONSTRAINED manifolds (A is None)
                if A is None and schedule_tick > 3 and trace_mcmc < 1e-3 * trace_anchors:
                    print("[WARNING] Severe covariance collapse detected (varying curvature). Freezing L-Matrix.")
                    # Force the schedule to complete immediately to prevent trapping
                    schedule_tick = k_max + 1 
                    continue
                
                
                # Safe Shrinkage: Blend statistical covariance with geometric Anchor covariance
                gamma_k = 1.0 / np.sqrt(schedule_tick)
                cov_safe = (1.0 - gamma_k) * cov_mcmc + gamma_k * cov_anchors
                
                # Adaptive Ridge to maintain strict positive-definiteness
                max_var = np.max(np.diag(cov_safe))
                adaptive_ridge = max(1e-8, 1e-5 * max_var)
                cov_safe += adaptive_ridge * np.eye(d)
                
                try:
                    self.L = np.linalg.cholesky(cov_safe)
                    self.L_inv = np.linalg.inv(self.L)
                except np.linalg.LinAlgError:
                    print("[WARNING] Matrix collapsed during refinement. Reverting to previous L.")
                    
            elif schedule_tick == k_max + 1:
                print("[PATT UPDATE] Schedule complete. L-Matrix locked for strict ergodicity. Running to budget...")

            schedule_tick += 1

        # Instead of np.vstack, return a 3D array: (p_chains, steps_per_chain, d)
        final_samples = np.array(all_samples)
        
        # Shift samples back to their original geometric coordinates
        if hasattr(self, 'safe_shift'):
            final_samples = final_samples + self.safe_shift
            
        return final_samples