import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass
from numba import njit

# Import the C-speed engines
from nsmc_sampling.methods.importance import importance_r_numba
from .utility import build_warp_matrix

# ==============================================================================
# NUMBA FACTORY FOR THE WARPED TARGET
# ==============================================================================
def create_warped_density(original_density, L_matrix, d):
    """
    Creates a highly optimized, C-compiled wrapper for the target density.
    Includes the critical mathematical correction for the spherical Jacobian.
    """
    @njit
    def warped_density(r_batch, theta):
        # 1. Warp the direction vector
        L_theta = np.dot(L_matrix, theta)
        norm_L_theta = np.linalg.norm(L_theta)
        theta_true = L_theta / norm_L_theta
        
        # 2. Warp the radius 
        r_true = r_batch * norm_L_theta
        
        # 3. Evaluate the original density (which automatically adds (d-1)*log(r_true))
        log_original = original_density(r_true, theta_true)
        
        # 4. MATH FIX: Subtract the artificial angular stretch added by the original Jacobian
        jacobian_correction = (d - 1.0) * np.log(norm_L_theta)
        
        return log_original - jacobian_correction
        
    return warped_density

@njit
def evaluate_density_batch(density_func, r_batch, theta_batch):
    n = len(r_batch)
    out = np.empty(n, dtype=np.float64)
    # This loop runs in C-speed, ZERO Python overhead
    for i in range(n):
        out[i] = density_func(r_batch[i], theta_batch[i])
    return out

@dataclass
class Samples:
    theta: np.ndarray
    r_batch: np.ndarray
    
    def extend(self, other):
        if len(self.theta) == 0:
            self.theta = other.theta
            self.r_batch = other.r_batch
        else:
            self.theta = np.concatenate([self.theta, other.theta], axis=0)
            self.r_batch = np.concatenate([self.r_batch, other.r_batch], axis=0)
            
    def filter(self, mask):
        return Samples(
            theta=self.theta[mask],
            r_batch=self.r_batch[mask]
        ) 

    def length(self):
        return len(self.r_batch)


class sampling:
    def _sampling_universal(self, density, batch_size=3256, fallback_proposer="vmf", switch_threshold=0.05, burn_in_samples=None, max_anchors=50):
        
        if burn_in_samples is None:
            burn_in_samples = batch_size
            
        from .proposal import PhaseManager
        proposer = PhaseManager(
            d=self.d, switch_threshold=switch_threshold, 
            fallback_strategy=fallback_proposer,
            burn_in_samples=burn_in_samples, max_anchors=max_anchors,
            exploration_batches=30 # Adapts for exploration before freezing
        )
        
        t_main = time.perf_counter()
        
        final_accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        rejected = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        
        # --- FIX 1: Pools to hold Phase 1 & 2 samples that do NOT need warping back ---
        early_accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        early_rejected = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        
        M_global = -np.inf 
        previous_phase = 1
        
        self.L = np.eye(self.d)
        self.L_inv = np.eye(self.d)
        is_warped = False
        
        R_func = self.R
        
        # UPDATE THE WHILE LOOP CONDITION to include early_accepted
        with tqdm(total=self.k, unit=' samples') as pbar:
            while (early_accepted.length() + final_accepted.length() + accepted.length()) < self.k:
                
                current_phase = proposer.phase
                
                # Transition 1 -> 2
                if current_phase == 2 and previous_phase == 1:
                    final_accepted.extend(accepted)
                    accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    M_global = -np.inf
                    previous_phase = 2
                    
                # Transition 2 -> 3 (Exploration -> Exact Sampling + WARP ENGINE)
                elif current_phase == 3 and previous_phase == 2:
                    
                    # --- FIX 1: Archive the unwarped samples! ---
                    early_accepted.extend(final_accepted)
                    early_accepted.extend(accepted)
                    early_rejected.extend(rejected)
                    
                    # Clear the active pools so they only collect Phase 3 warped data
                    final_accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    rejected = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    # ------------------------------------------
                    
# ==========================================================
                    # --- [PHASE 2.5] STOCHASTIC LOCAL ZOOM & DEDUPLICATION ---
                    print("\n[PHASE 2.5] Zooming in on top peaks to prevent Phase 3 ratchets...")
                    
                    # Pull directly from your proposer's existing state
                    initial_anchors = np.array(proposer.best_thetas) 
                    refined_anchors = []
                    zoom_rays = 100
                    zoom_kappa = 200.0  
                    
                    for ray in initial_anchors:
                        # 1. Generate local cluster
                        noise = np.random.normal(0, 1 / np.sqrt(zoom_kappa), size=(zoom_rays, self.d))
                        local_rays = ray + noise
                        local_rays = local_rays / np.linalg.norm(local_rays, axis=1, keepdims=True)
                        
                        # 2. Evaluate true directional mass using your exact Phase 3 logic
                        local_R = self.R(local_rays)
                        _, _, _, local_masses = self.importance_r(density, local_R, local_rays)
                        
                        # 3. Find the exact highest summit
                        best_local_idx = np.argmax(local_masses)
                        refined_anchors.append(local_rays[best_local_idx])
                        
                    # 4. FILTER DUPLICATES (Prevent ESS collapse)
                    unique_anchors = []
                    for anchor in refined_anchors:
                        if len(unique_anchors) == 0:
                            unique_anchors.append(anchor)
                        else:
                            # Check similarity against already accepted unique anchors
                            similarities = np.dot(unique_anchors, anchor)
                            if np.max(similarities) < 0.85: # 0.85 = distinct enough
                                unique_anchors.append(anchor)
                                
# If deduplication dropped us below max_anchors, pad with new random directions
                    attempts = 0
                    while len(unique_anchors) < proposer.max_anchors and attempts < 1000:
                        # 1. Generate a purely random Uniform ray
                        random_ray = np.random.normal(0, 1, size=self.d)
                        random_ray = random_ray / np.linalg.norm(random_ray)
                        
                        # 2. Check for overlap
                        similarities = np.dot(unique_anchors, random_ray)
                        if np.max(similarities) < 0.85:
                            unique_anchors.append(random_ray)
                            
                        attempts += 1
                    
                    if attempts >= 1000:
                        print(f"[WARNING] Sphere is crowded. Proceeding with {len(unique_anchors)} anchors.")
                        
                    # Lock in the final, perfect, unique anchors
                    best_anchors = np.array(unique_anchors[:proposer.max_anchors])
                    proposer.best_thetas = list(best_anchors)
                    print(f"[PHASE 2.5] Zoom complete. {len(best_anchors)} distinct summits locked.")
                    # ==========================================================

                    # 1. BUILD THE WARP MATRIX (Using the perfected anchors)
                    self.L, self.L_inv = build_warp_matrix(best_anchors, density, self.R, self.d)
                    
                    # 2. CREATE THE JIT-COMPILED WARPED DENSITY
                    density = create_warped_density(density, self.L, self.d)
                    is_warped = True
                    
                    # 3. TRANSLATE THE ANCHORS (The Starting Point)
                    warped_anchors = np.dot(best_anchors, self.L_inv.T)
                    warped_anchors = warped_anchors / np.linalg.norm(warped_anchors, axis=1, keepdims=True)
                    proposer.active_proposer.anchors_mu = warped_anchors
                    
                    # 4. FIX THE BOUNDING BOX
                    original_a = self.a
                    L_matrix_T = self.L.T
                    def warped_R(theta_batch):
                        L_theta = np.dot(theta_batch, L_matrix_T)
                        inf_norm = np.max(np.abs(L_theta), axis=1)
                        return original_a / (2 * inf_norm)
                    
                    R_func = warped_R
                    
                    
                    # =====================================================================
                    # --- [START] ENVELOPE DIAGNOSTIC PROBE ---
                    try:
                        anchors_to_check = proposer.active_proposer.anchors_mu
                        kappas_to_check = proposer.active_proposer.kappas
                        print("\n[ENVELOPE DIAGNOSTIC]")
                        print(f"Total Anchors Active : {len(kappas_to_check)}")
                        print(f"Kappas (Sharpness)   : Min={kappas_to_check.min():.2f} | Mean={kappas_to_check.mean():.2f} | Max={kappas_to_check.max():.2f}")
                        
                        # Calculate pairwise overlap (dot products) of the anchors
                        dots = np.dot(anchors_to_check, anchors_to_check.T)
                        np.fill_diagonal(dots, -1.0) # Ignore self-overlap
                        
                        max_overlap = np.max(dots)
                        mean_overlap = np.mean(dots[dots > -1.0])
                        
                        print(f"Max Anchor Overlap   : {max_overlap:.4f} (1.0 = identical)")
                        print(f"Mean Anchor Overlap  : {mean_overlap:.4f}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"[ENVELOPE DIAGNOSTIC FAILED]: {e}")
                    # --- [END] ENVELOPE DIAGNOSTIC PROBE --------------------------------
                    # =====================================================================

                    accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    M_global = -np.inf 
                    previous_phase = 3
                
                # --- CORE SAMPLING STEP ---
                theta_batch, log_q_batch = proposer.generate_batch(batch_size)
                R_batch = R_func(theta_batch)  
                a_batch, b_batch, log_peak_batch, log_mass_batch = self.importance_r(density, R_batch, theta_batch)
                
                # --- FIX 2: EXACT REJECTION LOGIC RUNS FIRST ---
                if proposer.phase != 2:
                    log_ratio_batch = log_mass_batch - log_q_batch
                    current_batch_M = np.max(log_ratio_batch)
                    
                    if current_batch_M > M_global:
                        if M_global != -np.inf and accepted.length() > 0:
                            u_retro = np.log(np.random.uniform(0, 1, accepted.length()))
                            keep_mask = u_retro <= (M_global - current_batch_M)
                            
                            rejected.extend(accepted.filter(~keep_mask))
                            accepted = accepted.filter(keep_mask)
                            
                        M_global = current_batch_M
                        
                        # =====================================================================
                        # --- [START] GAP DETECTOR PROBE ---
                        try:
                            import math
                            from scipy.special import loggamma
                            
                            # Find the exact ray that caused the ratchet
                            idx_max = np.argmax(log_ratio_batch)
                            trigger_q = log_q_batch[idx_max]
                            trigger_mass = log_mass_batch[idx_max]
                            
                            # Calculate the mathematical baseline of the Uniform Safety Net
                            # (0.25 weight * uniform spherical density)
                            log_unif_baseline = np.log(0.25) + loggamma(self.d / 2.0) - np.log(2.0) - (self.d / 2.0) * np.log(math.pi)
                            
                            # If log_q is within 0.1 of the baseline, the vMFs missed it entirely
                            is_gap = (trigger_q <= log_unif_baseline + 0.1)
                            
                            print(f"\n[DIAGNOSTIC] M_global ratcheted up to log(M) = {M_global:.2f}")
                            print(f"   -> Trigger Ray log_mass : {trigger_mass:.2f} | log_q : {trigger_q:.2f}")
                            print(f"   -> Uniform Baseline     : {log_unif_baseline:.2f}")
                            print(f"   -> Cause of Ratchet     : {'GAP DETECTED (Uniform Net Caught It)' if is_gap else 'vMF Tali Failure'}")
                        except Exception as e:
                            print(f"[DIAGNOSTIC] M_global ratcheted up to log(M) = {M_global:.2f} (Probe Failed: {e})")
                        # --- [END] GAP DETECTOR PROBE --------------------------------------
                        # =====================================================================
                        
                    r_batch = np.random.uniform(a_batch, b_batch)
                    
                    log_f_r =evaluate_density_batch(density, r_batch, theta_batch)
                    
                    log_mask1 = log_f_r - log_peak_batch
                    log_mask2 = log_ratio_batch - M_global
                    
                    log_u_batch = np.log(np.random.uniform(0, 1, batch_size))
                    final_accept_mask = log_u_batch <= (log_mask1 + log_mask2)
                    
                    batch_samples = Samples(theta=theta_batch, r_batch=r_batch)
                    accepted.extend(batch_samples.filter(final_accept_mask))
                    rejected.extend(batch_samples.filter(~final_accept_mask))
                    
                    # Record acceptances so Phase 1 knows its true efficiency!
                    proposer.register_acceptances(np.sum(final_accept_mask))
                
                # --- FIX 2: SIR UPDATE RUNS AFTER REJECTION ---
                proposer.update_knowledge(theta_batch, log_mass_batch, log_q_batch)
                
                # Track total progress across both pre-warp and post-warp pools
                total_current = early_accepted.length() + final_accepted.length() + accepted.length()
                pbar.n = min(self.k, total_current)
                pbar.refresh()
        
        t_main_end = time.perf_counter()
        print(f'Done! Total sampling time: {t_main_end - t_main:.2f}s')
        
        final_accepted.extend(accepted)
        
        # --- FIX 1: THE FINAL REVERSAL LOGIC ---
        def reverse_samples(samples_obj):
            true_coords = []
            for i in range(len(samples_obj.theta)):
                y_cartesian = samples_obj.r_batch[i] * samples_obj.theta[i]
                x_cartesian = np.dot(self.L, y_cartesian)
                r_true = np.linalg.norm(x_cartesian)
                theta_true = x_cartesian / (r_true + 1e-15)
                true_coords.append((theta_true, r_true))
            return true_coords

        if is_warped:
            # 1. Reverse ONLY the Phase 3 samples
            ans_accepted_warped = reverse_samples(final_accepted)
            ans_rejected_warped = reverse_samples(rejected)
            
            # 2. Format the Phase 1 & 2 samples (already in true space)
            ans_accepted_early = list(zip(early_accepted.theta, early_accepted.r_batch))
            ans_rejected_early = list(zip(early_rejected.theta, early_rejected.r_batch))
            
            # 3. Combine them
            ans_accepted = ans_accepted_early + ans_accepted_warped
            ans_rejected = ans_rejected_early + ans_rejected_warped
        else:
            # Fallback if it finished perfectly in Phase 1 before warping
            early_accepted.extend(final_accepted)
            ans_accepted = list(zip(early_accepted.theta, early_accepted.r_batch))
            
            early_rejected.extend(rejected)
            ans_rejected = list(zip(early_rejected.theta, early_rejected.r_batch))
            
        return ans_accepted, ans_rejected
