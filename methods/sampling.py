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
                    
                    # 1. BUILD THE WARP MATRIX
                    best_anchors = np.array(proposer.best_thetas)
                    self.L, self.L_inv = build_warp_matrix(best_anchors, density, self.R, self.d)
                    
                    # 2. CREATE THE JIT-COMPILED WARPED DENSITY
                    density = create_warped_density(density, self.L, self.d)
                    is_warped = True
                    
                    # 3. TRANSLATE THE ANCHORS
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
                        print(f"\n[DIAGNOSTIC] M_global ratcheted up to log(M) = {M_global:.2f}")
                        
                    r_batch = np.random.uniform(a_batch, b_batch)
                    
                    log_f_r = np.array([density(r_batch[i], theta_batch[i]) for i in range(batch_size)])
                    
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
