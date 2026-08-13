
import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass

# The C-speed 1D engine
from nsmc_sampling.methods.importance import importance_r_numba

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
            exploration_batches=5 # It will adapt for 5 batches (~16k rays) before freezing
        )
        
        
        t_main = time.perf_counter()
        
        # Vaults
        final_accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        rejected = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        
        M_global = -np.inf 
        previous_phase = 1
        
        with tqdm(total=self.k, unit=' samples') as pbar:
            while (final_accepted.length() + accepted.length()) < self.k:
                
                current_phase = proposer.phase
                
                # ==========================================
                # PHASE TRANSITION GATES
                # ==========================================
                # Transition 1 -> 2 (Uniform -> Exploration)
                if current_phase == 2 and previous_phase == 1:
                    final_accepted.extend(accepted) # Lock in uniform samples
                    accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    M_global = -np.inf
                    previous_phase = 2
                    
                # Transition 2 -> 3 (Exploration -> Exact Sampling)
                elif current_phase == 3 and previous_phase == 2:
                    accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
                    M_global = -np.inf # Fresh start for exact sampling
                    previous_phase = 3
                
                # ==========================================
                # RAY GENERATION & EVALUATION
                # ==========================================
                theta_batch, log_q_batch = proposer.generate_batch(batch_size)
                R_batch = self.R(theta_batch)  
                a_batch, b_batch, log_peak_batch, log_mass_batch = self.importance_r(density, R_batch, theta_batch)
                
                proposer.update_knowledge(theta_batch, log_mass_batch)
                
                # If we are in Phase 2 (Hunting), skip the math and don't save samples!
                if proposer.phase == 2:
                    continue
                
                # ==========================================
                # EXACT SAMPLING MATH (Phase 1 & Phase 3 only)
                # ==========================================
                log_ratio_batch = log_mass_batch - log_q_batch
                current_batch_M = np.max(log_ratio_batch)
                
                if current_batch_M > M_global:
                    if M_global != -np.inf and accepted.length() > 0:
                        u_retro = np.log(np.random.uniform(0, 1, accepted.length()))
                        keep_mask = u_retro <= (M_global - current_batch_M)
                        
                        rejected.extend(accepted.filter(~keep_mask))
                        accepted = accepted.filter(keep_mask)
                        
                    M_global = current_batch_M
                    
                r_batch = np.random.uniform(a_batch, b_batch)
                log_f_r = density(r_batch, theta_batch)
                
                log_mask1 = log_f_r - log_peak_batch
                log_mask2 = log_ratio_batch - M_global
                
                log_u_batch = np.log(np.random.uniform(0, 1, batch_size))
                final_accept_mask = log_u_batch <= (log_mask1 + log_mask2)
                
                batch_samples = Samples(theta=theta_batch, r_batch=r_batch)
                accepted.extend(batch_samples.filter(final_accept_mask))
                rejected.extend(batch_samples.filter(~final_accept_mask))
                
                proposer.register_acceptances(np.sum(final_accept_mask))
                
                total_current = final_accepted.length() + accepted.length()
                pbar.n = min(self.k, total_current)
                pbar.refresh()
        
        t_main_end = time.perf_counter()
        print(f'Done! Total sampling time: {t_main_end - t_main:.2f}s')
        
        final_accepted.extend(accepted)
        ans_accepted = list(zip(final_accepted.theta, final_accepted.r_batch))
        ans_rejected = list(zip(rejected.theta, rejected.r_batch))
        return ans_accepted, ans_rejected
