
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
        """
        The Universal Exact Rejection Sampler.
        """
        # 1. Initialize the PhaseManager (Conductor) inside the class
        # Assuming proposal.py is in the same directory or properly routed in your module structure
        if burn_in_samples is None:
            burn_in_samples = batch_size
        from proposal import PhaseManager
        proposer = PhaseManager(
            d=self.d, 
            switch_threshold=switch_threshold, 
            fallback_strategy=fallback_proposer,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        
        t_main = time.perf_counter()
        
        accepted = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        rejected = Samples(theta=np.empty((0, self.d)), r_batch=np.array([]))
        
        # Track the absolute maximum ratio (Mass / q_theta) ever seen
        M_global = -np.inf 
        
        with tqdm(total=self.k, unit=' samples') as pbar:
            while accepted.length() < self.k:
                
                # ==========================================
                # A. THE PROPOSAL & 1D SEARCH
                # ==========================================
                theta_batch, log_q_batch = proposer.generate_batch(batch_size)
                
                R_batch = self.R(theta_batch)  # Calls self.R from your base class
                a_batch, b_batch, log_peak_batch, log_mass_batch = self.importance_r(density, R_batch, theta_batch)
                
                # Feedback loop: Update anchors/orthants based on new terrain
                proposer.update_knowledge(theta_batch, log_mass_batch)
                
                # ==========================================
                # B. THE GLOBAL TRACKER & RETROSPECTIVE GATE
                # ==========================================
                log_ratio_batch = log_mass_batch - log_q_batch
                current_batch_M = np.max(log_ratio_batch)
                
                if current_batch_M > M_global:
                    if M_global != -np.inf and accepted.length() > 0:
                        # M went up! Retrospectively drop early samples
                        # Keep prob = exp(M_old - M_new). In log space: log_u <= M_old - M_new
                        u_retro = np.log(np.random.uniform(0, 1, accepted.length()))
                        keep_mask = u_retro <= (M_global - current_batch_M)
                        
                        rejected.extend(accepted.filter(~keep_mask))
                        accepted = accepted.filter(keep_mask)
                        
                        # Correct the progress bar backwards
                        pbar.n = accepted.length()
                        pbar.refresh()
                        
                    M_global = current_batch_M
                    
                # ==========================================
                # C. THE DOUBLE MASK EVALUATION
                # ==========================================
                r_batch = np.random.uniform(a_batch, b_batch)
                log_f_r = density(r_batch, theta_batch)
                
                # Mask 1: Ray Density vs Ray Peak
                log_mask1 = log_f_r - log_peak_batch
                
                # Mask 2: Ray Ratio vs Global Max Ratio
                log_mask2 = log_ratio_batch - M_global
                
                # Total Probability of Acceptance is Mask1 * Mask2. 
                # In log-space, we add them: log(Mask1) + log(Mask2)
                log_u_batch = np.log(np.random.uniform(0, 1, batch_size))
                final_accept_mask = log_u_batch <= (log_mask1 + log_mask2)
                
                # ==========================================
                # D. RECORD KEEPING
                # ==========================================
                batch_samples = Samples(theta=theta_batch, r_batch=r_batch)
                accepted.extend(batch_samples.filter(final_accept_mask))
                rejected.extend(batch_samples.filter(~final_accept_mask))
                
                # Tell the proposer how many survived
                proposer.register_acceptances(np.sum(final_accept_mask))
                
                pbar.n = min(self.k, accepted.length())
                pbar.refresh()
        
        t_main_end = time.perf_counter()
        print(f'Done! Total sampling time: {t_main_end - t_main:.2f}s')
        
        ans_accepted = list(zip(accepted.theta, accepted.r_batch))
        ans_rejected = list(zip(rejected.theta, rejected.r_batch))
        return ans_accepted, ans_rejected