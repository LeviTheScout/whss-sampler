# proposal.py

import numpy as np
import math
from scipy.special import logsumexp, loggamma

# Import the C-speed calculators from your updated utility.py
from .utility import parallel_generate_and_evaluate, update_vmf_parameters

# =====================================================================
# 1. THE BLUEPRINT
# =====================================================================
class BaseProposer:
    def generate_batch(self, batch_size):
        raise NotImplementedError
        
    def update_knowledge(self, theta_batch, log_mass_batch):
        raise NotImplementedError

# =====================================================================
# 2. PURE UNIFORM STRATEGY
# =====================================================================
class UniformProposer(BaseProposer):
    def __init__(self, d):
        self.d = d
        self.log_C_unif = loggamma(d / 2.0) - np.log(2.0) - (d / 2.0) * np.log(math.pi)
        
    def generate_batch(self, batch_size):
        vec = np.random.randn(batch_size, self.d)
        theta_batch = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        log_q_batch = np.full(batch_size, self.log_C_unif)
        return theta_batch, log_q_batch
        
    def update_knowledge(self, theta_batch, log_mass_batch):
        pass  # Uniform learns nothing

# =====================================================================
# 3. ORTHANT STRATEGY (Legacy fallback)
# =====================================================================
class OrthantProposer(BaseProposer):
    def __init__(self, d, initial_orthants, initial_log_weights):
        self.d = d
        self.orthants = np.array(initial_orthants)  
        self.log_weights = np.array(initial_log_weights)
        
        # 1. Density of a standard full sphere
        self.log_uniform_density = loggamma(d / 2.0) - np.log(2.0) - (d / 2.0) * np.log(math.pi)
        
        # 2. Density of an orthant (2^d times MORE concentrated than a full sphere)
        self.log_orthant_density = self.log_uniform_density + (self.d * np.log(2.0))
        
    def generate_batch(self, batch_size):
        probs = np.exp(self.log_weights - logsumexp(self.log_weights))
        chosen_indices = np.random.choice(len(self.orthants), size=batch_size, p=probs)
        chosen_orthants = self.orthants[chosen_indices]
        
        vec = np.abs(np.random.randn(batch_size, self.d))
        theta_batch = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        
        unpacked = np.unpackbits(chosen_orthants, axis=1)[:, :self.d]
        target_signs = (unpacked.astype(int) * 2) - 1 
        theta_batch = theta_batch * target_signs
        
        # 3. Exact q(theta): Probability of picking the orthant + Density inside the orthant
        log_selection_prob = self.log_weights[chosen_indices] - logsumexp(self.log_weights)
        log_q_batch = log_selection_prob + self.log_orthant_density
        
        return theta_batch, log_q_batch
# =====================================================================
# 4. ADAPTIVE vMF STRATEGY (Phase 2)
# =====================================================================
class vMFProposer(BaseProposer):
    def __init__(self, d, initial_anchors_mu, initial_anchors_mass, initial_kappas=None):
        self.d = d
        self.anchors_mu = np.array(initial_anchors_mu)
        self.anchors_mass = np.array(initial_anchors_mass)
        
        # If no kappas provided (Phase 1 to Phase 1.5 transition), default to 15.0
        if initial_kappas is None:
            self.kappas = np.full(len(self.anchors_mu), 15.0)
        else:
            self.kappas = np.array(initial_kappas)
            
        self.cos_threshold = 0.85 
        
    def generate_batch(self, batch_size):
        num_vmf = int(0.90 * batch_size)
        
        # --- THE FIX IS HERE: Add self.kappas as the 3rd argument ---
        log_alpha, kappas, log_C, log_C_unif = update_vmf_parameters(self.anchors_mass, self.d, self.kappas)
        # ------------------------------------------------------------
        
        alpha_probs = np.exp(log_alpha)
        parent_indices = np.random.choice(len(self.anchors_mu), size=num_vmf, p=alpha_probs)
        
        log_w_vmf = np.log(0.75)
        log_w_unif = np.log(0.25)
        
        theta_batch, log_q_batch = parallel_generate_and_evaluate(
            batch_size, self.d, parent_indices, 
            self.anchors_mu, log_alpha, kappas, log_C, 
            num_vmf, log_w_vmf, log_w_unif, log_C_unif
        )
        return theta_batch, log_q_batch

    def update_knowledge(self, theta_batch, log_mass_batch):
        pass


class PhaseManager:
    def __init__(self, d, switch_threshold=0.05, fallback_strategy="vmf", burn_in_samples=10000, max_anchors=50, exploration_batches=5):
        self.d = d
        self.switch_threshold = switch_threshold
        self.burn_in_samples = burn_in_samples
        self.max_anchors = max_anchors
        self.exploration_batches = exploration_batches
        self.exploration_count = 0
        
        self.phase = 1  # 1: Uniform, 2: Explore vMF, 3: WARM-UP COMPLETE
        
        from .proposal import UniformProposer
        self.active_proposer = UniformProposer(d)
        
        self.best_thetas = []
        self.best_masses = []
        self.total_accepted = 0
        self.total_proposed = 0

    def generate_batch(self, batch_size):
        if self.phase == 3:
            raise RuntimeError("Warm-Up is complete. Do not call proposer in Phase 3.")
            
        if self.phase == 1:
            self.total_proposed += batch_size
            
        return self.active_proposer.generate_batch(batch_size)

    def register_acceptances(self, count):
        if self.phase == 1:
            self.total_accepted += count

    def update_knowledge(self, theta_batch, log_mass_batch, log_q_batch=None):
        if self.phase == 1:
            # Grab top 10 per batch to speed up initial gathering
            top_idx = np.argsort(log_mass_batch)[::-1][:10]
            self.best_thetas.extend(theta_batch[top_idx])
            self.best_masses.extend(np.exp(log_mass_batch[top_idx]))
            
            if len(self.best_thetas) > self.max_anchors:
                sorted_indices = np.argsort(self.best_masses)[::-1][:self.max_anchors]
                self.best_thetas = [self.best_thetas[i] for i in sorted_indices]
                self.best_masses = [self.best_masses[i] for i in sorted_indices]
                
            if self.total_proposed >= self.burn_in_samples:
                # We can just check mass improvement or use a fixed threshold to switch
                print(f"\n[WARM-UP] Starting Phase 2 (Adaptive vMF Exploration)...")
                from .proposal import vMFProposer
                self.active_proposer = vMFProposer(self.d, self.best_thetas, self.best_masses)
                self.phase = 2

        elif self.phase == 2:
            log_ratio = log_mass_batch - log_q_batch if log_q_batch is not None else log_mass_batch
            
            max_ratio = np.max(log_ratio)
            weights = np.exp(log_ratio - max_ratio)
            probs = weights / np.sum(weights)
            
            chosen_indices = np.random.choice(len(theta_batch), size=self.max_anchors, p=probs, replace=True)
            self.best_thetas = [theta_batch[i] for i in chosen_indices]
            self.best_masses = [np.exp(log_mass_batch[i]) for i in chosen_indices]
            
            # Use your brilliant Banerjee dynamic kappas here...
            best_thetas_np = np.array(self.best_thetas)
            similarities = np.dot(theta_batch, best_thetas_np.T)
            cluster_assignments = np.argmax(similarities, axis=1)
            dynamic_kappas = np.zeros(self.max_anchors)
            
            for k in range(self.max_anchors):
                cluster_rays = theta_batch[cluster_assignments == k]
                if len(cluster_rays) > 0:
                    mean_vec = np.mean(cluster_rays, axis=0)
                    R_bar = min(np.linalg.norm(mean_vec), 0.999)
                    kappa_k = (R_bar * (self.d - R_bar**2)) / (1.0 - R_bar**2)
                    dynamic_kappas[k] = min(kappa_k, 150.0)
                else:
                    dynamic_kappas[k] = 15.0 
                    
            from .proposal import vMFProposer
            self.active_proposer = vMFProposer(self.d, self.best_thetas, self.best_masses, dynamic_kappas)
            
            self.exploration_count += 1
            if self.exploration_count >= self.exploration_batches:
                print(f"\n[WARM-UP] Phase 2 Complete. Structural anchors locked.")
                self.phase = 3 # Signal to sampling.py that Warm-Up is done!
