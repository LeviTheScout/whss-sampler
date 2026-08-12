# proposal.py

import numpy as np
import math
from scipy.special import logsumexp, loggamma

# Import the C-speed calculators from your updated utility.py
from utility import parallel_generate_and_evaluate, update_vmf_parameters

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
    def __init__(self, d, initial_anchors_mu, initial_anchors_mass):
        self.d = d
        self.anchors_mu = np.array(initial_anchors_mu)
        self.anchors_mass = np.array(initial_anchors_mass)
        self.cos_threshold = 0.85 
        
    def generate_batch(self, batch_size):
        num_vmf = int(0.90 * batch_size)
        log_alpha, kappas, log_C, log_C_unif = update_vmf_parameters(self.anchors_mass, self.d)
        
        alpha_probs = np.exp(log_alpha)
        parent_indices = np.random.choice(len(self.anchors_mu), size=num_vmf, p=alpha_probs)
        
        log_w_vmf = np.log(0.90)
        log_w_unif = np.log(0.10)
        
        theta_batch, log_q_batch = parallel_generate_and_evaluate(
            batch_size, self.d, parent_indices, 
            self.anchors_mu, log_alpha, kappas, log_C, 
            num_vmf, log_w_vmf, log_w_unif, log_C_unif
        )
        return theta_batch, log_q_batch
        
    def update_knowledge(self, theta_batch, log_mass_batch):
        best_indices = np.argsort(log_mass_batch)[::-1]
        top_k = max(1, len(log_mass_batch) // 20)
        
        for idx in best_indices[:top_k]:
            new_mass = np.exp(log_mass_batch[idx])
            new_theta = theta_batch[idx]
            
            similarities = np.dot(self.anchors_mu, new_theta)
            closest_idx = np.argmax(similarities)
            
            if similarities[closest_idx] > self.cos_threshold:
                if new_mass > self.anchors_mass[closest_idx]:
                    self.anchors_mu[closest_idx] = new_theta
                    self.anchors_mass[closest_idx] = new_mass
            else:
                worst_idx = np.argmin(self.anchors_mass)
                if new_mass > self.anchors_mass[worst_idx]:
                    self.anchors_mu[worst_idx] = new_theta
                    self.anchors_mass[worst_idx] = new_mass

# =====================================================================
# 5. THE PHASE MANAGER (The Conductor)
# =====================================================================
class PhaseManager(BaseProposer):
    def __init__(self, d, switch_threshold=0.05, fallback_strategy="vmf", burn_in_samples=10000, max_anchors=50):
        self.d = d
        self.switch_threshold = switch_threshold
        self.fallback_strategy = fallback_strategy.lower()
        
        self.burn_in_samples = burn_in_samples
        self.max_anchors = max_anchors
        
        self.active_proposer = UniformProposer(d)
        self.is_phase_2_active = False
        
        self.best_thetas = []
        self.best_masses = []
        self.total_accepted = 0
        self.total_proposed = 0
        
    def generate_batch(self, batch_size):
        self.total_proposed += batch_size
        return self.active_proposer.generate_batch(batch_size)
        
    def update_knowledge(self, theta_batch, log_mass_batch):
        self.active_proposer.update_knowledge(theta_batch, log_mass_batch)
        
        if not self.is_phase_2_active:
            best_idx = np.argmax(log_mass_batch)
            self.best_thetas.append(theta_batch[best_idx])
            self.best_masses.append(np.exp(log_mass_batch[best_idx]))
            
            if len(self.best_thetas) > self.max_anchors:
                sorted_indices = np.argsort(self.best_masses)[::-1][:50]
                self.best_thetas = [self.best_thetas[i] for i in sorted_indices]
                self.best_masses = [self.best_masses[i] for i in sorted_indices]
            
            if self.total_proposed > self.burn_in_samples:
                acceptance_rate = self.total_accepted / self.total_proposed
                if acceptance_rate < self.switch_threshold:
                    print(f"\nEfficiency dropped to {acceptance_rate:.4f}. Switching to {self.fallback_strategy.upper()} Phase 2!")
                    
                    if self.fallback_strategy == "vmf":
                        self.active_proposer = vMFProposer(self.d, self.best_thetas, self.best_masses)
                    elif self.fallback_strategy == "orthant":
                        mask = np.array(self.best_thetas) >= 0
                        best_orthants = np.packbits(mask, axis=1)
                        # We use np.unique to group redundant orthants
                        unique_orthants, indices = np.unique(best_orthants, axis=0, return_index=True)
                        unique_masses = np.array(self.best_masses)[indices]
                        self.active_proposer = OrthantProposer(self.d, unique_orthants, np.log(unique_masses + 1e-100))
                        
                    self.is_phase_2_active = True
                    
    def register_acceptances(self, count):
        self.total_accepted += count