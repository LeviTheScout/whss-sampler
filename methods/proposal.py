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
# =====================================================================
# 5. THE PHASE MANAGER (The Conductor)
# =====================================================================
class PhaseManager:
    def __init__(self, d, switch_threshold=0.05, fallback_strategy="vmf", burn_in_samples=10000, max_anchors=50, exploration_batches=5):
        self.d = d
        self.switch_threshold = switch_threshold
        self.fallback_strategy = fallback_strategy.lower()
        self.burn_in_samples = burn_in_samples
        self.max_anchors = max_anchors
        
        # Phase parameters
        self.exploration_batches = exploration_batches
        self.exploration_count = 0
        self.phase = 1  # 1: Uniform, 2: Explore, 3: Exact
        
        from .proposal import UniformProposer
        self.active_proposer = UniformProposer(d)
        
        self.best_thetas = []
        self.best_masses = []
        self.total_accepted = 0
        self.total_proposed = 0

    def generate_batch(self, batch_size):
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
                acceptance_rate = self.total_accepted / (self.total_proposed + 1e-9)
                if acceptance_rate < self.switch_threshold:
                    print(f"\nEfficiency dropped to {acceptance_rate:.4f}. Starting Phase 1.5 (Adaptive Exploration)!")
                    from .proposal import vMFProposer
                    self.active_proposer = vMFProposer(self.d, self.best_thetas, self.best_masses)
                    self.phase = 2

        elif self.phase == 2:
            if log_q_batch is not None:
                log_ratio = log_mass_batch - log_q_batch
            else:
                log_ratio = log_mass_batch

            # ====================================================
            # DYNAMIC OVERLAP FIX: DECAYING SPATIAL FILTERING
            # ====================================================
            sorted_idx = np.argsort(log_ratio)[::-1]
            
            threshold = 0.85
            new_thetas = []
            new_masses = []
            
            # Keep relaxing the threshold if space is too tight (e.g. low dimensions)
            # threshold approaches 1.0 (closer together), capped at 0.999 to prevent identical copies
            while len(new_thetas) < self.max_anchors and threshold <= 0.999:
                new_thetas = []
                new_masses = []
                
                for idx in sorted_idx:
                    candidate = theta_batch[idx]
                    
                    if len(new_thetas) > 0:
                        overlaps = np.dot(new_thetas, candidate)
                        if np.max(overlaps) > threshold:
                            continue
                            
                    new_thetas.append(candidate)
                    new_masses.append(np.exp(log_mass_batch[idx]))
                    
                    if len(new_thetas) == self.max_anchors:
                        break
                        
                # If we couldn't find 50 separate anchors, relax the spread constraint
                if len(new_thetas) < self.max_anchors:
                    threshold += 0.05
                    
            self.best_thetas = new_thetas
            self.best_masses = new_masses
# GEOMETRIC LOCK: 0.7 * d prevents all Tail Failures while keeping M low
            fixed_kappa = max(5.0, float(self.d) * 0.7)
            dynamic_kappas = np.full(self.max_anchors, fixed_kappa)
            
            from .proposal import vMFProposer
            self.active_proposer = vMFProposer(self.d, self.best_thetas, self.best_masses, dynamic_kappas)
            
            self.exploration_count += 1
            if self.exploration_count >= self.exploration_batches:
                theta_array = np.array(self.best_thetas)
                unique_anchors = len(np.unique(theta_array, axis=0))
                print(f"\n[DIAGNOSTIC] Exploration complete! Unique anchors: {unique_anchors}/{self.max_anchors}")
                print("Anchors frozen. Starting Phase 3 (Exact Sampling)!")
                self.phase = 3

        elif self.phase == 3:
            # PROTECT THE EXACT SAMPLER: Do nothing here.
            pass

    def warmup_kappas(self, theta_batch):
        """
        DEDICATED PHASE 2.5 WARMUP FUNCTION.
        Safely updates kappas in the warped space WITHOUT touching the frozen anchors.
        """
        if self.phase != 3: return
        
        frozen_anchors = np.array(self.active_proposer.anchors_mu)
        similarities = np.dot(theta_batch, frozen_anchors.T)
        cluster_assignments = np.argmax(similarities, axis=1)
        
        new_kappas = np.zeros(len(frozen_anchors))
        
        for k in range(len(frozen_anchors)):
            cluster_rays = theta_batch[cluster_assignments == k]
            if len(cluster_rays) > 0:
                mean_vec = np.mean(cluster_rays, axis=0)
                R_bar = min(np.linalg.norm(mean_vec), 0.999)
                kappa_k = (R_bar * (self.d - R_bar**2)) / (1.0 - R_bar**2)
                
                # Smooth learning: average old and new
                old_kappa = self.active_proposer.kappas[k]
                new_kappas[k] = min(0.5 * old_kappa + 0.5 * kappa_k, 150.0)
            else:
                new_kappas[k] = self.active_proposer.kappas[k]
                
        self.active_proposer.kappas = new_kappas
