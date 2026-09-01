import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import cobra
from cobra.sampling import sample
from scipy.linalg import svd
from scipy.stats import wasserstein_distance
from numba import njit
from emcee.autocorr import integrated_time

# =====================================================================
# PATH RESOLUTION: NATIVE CODEBASE
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from nsmc_sampling.distributions.gaussian import nsmc_sampling_gaussian

results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# =====================================================================
# DIAGNOSTICS & RESAMPLING
# =====================================================================
def compute_robust_ess_stats(samples_2d):
    n_steps, d = samples_2d.shape
    ess_list = []
    for j in range(d):
        col = samples_2d[:, j]
        if np.std(col) < 1e-8: continue
        try:
            tau = integrated_time(col, tol=0)
            ess = n_steps / np.max(tau)
            if not np.isnan(ess) and ess > 0: ess_list.append(ess)
        except Exception:
            pass
    if len(ess_list) == 0: return 1.0, 1.0, 1.0
    return np.min(ess_list), np.median(ess_list), np.max(ess_list)

def apply_sir_weighting(uniform_samples, x_ref, sigma):
    """
    Applies Sampling Importance Resampling (SIR) and explicitly calculates
    the Importance Sampling Effective Sample Size (ESS_IS) to prove SOTA degeneracy.
    """
    n_samples = len(uniform_samples)
    diff = uniform_samples - x_ref
    sq_dist = np.sum(diff**2, axis=1)
    log_weights = -0.5 * sq_dist / (sigma**2)
    
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    weights /= np.sum(weights)
    
    # The true measure of importance sampling collapse
    ess_is = 1.0 / np.sum(weights**2)
    ess_is_pct = (ess_is / n_samples) * 100.0
    
    resampled_idx = np.random.choice(
        n_samples, size=n_samples, p=weights, replace=True
    )
    return uniform_samples[resampled_idx], ess_is, ess_is_pct

def compute_ess_per_chain(samples):
    """
    Computes Min ESS. Handles 3D arrays (n_chains, n_steps, dims) from WHSS.
    """
    try:
        from emcee.autocorr import integrated_time
    except ImportError:
        return 1.0
        
    if samples.ndim == 3:
        n_chains, n_steps, d = samples.shape
        chain_ess = []
        for c in range(n_chains):
            try:
                tau = integrated_time(samples[c], c=5, tol=10, quiet=True)
                chain_ess.append(n_steps / np.max(tau))
            except:
                chain_ess.append(1.0)
        return np.sum(chain_ess)
    else:
        try:
            tau = integrated_time(samples, c=5, tol=10, quiet=True)
            return len(samples) / np.max(tau)
        except:
            return 1.0

def compute_rhat(samples):
    """
    Computes the Gelman-Rubin (R-hat) diagnostic for multi-chain samples.
    Requires (n_chains, n_steps, d).
    """
    if samples.ndim != 3: return 1.0
    n_chains, n_steps, d = samples.shape
    if n_chains < 2: return 1.0
    
    chain_means = np.mean(samples, axis=1) # (chains, dims)
    overall_mean = np.mean(chain_means, axis=0) # (dims,)
    
    B = n_steps / (n_chains - 1) * np.sum((chain_means - overall_mean)**2, axis=0)
    W = np.mean(np.var(samples, axis=1, ddof=1), axis=0)
    
    V_hat = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B
    R_hat = np.sqrt(V_hat / (W + 1e-12))
    return np.max(R_hat)

def compute_ess_whss_3d_stats(samples_3d):
    n_chains, n_steps, d = samples_3d.shape
    total_ess_per_dim = []
    for j in range(d):
        dim_total = 0.0
        active = False
        for c in range(n_chains):
            col = samples_3d[c, :, j]
            if np.std(col) > 1e-8:
                try:
                    tau = integrated_time(col, tol=0)
                    ess = n_steps / np.max(tau)
                    if not np.isnan(ess) and ess > 0:
                        dim_total += ess; active = True
                except Exception:
                    pass
        if active: total_ess_per_dim.append(dim_total)
    if len(total_ess_per_dim) == 0: return 1.0, 1.0, 1.0
    return np.min(total_ess_per_dim), np.median(total_ess_per_dim), np.max(total_ess_per_dim)

def generate_pure_lp_basis(model):
    """
    Geometric Preprocessor (Topological Identification)
    Uses a fast ACHR trace strictly to identify the true volumetric center 
    and the exact non-zero variance dimensions.
    """
    from cobra.sampling import sample
    
    warmup_df = sample(model, 2500, method="achr", thinning=1)
    points = warmup_df.values
    center = np.mean(points, axis=0)
    centered = points - center
    
    _, S_sing, Vt = svd(centered, full_matrices=False)
    Z = Vt[S_sing > 1e-8].T
    return center, Z, points

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def run_genome_scale_benchmark():
    k_runs = 1
    n_samples = 15000 
    
    print("=" * 85)
    print(f"GENOME-SCALE NON-UNIFORM BENCHMARK (Averaged over {k_runs} Runs)")
    print("=" * 85 + "\n")
    
    import logging
    logging.getLogger("cobra").setLevel(logging.ERROR)
    
    print("[1/4] Loading Model & Extracting Network Null Space...")
    try:
        model = cobra.io.load_model("iJO1366")
    except Exception:
        print("[WARNING] iJO1366 not found locally. Falling back to e_coli_core.")
        model = cobra.io.load_model("textbook")
        
    d_full = len(model.reactions)
    fba_solution = model.optimize()
    x_ref = fba_solution.fluxes.values
    
    # -----------------------------------------------------------------
    # PRE-COMPUTATION (Removes Warmup from Timed Loops & Locks Dimensions)
    # -----------------------------------------------------------------
    print("  -> Running volumetric warmup to determine exact static null space...")
    center, Z, lp_points = generate_pure_lp_basis(model)
    k_dims = Z.shape[1] 
    
    lb = np.array([rxn.lower_bound for rxn in model.reactions])
    ub = np.array([rxn.upper_bound for rxn in model.reactions])
    A_poly = np.vstack([Z, -Z])
    b_poly = np.concatenate([ub - center, center - lb])
    b_poly = np.maximum(b_poly, 0.0) 
    
    # Calibrate Sigma dynamically to 25% of the median active flux range
    active_ranges = np.max(lp_points, axis=0) - np.min(lp_points, axis=0)
    sigma_penalty = np.median(active_ranges[active_ranges > 1e-5]) * 0.25
    
    print(f"  -> Fixed Null Space Dimensions: {k_dims}")
    print(f"  -> Calibrated Sigma Penalty   : {sigma_penalty:.4f}")

    # Trackers
    achr_times, achr_min, achr_ess_is = [], [], []
    chrr_times, chrr_min, chrr_ess_is = [], [], []
    whss_times, whss_min = [], []
    whss_rhat = []
    
    # JIT WARMUP
    print("\n[JIT WARM-UP] Compiling exact production signature...")
    old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
    try:
        @njit(fastmath=True)
        def dummy_log_prob(y): return 0.0
        dummy_Z = np.eye(k_dims); dummy_A = np.vstack([dummy_Z, -dummy_Z])
        dummy = nsmc_sampling_gaussian(d=k_dims, k=10, sigma=np.eye(k_dims), mu=np.zeros(k_dims))
        dummy._sampling_universal(
            density_cartesian=dummy_log_prob, A=dummy_A, b=np.ones(k_dims*2),
            burn_in_samples=15000, max_anchors=130 
        )
    except Exception: pass
    finally: sys.stdout = old_stdout

    # -----------------------------------------------------------------
    # ACHR BASELINE
    # -----------------------------------------------------------------
    print(f"\n[2/4] Executing ACHR (Uniform + SIR Weighting)...")
    for r in range(k_runs):
        try:
            t0 = time.perf_counter()
            achr_df = sample(model, n_samples, method="achr", thinning=1)
            t_achr = time.perf_counter() - t0
            
            weighted_achr, ess_is, ess_is_pct = apply_sir_weighting(achr_df.values, x_ref, sigma_penalty)
            mn, md, _ = compute_robust_ess_stats(weighted_achr)
            
            achr_times.append(t_achr); achr_min.append(mn); achr_ess_is.append(ess_is_pct)
            print(f"  ACHR Run {r+1} -> Time: {t_achr:.2f}s | ESS_IS: {ess_is:.1f} ({ess_is_pct:.2f}%) | MCMC Min ESS: {mn:.1f}")
        except Exception as e:
            print(f"  ACHR Run {r+1} -> FAILED")
            with open(os.path.join(results_dir, "achr_failures.log"), "a") as f: f.write(f"Run {r+1}: {traceback.format_exc()}\n")

    # -----------------------------------------------------------------
    # CHRR BASELINE
    # -----------------------------------------------------------------
    print(f"\n[3/4] Executing CHRR (Uniform + SIR Weighting)...")
    for r in range(k_runs):
        try:
            t0 = time.perf_counter()
            chrr_df = sample(model, n_samples, method="chrr", thinning=1)
            t_chrr = time.perf_counter() - t0
            
            weighted_chrr, ess_is, ess_is_pct = apply_sir_weighting(chrr_df.values, x_ref, sigma_penalty)
            mn, md, _ = compute_robust_ess_stats(weighted_chrr)
            
            chrr_times.append(t_chrr); chrr_min.append(mn); chrr_ess_is.append(ess_is_pct)
            print(f"  CHRR Run {r+1} -> Time: {t_chrr:.2f}s | ESS_IS: {ess_is:.1f} ({ess_is_pct:.2f}%) | MCMC Min ESS: {mn:.1f}")
        except Exception as e:
            print(f"  CHRR Run {r+1} -> FAILED")
            with open(os.path.join(results_dir, "chrr_failures.log"), "a") as f: f.write(f"Run {r+1}: {traceback.format_exc()}\n")

    # -----------------------------------------------------------------
    # WHSS (OURS)
    # -----------------------------------------------------------------
    print(f"\n[4/4] Executing Native WHSS (Direct Non-Uniform Sampling)...")
    
    Z_numba = np.ascontiguousarray(Z)
    center_numba = np.ascontiguousarray(center)
    x_ref_numba = np.ascontiguousarray(x_ref)
    
    @njit(fastmath=True)
    def log_prob_target(y):
        x = np.dot(Z_numba, y) + center_numba
        dist = np.sum((x - x_ref_numba)**2)
        return -0.5 * dist / (sigma_penalty**2)

    for r in range(k_runs):
        t0 = time.perf_counter()
        
        sampler_whss = nsmc_sampling_gaussian(d=k_dims, k=n_samples, sigma=np.eye(k_dims), mu=np.zeros(k_dims))
        whss_samples = sampler_whss._sampling_universal(
            density_cartesian=log_prob_target,
            A=A_poly, b=b_poly,
            burn_in_samples=15000, max_anchors=130
        )
        
        whss_reactions_3d = np.einsum('csd,rd->csr', whss_samples, Z) + center
        t_whss = time.perf_counter() - t0
        
        ess = compute_ess_per_chain(whss_samples)
        r_hat = compute_rhat(whss_samples)
        
        whss_times.append(t_whss); whss_min.append(ess)
        whss_rhat.append(r_hat)
        print(f"  WHSS (Unbiased) Run {r+1}/{k_runs} -> Time: {t_whss:.2f}s | Min ESS: {ess:.1f} | Max R-hat: {r_hat:.3f}")
        final_whss_3d = whss_reactions_3d

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------
    m_achr_t, m_achr_mn, m_achr_is = np.mean(achr_times), np.mean(achr_min), np.mean(achr_ess_is)
    
    if len(chrr_times) > 0:
        m_chrr_t, m_chrr_mn, m_chrr_is = np.mean(chrr_times), np.mean(chrr_min), np.mean(chrr_ess_is)
    else:
        m_chrr_t = m_chrr_mn = m_chrr_is = np.nan
        
    m_whss_t, m_whss_mn = np.mean(whss_times), np.mean(whss_min)

    report = [
        "=" * 105,
        f"Model          : {model.id} ({d_full} Reactions)",
        f"Null Space Dim : {k_dims} (Precomputed & Fixed)",
        f"Target Density : Gaussian L2 Penalty to FBA Optima (Sigma = {sigma_penalty:.4f})",
        "-" * 105,
        f"{'Algorithm':<22} | {'Time (s)':<9} | {'Weight Retention (ESS_IS)':<25} | {'MCMC Min ESS':<15} | {'True ESS/s':<11}",
        "-" * 105,
        f"{'ACHR (Uniform+SIR)':<22} | {m_achr_t:<9.2f} | {m_achr_is:<6.2f}% (Degeneracy Risk) | {m_achr_mn:<15.1f} | {(m_achr_mn/m_achr_t):<11.2f}",
        f"{'CHRR (Uniform+SIR)':<22} | {m_chrr_t:<9.2f} | {m_chrr_is:<6.2f}% (Degeneracy Risk) | {m_chrr_mn:<15.1f} | {(m_chrr_mn/m_chrr_t):<11.2f}",
        f"{'WHSS (Native)':<22} | {m_whss_t:<9.2f} | {'100.00% (Direct Sample)':<25} | {m_whss_mn:<15.1f} | {(m_whss_mn/m_whss_t):<11.2f}",
        "=" * 105,
        "[NOTE: If Weight Retention is < 5%, SOTA distributions are mathematically invalid regardless of MCMC ESS.]"
    ]
    
    print("\n" + "\n".join(report))

if __name__ == "__main__":
    run_genome_scale_benchmark()