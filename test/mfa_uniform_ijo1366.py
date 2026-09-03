import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import cobra
from cobra.sampling import sample
from scipy.linalg import svd
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

def compute_ess_per_chain(samples):
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
                max_tau = np.nanmax(tau)
                if np.isnan(max_tau) or max_tau <= 0:
                    chain_ess.append(1.0)
                else:
                    chain_ess.append(n_steps / max_tau)
            except:
                chain_ess.append(1.0)
        return np.sum(chain_ess)
    else:
        try:
            tau = integrated_time(samples, c=5, tol=10, quiet=True)
            max_tau = np.nanmax(tau)
            if np.isnan(max_tau) or max_tau <= 0: return 1.0
            return len(samples) / max_tau
        except:
            return 1.0

def compute_rhat(samples):
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

def generate_pure_lp_basis(model):
    from cobra.sampling import sample
    
    warmup_df = sample(model, 2500, method="achr", thinning=1)
    points = warmup_df.values
    center = np.mean(points, axis=0)
    centered = points - center
    
    _, S_sing, Vt = svd(centered, full_matrices=False)
    Z = Vt[S_sing > 1e-8].T
    return center, Z, points

@njit(fastmath=True)
def uniform_poly(y):
    return 0.0

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def run_genome_scale_uniform_benchmark():
    np.random.seed(42)
    k_runs = 1
    n_samples = 150000 
    
    print("=" * 85)
    print(f"GENOME-SCALE UNIFORM BENCHMARK (Averaged over {k_runs} Runs)")
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
    
    print("  -> Running volumetric warmup to determine exact static null space...")
    center, Z, lp_points = generate_pure_lp_basis(model)
    k_dims = Z.shape[1] 
    
    lb = np.array([rxn.lower_bound for rxn in model.reactions])
    ub = np.array([rxn.upper_bound for rxn in model.reactions])
    A_poly = np.vstack([Z, -Z])
    b_poly = np.concatenate([ub - center, center - lb])
    b_poly = np.maximum(b_poly, 0.0) 
    
    print(f"  -> Fixed Null Space Dimensions: {k_dims}")
    print(f"  -> Target Density             : UNIFORM (Flat Polytope)\n")

    achr_times, achr_min = [], []
    chrr_times, chrr_min = [], []
    whss_times, whss_min = [], []
    whss_rhat = []
    
    print("[JIT WARM-UP] Compiling exact production signature...")
    old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
    try:
        dummy_Z = np.eye(k_dims); dummy_A = np.vstack([dummy_Z, -dummy_Z])
        dummy = nsmc_sampling_gaussian(d=k_dims, k=10, sigma=np.eye(k_dims), mu=np.zeros(k_dims))
        dummy._sampling_universal(
            density_cartesian=uniform_poly, A=dummy_A, b=np.ones(k_dims*2),
            burn_in_samples=5000, max_anchors=50 
        )
    except Exception: pass
    finally: sys.stdout = old_stdout

    print(f"\n[2/4] Executing ACHR (Uniform Baseline)...")
    for r in range(k_runs):
        try:
            t0 = time.perf_counter()
            achr_df = sample(model, n_samples, method="achr", thinning=1)
            t_achr = time.perf_counter() - t0
            
            # Unbiased: Discard 50% burn-in to match WHSS
            achr_stationary = achr_df.values[len(achr_df)//2:]
            mn, md, _ = compute_robust_ess_stats(achr_stationary)
            achr_times.append(t_achr); achr_min.append(mn)
            print(f"  ACHR Run {r+1} -> Time: {t_achr:.2f}s | MCMC Min ESS: {mn:.1f}")
        except Exception as e:
            print(f"  ACHR Run {r+1} -> FAILED")

    print(f"\n[3/4] Executing CHRR (Uniform SOTA)...")
    for r in range(k_runs):
        try:
            t0 = time.perf_counter()
            chrr_df = sample(model, n_samples, method="chrr", thinning=1)
            t_chrr = time.perf_counter() - t0
            
            # Unbiased: Discard 50% burn-in to match WHSS
            chrr_stationary = chrr_df.values[len(chrr_df)//2:]
            mn, md, _ = compute_robust_ess_stats(chrr_stationary)
            chrr_times.append(t_chrr); chrr_min.append(mn)
            print(f"  CHRR Run {r+1} -> Time: {t_chrr:.2f}s | MCMC Min ESS: {mn:.1f}")
        except Exception as e:
            print(f"  CHRR Run {r+1} -> FAILED")

    print(f"\n[4/4] Executing Native WHSS (Uniform Sampling)...")
    
    for r in range(k_runs):
        t0 = time.perf_counter()
        
        # Native WHSS for uniform space (no L_override needed, uses native ray-shooting)
        sampler_whss = nsmc_sampling_gaussian(d=k_dims, k=n_samples, sigma=np.eye(k_dims), mu=np.zeros(k_dims))
        whss_samples = sampler_whss._sampling_universal(
            density_cartesian=uniform_poly,
            A=A_poly, b=b_poly,
            burn_in_samples=15000, max_anchors=130
        )
        
        t_whss = time.perf_counter() - t0
        
        n_steps = whss_samples.shape[1]
        whss_stationary = whss_samples[:, (n_steps // 2):, :]
        
        ess = compute_ess_per_chain(whss_stationary)
        r_hat = compute_rhat(whss_stationary)
        
        whss_times.append(t_whss); whss_min.append(ess)
        whss_rhat.append(r_hat)
        print(f"  WHSS Run {r+1}/{k_runs} -> Time: {t_whss:.2f}s | Min ESS: {ess:.1f} | Max R-hat: {r_hat:.3f}")

    m_achr_t, m_achr_mn = np.mean(achr_times), np.mean(achr_min)
    if len(chrr_times) > 0:
        m_chrr_t, m_chrr_mn = np.mean(chrr_times), np.mean(chrr_min)
    else:
        m_chrr_t = m_chrr_mn = np.nan
        
    m_whss_t, m_whss_mn = np.mean(whss_times), np.mean(whss_min)

    eff_achr = m_achr_mn / (n_samples / 1000.0)
    eff_chrr = m_chrr_mn / (n_samples / 1000.0) if not np.isnan(m_chrr_mn) else np.nan
    eff_whss = m_whss_mn / (n_samples / 1000.0)
    
    report = [
        "=" * 95,
        f"Model          : {model.id} ({d_full} Reactions -> {k_dims} Active Flux Dimensions)",
        f"Null Space Dim : {k_dims} (Precomputed & Fixed)",
        f"Target Density : UNIFORM (No Penalty)",
        "-" * 95,
        f"{'Algorithm':<22} | {'Time (s)':<9} | {'MCMC Min ESS':<15} | {'ESS/1k NFE':<11}",
        "-" * 95,
        f"{'ACHR (Uniform)':<22} | {m_achr_t:<9.2f} | {m_achr_mn:<15.1f} | {eff_achr:<11.2f}",
        f"{'CHRR (Uniform SOTA)':<22} | {m_chrr_t:<9.2f} | {m_chrr_mn:<15.1f} | {eff_chrr:<11.2f}",
        f"{'WHSS (Native)':<22} | {m_whss_t:<9.2f} | {m_whss_mn:<15.1f} | {eff_whss:<11.2f}",
        "=" * 95
    ]
    final_output = "\n" + "\n".join(report)
    print(final_output)
    
    report_path = os.path.join(results_dir, "mfa_uniform_ijo1366_report.txt")
    with open(report_path, "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    run_genome_scale_uniform_benchmark()