import os
import sys
import time
import numpy as np
import emcee
from numba import njit

# =====================================================================
# PATH RESOLUTION: ALLOW IMPORTING FROM NATIVE CODEBASE
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

from whss.distributions.gaussian import whss_gaussian

# =====================================================================
# DIRECTORY SETUP
# =====================================================================
results_dir = os.path.join(current_dir, "results")
diag_dir = os.path.join(current_dir, "diagnostics")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)

# =====================================================================
# DIAGNOSTIC TOOLS
# =====================================================================
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
                tau = integrated_time(samples[c], tol=0)
                ess = n_steps / np.max(tau)
                chain_ess.append(ess)
            except Exception:
                chain_ess.append(1.0)
        return np.sum(chain_ess)
    elif samples.ndim == 2:
        n_steps, d = samples.shape
        try:
            tau = integrated_time(samples, tol=0)
            return n_steps / np.max(tau)
        except Exception:
            return 1.0
    return 1.0

def compute_gelman_rubin(chains_3d):
    """
    Computes Gelman-Rubin R-hat across parallel chains.
    """
    if chains_3d.ndim != 3 or chains_3d.shape[0] < 2:
        return 1.0
    
    m, n, d = chains_3d.shape
    chain_means = np.mean(chains_3d, axis=1)
    chain_vars = np.var(chains_3d, axis=1, ddof=1)
    
    B = (n / (m - 1)) * np.sum((chain_means - np.mean(chain_means, axis=0))**2, axis=0)
    W = np.mean(chain_vars, axis=0)
    
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    r_hat = np.sqrt(var_hat / np.maximum(W, 1e-12))
    return np.max(r_hat)

def run_silent_jit_warmup(sampler, density_func, A=None, b=None):
    """
    Forces Numba to compile C-code before the timer starts.
    Redirects stdout to devnull so the terminal stays perfectly clean.
    """
    print("  [JIT WARM-UP] Compiling WHSS (Untimed)... This may take a few seconds.")
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        _ = sampler._sampling_universal(density_cartesian=density_func, A=A, b=b)
    finally:
        sys.stdout = old_stdout
        devnull.close()

# =====================================================================
# UNIFIED BENCHMARK SUITE
# =====================================================================
def run_all_benchmarks():
    report = []
    report.append("=" * 80)
    report.append("WHSS NATIVE CODEBASE EMPIRICAL VALIDATION REPORT")
    report.append("=" * 80 + "\n")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: 20D NEAL'S FUNNEL (VARYING CURVATURE)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 1: 20D NEAL'S FUNNEL (NFE BUDGET: 500,000)")
    print("="*70)
    
    d_funnel = 20
    k_funnel = 400000
    
    @njit
    def funnel_log_prob(x):
        v = x[0]
        if v < -15.0 or v > 15.0:
            return -np.inf
        log_prior = -0.5 * (v**2) / 9.0
        var = np.exp(v)
        log_lik = -0.5 * np.sum(x[1:]**2) / var - 0.5 * (d_funnel - 1) * v
        return log_prior + log_lik

    n_walkers = 40
    steps_per_walker = 12500
    emcee_sampler = emcee.EnsembleSampler(n_walkers, d_funnel, funnel_log_prob)
    p0 = np.random.randn(n_walkers, d_funnel) * 0.1
    p0[:, 0] = np.random.normal(0, 3.0, size=n_walkers)
    
    t0 = time.perf_counter()
    emcee_sampler.run_mcmc(p0, steps_per_walker, progress=True)
    t_emcee_funnel = time.perf_counter() - t0
    
    chain_flat = emcee_sampler.get_chain(discard=2500, flat=True)
    try:
        tau_emcee = emcee_sampler.get_autocorr_time(discard=2500, tol=0)
        ess_emcee_funnel = chain_flat.shape[0] / np.max(tau_emcee)
    except Exception:
        ess_emcee_funnel = compute_ess_per_chain(chain_flat)

    # Numba JIT Warmup (Untimed)
    sampler_whss_dummy = whss_gaussian(d=d_funnel, k=10, sigma=np.eye(d_funnel), mu=np.zeros(d_funnel))
    run_silent_jit_warmup(sampler_whss_dummy, funnel_log_prob)

    # True Timed Benchmark
    sampler_whss = whss_gaussian(d=d_funnel, k=k_funnel, sigma=np.eye(d_funnel), mu=np.zeros(d_funnel))
    t0 = time.perf_counter()
    whss_funnel_samples = sampler_whss._sampling_universal(density_cartesian=funnel_log_prob, A=None, b=None)
    t_whss_funnel = time.perf_counter() - t0
    
    ess_whss_funnel = compute_ess_per_chain(whss_funnel_samples)
    r_hat_funnel = compute_gelman_rubin(whss_funnel_samples)
    np.save(os.path.join(diag_dir, "exp1_funnel_samples.npy"), whss_funnel_samples)
    
    report.append("[EXPERIMENT 1: 20D NEAL'S FUNNEL (VARYING CURVATURE)]")
    report.append(f"{'Algorithm':<22} | {'Time (s)':<10} | {'Min ESS':<10} | {'ESS/s':<10} | {'R-hat':<10}")
    report.append("-" * 75)
    report.append(f"{'emcee (Ensemble)':<22} | {t_emcee_funnel:<10.2f} | {ess_emcee_funnel:<10.1f} | {(ess_emcee_funnel/t_emcee_funnel):<10.2f} | {'N/A':<10}")
    report.append(f"{'WHSS (Native Codebase)':<22} | {t_whss_funnel:<10.2f} | {ess_whss_funnel:<10.1f} | {(ess_whss_funnel/t_whss_funnel):<10.2f} | {r_hat_funnel:<10.4f}\n")

    # -----------------------------------------------------------------
    # EXPERIMENT 2: 20D ROSENBROCK (BANANA-SHAPED CURVATURE)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 2: 20D ROSENBROCK (NFE BUDGET: 500,000)")
    print("="*70)
    
    d_rosen = 20
    k_rosen = 400000
    
    @njit
    def rosenbrock_log_prob(x):
        for i in range(d_rosen):
            if x[i] < -10.0 or x[i] > 10.0:
                return -np.inf
        term1 = 100.0 * (x[1:] - x[:-1]**2)**2
        term2 = (1.0 - x[:-1])**2
        return -np.sum(term1 + term2) / 20.0 

    sampler_emcee_rosen = emcee.EnsembleSampler(n_walkers, d_rosen, rosenbrock_log_prob)
    p0_rosen = np.random.randn(n_walkers, d_rosen) * 0.1 + 1.0 
    
    t0 = time.perf_counter()
    sampler_emcee_rosen.run_mcmc(p0_rosen, steps_per_walker, progress=True)
    t_emcee_rosen = time.perf_counter() - t0
    
    chain_flat_rosen = sampler_emcee_rosen.get_chain(discard=2500, flat=True)
    try:
        tau_emcee_rosen = sampler_emcee_rosen.get_autocorr_time(discard=2500, tol=0)
        ess_emcee_rosen = chain_flat_rosen.shape[0] / np.max(tau_emcee_rosen)
    except Exception:
        ess_emcee_rosen = compute_ess_per_chain(chain_flat_rosen)

    # Numba JIT Warmup (Untimed)
    sampler_whss_dummy_rosen = whss_gaussian(d=d_rosen, k=10, sigma=np.eye(d_rosen), mu=np.ones(d_rosen))
    run_silent_jit_warmup(sampler_whss_dummy_rosen, rosenbrock_log_prob)

    # True Timed Benchmark
    sampler_whss_rosen = whss_gaussian(d=d_rosen, k=k_rosen, sigma=np.eye(d_rosen), mu=np.ones(d_rosen))
    t0 = time.perf_counter()
    whss_rosen_samples = sampler_whss_rosen._sampling_universal(density_cartesian=rosenbrock_log_prob, A=None, b=None)
    t_whss_rosen = time.perf_counter() - t0
    
    ess_whss_rosen = compute_ess_per_chain(whss_rosen_samples)
    r_hat_rosen = compute_gelman_rubin(whss_rosen_samples)
    np.save(os.path.join(diag_dir, "exp2_rosenbrock_samples.npy"), whss_rosen_samples)
    
    report.append("[EXPERIMENT 2: 20D ROSENBROCK FUNCTION (NON-LINEAR CURVATURE)]")
    report.append(f"{'Algorithm':<22} | {'Time (s)':<10} | {'Min ESS':<10} | {'ESS/s':<10} | {'R-hat':<10}")
    report.append("-" * 75)
    report.append(f"{'emcee (Ensemble)':<22} | {t_emcee_rosen:<10.2f} | {ess_emcee_rosen:<10.1f} | {(ess_emcee_rosen/t_emcee_rosen):<10.2f} | {'N/A':<10}")
    report.append(f"{'WHSS (Native Codebase)':<22} | {t_whss_rosen:<10.2f} | {ess_whss_rosen:<10.1f} | {(ess_whss_rosen/t_whss_rosen):<10.2f} | {r_hat_rosen:<10.4f}\n")

    # -----------------------------------------------------------------
    # EXPERIMENT 3: 30D PATHOLOGICAL POLYTOPE (STRICT CONSTRAINTS)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 3: 30D PATHOLOGICAL POLYTOPE (Ax <= b, 500:1 SKEW)")
    print("="*70)
    
    d_poly = 30
    k_poly = 300000
    
    cov_poly = np.eye(d_poly)
    cov_poly[0, 0] = 500.0
    inv_cov_poly = np.linalg.inv(cov_poly)
    
    A_poly = np.vstack([np.eye(d_poly), -np.eye(d_poly)])
    b_upper = np.full(d_poly, 0.1); b_upper[0] = 50.0
    b_lower = np.full(d_poly, 0.1); b_lower[0] = 50.0
    b_poly = np.concatenate([b_upper, b_lower])

    def poly_log_prob(x):
        if np.any(A_poly @ x > b_poly):
            return -np.inf
        return -0.5 * x.T @ inv_cov_poly @ x

    n_walkers_poly = 60
    steps_poly = 5000
    sampler_emcee_poly = emcee.EnsembleSampler(n_walkers_poly, d_poly, poly_log_prob)
    p0_poly = np.random.uniform(-0.05, 0.05, size=(n_walkers_poly, d_poly))
    
    t0 = time.perf_counter()
    sampler_emcee_poly.run_mcmc(p0_poly, steps_poly, progress=True)
    t_emcee_poly = time.perf_counter() - t0
    
    try:
        tau_emcee_p = sampler_emcee_poly.get_autocorr_time(discard=1000, tol=0)
        ess_emcee_poly = (n_walkers_poly * (steps_poly - 1000)) / np.max(tau_emcee_p)
    except Exception:
        ess_emcee_poly = np.nan

    # Numba JIT Warmup (Untimed)
    sampler_whss_dummy_poly = whss_gaussian(d=d_poly, k=10, sigma=cov_poly, mu=np.zeros(d_poly))
    run_silent_jit_warmup(sampler_whss_dummy_poly, sampler_whss_dummy_poly.f_cartesian_gaussian(), A=A_poly, b=b_poly)

    # True Timed Benchmark
    sampler_whss_poly = whss_gaussian(d=d_poly, k=k_poly, sigma=cov_poly, mu=np.zeros(d_poly))
    t0 = time.perf_counter()
    whss_poly_samples = sampler_whss_poly._sampling_universal(
        density_cartesian=sampler_whss_poly.f_cartesian_gaussian(), A=A_poly, b=b_poly
    )
    t_whss_poly = time.perf_counter() - t0
    ess_whss_poly = compute_ess_per_chain(whss_poly_samples)
    np.save(os.path.join(diag_dir, "exp3_poly30d_samples.npy"), whss_poly_samples)
    
    ess_sec_emcee = (ess_emcee_poly / t_emcee_poly) if not np.isnan(ess_emcee_poly) else np.nan
    report.append("[EXPERIMENT 3: 30D PATHOLOGICAL POLYTOPE (LINEAR CONSTRAINTS)]")
    report.append(f"{'Algorithm':<22} | {'Time (s)':<10} | {'Min ESS':<10} | {'ESS/s':<10}")
    report.append("-" * 60)
    report.append(f"{'emcee (Ensemble)':<22} | {t_emcee_poly:<10.2f} | {ess_emcee_poly:<10.1f} | {ess_sec_emcee:<10.2f}")
    report.append(f"{'WHSS (Native Codebase)':<22} | {t_whss_poly:<10.2f} | {ess_whss_poly:<10.1f} | {(ess_whss_poly/t_whss_poly):<10.2f}\n")

    # -----------------------------------------------------------------
    # EXPERIMENT 4: 100D SCALABILITY TEST
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 4: 100D PATHOLOGICAL POLYTOPE SCALABILITY")
    print("="*70)
    
    d_100 = 100
    k_100 = 500000
    cov_100 = np.eye(d_100); cov_100[0, 0] = 500.0
    A_100 = np.vstack([np.eye(d_100), -np.eye(d_100)])
    b_upper_100 = np.full(d_100, 0.1); b_upper_100[0] = 50.0
    b_lower_100 = np.full(d_100, 0.1); b_lower_100[0] = 50.0
    b_100 = np.concatenate([b_upper_100, b_lower_100])

    # Numba JIT Warmup (Untimed)
    sampler_whss_dummy_100 = whss_gaussian(d=d_100, k=10, sigma=cov_100, mu=np.zeros(d_100))
    run_silent_jit_warmup(sampler_whss_dummy_100, sampler_whss_dummy_100.f_cartesian_gaussian(), A=A_100, b=b_100)

    # True Timed Benchmark
    sampler_whss_100 = whss_gaussian(d=d_100, k=k_100, sigma=cov_100, mu=np.zeros(d_100))
    t0 = time.perf_counter()
    whss_100_samples = sampler_whss_100._sampling_universal(
        density_cartesian=sampler_whss_100.f_cartesian_gaussian(), A=A_100, b=b_100
    )
    t_whss_100 = time.perf_counter() - t0
    ess_whss_100 = compute_ess_per_chain(whss_100_samples)
    np.save(os.path.join(diag_dir, "exp4_poly100d_samples.npy"), whss_100_samples)
    
    report.append("[EXPERIMENT 4: 100D PATHOLOGICAL POLYTOPE SCALABILITY]")
    report.append(f"{'Algorithm':<22} | {'Time (s)':<10} | {'Min ESS':<10} | {'ESS/s':<10}")
    report.append("-" * 60)
    report.append(f"{'WHSS (Native Codebase)':<22} | {t_whss_100:<10.2f} | {ess_whss_100:<10.1f} | {(ess_whss_100/t_whss_100):<10.2f}\n")

    # =================================================================
    # WRITE REPORT
    # =================================================================
    final_output = "\n".join(report)
    print("\n" + final_output)
    
    report_path = os.path.join(results_dir, "publication_report.txt")
    with open(report_path, "w") as f:
        f.write(final_output)
        
    print(f"\n[COMPLETE] Report saved to {report_path}")
    print(f"[COMPLETE] Arrays saved to {diag_dir}/")

if __name__ == "__main__":
    run_all_benchmarks()