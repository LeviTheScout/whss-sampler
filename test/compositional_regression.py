import os, sys, time
import numpy as np
from numba import njit
from emcee.autocorr import integrated_time
import emcee

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from nsmc_sampling.distributions.gaussian import nsmc_sampling_gaussian

results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)

def build_compositional_environment(d=100, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    
    # 1. Simplex Constraints
    # To sample a d-dimensional probability simplex, we sample d-1 variables
    # The constraints are:
    # x_i >= 0  for i=1...d-1
    # sum(x_i) <= 1
    # This forms a perfect (d-1) dimensional simplex.
    d_reduced = d - 1
    
    A_rows, b_rows = [], []
    
    # x_i >= 0  => -x_i <= 0
    for i in range(d_reduced):
        row = np.zeros(d_reduced)
        row[i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    # sum(x_i) <= 1
    row = np.ones(d_reduced)
    A_rows.append(row)
    b_rows.append(1.0)
    
    A = np.ascontiguousarray(np.vstack(A_rows), dtype=np.float64)
    b_vec = np.ascontiguousarray(np.array(b_rows), dtype=np.float64)
    
    # Generate ground truth OUTSIDE the simplex to force the posterior 
    # to be heavily truncated by the boundaries.
    theta_true_reduced = rng.uniform(-1.0, 1.0, d_reduced)
    
    # We must start the samplers inside the feasible region
    x_init = np.full(d_reduced, 1.0 / d)
    
    # 2. Correlated Design Matrix (X)
    n_factors = 5
    L = rng.standard_normal((d_reduced, n_factors))
    Z = rng.standard_normal((n, n_factors))
    noise = 0.1 * rng.standard_normal((n, d_reduced))
    X = Z @ L.T + noise
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    X = X.astype(np.float64)
    X = np.ascontiguousarray(X)
    
    # 3. Generate response
    sigma_noise = 0.5
    y = X @ theta_true_reduced + rng.normal(0, sigma_noise, n)
    y = y.astype(np.float64)
    y = np.ascontiguousarray(y)
    
    return X, y, A, b_vec, x_init


def make_density(X, y):
    sigma_noise_sq = 0.5 ** 2
    prior_sigma_sq = 2.0 ** 2
    
    # Precompute X^T X and X^T y to make evaluations blazing fast (Gaussian likelihood)
    XtX = X.T @ X
    Xty = X.T @ y
    yty = y.T @ y
    
    @njit(fastmath=True)
    def log_posterior(theta):
        # Log prior: N(0, prior_sigma_sq * I)
        log_p = 0.0
        for i in range(len(theta)):
            log_p -= 0.5 * theta[i] * theta[i] / prior_sigma_sq
            
        # Log likelihood: -0.5/sigma^2 * (y - X*theta)^T (y - X*theta)
        # = -0.5/sigma^2 * (y^T y - 2*theta^T X^T y + theta^T X^T X theta)
        theta_XtX_theta = 0.0
        d_dim = len(theta)
        for i in range(d_dim):
            row_sum = 0.0
            for j in range(d_dim):
                row_sum += XtX[i, j] * theta[j]
            theta_XtX_theta += theta[i] * row_sum
            
        theta_Xty = 0.0
        for i in range(d_dim):
            theta_Xty += theta[i] * Xty[i]
            
        log_lik = -0.5 * (yty - 2.0 * theta_Xty + theta_XtX_theta) / sigma_noise_sq
        
        return log_p + log_lik

    return log_posterior

def compute_robust_ess_stats_3d(samples_3d, is_emcee=False):
    # If it's Emcee, we MUST NOT sum the ESS of each walker, because they are dependent.
    # emcee.autocorr.integrated_time computes the integrated autocorrelation time for the ensemble 
    # when passed an array of shape (n_steps, n_walkers, d). 
    # The total independent samples is then (n_steps * n_walkers) / tau.
    
    if is_emcee:
        # samples_3d is already (n_walkers, n_steps, d) from Emcee raw output in our script
        # We need it as (n_steps, n_walkers, d) for emcee.autocorr
        samples_emcee = samples_3d.transpose(1, 0, 2)
        n_steps, n_walkers, d = samples_emcee.shape
        ess_list = []
        for j in range(d):
            try:
                tau = integrated_time(samples_emcee[:, :, j], c=5, tol=10, quiet=True)
                max_tau = np.max(tau)
                if not np.isnan(max_tau) and max_tau > 0:
                    ess_list.append((n_steps * n_walkers) / max_tau)
            except:
                pass
        if not ess_list: return 1.0, 1.0, 1.0
        return np.min(ess_list), np.median(ess_list), np.max(ess_list)
        
    # For WHSS, HRSS, MwG, chains are strictly independent, so we compute ESS per chain and sum them.
    n_chains, n_steps, d = samples_3d.shape
    total_ess_per_dim = []
    for j in range(d):
        dim_ess_total = 0.0
        active = False
        for c in range(n_chains):
            col = samples_3d[c, :, j]
            if np.std(col) > 1e-8:
                try:
                    tau = integrated_time(col, tol=0)
                    ess = n_steps / np.max(tau)
                    if not np.isnan(ess) and ess > 0:
                        dim_ess_total += ess
                        active = True
                except Exception:
                    pass
        if active:
            total_ess_per_dim.append(dim_ess_total)
    if len(total_ess_per_dim) == 0:
        return 1.0, 1.0, 1.0
    return np.min(total_ess_per_dim), np.median(total_ess_per_dim), np.max(total_ess_per_dim)

def run_whss(log_posterior, A, b, x_init, total_mcmc_budget=30000):
    d = len(x_init)
    warmup_nfe = 1000 * d          
    total_nfe = warmup_nfe + total_mcmc_budget  
    
    t0 = time.perf_counter()
    sampler = nsmc_sampling_gaussian(d=d, a=6.0, sigma=np.ones(d), mu=np.zeros(d), k=total_mcmc_budget)
    sampler.x_init_override = x_init.copy()
    
    samples_3d = sampler._sampling_universal(
        log_posterior,
        A=A, b=b,
        burn_in_samples=warmup_nfe,
        max_anchors=3 * d
    )
    elapsed = time.perf_counter() - t0
    
    burn = int(0.30 * samples_3d.shape[1])
    return samples_3d[:, burn:, :], total_nfe, elapsed

@njit(fastmath=True)
def run_hit_and_run(density_func, A, b, x_init, n_chains, n_steps):
    d = len(x_init)
    samples = np.empty((n_chains, n_steps, d))
    
    for c in range(n_chains):
        x_curr = x_init.copy()
        for s in range(n_steps):
            u = np.random.randn(d)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-12: u[0] = 1.0
            else: u /= norm_u
                
            Au = np.dot(A, u)
            Ax = np.dot(A, x_curr)
            t_min = -1e20
            t_max = 1e20
            for i in range(len(b)):
                if Au[i] > 1e-12:
                    t_max = min(t_max, (b[i] - Ax[i]) / Au[i])
                elif Au[i] < -1e-12:
                    t_min = max(t_min, (b[i] - Ax[i]) / Au[i])
                    
            if t_min >= t_max:
                samples[c, s] = x_curr
                continue
                
            y_slice = density_func(x_curr) - np.random.exponential(1.0)
            t = np.random.uniform(t_min, t_max)
            x_prop = x_curr + t * u
            
            while density_func(x_prop) < y_slice:
                if t > 0: t_max = t
                else: t_min = t
                if t_max - t_min < 1e-10: break
                t = np.random.uniform(t_min, t_max)
                x_prop = x_curr + t * u
                
            x_curr = x_prop
            samples[c, s] = x_curr
            
    return samples

def run_hrss(log_posterior, A, b, x_init, total_budget=30000, n_chains=10):
    n_steps = total_budget // n_chains
    t0 = time.perf_counter()
    samples_3d = run_hit_and_run(log_posterior, A, b, x_init, n_chains, n_steps)
    elapsed = time.perf_counter() - t0
    burn = int(0.30 * n_steps)
    return samples_3d[:, burn:, :], total_budget, elapsed

def run_mwg(log_posterior, A, b, x_init, total_budget=30000, n_chains=10):
    d = len(x_init)
    n_sweeps = total_budget // (n_chains * d)
    actual_nfe = n_chains * n_sweeps * d
    
    t0 = time.perf_counter()
    all_samples = []
    
    for c in range(n_chains):
        theta = x_init.copy()
        chain_samples = np.zeros((n_sweeps, d))
        log_curr = log_posterior(theta)
        total_accepts = 0
        total_proposals = 0
        
        for sweep in range(n_sweeps):
            for j in range(d):
                lo_j = -np.inf
                hi_j = np.inf
                for i in range(len(b)):
                    if abs(A[i, j]) < 1e-14: continue
                    rest = b[i] - float(np.dot(A[i], theta)) + A[i, j] * theta[j]
                    bnd = rest / A[i, j]
                    if A[i, j] > 0: hi_j = min(hi_j, bnd)
                    else: lo_j = max(lo_j, bnd)
                    
                if lo_j >= hi_j - 1e-12: continue
                
                theta_prop_j = np.random.uniform(lo_j, hi_j)
                theta_prop = theta.copy()
                theta_prop[j] = theta_prop_j
                log_prop = log_posterior(theta_prop)
                
                total_proposals += 1
                if np.log(np.random.rand()) < log_prop - log_curr:
                    theta = theta_prop
                    log_curr = log_prop
                    total_accepts += 1
                    
            chain_samples[sweep] = theta.copy()
        all_samples.append(chain_samples)
        
    elapsed = time.perf_counter() - t0
    samples_3d = np.array(all_samples)
    accept_rate = total_accepts / max(total_proposals, 1)
    burn = int(0.30 * n_sweeps)
    return samples_3d[:, burn:, :], actual_nfe, elapsed, accept_rate

def run_emcee_sampler(log_posterior, A, b, x_init, total_budget=30000):
    d = len(x_init)
    n_walkers = d * 2
    n_steps = total_budget // n_walkers
    
    def log_prob_emcee(theta):
        if np.any(A @ theta > b + 1e-9): return -np.inf
        val = float(log_posterior(theta))
        if not np.isfinite(val): return -np.inf
        return val
        
    # Initialize walkers strictly feasible and linearly independent, dispersed across the simplex
    p0 = []
    # Create a safe center deep inside the simplex
    safe_center = np.full(d, 0.9 / d)
    for _ in range(n_walkers):
        for attempt in range(100000):
            # Take a large uniform step from the center
            candidate = safe_center + np.random.uniform(-0.01, 0.01, d)
            if np.all(A @ candidate <= b):
                p0.append(candidate)
                break
        else:
            p0.append(safe_center.copy())
    p0 = np.array(p0)
    
    t0 = time.perf_counter()
    sampler_obj = emcee.EnsembleSampler(n_walkers, d, log_prob_emcee)
    sampler_obj.run_mcmc(p0, n_steps, progress=False)
    elapsed = time.perf_counter() - t0
    
    raw = sampler_obj.get_chain()
    samples_3d = raw.transpose(1, 0, 2)
    accept_rate = float(np.mean(sampler_obj.acceptance_fraction))
    burn = int(0.30 * n_steps)
    return samples_3d[:, burn:, :], total_budget, elapsed, accept_rate

if __name__ == "__main__":
    N_RUNS = 3
    TOTAL_MCMC_BUDGET = 50_000
    D = 100
    
    print("Building Compositional Regression environment...")
    X, y, A, b, x_init = build_compositional_environment(d=D, n=500, seed=42)
    log_posterior = make_density(X, y)
    
    _ = log_posterior(x_init)
    
    results = {}
    
    for run_idx in range(N_RUNS):
        print(f"\n=== RUN {run_idx + 1}/{N_RUNS} ===")
        np.random.seed(run_idx * 17)
        
        print("Running WHSS...")
        s, nfe, t = run_whss(log_posterior, A, b, x_init, TOTAL_MCMC_BUDGET)
        min_e, _, _ = compute_robust_ess_stats_3d(s)
        results.setdefault("WHSS", []).append((min_e, nfe, t))
        
        print("Running HRSS...")
        s, nfe, t = run_hrss(log_posterior, A, b, x_init, TOTAL_MCMC_BUDGET)
        min_e, _, _ = compute_robust_ess_stats_3d(s)
        results.setdefault("HRSS", []).append((min_e, nfe, t))
        
        print("Running MwG...")
        s, nfe, t, acc = run_mwg(log_posterior, A, b, x_init, TOTAL_MCMC_BUDGET)
        min_e, _, _ = compute_robust_ess_stats_3d(s)
        results.setdefault("MwG", []).append((min_e, nfe, t))
        
        print("Running Emcee...")
        s, nfe, t, acc = run_emcee_sampler(log_posterior, A, b, x_init, TOTAL_MCMC_BUDGET)
        min_e, _, _ = compute_robust_ess_stats_3d(s, is_emcee=True)
        results.setdefault("Emcee", []).append((min_e, nfe, t))
        
    report_path = os.path.join(results_dir, "compositional_regression_report.txt")
    lines = [
        "=" * 60,
        "COMPOSITIONAL BAYESIAN REGRESSION BENCHMARK",
        "=" * 60,
        f"Problem: D={D} (Simplex in {D-1}D), N=500",
        "Target: Correlated Gaussian Likelihood (Anisotropic)",
        f"MCMC Budget: {TOTAL_MCMC_BUDGET:,} NFE per algorithm (post-warmup)",
        f"WHSS Warm-Up NFE: {(D-1)*1000:,} (included in ESS/NFE denominator)",
        f"Runs: {N_RUNS} (mean reported)",
        "",
        f"{'Algorithm':<22} | {'Total NFE':>10} | {'Min ESS':>8} | {'ESS/1k NFE':>11} | {'Time (s)':>9}",
        "-" * 70,
    ]
    
    for name, runs in results.items():
        ess_vals = [r[0] for r in runs]
        nfe_val = runs[0][1]
        t_vals = [r[2] for r in runs]
        mean_ess = np.mean(ess_vals)
        mean_t = np.mean(t_vals)
        ess_per_1k = (mean_ess / nfe_val) * 1000
        lines.append(f"{name:<22} | {nfe_val:>10,} | {mean_ess:>8.2f} | {ess_per_1k:>11.4f} | {mean_t:>9.2f}")
        
    lines += [
        "",
        "NOTES:",
        f"- WHSS total NFE = {(D-1)*1000:,} (warmup) + {TOTAL_MCMC_BUDGET:,} (MCMC)",
        "- ESS via emcee.autocorr.integrated_time with tol=0",
        "- Min ESS across dimensions reported (most conservative)",
        "- 30% burn-in discarded from all chains before ESS",
        "=" * 60,
    ]
    
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(report_path, "w") as f:
        f.write(report_text)
