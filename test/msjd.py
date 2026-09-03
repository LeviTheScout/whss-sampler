import os
import sys
import numpy as np
import emcee
import matplotlib.pyplot as plt
from numba import njit

# =====================================================================
# PATH RESOLUTION & SETUP
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from nsmc_sampling.distributions.gaussian import nsmc_sampling_gaussian

plots_dir = os.path.join(current_dir, "results")
os.makedirs(plots_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "lines.linewidth": 3.5,
    "lines.markersize": 10,
    "figure.figsize": (10, 7),
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# =====================================================================
# FINANCE ENVIRONMENT SETUP
# =====================================================================
def build_finance_environment(n_assets):
    np.random.seed(42)
    n_scenarios = 1000
    n_factors = 3
    
    factor_loadings = np.random.randn(n_assets, n_factors)
    factor_returns = np.random.randn(n_scenarios, n_factors) * 0.03
    idiosyncratic = np.random.randn(n_scenarios, n_assets) * 0.01
    
    d_sub = n_assets - 1
    A_rows, b_rows = [], []
    
    for i in range(d_sub):
        e_i = np.zeros(d_sub)
        e_i[i] = 1.0
        A_rows.append(-e_i); b_rows.append(0.0)      # w_i >= 0
        A_rows.append(e_i);  b_rows.append(0.15)     # w_i <= 0.15
        
    ones = np.ones(d_sub)
    A_rows.append(ones);  b_rows.append(1.0)         # w_d >= 0
    A_rows.append(-ones); b_rows.append(-(1.0 - 0.15)) # w_d <= 0.15
    
    tech_mask = np.zeros(d_sub)
    tech_mask[:min(5, d_sub)] = 1.0
    A_rows.append(tech_mask); b_rows.append(0.30)
    
    A_poly = np.ascontiguousarray(np.vstack(A_rows))
    b_poly = np.ascontiguousarray(np.array(b_rows))
    
    # --- NEW: Build the true Risk Covariance Matrix ---
    factor_cov = np.cov(factor_returns, rowvar=False)
    idio_var = np.var(idiosyncratic, axis=0)
    full_cov = factor_loadings @ factor_cov @ factor_loadings.T + np.diag(idio_var)
    
    # Project the full covariance into the (n_assets - 1) subspace
    # since w_last = 1 - sum(w_sub)
    P = np.vstack([np.eye(d_sub), -np.ones(d_sub)])
    risk_matrix = P.T @ full_cov @ P
    risk_matrix = np.ascontiguousarray(risk_matrix)
    
    return d_sub, A_poly, b_poly, risk_matrix

# =====================================================================
# SOTA BASELINES (HRSS & Dikin)
# =====================================================================
@njit(fastmath=True)
def run_hit_and_run(density_func, A, b, x_init, n_chains, n_steps):
    d = len(x_init)
    samples = np.empty((n_chains, n_steps, d))
    for c in range(n_chains):
        x_curr = x_init.copy()
        for s in range(n_steps):
            u = np.random.randn(d)
            u /= np.linalg.norm(u)
            Au, Ax = np.dot(A, u), np.dot(A, x_curr)
            t_min, t_max = -1e20, 1e20
            for i in range(len(b)):
                if Au[i] > 1e-12: t_max = min(t_max, (b[i] - Ax[i]) / Au[i])
                elif Au[i] < -1e-12: t_min = max(t_min, (b[i] - Ax[i]) / Au[i])
            if t_min >= t_max:
                samples[c, s] = x_curr; continue
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

@njit(fastmath=True)
def run_dikin_walk(density_func, A, b, x_init, n_chains, n_steps, r_step=0.15):
    d, m = len(x_init), len(b)
    samples = np.empty((n_chains, n_steps, d))
    for c in range(n_chains):
        x_curr = x_init.copy()
        for s in range(n_steps):
            slack_x = np.maximum(b - np.dot(A, x_curr), 1e-10)
            H_x = np.zeros((d, d))
            for i in range(m):
                row = A[i] / slack_x[i]
                for j in range(d):
                    for k in range(d): H_x[j, k] += row[j] * row[k]
            for j in range(d): H_x[j, j] += 1e-6
            L_x = np.linalg.cholesky(H_x)
            z = np.random.randn(d)
            x_prop = x_curr + np.linalg.solve(L_x.T, z) * r_step
            
            slack_prop = b - np.dot(A, x_prop)
            feasible = True
            for i in range(m):
                if slack_prop[i] <= 0.0: feasible = False; break
            if not feasible:
                samples[c, s] = x_curr; continue
                
            H_p = np.zeros((d, d))
            for i in range(m):
                row = A[i] / np.maximum(slack_prop, 1e-10)[i]
                for j in range(d):
                    for k in range(d): H_p[j, k] += row[j] * row[k]
            for j in range(d): H_p[j, j] += 1e-6
            L_p = np.linalg.cholesky(H_p)
            
            log_alpha = (density_func(x_prop) - density_func(x_curr)) + \
                        (np.sum(np.log(np.diag(L_p))) - np.sum(np.log(np.diag(L_x)))) - \
                        (np.dot((x_curr - x_prop), np.dot(H_p - H_x, (x_curr - x_prop))) / (2.0 * r_step**2))
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha: x_curr = x_prop
            samples[c, s] = x_curr
    return samples

# =====================================================================
# DIAGNOSTICS & EXECUTION
# =====================================================================
def compute_msjd(samples_3d):
    diffs = np.diff(samples_3d, axis=1)
    squared_jumps = np.sum(diffs**2, axis=-1)
    return np.mean(squared_jumps)

def run_silent_jit_warmup(sampler, density_func, A, b):
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        _ = sampler._sampling_universal(density_cartesian=density_func, A=A, b=b)
    finally:
        sys.stdout = old_stdout
        devnull.close()

def run_msjd_scaling():
    dimensions = [10, 20, 30, 40]
    n_samples = 15000
    k_runs = 10
    
    msjd_whss_mu, msjd_whss_std = [], []
    msjd_hrss_mu, msjd_hrss_std = [], []
    msjd_dikin_mu, msjd_dikin_std = [], []
    msjd_emcee_mu, msjd_emcee_std = [], []

    print("=" * 80)
    print(f"EXECUTING MSJD MOBILITY BENCHMARK (Averaged over {k_runs} Runs)")
    print("=" * 80)

    for n_assets in dimensions:
        print(f"\n[Testing Portfolio Size: {n_assets} Assets]")
        d_sub, A_poly, b_poly, risk_matrix = build_finance_environment(n_assets)
        
        w_center = np.full(d_sub, 1.0 / n_assets)
        b_shifted = np.maximum(b_poly - A_poly @ w_center, 1e-7)
        
        @njit
        def log_prob_shifted(w_sub):
            for i in range(len(b_shifted)):
                val = 0.0
                for j in range(len(w_sub)): val += A_poly[i, j] * w_sub[j]
                if val > b_shifted[i]: return -np.inf
                
            # --- NEW: Non-Uniform Skewed Risk Target ---
            risk = 0.0
            for i in range(len(w_sub)):
                for j in range(len(w_sub)):
                    risk += w_sub[i] * risk_matrix[i, j] * w_sub[j]
            # Multiply by a scalar to make the risk landscape steep
            return -500.0 * risk

        # JIT WARMUP for current dimension
        print("  [JIT] Warming up compilers...")
        dummy = nsmc_sampling_gaussian(d=d_sub, k=10, sigma=np.eye(d_sub), mu=np.zeros(d_sub))
        run_silent_jit_warmup(dummy, log_prob_shifted, A_poly, b_shifted)

        w_runs, h_runs, d_runs, e_runs = [], [], [], []

        for r in range(k_runs):
            print(f"  --- Run {r+1}/{k_runs} ---", end="\r")
            
            # 1. WHSS
            sampler_whss = nsmc_sampling_gaussian(d=d_sub, k=n_samples, sigma=np.eye(d_sub), mu=np.zeros(d_sub))
            whss_samples = sampler_whss._sampling_universal(
                density_cartesian=log_prob_shifted, A=A_poly, b=b_shifted,
                burn_in_samples=2500, max_anchors=60
            )
            w_runs.append(compute_msjd(whss_samples))

            # 2. Hit-and-Run (HRSS)
            hr_samples = run_hit_and_run(log_prob_shifted, A_poly, b_shifted, np.zeros(d_sub), 10, n_samples // 10)
            h_runs.append(compute_msjd(hr_samples))

            # 3. Dikin Walk
            print("  Running Dikin Walk...")
            try:
                dikin_samples = run_dikin_walk(log_prob_shifted, A_poly, b_shifted, np.zeros(d_sub), 10, n_samples // 10)
                d_runs.append(compute_msjd(dikin_samples))
            except Exception as e:
                print(f"  [!] Dikin Walk mathematically collapsed: {e}")
                d_runs.append(np.nan)

            # 4. emcee
            n_walkers = max(40, 2*d_sub)
            sampler_emcee = emcee.EnsembleSampler(n_walkers, d_sub, log_prob_shifted)
            p0 = np.random.uniform(-1e-4, 1e-4, size=(n_walkers, d_sub))
            sampler_emcee.run_mcmc(p0, n_samples // n_walkers, progress=False)
            emcee_samples = np.transpose(sampler_emcee.get_chain(), (1, 0, 2))
            e_runs.append(compute_msjd(emcee_samples))

        print(f"  --- Completed {k_runs} Runs ---")
        
        msjd_whss_mu.append(np.mean(w_runs)); msjd_whss_std.append(np.std(w_runs))
        msjd_hrss_mu.append(np.mean(h_runs)); msjd_hrss_std.append(np.std(h_runs))
        msjd_dikin_mu.append(np.nanmean(d_runs)); msjd_dikin_std.append(np.nanstd(d_runs))
        msjd_emcee_mu.append(np.mean(e_runs)); msjd_emcee_std.append(np.std(e_runs))
        
        print(f"  WHSS  MSJD -> {msjd_whss_mu[-1]:.2e} ± {msjd_whss_std[-1]:.2e}")
        print(f"  HRSS  MSJD -> {msjd_hrss_mu[-1]:.2e} ± {msjd_hrss_std[-1]:.2e}")
        print(f"  Dikin MSJD -> {msjd_dikin_mu[-1]:.2e} ± {msjd_dikin_std[-1]:.2e}")
        print(f"  emcee MSJD -> {msjd_emcee_mu[-1]:.2e} ± {msjd_emcee_std[-1]:.2e}")

    # Plotting
    print("\n[Plotting] Building Log-Scale Error Bar Curve...")
    plt.figure()
    
    plt.errorbar(dimensions, msjd_whss_mu, yerr=msjd_whss_std, marker='o', capsize=5, color="#4285F4", label="WHSS (Ours)")
    plt.errorbar(dimensions, msjd_hrss_mu, yerr=msjd_hrss_std, marker='s', capsize=5, color="#FBBC04", label="Hit-and-Run (HRSS)")
    plt.errorbar(dimensions, msjd_dikin_mu, yerr=msjd_dikin_std, marker='^', capsize=5, color="#34A853", label="Dikin Walk")
    plt.errorbar(dimensions, msjd_emcee_mu, yerr=msjd_emcee_std, marker='X', capsize=5, color="#EA4335", linestyle="--", label="emcee (Trapped)")

    plt.yscale("log")
    plt.xlabel("Portfolio Size (Number of Assets)")
    plt.ylabel("Mean Squared Jump Distance (log scale)")
    plt.title(f"Algorithm Mobility in Constrained Portfolio Space (10 Runs)")
    plt.legend()
    
    png_path = os.path.join(plots_dir, "msjd_plot.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    
    report_text = []
    report_text.append("MSJD Benchmark Results (Log Scale Mobility):")
    for d, w_mu, h_mu, d_mu in zip(dimensions, msjd_whss_mu, msjd_hrss_mu, msjd_dikin_mu):
        report_text.append(f"Dimension: {d} | WHSS: {w_mu:.2e} | HRSS: {h_mu:.2e} | Dikin: {d_mu:.2e}")
    
    with open(os.path.join(plots_dir, "msjd_report.txt"), "w") as f:
        f.write("\n".join(report_text))
        
    print(f"\nPlot and report saved to: {plots_dir}")

if __name__ == "__main__":
    run_msjd_scaling()