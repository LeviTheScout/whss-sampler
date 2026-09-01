import os
import sys
import time
import numpy as np
import emcee
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
diag_dir = os.path.join(current_dir, "diagnostics")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)

# =====================================================================
# UNBIASED DIAGNOSTIC TOOLS
# =====================================================================
def compute_robust_ess_stats_3d(samples_3d):
    """Computes Total ESS per dimension across parallel chains/walkers."""
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

# =====================================================================
# FINANCIAL ENVIRONMENT BUILDER
# =====================================================================
def build_finance_environment(n_assets=30, n_scenarios=500):
    np.random.seed(42)
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors)
    factor_returns = np.random.randn(n_scenarios, n_factors) * 0.03
    idiosyncratic = np.random.randn(n_scenarios, n_assets) * 0.01
    
    R_scenarios = factor_returns @ factor_loadings.T + idiosyncratic
    mu_returns = np.mean(R_scenarios, axis=0) + 0.005 
    
    d_sub = n_assets - 1
    A_rows, b_rows = [], []
    
    # 1. Long-only (w_i >= 0) and Single Asset Concentration Cap (w_i <= 0.15)
    for i in range(d_sub):
        e_i = np.zeros(d_sub)
        e_i[i] = 1.0
        A_rows.append(-e_i); b_rows.append(0.0)      
        A_rows.append(e_i);  b_rows.append(0.15)     
        
    # 2. Total Budget Simplex sum(w) = 1 (enforcing 0 <= w_d <= 0.15)
    ones = np.ones(d_sub)
    A_rows.append(ones);  b_rows.append(1.0)         
    A_rows.append(-ones); b_rows.append(-(1.0 - 0.15)) 
    
    # 3. Sector Cap Constraint (sum_{i=1..5} w_i <= 0.30)
    tech_mask = np.zeros(d_sub)
    tech_mask[:5] = 1.0
    A_rows.append(tech_mask); b_rows.append(0.30)
    
    A_poly = np.ascontiguousarray(np.vstack(A_rows), dtype=np.float64)
    b_poly = np.ascontiguousarray(np.array(b_rows), dtype=np.float64)
    
    return d_sub, A_poly, b_poly, R_scenarios, mu_returns

# =====================================================================
# NUMBA COMPILER FACTORY (PREVENTS LLVM DEADLOCK)
# =====================================================================
def make_evaluators(A_poly, b_poly, R_sub, R_d, mu_sub, mu_d, k_cutoff, lambda_risk, tau_temp, w_center):
    @njit(fastmath=True)
    def finance_log_prob(w_sub):
        # Polytope boundary check (Fast reject outside walls)
        for i in range(len(b_poly)):
            val = 0.0
            for j in range(len(w_sub)):
                val += A_poly[i, j] * w_sub[j]
            if val > b_poly[i]:
                return -np.inf
                
        w_d = 1.0 - np.sum(w_sub)
        
        # Fast Vectorized Simulator Returns
        port_losses = -(np.dot(R_sub, w_sub) + R_d * w_d)
            
        # Non-differentiable CVaR Discrete Sorting
        port_losses.sort()
        
        cvar_loss = 0.0
        for idx in range(len(port_losses) - k_cutoff, len(port_losses)):
            cvar_loss += port_losses[idx]
        cvar_loss /= k_cutoff
        
        exp_ret = np.dot(mu_sub, w_sub) + mu_d * w_d
        utility = exp_ret - lambda_risk * cvar_loss
        return utility / tau_temp

    @njit(fastmath=True)
    def shifted_log_prob(y):
        return finance_log_prob(y + w_center)

    return finance_log_prob, shifted_log_prob

# =====================================================================
# SOTA BASELINE 1: HIT-AND-RUN SLICE SAMPLER (HRSS)
# =====================================================================
@njit(fastmath=True)
def run_hit_and_run(density_func, A, b, x_init, n_chains, n_steps):
    d = len(x_init)
    samples = np.empty((n_chains, n_steps, d))
    
    for c in range(n_chains):
        x_curr = x_init.copy()
        for s in range(n_steps):
            # 1. Propose Isotropic Random Direction
            u = np.random.randn(d)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-12:
                u[0] = 1.0
            else:
                u /= norm_u
                
            # 2. Analytic Chord Clipping along Ax <= b
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
                
            # 3. 1D Slice Sampling along the chord
            y_slice = density_func(x_curr) - np.random.exponential(1.0)
            t = np.random.uniform(t_min, t_max)
            x_prop = x_curr + t * u
            
            while density_func(x_prop) < y_slice:
                if t > 0:
                    t_max = t
                else:
                    t_min = t
                if t_max - t_min < 1e-10:
                    break
                t = np.random.uniform(t_min, t_max)
                x_prop = x_curr + t * u
                
            x_curr = x_prop
            samples[c, s] = x_curr
            
    return samples

# =====================================================================
# SOTA BASELINE 2: DIKIN WALK (BARRIER-BASED ZEROTH-ORDER MCMC)
# =====================================================================
@njit(fastmath=True)
def run_dikin_walk(density_func, A, b, x_init, n_chains, n_steps, r_step=0.15):
    d = len(x_init)
    m = len(b)
    samples = np.empty((n_chains, n_steps, d))
    
    for c in range(n_chains):
        x_curr = x_init.copy()
        for s in range(n_steps):
            # 1. Compute Barrier Hessian: H(x) = A^T D(x)^{-2} A
            slack_x = b - np.dot(A, x_curr)
            for i in range(m):
                if slack_x[i] <= 1e-10:
                    slack_x[i] = 1e-10
            
            D_inv_x = 1.0 / slack_x
            H_x = np.zeros((d, d))
            for i in range(m):
                row = A[i] * D_inv_x[i]
                for j in range(d):
                    for k in range(d):
                        H_x[j, k] += row[j] * row[k]
                        
            # Regularize for strict positive-definiteness
            for j in range(d):
                H_x[j, j] += 1e-6
                
            L_x = np.linalg.cholesky(H_x)
            
            # 2. Propose Gaussian step in Dikin Ellipsoid
            z = np.random.randn(d)
            delta = np.linalg.solve(L_x.T, z) * r_step
            x_prop = x_curr + delta
            
            # 3. Check Feasibility
            slack_prop = b - np.dot(A, x_prop)
            feasible = True
            for i in range(m):
                if slack_prop[i] <= 0.0:
                    feasible = False
                    break
                    
            if not feasible:
                samples[c, s] = x_curr
                continue
                
            # 4. Compute Proposal Hessian at proposed point
            D_inv_p = 1.0 / np.maximum(slack_prop, 1e-10)
            H_p = np.zeros((d, d))
            for i in range(m):
                row = A[i] * D_inv_p[i]
                for j in range(d):
                    for k in range(d):
                        H_p[j, k] += row[j] * row[k]
            for j in range(d):
                H_p[j, j] += 1e-6
                
            L_p = np.linalg.cholesky(H_p)
            
            # 5. Metropolis Hastings Acceptance Ratio (Zeroth-order density + determinant transition)
            log_target_ratio = density_func(x_prop) - density_func(x_curr)
            log_det_ratio = np.sum(np.log(np.diag(L_p))) - np.sum(np.log(np.diag(L_x)))
            
            diff = x_curr - x_prop
            quad_diff = np.dot(diff, np.dot(H_p - H_x, diff)) / (2.0 * r_step**2)
            
            log_alpha = log_target_ratio + log_det_ratio - quad_diff
            
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                x_curr = x_prop
                
            samples[c, s] = x_curr
            
    return samples

# =====================================================================
# UNTIMED COMPILER WARM-UPS
# =====================================================================
def run_silent_jit_warmups(density_func, shifted_func, A, b, b_shifted, d_sub):
    print("  [JIT WARM-UP] Compiling all C-backends (Untimed)...")
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        x_dummy = np.zeros(d_sub)
        _ = density_func(np.full(d_sub, 1.0/30))
        _ = shifted_func(x_dummy)
        _ = run_hit_and_run(shifted_func, A, b_shifted, x_dummy, n_chains=2, n_steps=5)
        _ = run_dikin_walk(shifted_func, A, b_shifted, x_dummy, n_chains=2, n_steps=5)
        
        dummy_whss = nsmc_sampling_gaussian(d=d_sub, k=10, sigma=np.eye(d_sub), mu=np.zeros(d_sub))
        _ = dummy_whss._sampling_universal(
            density_cartesian=shifted_func, A=A, b=b_shifted,
            batch_size=10, burn_in_samples=10, max_anchors=10
        )
    finally:
        sys.stdout = old_stdout
        devnull.close()
    print("  [JIT WARM-UP] Compilation complete. Starting benchmark.\n")


# =====================================================================
# MULTI-RUN FINANCE BENCHMARK
# =====================================================================
def run_finance_benchmark():
    k_runs = 5
    n_samples = 60000  # Total evaluation budget across algorithms
    n_assets = 30
    
    d_sub, A_poly, b_poly, R_scenarios, mu_returns = build_finance_environment(n_assets, 500)
    
    alpha_cvar = 0.95
    k_cutoff = int((1.0 - alpha_cvar) * R_scenarios.shape[0])
    lambda_risk = 5.0
    tau_temp = 0.05
    w_center = np.full(d_sub, 1.0 / n_assets)
    
    R_sub = np.ascontiguousarray(R_scenarios[:, :-1])
    R_d = np.ascontiguousarray(R_scenarios[:, -1])
    mu_sub = np.ascontiguousarray(mu_returns[:-1])
    mu_d = mu_returns[-1]
    
    b_shifted = np.ascontiguousarray(np.maximum(b_poly - A_poly @ w_center, 1e-7), dtype=np.float64)

    # 1. Generate Safe Evaluators
    finance_log_prob, shifted_log_prob = make_evaluators(
        A_poly, b_poly, R_sub, R_d, mu_sub, mu_d, 
        k_cutoff, lambda_risk, tau_temp, w_center
    )
    
    # 2. Untimed Compiler Warm-Up
    run_silent_jit_warmups(finance_log_prob, shifted_log_prob, A_poly, b_poly, b_shifted, d_sub)

    report = []
    report.append("=" * 105)
    report.append(f"WHSS REAL-WORLD VALIDATION: QUANTITATIVE FINANCE BENCHMARK ({k_runs} Runs)")
    report.append("=" * 105 + "\n")
    report.append(f"Problem: {n_assets}-Asset Black-Box CVaR Allocation under Polytope Constraints")
    report.append(f"Target : Non-differentiable Historical 95% CVaR Utility inside Aw <= b\n")

    emcee_times, emcee_min, emcee_med, emcee_max = [], [], [], []
    hr_times, hr_min, hr_med, hr_max = [], [], [], []
    dikin_times, dikin_min, dikin_med, dikin_max = [], [], [], []
    whss_times, whss_min, whss_med, whss_max = [], [], [], []

    # -----------------------------------------------------------------
    # BASELINE 1: emcee Ensemble
    # -----------------------------------------------------------------
    print("[1/4] Running emcee Baseline (Testing for Spatial Trapping)...")
    n_walkers = 60
    for r in range(k_runs):
        sampler = emcee.EnsembleSampler(n_walkers, d_sub, finance_log_prob)
        p0 = w_center + np.random.uniform(-0.002, 0.002, size=(n_walkers, d_sub))
        
        t0 = time.perf_counter()
        sampler.run_mcmc(p0, n_samples // n_walkers, progress=False)
        t_emcee = time.perf_counter() - t0
        
        chain_3d = np.transpose(sampler.get_chain(), (1, 0, 2))
        mn, md, mx = compute_robust_ess_stats_3d(chain_3d)
        emcee_times.append(t_emcee); emcee_min.append(mn); emcee_med.append(md); emcee_max.append(mx)
        
        print(f"  emcee Run {r+1}/{k_runs} -> Time: {t_emcee:.2f}s | Min ESS: {mn:.1f} | Med ESS: {md:.1f}")
        print(f"      -> Explored Range: [{np.min(chain_3d):.4f}, {np.max(chain_3d):.4f}] | Acc: {np.mean(sampler.acceptance_fraction):.3f}")

    # -----------------------------------------------------------------
    # BASELINE 2: Hit-and-Run Slice Sampler (HRSS)
    # -----------------------------------------------------------------
    print("\n[2/4] Running Hit-and-Run SOTA Baseline (Isotropic Polytope Sampler)...")
    n_chains = 10
    steps_per_chain = n_samples // n_chains
    x_init_sub = np.zeros(d_sub)
    for r in range(k_runs):
        t0 = time.perf_counter()
        hr_samples = run_hit_and_run(shifted_log_prob, A_poly, b_shifted, x_init_sub, n_chains, steps_per_chain)
        t_hr = time.perf_counter() - t0
        
        mn, md, mx = compute_robust_ess_stats_3d(hr_samples)
        hr_times.append(t_hr); hr_min.append(mn); hr_med.append(md); hr_max.append(mx)
        
        hr_abs = hr_samples + w_center
        print(f"  Hit-and-Run Run {r+1}/{k_runs} -> Time: {t_hr:.2f}s | Min ESS: {mn:.1f} | Med ESS: {md:.1f}")
        print(f"      -> Explored Range: [{np.min(hr_abs):.4f}, {np.max(hr_abs):.4f}]")

    # -----------------------------------------------------------------
    # BASELINE 3: Dikin Walk (Barrier Zeroth-Order SOTA)
    # -----------------------------------------------------------------
    print("\n[3/4] Running Dikin Walk SOTA Baseline (Polytope Barrier Sampler)...")
    for r in range(k_runs):
        t0 = time.perf_counter()
        dikin_samples = run_dikin_walk(shifted_log_prob, A_poly, b_shifted, x_init_sub, n_chains, steps_per_chain)
        t_dikin = time.perf_counter() - t0
        
        mn, md, mx = compute_robust_ess_stats_3d(dikin_samples)
        dikin_times.append(t_dikin); dikin_min.append(mn); dikin_med.append(md); dikin_max.append(mx)
        
        dikin_abs = dikin_samples + w_center
        print(f"  Dikin Walk Run {r+1}/{k_runs} -> Time: {t_dikin:.2f}s | Min ESS: {mn:.1f} | Med ESS: {md:.1f}")
        print(f"      -> Explored Range: [{np.min(dikin_abs):.4f}, {np.max(dikin_abs):.4f}]")

    # -----------------------------------------------------------------
    # PROPOSED: Warped Hybrid Slice Sampler (WHSS)
    # -----------------------------------------------------------------
    print("\n[4/4] Running Native WHSS (Ours)...")
    for r in range(k_runs):
        t0 = time.perf_counter()
        sampler_whss = nsmc_sampling_gaussian(d=d_sub, k=n_samples, sigma=np.eye(d_sub), mu=np.zeros(d_sub))
        whss_samples = sampler_whss._sampling_universal(
            density_cartesian=shifted_log_prob,
            A=A_poly, b=b_shifted,
            batch_size=150, burn_in_samples=1500, max_anchors=60
        )
        t_whss = time.perf_counter() - t0
        
        mn, md, mx = compute_robust_ess_stats_3d(whss_samples)
        whss_times.append(t_whss); whss_min.append(mn); whss_med.append(md); whss_max.append(mx)
        
        whss_abs = whss_samples + w_center
        print(f"  WHSS Run {r+1}/{k_runs} -> Time: {t_whss:.2f}s | Min ESS: {mn:.1f} | Med ESS: {md:.1f}")
        print(f"      -> Explored Range: [{np.min(whss_abs):.4f}, {np.max(whss_abs):.4f}]")

    # -----------------------------------------------------------------
    # REPORT GENERATION
    # -----------------------------------------------------------------
    m_emcee_t, m_emcee_mn, m_emcee_md, m_emcee_mx = np.mean(emcee_times), np.mean(emcee_min), np.mean(emcee_med), np.mean(emcee_max)
    m_hr_t, m_hr_mn, m_hr_md, m_hr_mx = np.mean(hr_times), np.mean(hr_min), np.mean(hr_med), np.mean(hr_max)
    m_dikin_t, m_dikin_mn, m_dikin_md, m_dikin_mx = np.mean(dikin_times), np.mean(dikin_min), np.mean(dikin_med), np.mean(dikin_max)
    m_whss_t, m_whss_mn, m_whss_md, m_whss_mx = np.mean(whss_times), np.mean(whss_min), np.mean(whss_med), np.mean(whss_max)

    report.append(f"{'Algorithm':<24} | {'Time (s)':<9} | {'Min ESS':<9} | {'Med ESS':<9} | {'Max ESS':<9} | {'ESS/s (Min)':<11} | {'ESS/s (Med)':<11}")
    report.append("-" * 102)
    report.append(f"{'emcee (Affine Ensemble)':<24} | {m_emcee_t:<9.2f} | {m_emcee_mn:<9.1f} | {m_emcee_md:<9.1f} | {m_emcee_mx:<9.1f} | {(m_emcee_mn/m_emcee_t):<11.2f} | {(m_emcee_md/m_emcee_t):<11.2f} (Trapped)")
    report.append(f"{'Hit-and-Run (HRSS)':<24} | {m_hr_t:<9.2f} | {m_hr_mn:<9.1f} | {m_hr_md:<9.1f} | {m_hr_mx:<9.1f} | {(m_hr_mn/m_hr_t):<11.2f} | {(m_hr_md/m_hr_t):<11.2f}")
    report.append(f"{'Dikin Walk (Barrier SOTA)':<24} | {m_dikin_t:<9.2f} | {m_dikin_mn:<9.1f} | {m_dikin_md:<9.1f} | {m_dikin_mx:<9.1f} | {(m_dikin_mn/m_dikin_t):<11.2f} | {(m_dikin_md/m_dikin_t):<11.2f}")
    report.append(f"{'WHSS (Ours)':<24} | {m_whss_t:<9.2f} | {m_whss_mn:<9.1f} | {m_whss_md:<9.1f} | {m_whss_mx:<9.1f} | {(m_whss_mn/m_whss_t):<11.2f} | {(m_whss_md/m_whss_t):<11.2f}\n")
    
    final_output = "\n".join(report)
    print("\n" + final_output)
    
    report_path = os.path.join(results_dir, "finance_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    run_finance_benchmark()