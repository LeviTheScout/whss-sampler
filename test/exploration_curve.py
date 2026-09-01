import os
import sys
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import emcee

# =====================================================================
# PATH RESOLUTION & SETUP
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from nsmc_sampling.distributions.gaussian import nsmc_sampling_gaussian

plots_dir = os.path.join(current_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "lines.linewidth": 3.5,
    "figure.figsize": (10, 7),
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# =====================================================================
# TARGET & SOTA SAMPLERS
# =====================================================================
def build_skewed_target(d=30, cond=1000.0):
    np.random.seed(42)
    evals = np.geomspace(10.0, 10.0 / cond, d)
    H = np.random.randn(d, d)
    Q, _ = np.linalg.qr(H)
    
    inv_cov = Q @ np.diag(1.0 / evals) @ Q.T
    inv_cov = np.ascontiguousarray(inv_cov, dtype=np.float64)
    
    A_rows, b_rows = [], []
    for i in range(d):
        e_i = np.zeros(d); e_i[i] = 1.0
        A_rows.append(e_i);  b_rows.append(5.0)
        A_rows.append(-e_i); b_rows.append(5.0)
        
    A_poly = np.ascontiguousarray(np.vstack(A_rows), dtype=np.float64)
    b_poly = np.ascontiguousarray(np.array(b_rows), dtype=np.float64)
    
    return inv_cov, A_poly, b_poly, np.sum(evals)

def make_evaluators(inv_cov, A_poly, b_poly):
    @njit(fastmath=True)
    def log_prob(x):
        for i in range(len(b_poly)):
            val = 0.0
            for j in range(len(x)): val += A_poly[i, j] * x[j]
            if val > b_poly[i]: return -np.inf
        return -0.5 * np.dot(x, np.dot(inv_cov, x))
    return log_prob

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

def get_running_trace_nfe(samples_3d, step_size=100):
    n_chains, n_steps, d = samples_3d.shape
    checkpoints = np.arange(step_size, n_steps, step_size)
    trace_vals = []
    
    for s in checkpoints:
        data_chunk = samples_3d[:, :s, :].reshape(-1, d)
        current_trace = np.sum(np.var(data_chunk, axis=0))
        trace_vals.append(current_trace)
        
    nfe_checkpoints = checkpoints * n_chains
    return nfe_checkpoints, np.array(trace_vals)

def run_silent_jit_warmup(sampler, density_func, A, b):
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        _ = sampler._sampling_universal(density_cartesian=density_func, A=A, b=b)
    finally:
        sys.stdout = old_stdout
        devnull.close()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def run_variance_dominance_plot():
    d = 30
    n_samples = 30000
    k_runs = 10
    
    inv_cov, A_poly, b_poly, true_trace = build_skewed_target(d, cond=1000.0)
    log_prob = make_evaluators(inv_cov, A_poly, b_poly)

    print("=" * 80)
    print(f"GENERATING DOMINANCE PLOT (Averaged over {k_runs} Runs)")
    print("=" * 80 + "\n")

    # Arrays to store aggregated traces
    traces_w, traces_h, traces_e, traces_d = [], [], [], []
    nfe_w, nfe_h, nfe_e, nfe_d = None, None, None, None

    # JIT WARMUP
    print("[JIT WARM-UP] Compiling internal functions...")
    dummy_sampler = nsmc_sampling_gaussian(d=d, k=10, sigma=np.eye(d), mu=np.zeros(d))
    run_silent_jit_warmup(dummy_sampler, log_prob, A_poly, b_poly)

    for r in range(k_runs):
        print(f"\n--- Executing Run {r+1}/{k_runs} ---")
        
        # 1. emcee
        print("  Running emcee...")
        n_walkers = 60
        sampler = emcee.EnsembleSampler(n_walkers, d, log_prob)
        p0 = np.random.uniform(-0.1, 0.1, size=(n_walkers, d))
        sampler.run_mcmc(p0, n_samples // n_walkers, progress=False)
        emcee_samples = np.transpose(sampler.get_chain(), (1, 0, 2))
        nfe_e, t_e = get_running_trace_nfe(emcee_samples)
        traces_e.append(t_e)

        # 2. Hit-and-Run
        print("  Running Hit-and-Run (HRSS)...")
        hr_samples = run_hit_and_run(log_prob, A_poly, b_poly, np.zeros(d), 10, n_samples // 10)
        nfe_h, t_h = get_running_trace_nfe(hr_samples)
        traces_h.append(t_h)

        # 3. Dikin Walk
        print("  Running Dikin Walk...")
        try:
            dikin_samples = run_dikin_walk(log_prob, A_poly, b_poly, np.zeros(d), 10, n_samples // 10)
            nfe_d, t_d = get_running_trace_nfe(dikin_samples)
            traces_d.append(t_d)
        except Exception as e:
            print(f"  [!] Dikin Walk mathematically collapsed: {e}")

        # 4. WHSS
        print("  Running WHSS (Ours)...")
        sampler_whss = nsmc_sampling_gaussian(d=d, k=n_samples, sigma=np.eye(d)*0.1, mu=np.zeros(d))
        whss_samples = sampler_whss._sampling_universal(
            density_cartesian=log_prob, A=A_poly, b=b_poly,
            burn_in_samples=2500, max_anchors=60
        )
        nfe_w, t_w = get_running_trace_nfe(whss_samples)
        traces_w.append(t_w)

    print("\n[Plotting] Building Multi-Run Exploration Curve...")
    
    # Calculate Means and Standard Deviations
    mean_w, std_w = np.mean(traces_w, axis=0), np.std(traces_w, axis=0)
    mean_h, std_h = np.mean(traces_h, axis=0), np.std(traces_h, axis=0)
    mean_e, std_e = np.mean(traces_e, axis=0), np.std(traces_e, axis=0)
    
    if len(traces_d) > 0:
        mean_d, std_d = np.mean(traces_d, axis=0), np.std(traces_d, axis=0)

    plt.figure()
    plt.axhline(y=true_trace, color='black', linestyle=':', label="Unconstrained Target Volume")
    
    # WHSS
    plt.plot(nfe_w, mean_w, label="WHSS (Ours)", color="#4285F4")
    plt.fill_between(nfe_w, np.maximum(0, mean_w - std_w), mean_w + std_w, color="#4285F4", alpha=0.2)
    
    # HRSS
    plt.plot(nfe_h, mean_h, label="Hit-and-Run (HRSS)", color="#FBBC04")
    plt.fill_between(nfe_h, np.maximum(0, mean_h - std_h), mean_h + std_h, color="#FBBC04", alpha=0.2)
    
    # Dikin Walk
    if len(traces_d) > 0:
        plt.plot(nfe_d, mean_d, label="Dikin Walk", color="#34A853", linestyle="-.")
        plt.fill_between(nfe_d, np.maximum(0, mean_d - std_d), mean_d + std_d, color="#34A853", alpha=0.2)
    
    # emcee
    plt.plot(nfe_e, mean_e, label="emcee", color="#EA4335", linestyle="--")
    plt.fill_between(nfe_e, np.maximum(0, mean_e - std_e), mean_e + std_e, color="#EA4335", alpha=0.2)

    plt.xlabel("Total Function Evaluations (NFE)")
    plt.ylabel(r"Captured Variance (Trace $\Sigma$)")
    plt.title(f"Global Exploration in 30D Space (Mean ± Std, 10 Runs)")
    plt.legend(loc="upper left")
    
    pdf_path = os.path.join(plots_dir, "variance_dominance.pdf")
    png_path = os.path.join(plots_dir, "variance_dominance.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"Plot successfully saved to:\n  - {png_path}")

if __name__ == "__main__":
    run_variance_dominance_plot()