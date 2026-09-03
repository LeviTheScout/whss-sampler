import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from emcee.autocorr import function_1d
from numba import njit

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from whss.distributions.gaussian import whss_gaussian

plots_dir = os.path.join(current_dir, "results")
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
    
    return inv_cov, A_poly, b_poly

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

def compute_mean_acf(samples, max_lag=250):
    n_chains, n_steps, d = samples.shape
    acfs = []
    for c in range(n_chains):
        for j in range(d):
            col = samples[c, :, j]
            if np.std(col) > 1e-8:
                f = function_1d(col)
                acfs.append(f[:max_lag])
    if len(acfs) == 0:
        return np.ones(max_lag)
    return np.mean(acfs, axis=0)

def generate_acf_hero_plot():
    d = 30
    n_samples = 25000
    k_runs = 10
    
    print("=" * 80)
    print(f"GENERATING AUTOCORRELATION HERO PLOT (ACF vs LAG, {k_runs} Runs)")
    print("=" * 80 + "\n")
    
    inv_cov, A_poly, b_poly = build_skewed_target(d, cond=1000.0)
    log_prob = make_evaluators(inv_cov, A_poly, b_poly)
    
    all_acf_h, all_acf_d, all_acf_w = [], [], []
    
    for r in range(k_runs):
        print(f"--- Run {r+1}/{k_runs} ---")
        
        # 1. Hit-and-Run
        print("  Running Hit-and-Run (HRSS)...")
        hr_samples = run_hit_and_run(log_prob, A_poly, b_poly, np.zeros(d), 1, n_samples)
        all_acf_h.append(compute_mean_acf(hr_samples))
        
        # 2. Dikin Walk
        print("  Running Dikin Walk...")
        try:
            dikin_samples = run_dikin_walk(log_prob, A_poly, b_poly, np.zeros(d), 1, n_samples)
            all_acf_d.append(compute_mean_acf(dikin_samples))
        except Exception as e:
            print(f"  [!] Dikin Walk collapsed: {e}")
            all_acf_d.append(np.ones_like(all_acf_h[-1]))
            
        # 3. WHSS
        print("  Running WHSS (Ours)...")
        sampler_whss = whss_gaussian(d=d, k=n_samples, sigma=np.eye(d)*0.1, mu=np.zeros(d))
        
        old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
        whss_samples = sampler_whss._sampling_universal(
            density_cartesian=log_prob, A=A_poly, b=b_poly,
            burn_in_samples=2500, max_anchors=60
        )
        sys.stdout = old_stdout
        all_acf_w.append(compute_mean_acf(whss_samples))
        
    print("\n[Plotting] Building Multi-Run ACF Hero Plot...")
    
    mean_h = np.mean(all_acf_h, axis=0); std_h = np.std(all_acf_h, axis=0)
    mean_d = np.mean(all_acf_d, axis=0); std_d = np.std(all_acf_d, axis=0)
    mean_w = np.mean(all_acf_w, axis=0); std_w = np.std(all_acf_w, axis=0)
    
    plt.figure()
    lags = np.arange(len(mean_w))
    
    plt.plot(lags, mean_h, label="Hit-and-Run (HRSS)", color="#FBBC04", alpha=0.9)
    plt.fill_between(lags, mean_h - std_h, mean_h + std_h, color="#FBBC04", alpha=0.2)
    
    plt.plot(lags, mean_d, label="Dikin Walk (Geometric Collapse)", color="#34A853", linestyle="-.", alpha=0.9)
    plt.fill_between(lags, mean_d - std_d, mean_d + std_d, color="#34A853", alpha=0.2)
    
    plt.plot(lags, mean_w, label="WHSS (Ours)", color="#4285F4", alpha=0.9)
    plt.fill_between(lags, mean_w - std_w, mean_w + std_w, color="#4285F4", alpha=0.2)
    
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    
    plt.xlabel("MCMC Step Lag (k)")
    plt.ylabel(r"Autocorrelation $\rho(k)$")
    plt.title(f"Mixing Autocorrelation in 30D Skewed Space (Cond=1000, 10 Runs)")
    plt.legend(loc="upper right")
    
    png_path = os.path.join(plots_dir, "acf_plot.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    
    report_text = ["ACF Mixing Decorrelation Benchmark (Mean over 10 Runs):"]
    report_text.append("Lag | WHSS | HRSS | Dikin")
    for i in range(len(lags)):
        report_text.append(f"{lags[i]} | {mean_w[i]:.4f} | {mean_h[i]:.4f} | {mean_d[i]:.4f}")
        
    with open(os.path.join(plots_dir, "acf_plot_report.txt"), "w") as f:
        f.write("\n".join(report_text))
        
    print(f"Plot and report successfully saved to {plots_dir}")

if __name__ == "__main__":
    generate_acf_hero_plot()
