import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from whss_sampling.distributions.gaussian import whss_sampling_gaussian
import emcee

def compute_ess_per_chain(samples, burn_in_pct=0.2):
    try:
        from emcee.autocorr import integrated_time
    except ImportError:
        return 1.0, len(samples)
        
    if samples.ndim == 3:
        n_chains, n_steps, d = samples.shape
        burn_in = int(n_steps * burn_in_pct)
        valid_samples = samples[:, burn_in:, :]
        valid_steps = valid_samples.shape[1]
        
        chain_ess = []
        for c in range(n_chains):
            try:
                tau = integrated_time(valid_samples[c], c=5, tol=10, quiet=True)
                chain_ess.append(valid_steps / np.max(tau))
            except:
                chain_ess.append(1.0)
        return np.sum(chain_ess), valid_steps * n_chains
    else:
        burn_in = int(len(samples) * burn_in_pct)
        valid_samples = samples[burn_in:]
        valid_steps = len(valid_samples)
        try:
            tau = integrated_time(valid_samples, c=5, tol=10, quiet=True)
            return valid_steps / np.max(tau), valid_steps
        except:
            return 1.0, valid_steps

# Dikin and HRSS implementations (C-Speed)
@njit(fastmath=True)
def run_hit_and_run(density_func, A, b, x_init, n_chains, n_steps):
    d = len(x_init)
    samples = np.empty((n_chains, n_steps, d))
    nfe = 0
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
            nfe += 1
            t = np.random.uniform(t_min, t_max)
            x_prop = x_curr + t * u
            
            nfe += 1
            while density_func(x_prop) < y_slice:
                if t > 0: t_max = t
                else: t_min = t
                if t_max - t_min < 1e-10: break
                t = np.random.uniform(t_min, t_max)
                x_prop = x_curr + t * u
                nfe += 1
            x_curr = x_prop
            samples[c, s] = x_curr
    return samples, nfe

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

def run_scaling_benchmark():
    np.random.seed(42)
    # 10, 50, 100, 200, 400
    dimensions = [10, 20, 30, 40, 50, 75,100]
    k_runs = 5
    
    results = {
        "WHSS": {"mu": [], "std": []},
        "HRSS": {"mu": [], "std": []},
        "Dikin": {"mu": [], "std": []}
    }
    
    print("=====================================================")
    print("STARTING DIMENSIONAL SCALING BENCHMARK")
    print("=====================================================")
    
    for d in dimensions:
        print(f"\n--- Testing Dimension: {d} ---")
        
        # 1. Create a Challenging Space
        # Highly Skewed Gaussian target inside a tight restricted bounding box
        # Condition Number = 1000 across ALL axes
        # Using 10.0 as max variance (std ~3.16) so it crashes heavily into the [-1.0, 1.0] box!
        cond = 1000.0
        eigenvalues = np.geomspace(10.0, 10.0 / cond, d)
        
        np.random.seed(42 + d)
        H = np.random.randn(d, d)
        Q, _ = np.linalg.qr(H)
        
        base_cov = np.diag(eigenvalues)
        base_inv_cov = np.diag(1.0 / eigenvalues)
        
        cov = Q @ base_cov @ Q.T
        inv_cov = Q @ base_inv_cov @ Q.T
        
        # Target density: Skewed Gaussian
        @njit(fastmath=True)
        def log_prob(x):
            val = 0.0
            for i in range(d):
                for j in range(d):
                    val += x[i] * inv_cov[i, j] * x[j]
            return -0.5 * val
            
        # Box constraints (Tight Bounding Box)
        # Using 1.0 to ensure the skewed Gaussian crashes heavily into the constraints
        A_poly = np.ascontiguousarray(np.vstack([np.eye(d), -np.eye(d)]))
        b_poly = np.ascontiguousarray(np.concatenate([np.ones(d)*1.0, np.ones(d)*1.0]))
        x0 = np.zeros(d)
        
        # DYNAMIC BUDGET: We MUST clear the 3*d anchor safeguard for the Warp matrix to be full-rank.
        # At 10% acceptance, 3*d anchors takes ~30*d steps. 
        burn_in_steps_per_chain = max(1000, 30 * d)
        post_warmup_steps = max(1000, 15 * d)
        steps_per_chain = burn_in_steps_per_chain + post_warmup_steps
        n_samples = steps_per_chain * 10
        
        run_whss, run_hrss, run_dikin = [], [], []
        
        for run in range(k_runs):
            if k_runs > 1:
                print(f"   --- Run {run+1}/{k_runs} ---")
                
            # ---------------- WHSS ----------------
            if k_runs == 1: print(f"[{d}D] Running WHSS...")
            sampler = whss_sampling_gaussian(d=d, k=n_samples, sigma=np.eye(d), mu=np.zeros(d))
            
            old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
            try:
                t0 = time.time()
                whss_samples = sampler._sampling_universal(
                    density_cartesian=log_prob, A=A_poly, b=b_poly,
                    burn_in_samples=burn_in_steps_per_chain * 10, max_anchors=3*d, bypass_safeguards=False
                )
                post_warmup_whss = whss_samples[:, burn_in_steps_per_chain:, :]
                ess, valid_samples = compute_ess_per_chain(post_warmup_whss, burn_in_pct=0.0)
                run_whss.append((ess / valid_samples) * 1000)
            except Exception as e:
                run_whss.append(0.0)
            finally:
                sys.stdout = old_stdout
                
            # ---------------- HRSS ----------------
            if k_runs == 1: print(f"[{d}D] Running Hit-and-Run (HRSS)...")
            t0 = time.time()
            hrss_samples, hrss_nfe = run_hit_and_run(log_prob, A_poly, b_poly, x0, 10, steps_per_chain)
            hrss_time = time.time() - t0
            post_warmup_hrss = hrss_samples[:, burn_in_steps_per_chain:, :]
            ess, valid_samples = compute_ess_per_chain(post_warmup_hrss, burn_in_pct=0.0)
            run_hrss.append((ess / valid_samples) * 1000)
            if k_runs == 1: print(f"      [!] HRSS required {hrss_nfe / (steps_per_chain*10):.1f} density evaluations per step!")
            
            # ---------------- Dikin Walk ----------------
            if k_runs == 1: print(f"[{d}D] Running Dikin Walk (SOTA)...")
            try:
                dikin_samples = run_dikin_walk(log_prob, A_poly, b_poly, x0, 10, steps_per_chain)
                post_warmup_dikin = dikin_samples[:, burn_in_steps_per_chain:, :]
                ess, valid_samples = compute_ess_per_chain(post_warmup_dikin, burn_in_pct=0.0)
                run_dikin.append((ess / valid_samples) * 1000)
            except Exception as e:
                run_dikin.append(0.0)
                
        # Aggregate statistics
        results["WHSS"]["mu"].append(np.mean(run_whss))
        results["WHSS"]["std"].append(np.std(run_whss) if k_runs > 1 else 0.0)
        results["HRSS"]["mu"].append(np.mean(run_hrss))
        results["HRSS"]["std"].append(np.std(run_hrss) if k_runs > 1 else 0.0)
        results["Dikin"]["mu"].append(np.mean(run_dikin))
        results["Dikin"]["std"].append(np.std(run_dikin) if k_runs > 1 else 0.0)
        
        print(f"   -> WHSS Efficiency: {results['WHSS']['mu'][-1]:.2f} ± {results['WHSS']['std'][-1]:.2f}")
        print(f"   -> HRSS Efficiency: {results['HRSS']['mu'][-1]:.2f} ± {results['HRSS']['std'][-1]:.2f}")
        print(f"   -> Dikin Efficiency: {results['Dikin']['mu'][-1]:.2f} ± {results['Dikin']['std'][-1]:.2f}")

    # ---------------- Plotting ----------------
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})
    
    plt.errorbar(dimensions, results["WHSS"]["mu"], yerr=results["WHSS"]["std"], label="WHSS (Ours)", marker='o', linewidth=3, color="#4285F4", markersize=8, capsize=5)
    plt.errorbar(dimensions, results["HRSS"]["mu"], yerr=results["HRSS"]["std"], label="Hit-and-Run (HRSS)", marker='s', linewidth=2, color="#FBBC04", linestyle="--", capsize=5)
    plt.errorbar(dimensions, results["Dikin"]["mu"], yerr=results["Dikin"]["std"], label="Dikin Walk (SOTA)", marker='^', linewidth=2, color="#EA4335", linestyle=":", capsize=5)
    
    plt.yscale('log')
    plt.xlabel("Dimensionality ($D$)")
    plt.ylabel("Efficiency (ESS / 1k Steps)")
    plt.title(f"Scaling Efficiency on Highly Skewed Gaussian (Cond={int(cond)})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    save_path = os.path.join(current_dir, "results", "scaling_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n[SUCCESS] Scaling benchmark plot saved to {save_path}")
    
    report_text = []
    report_text.append(f"Scaling Benchmark Results ({k_runs} Runs):")
    for i, d in enumerate(dimensions):
        w_mu, w_std = results["WHSS"]["mu"][i], results["WHSS"]["std"][i]
        h_mu, h_std = results["HRSS"]["mu"][i], results["HRSS"]["std"][i]
        dk_mu, dk_std = results["Dikin"]["mu"][i], results["Dikin"]["std"][i]
        report_text.append(f"Dim: {d:3d} | WHSS: {w_mu:.2f} ± {w_std:.2f} | HRSS: {h_mu:.2f} ± {h_std:.2f} | Dikin: {dk_mu:.2f} ± {dk_std:.2f}")
        
    with open(os.path.join(current_dir, "results", "scaling_report.txt"), "w") as f:
        f.write("\n".join(report_text))

if __name__ == "__main__":
    run_scaling_benchmark()
