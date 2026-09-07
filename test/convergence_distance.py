import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.stats import truncnorm, wasserstein_distance
from numba import njit
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from whss_sampling.distributions.gaussian import whss_gaussian

@njit(fastmath=True)
def run_hit_and_run(density_func, A, b, x0, num_samples):
    d = len(x0)
    samples = np.zeros((num_samples, d))
    x = np.copy(x0)
    for i in range(num_samples):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        t_min, t_max = -1e20, 1e20
        for j in range(len(b)):
            A_j_u = np.dot(A[j], u)
            slack = b[j] - np.dot(A[j], x)
            if A_j_u > 1e-12: t_max = min(t_max, slack / A_j_u)
            elif A_j_u < -1e-12: t_min = max(t_min, slack / A_j_u)
        
        if t_min >= t_max:
            samples[i] = x; continue
            
        y_slice = density_func(x) - np.random.exponential(1.0)
        t = np.random.uniform(t_min, t_max)
        x_prop = x + t * u
        
        while density_func(x_prop) < y_slice:
            if t > 0: t_max = t
            else: t_min = t
            if t_max - t_min < 1e-10: break
            t = np.random.uniform(t_min, t_max)
            x_prop = x + t * u
            
        x = x_prop
        samples[i] = x
    return samples

@njit(fastmath=True)
def run_dikin_walk(density_func, A, b, x0, num_samples, r_step=0.15):
    d, m = len(x0), len(b)
    samples = np.zeros((num_samples, d))
    x = np.copy(x0)
    for i in range(num_samples):
        slack_x = np.maximum(b - np.dot(A, x), 1e-10)
        H_x = np.zeros((d, d))
        for j in range(m):
            row = A[j] / slack_x[j]
            for k in range(d):
                for l in range(d): H_x[k, l] += row[k] * row[l]
        for j in range(d): H_x[j, j] += 1e-6
        L_x = np.linalg.cholesky(H_x)
        z = np.random.randn(d)
        x_prop = x + np.linalg.solve(L_x.T, z) * r_step
        
        slack_prop = b - np.dot(A, x_prop)
        feasible = True
        for j in range(m):
            if slack_prop[j] <= 0.0: feasible = False; break
        if not feasible:
            samples[i] = x; continue
            
        H_p = np.zeros((d, d))
        for j in range(m):
            row = A[j] / np.maximum(slack_prop, 1e-10)[j]
            for k in range(d):
                for l in range(d): H_p[k, l] += row[k] * row[l]
        for j in range(d): H_p[j, j] += 1e-6
        L_p = np.linalg.cholesky(H_p)
        
        log_alpha = (density_func(x_prop) - density_func(x)) + \
                    (np.sum(np.log(np.diag(L_p))) - np.sum(np.log(np.diag(L_x)))) - \
                    (np.dot((x - x_prop), np.dot(H_p - H_x, (x - x_prop))) / (2.0 * r_step**2))
        if np.log(np.random.uniform(0.0, 1.0)) < log_alpha: x = x_prop
        samples[i] = x
    return samples

def run_convergence_benchmark():
    d = 20
    mu = np.zeros(d)
    
    # Highly skewed target across ALL axes (geomspace), keeping it diagonal 
    # so the independent marginal ground truth remains analytically exact.
    # Max variance 5.0 (std ~2.2) perfectly fits inside the [-5, 5] bounding box.
    # Condition Number = 5.0 / 0.01 = 500.
    eigenvalues = np.geomspace(5.0, 0.01, d)
    cov = np.diag(eigenvalues)
    inv_cov = np.diag(1.0 / eigenvalues)
    
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.concatenate([np.full(d, 5.0), np.full(d, 5.0)])
    
    @njit(fastmath=True)
    def log_prob(x):
        for j in range(len(b)):
            if np.dot(A[j], x) > b[j]: return -np.inf
        return -0.5 * np.sum(x * (inv_cov @ x))

    # 1. GENERATE GROUND TRUTH
    print("Generating exact Ground Truth samples from true mathematical marginal...")
    std_0 = np.sqrt(5.0)
    a, b_trunc = -5.0 / std_0, 5.0 / std_0
    ground_truth_samples = truncnorm.rvs(a, b_trunc, loc=0, scale=std_0, size=50000)

    max_steps = 30000
    step_intervals = np.arange(1000, max_steps + 1, 1000)
    k_runs = 10
    
    all_whss_distances = []
    all_hrss_distances = []
    
    print(f"Executing {k_runs} Independent Runs of {max_steps} steps each...")
    
    for run in range(k_runs):
        print(f"--- Run {run+1}/{k_runs} ---", end="\r")
        
        # WHSS
        sampler = whss_gaussian(d=d, k=max_steps * 10, sigma=np.eye(d), mu=mu)
        old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
        try:
            # We want WHSS to activate Phase 2 (Warp matrix) early so we can see its true power.
            # 60 anchors is plenty for a 20D space. We set burn-in to 2,000 steps and bypass 
            # the massive 20,000 step safeguard.
            whss_full = sampler._sampling_universal(
                density_cartesian=log_prob, A=A, b=b, 
                burn_in_samples=2000, max_anchors=60, bypass_safeguards=True
            )
        finally:
            sys.stdout = old_stdout
        whss_chain = whss_full[0]
        
        # HRSS
        hrss_chain = run_hit_and_run(log_prob, A, b, np.zeros(d), max_steps)
        
        whss_dist_run = []
        hrss_dist_run = []
        for t in step_intervals:
            dist_whss = wasserstein_distance(whss_chain[:t, 0], ground_truth_samples)
            dist_hrss = wasserstein_distance(hrss_chain[:t, 0], ground_truth_samples)
            whss_dist_run.append(dist_whss)
            hrss_dist_run.append(dist_hrss)
            
        all_whss_distances.append(whss_dist_run)
        all_hrss_distances.append(hrss_dist_run)

    print(f"\nCompleted {k_runs} runs. Calculating statistics and plotting...")
    
    whss_mean = np.mean(all_whss_distances, axis=0)
    whss_std = np.std(all_whss_distances, axis=0)
    hrss_mean = np.mean(all_hrss_distances, axis=0)
    hrss_std = np.std(all_hrss_distances, axis=0)

    # 4. PLOT RESULTS
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(step_intervals, whss_mean, color='#4285F4', linewidth=3, marker='o', label='WHSS (Our Algorithm)')
    plt.fill_between(step_intervals, whss_mean - whss_std, whss_mean + whss_std, color='#4285F4', alpha=0.2)
    
    plt.plot(step_intervals, hrss_mean, color='#FBBC05', linewidth=3, marker='s', label='HRSS (SOTA Baseline)')
    plt.fill_between(step_intervals, hrss_mean - hrss_std, hrss_mean + hrss_std, color='#FBBC05', alpha=0.2)
    
    plt.title(f'Distributional Convergence (Averaged over {k_runs} Runs)', fontsize=14, fontweight='bold')
    plt.xlabel('MCMC Iterations (Chain Length)', fontsize=12)
    plt.ylabel('Wasserstein Distance (Lower is Better)', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('test/results', exist_ok=True)
    out_path = 'test/results/convergence_distance_plot.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    print(f"Benchmark complete. Artifacts saved to {out_path}")

if __name__ == "__main__":
    run_convergence_benchmark()
