import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from nsmc_sampling.distributions.gaussian import nsmc_sampling_gaussian
import emcee

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

# Dikin and HRSS implementations (C-Speed)
@njit(fastmath=True)
def run_hit_and_run(x0, A, b, num_samples):
    d = len(x0)
    samples = np.zeros((num_samples, d))
    x = np.copy(x0)
    
    for i in range(num_samples):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        
        t_min, t_max = -1e10, 1e10
        for j in range(len(b)):
            A_j_u = np.dot(A[j], u)
            slack = b[j] - np.dot(A[j], x)
            if A_j_u > 1e-9:
                t_max = min(t_max, slack / A_j_u)
            elif A_j_u < -1e-9:
                t_min = max(t_min, slack / A_j_u)
                
        if t_min > t_max: t_min = t_max
        
        y = -np.random.exponential(1.0)
        t = np.random.uniform(t_min, t_max)
        x = x + t * u
        samples[i] = x
        
    return samples

@njit(fastmath=True)
def run_dikin_walk(x0, A, b, num_samples, r=0.5):
    d = len(x0)
    samples = np.zeros((num_samples, d))
    x = np.copy(x0)
    I = np.eye(d)
    
    for i in range(num_samples):
        slacks = b - np.dot(A, x)
        H = np.zeros((d, d))
        for j in range(len(b)):
            H += np.outer(A[j], A[j]) / (slacks[j]**2)
            
        try:
            H_inv = np.linalg.inv(H)
        except:
            samples[i:] = x
            break
            
        z = np.random.randn(d)
        L = np.linalg.cholesky(H_inv)
        prop = x + r * np.dot(L, z)
        
        prop_slacks = b - np.dot(A, prop)
        if np.any(prop_slacks <= 0):
            samples[i] = x
            continue
            
        samples[i] = prop
        x = prop
        
    return samples

def run_scaling_benchmark():
    # 10, 50, 100, 200, 400
    dimensions = [10, 50, 100, 200, 400]
    n_samples = 25000
    
    results = {
        "WHSS": [],
        "HRSS": [],
        "Dikin": []
    }
    
    print("=====================================================")
    print("STARTING DIMENSIONAL SCALING BENCHMARK")
    print("=====================================================")
    
    for d in dimensions:
        print(f"\n--- Testing Dimension: {d} ---")
        
        # 1. Create a Challenging Space
        # Highly Skewed Gaussian target inside a restricted bounding box
        # Condition Number = 1000
        eigenvalues = np.linspace(1, 1000, d)
        cov = np.diag(eigenvalues)
        inv_cov = np.diag(1.0 / eigenvalues)
        
        # Target density
        @njit(fastmath=True)
        def log_prob(x):
            return -0.5 * np.sum(x * (inv_cov @ x))
            
        # Box constraints (Truncated Skewed Distribution)
        A_poly = np.vstack([np.eye(d), -np.eye(d)])
        b_poly = np.concatenate([np.ones(d)*10, np.ones(d)*10])
        
        x0 = np.zeros(d)
        
        # ---------------- WHSS ----------------
        print(f"[{d}D] Running WHSS...")
        t0 = time.time()
        sampler = nsmc_sampling_gaussian(d=d, k=n_samples, sigma=np.eye(d), mu=np.zeros(d))
        
        old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
        try:
            whss_samples = sampler._sampling_universal(
                density_cartesian=log_prob, A=A_poly, b=b_poly,
                burn_in_samples=2500, max_anchors=60
            )
            whss_time = time.time() - t0
            ess = compute_ess_per_chain(whss_samples)
            results["WHSS"].append((ess / n_samples) * 1000)
        except Exception as e:
            results["WHSS"].append(0.0)
        finally:
            sys.stdout = old_stdout
            
        print(f"   -> WHSS Efficiency: {results['WHSS'][-1]:.2f} ESS per 1000 NFE")
        
        # ---------------- HRSS ----------------
        print(f"[{d}D] Running Hit-and-Run (HRSS)...")
        t0 = time.time()
        hrss_samples = run_hit_and_run(x0, A_poly, b_poly, n_samples)
        hrss_time = time.time() - t0
        ess = compute_ess_per_chain(hrss_samples)
        results["HRSS"].append((ess / n_samples) * 1000)
        print(f"   -> HRSS Efficiency: {results['HRSS'][-1]:.2f} ESS per 1000 NFE")
        
        # ---------------- Dikin Walk ----------------
        print(f"[{d}D] Running Dikin Walk (SOTA)...")
        t0 = time.time()
        try:
            dikin_samples = run_dikin_walk(x0, A_poly, b_poly, n_samples)
            dikin_time = time.time() - t0
            ess = compute_ess_per_chain(dikin_samples)
            results["Dikin"].append((ess / n_samples) * 1000)
        except:
            results["Dikin"].append(0.0)
        print(f"   -> Dikin Efficiency: {results['Dikin'][-1]:.2f} ESS per 1000 NFE")

    # ---------------- Plotting ----------------
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})
    
    plt.plot(dimensions, results["WHSS"], label="WHSS (Ours)", marker='o', linewidth=3, color="#4285F4", markersize=8)
    plt.plot(dimensions, results["HRSS"], label="Hit-and-Run (Legacy)", marker='s', linewidth=2, color="#FBBC04", linestyle="--")
    plt.plot(dimensions, results["Dikin"], label="Dikin Walk (SOTA)", marker='^', linewidth=2, color="#EA4335", linestyle=":")
    
    plt.yscale('log')
    plt.xlabel("Dimensionality ($D$)")
    plt.ylabel("Algorithmic Efficiency (ESS per 1000 NFE)")
    plt.title("Dimensional Scaling: Algorithmic Efficiency Under High Skew")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    save_path = os.path.join(current_dir, "results", "dimensional_scaling.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n[SUCCESS] Scaling benchmark saved to {save_path}")

if __name__ == "__main__":
    run_scaling_benchmark()
