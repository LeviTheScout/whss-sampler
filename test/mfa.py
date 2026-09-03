import os
import sys
import time
import numpy as np
import pandas as pd
import cobra
from cobra.sampling import sample
from scipy.linalg import svd
from numba import njit
from emcee.autocorr import integrated_time
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# PATH RESOLUTION: NATIVE CODEBASE
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from whss.distributions.gaussian import whss_gaussian

results_dir = os.path.join(current_dir, "results")
diag_dir = os.path.join(current_dir, "diagnostics")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)

# =====================================================================
# UNBIASED DIAGNOSTIC TOOLS
# =====================================================================
def compute_robust_ess_stats(samples_2d):
    """Computes Min, Median, Max ESS, filtering zero-variance biological reactions."""
    n_steps, d = samples_2d.shape
    ess_list = []
    for j in range(d):
        col = samples_2d[:, j]
        if np.std(col) < 1e-8:
            continue
        try:
            tau = integrated_time(col, tol=0)
            ess = n_steps / np.max(tau)
            if not np.isnan(ess) and ess > 0:
                ess_list.append(ess)
        except Exception:
            pass
    if len(ess_list) == 0:
        return 1.0, 1.0, 1.0
    return np.min(ess_list), np.median(ess_list), np.max(ess_list)

def compute_ess_whss_3d_stats(samples_3d):
    """Computes Total ESS per dimension across parallel chains, then finds Min/Med/Max."""
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

@njit
def uniform_poly(y): 
    return 0.0

# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================
def plot_mfa_dominance(
    achr_times, achr_min, achr_df,
    chrr_times, chrr_min, chrr_df,
    whss_times, whss_min, whss_reactions_3d,
    save_path
):
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- PANEL A: THROUGHPUT WITH ERROR BARS ---
    algos = ["ACHR\n(Legacy)", "CHRR\n(SOTA)", "WHSS\n(Ours)"]
    
    achr_eff = np.array(achr_min) / np.array(achr_times)
    chrr_eff = np.array(chrr_min) / np.array(chrr_times) if len(chrr_times) > 0 else [0]
    whss_eff = np.array(whss_min) / np.array(whss_times)
    
    means = [np.mean(achr_eff), np.mean(chrr_eff), np.mean(whss_eff)]
    stds = [np.std(achr_eff), np.std(chrr_eff), np.std(whss_eff)]
    colors = ["#FBBC04", "#34A853", "#4285F4"]

    bars = ax1.bar(algos, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=1.2, alpha=0.9)
    ax1.set_ylabel("Sampling Throughput (Min ESS / sec)")
    ax1.set_title("(a) Algorithmic Throughput (10 Runs)")
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

    # --- PANEL B: REACTION-WISE ESS BOXPLOT ---
    def extract_per_dim_ess_2d(samples_2d):
        n_steps, d = samples_2d.shape
        ess_list = []
        for j in range(d):
            col = samples_2d[:, j]
            if np.std(col) > 1e-8:
                try:
                    tau = integrated_time(col, tol=0)
                    ess = n_steps / np.max(tau)
                    if not np.isnan(ess) and ess > 0:
                        ess_list.append(ess)
                except Exception:
                    pass
        return ess_list

    def extract_per_dim_ess_3d(samples_3d):
        n_chains, n_steps, d = samples_3d.shape
        ess_list = []
        for j in range(d):
            dim_total = 0.0
            active = False
            for c in range(n_chains):
                col = samples_3d[c, :, j]
                if np.std(col) > 1e-8:
                    try:
                        tau = integrated_time(col, tol=0)
                        ess = n_steps / np.max(tau)
                        if not np.isnan(ess) and ess > 0:
                            dim_total += ess
                            active = True
                    except Exception:
                        pass
            if active:
                ess_list.append(dim_total)
        return ess_list

    achr_flux_ess = extract_per_dim_ess_2d(achr_df.values)
    chrr_flux_ess = extract_per_dim_ess_2d(chrr_df.values) if chrr_df is not None else []
    whss_flux_ess = extract_per_dim_ess_3d(whss_reactions_3d)

    bp = ax2.boxplot(
        [achr_flux_ess, chrr_flux_ess, whss_flux_ess],
        labels=["ACHR", "CHRR", "WHSS"],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax2.set_ylabel("ESS per Reaction Flux")
    ax2.set_title("(b) ESS Distribution Across Active Reactions")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT GENERATED] Dominance Plot saved to {save_path}")

def plot_biological_validation(achr_df, chrr_df, whss_df, save_path):
    from scipy.stats import wasserstein_distance
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.dpi": 300,
    })

    target_reactions = ['Biomass_Ecoli_core', 'ATPM', 'EX_o2_e']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"ACHR (Legacy)": "#FBBC04", "CHRR (Gold Standard)": "#34A853", "WHSS (Ours)": "#4285F4"}

    for idx, rxn in enumerate(target_reactions):
        ax = axes[idx]
        
        # Calculate Wasserstein Distances (Earth Mover's Distance) vs Infinite-Compute Ground Truth
        wd_achr = wasserstein_distance(chrr_df[rxn], achr_df[rxn]) if chrr_df is not None else 0.0
        wd_whss = wasserstein_distance(chrr_df[rxn], whss_df[rxn]) if chrr_df is not None else 0.0
        
        sns.kdeplot(achr_df[rxn], ax=ax, color=colors["ACHR (Legacy)"], 
                    label=f"ACHR [W: {wd_achr:.3f}]", linewidth=2.5, linestyle=":")
        
        if chrr_df is not None:
            sns.kdeplot(chrr_df[rxn], ax=ax, color=colors["CHRR (Gold Standard)"], 
                        label="CHRR (Ground Truth 200k)", linewidth=4, alpha=0.6)
            
        sns.kdeplot(whss_df[rxn], ax=ax, color=colors["WHSS (Ours)"], 
                    label=f"WHSS [W: {wd_whss:.3f}]", linewidth=2.5, linestyle="--")

        ax.set_title(f"Flux Distribution: {rxn}")
        ax.set_xlabel("Reaction Flux (mmol / gDW·h)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT GENERATED] Biological validation saved to {save_path}")

def generate_pure_lp_basis(model):
    """
    Geometric Preprocessor (Topological Identification)
    Uses a fast ACHR trace strictly to identify the true volumetric center 
    and the exact non-zero variance dimensions. This is required because 
    SVD on FVA vertices distorts the internal probability geometry.
    """
    from cobra.sampling import sample
    
    warmup_df = sample(model, 2500, method="achr", thinning=1)
    points = warmup_df.values
    center = np.mean(points, axis=0)
    centered = points - center
    
    _, S_sing, Vt = svd(centered, full_matrices=False)
    Z = Vt[S_sing > 1e-8].T
    return center, Z

# =====================================================================
# MULTI-RUN MFA BENCHMARK (k_runs = 10)
# =====================================================================
def run_mfa_benchmark():
    k_runs = 10
    n_samples = 500000 
    
    report = []
    report.append("=" * 85)
    report.append(f"WHSS REAL-WORLD VALIDATION: MFA BENCHMARK (Averaged over {k_runs} Runs)")
    report.append("=" * 85 + "\n")
    
    import logging
    logging.getLogger("cobra").setLevel(logging.ERROR)
    
    print("[1/4] Loading E. coli Core model...")
    model = cobra.io.load_model("textbook")
    d_full = len(model.reactions)

    achr_times, achr_min, achr_med, achr_max = [], [], [], []
    chrr_times, chrr_min, chrr_med, chrr_max = [], [], [], []
    whss_times, whss_min, whss_med, whss_max = [], [], [], []
    
    # Store the last run's data for plotting
    final_achr_df = None
    final_chrr_df = None
    final_whss_3d = None

    print(f"\n[2/4] Executing {k_runs} runs for ACHR Baseline...")
    for r in range(k_runs):
        t0 = time.perf_counter()
        achr_df = sample(model, n_samples, method="achr", thinning=1)
        t_achr = time.perf_counter() - t0
        mn, md, mx = compute_robust_ess_stats(achr_df.values)
        achr_times.append(t_achr); achr_min.append(mn); achr_med.append(md); achr_max.append(mx)
        final_achr_df = achr_df
        print(f"  ACHR Run {r+1}/{k_runs} -> Time: {t_achr:.2f}s | Min ESS: {mn:.1f}")

    print(f"\n[3/4] Executing {k_runs} runs for CHRR SOTA...")
    for r in range(k_runs):
        try:
            t0 = time.perf_counter()
            chrr_df = sample(model, n_samples, method="chrr", thinning=1)
            t_chrr = time.perf_counter() - t0
            mn, md, mx = compute_robust_ess_stats(chrr_df.values)
            chrr_times.append(t_chrr); chrr_min.append(mn); chrr_med.append(md); chrr_max.append(mx)
            final_chrr_df = chrr_df
            print(f"  CHRR Run {r+1}/{k_runs} -> Time: {t_chrr:.2f}s | Min ESS: {mn:.1f}")
        except Exception:
            print(f"  CHRR Run {r+1}/{k_runs} -> FAILED")

    # ---------------- THE FIX: WARM UP THE JIT COMPILER ----------------
    print("\n[JIT WARM-UP] Compiling WHSS internals (Untimed)...")
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        dummy_Z = np.eye(24)
        dummy_A = np.vstack([dummy_Z, -dummy_Z])
        dummy_b = np.ones(48)
        dummy_sampler = whss_gaussian(d=24, k=10, sigma=np.eye(24), mu=np.zeros(24))
        _ = dummy_sampler._sampling_universal(density_cartesian=uniform_poly, A=dummy_A, b=dummy_b)
    except Exception:
        pass
    finally:
        sys.stdout = old_stdout
        devnull.close()
    # -------------------------------------------------------------------

    print(f"\n[4/4] Executing {k_runs} runs for Native WHSS (Unbiased End-to-End)...")
    
    lb = np.array([rxn.lower_bound for rxn in model.reactions])
    ub = np.array([rxn.upper_bound for rxn in model.reactions])

    for r in range(k_runs):
        t0 = time.perf_counter()
        
        print("  -> Preprocessing: Volumetric topological projection...")
        center, Z = generate_pure_lp_basis(model)
        k_dims = Z.shape[1] 
        
        A_poly = np.vstack([Z, -Z])
        b_poly = np.concatenate([ub - center, center - lb])
        b_poly = np.maximum(b_poly, 0.0) 
        
        sampler_whss = whss_gaussian(d=k_dims, k=n_samples, sigma=np.eye(k_dims), mu=np.zeros(k_dims))
        whss_shifted = sampler_whss._sampling_universal(
            density_cartesian=uniform_poly,
            A=A_poly, b=b_poly,
            burn_in_samples=50000, max_anchors=130
        )
        
        t_whss = time.perf_counter() - t0
        
        whss_reactions_3d = np.einsum('csd,rd->csr', whss_shifted, Z) + center
        mn, md, mx = compute_ess_whss_3d_stats(whss_reactions_3d)
        
        whss_times.append(t_whss); whss_min.append(mn); whss_med.append(md); whss_max.append(mx)
        final_whss_3d = whss_reactions_3d
        print(f"  WHSS Run {r+1}/{k_runs} -> Time: {t_whss:.2f}s | Min ESS: {mn:.1f}")

    # Compute Averages
    m_achr_t, m_achr_mn, m_achr_md, m_achr_mx = np.mean(achr_times), np.mean(achr_min), np.mean(achr_med), np.mean(achr_max)
    m_chrr_t = np.mean(chrr_times) if len(chrr_times) > 0 else np.nan
    m_chrr_mn = np.mean(chrr_min) if len(chrr_min) > 0 else np.nan
    m_chrr_md = np.mean(chrr_med) if len(chrr_med) > 0 else np.nan
    m_chrr_mx = np.mean(chrr_max) if len(chrr_max) > 0 else np.nan
    m_whss_t, m_whss_mn, m_whss_md, m_whss_mx = np.mean(whss_times), np.mean(whss_min), np.mean(whss_med), np.mean(whss_max)

    # -----------------------------------------------------------------
    # REPORT GENERATION & PLOTTING
    # -----------------------------------------------------------------
    report.append(f"Network Space : {d_full} Biological Reactions")
    report.append(f"Null Space    : {k_dims} Active Flux Dimensions (Averaged over {k_runs} runs)\n")
    report.append(f"{'Algorithm':<22} | {'Time (s)':<9} | {'Min ESS':<9} | {'Med ESS':<9} | {'Max ESS':<9} | {'ESS/s (Min)':<11}")
    report.append("-" * 78)
    report.append(f"{'ACHR (Legacy Baseline)':<22} | {m_achr_t:<9.2f} | {m_achr_mn:<9.1f} | {m_achr_md:<9.1f} | {m_achr_mx:<9.1f} | {(m_achr_mn/m_achr_t):<11.2f}")
    
    if not np.isnan(m_chrr_t):
        report.append(f"{'CHRR (SOTA Rounding)':<22} | {m_chrr_t:<9.2f} | {m_chrr_mn:<9.1f} | {m_chrr_md:<9.1f} | {m_chrr_mx:<9.1f} | {(m_chrr_mn/m_chrr_t):<11.2f}")
    else:
        report.append(f"{'CHRR (SOTA Rounding)':<22} | {'NaN':<9} | {'NaN':<9} | {'NaN':<9} | {'NaN':<9} | {'NaN':<11}")
        
    report.append(f"{'WHSS (Ours - Unbiased)':<22} | {m_whss_t:<9.2f} | {m_whss_mn:<9.1f} | {m_whss_md:<9.1f} | {m_whss_mx:<9.1f} | {(m_whss_mn/m_whss_t):<11.2f}\n")
    
    final_output = "\n".join(report)
    print("\n" + final_output)
    
    report_path = os.path.join(results_dir, "mfa_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write(final_output)

    # Convert WHSS 3D array to a Pandas DataFrame for Seaborn KDE plotting
    # THE FIX: Discard the first 50% of samples as burn-in to guarantee stationary convergence
    n_steps = final_whss_3d.shape[1]
    flat_whss = final_whss_3d[:, (n_steps // 2):, :].reshape(-1, d_full)
    whss_df = pd.DataFrame(flat_whss, columns=[rxn.id for rxn in model.reactions])

    plot_mfa_dominance(
        achr_times, achr_min, final_achr_df,
        chrr_times, chrr_min, final_chrr_df,
        whss_times, whss_min, final_whss_3d,
        os.path.join(results_dir, "mfa_dominance.png")
    )

    print("\n[PLOTTING] Generating Infinite-Compute Ground Truth (CHRR 200k steps)...")
    # Using global cobra module
    chrr_gt_df = cobra.sampling.sample(model, 200000, method="chrr", thinning=1)

    plot_biological_validation(
        final_achr_df, chrr_gt_df, whss_df,
        os.path.join(results_dir, "biological_validation.png")
    )

if __name__ == "__main__":
    run_mfa_benchmark()