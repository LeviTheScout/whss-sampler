# WHSS (Warped Hybrid Slice Sampler) - Project Context for AI Agents

**Target Goal:** Finalize the algorithm and its benchmarks for submission to a top-tier peer-reviewed journal. 

## 1. The Algorithm (WHSS)
WHSS is a state-of-the-art Markov Chain Monte Carlo (MCMC) algorithm designed strictly for sampling from high-dimensional, highly skewed constrained spaces (e.g., Biological Metabolic Networks bounded by polytopes $Ax \le b$). 

It solves the "Boundary Trapping" and "Matrix Inversion Collapse" problems that plague legacy SOTA algorithms like Dikin Walk, ACHR, and CHRR.

### Core Mathematical Novelties
1. **Spherical Jacobian Warm-Up:** Uses golden-section line searches along random spherical rays to find density peaks. It explicitly corrects for high-dimensional spherical volumes by adding the Jacobian $(d-1)\log(r)$ to the log-density.
2. **vMF Mixture with Banerjee MLE & Rank-Based Elitism:** Groups discovered peaks into a von Mises-Fisher (vMF) mixture model. It uses the exact Banerjee Maximum Likelihood Estimate for concentration ($\kappa$). To prevent floating-point underflow in spaces $D > 200$, it uses a novel **Rank-Based Elitism** sort instead of probabilistic softmax. 
3. **Phase-Locked Ergodicity:** It builds an empirical covariance matrix ($L$) dynamically using Welford's recursion. Crucially, to preserve Detailed Balance, the algorithm uses **Finite Adaptation**—it explicitly freezes the $L$ matrix at Tick 11. From then on, it is a perfectly symmetric, mathematically unbiased Markov Chain.
4. **3-Part Hybrid Proposal:** Proposes slice directions using a blend of (a) Cholesky Warped Moves, (b) Skeleton Moves between geometric anchors, and (c) Coordinate-aligned fallback moves. 

## 2. Codebase Architecture (`methods/`)
*   `sampling.py`: The orchestrator. Contains `_sampling_universal`, which handles the 4-Phase loop (Ray casting $\rightarrow$ vMF $\rightarrow$ L-Matrix Extraction $\rightarrow$ Parallel Hybrid Slice Sampling).
*   `proposal.py`: Handles the vMF mathematics, mixture weight calculations, and the `PhaseManager` logic.
*   `utility.py`: Highly optimized `@njit` Numba functions for generating hybrid rays, calculating Bessel function normalizations, and running parallel golden-section searches.
*   `importance.py`: Contains the core `slice_step_polytope` logic (bounding chords against $Ax \le b$).

## 3. Benchmarking Suite (`test/`)
We evaluate algorithms purely on **Algorithmic Efficiency (ESS per 1000 NFE)** to isolate mathematical power from Python vs. C wall-clock differences.

1. **`scaling.py` (Dimensional Robustness):** Benchmarks WHSS against Hit-and-Run (HRSS) and Dikin Walk from 10D to 400D under a skewed condition number of 1000. **Status: Proves WHSS dominates HRSS by 10x at 400D (4.11 vs 0.39 ESS/1k NFE) while Dikin matrix inversions collapse.**
2. **`mfa.py` (Biological Correctness):** Samples the core `e_coli_core` model. Proves uniform sampling accuracy using **Wasserstein (EMD) Distance** against an "Infinite Compute" (200,000 step) CHRR ground truth. **Status: Pending final execution. Code is ready.**
3. **`mfa_non_uniform_ijo1366.py` (Genome-Scale Convergence):** Samples a massive 577-dimensional biological model. Uses the **Gelman-Rubin Diagnostic ($\hat{R} < 1.1$)** to prove that the multi-chain parallel structure of WHSS correctly converges to a stationary distribution. **Status: Code injected, pending execution.**

## 4. Operational Rules (CRITICAL FOR AGENTS)
1. **Obey `.agents/rules/rules.md`:** All user-defined rules in this repository must be strictly obeyed.
2. **Process Management (No Overlapping):** Never trigger a long-running benchmark script if another terminal process is already running. The CPU must not be overloaded. Check `command_status` first.
3. **Permission to Kill:** NEVER terminate a background run without explicitly asking the user for permission and providing a clear rationale.
4. **Scientific Honesty:** Do not sugarcoat metrics. Frame limitations (like the `emcee` short-chain penalty at 400D or the high per-step computational overhead) accurately. Academic reviewers respect transparent mathematical limitations.