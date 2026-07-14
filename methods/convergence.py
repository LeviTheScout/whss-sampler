import numpy as np
from scipy.spatial.distance import pdist, squareform


class convergence():
    

    def ksd(self, accepted, density):
        """
        accepted: list of samples each sampling being (theta, r) 
                  Note: theta is assumed to be a d-dimensional unit vector to match `r * theta` 
                  in the provided density function. If it is d-1 angles, convert to Cartesian first.
        density: the density function yielding log(f(r)) + (d-1)*log(r)
        """
        N = len(accepted)
        if N < 2:
            raise ValueError("KSD requires at least 2 samples.")
            
        # 1. Convert samples to Cartesian coordinates (X)
        X = np.array([r * theta for theta, r in accepted])
        d = X.shape[1]
        
        # 2. Compute the Score Vectors using finite differences
        # s_i = \nabla_x log p(x) | x = x_i
        S = np.zeros_like(X)
        eps = 1e-5
        
        for i in range(N):
            x = X[i]
            grad = np.zeros(d)
            for j in range(d):
                # Forward step
                x_plus = x.copy()
                x_plus[j] += eps
                r_plus = np.linalg.norm(x_plus)
                theta_plus = x_plus / r_plus
                # Evaluate Cartesian log-density: log_p(x) = polar_density - log_volume
                polar_log_plus = density(np.array([r_plus]), np.array([theta_plus]))[0]
                log_p_plus = polar_log_plus - (d - 1) * np.log(r_plus + 1e-10)
                
                # Backward step
                x_minus = x.copy()
                x_minus[j] -= eps
                r_minus = np.linalg.norm(x_minus)
                theta_minus = x_minus / r_minus
                polar_log_minus = density(np.array([r_minus]), np.array([theta_minus]))[0]
                log_p_minus = polar_log_minus - (d - 1) * np.log(r_minus + 1e-10)
                
                # Central difference
                grad[j] = (log_p_plus - log_p_minus) / (2 * eps)
            S[i] = grad

        # 3. Choose and Tune the Kernel (Median Heuristic)
        # Compute the Euclidean distance ||x_i - x_j||_2 for all unique pairs
        dist_condensed = pdist(X, metric='euclidean')
        # Set c equal to the median of these distances
        c = np.median(dist_condensed)
        c2 = c**2
        dist_sq_matrix = squareform(dist_condensed**2)

        # 4. Compute the Pairwise Stein Kernel Matrix (U)
        U = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue # Skip the diagonal for the unbiased U-statistic
                    
                diff_x = X[i] - X[j]
                diff_s = S[i] - S[j]
                dist2 = dist_sq_matrix[i, j]
                
                # Base IMQ Kernel: (c^2 + ||x_i - x_j||_2^2)^(-0.5)
                k_val = (c2 + dist2)**(-0.5)
                
                # Term 1: (s_i^T s_j) * k(x_i, x_j)
                term1 = np.dot(S[i], S[j]) * k_val
                
                # Term 2: (x_i - x_j)^T (s_i - s_j) / (c^2 + ||x_i - x_j||_2^2)^1.5
                term2 = np.dot(diff_x, diff_s) / ((c2 + dist2)**1.5)
                
                # Term 3 (Trace term): d / (c^2 + dist2)^1.5 - 3 * dist2 / (c^2 + dist2)^2.5
                trace_term = (d / ((c2 + dist2)**1.5)) - (3 * dist2 / ((c2 + dist2)**2.5))
                
                # Construct U_{i,j}
                U[i, j] = term1 + term2 + trace_term

        # 5. Calculate the Final KSD Score
        # Aggregate matrix U using the unbiased U-statistic to remove self-correlation bias
        ksd_sq = np.sum(U) / (N * (N - 1))
        
        # Final evaluation metric is \sqrt{KSD^2}
        return np.sqrt(max(ksd_sq, 0.0))


    def projection_testing(self,accepted,g_r):
        """
        - Generte random direction.
        - Take projection of samples along that direction.
        - Projection of density (dont know how to do this).
        - measure distance (Primarily KS, but could use others too).
        """

        return
