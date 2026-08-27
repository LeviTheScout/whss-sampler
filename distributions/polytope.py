import numpy as np
from numba import njit
from ..base.nsmc_sampling import nsmc_sampling

@njit
def get_polytope_R_max_batch(A, b, thetas):
    """
    Calculates the exact maximum radius R from the origin 
    to the polytope walls for a batch of scout rays.
    """
    n_rays, d = thetas.shape
    m = A.shape[0]
    R_max = np.full(n_rays, 1e20)
    
    for i in range(n_rays):
        for j in range(m):
            denom = np.dot(A[j], thetas[i])
            # We only care about forward intersections from the origin
            if denom > 1e-14:
                r = b[j] / denom
                if r < R_max[i]:
                    R_max[i] = r
    return R_max


class nsmc_sampling_polytope(nsmc_sampling):
    """
    WHSS Target Class for a Linearly Constrained Polytope (Ax <= b).
    """
    def __init__(self, d, k, A, b):
        super().__init__(d, 100.0, k) 
        self.A = A
        self.b = b

    def R(self, thetas):
        """
        OVERRIDE: Forces the Warm-Up Scouts to bounce off the Polytope walls 
        instead of the dummy hypercube.
        """
        return get_polytope_R_max_batch(self.A, self.b, thetas)

    def f_cartesian_polytope(self):
        @njit
        def density_cartesian(x):
            return 0.0
        return density_cartesian

    def get_samples(self, batch_size=3256, burn_in_samples=10000, max_anchors=50):
        target_density = self.f_cartesian_polytope()
        
        mcmc_samples = self._sampling_universal(
            density_cartesian=target_density,
            A=self.A,
            b=self.b,
            batch_size=batch_size,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )
        return mcmc_samples