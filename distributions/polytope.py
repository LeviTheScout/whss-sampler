import numpy as np
from numba import njit, prange
from ..base.nsmc_sampling import nsmc_sampling
from ..methods.importance import get_polytope_bounds

class nsmc_sampling_polytope(nsmc_sampling):
    def __init__(self, d=30, k=100000, A=None, b=None):
        super().__init__(d, 1e6, k)
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

    def R(self, thetas):
        """
        OVERRIDE: Forces the Phase 1.5 Scout to bound its ray-casting strictly
        by the exact geometric walls of the Polytope, preventing it from 
        exploring impossible space.
        """
        N = thetas.shape[0]
        R_vec = np.empty(N)
        
        # We must use a Numba wrapper to efficiently check thousands of rays
        @njit(parallel=True)
        def compute_bounds_parallel(thetas_in, A_in, b_in):
            out = np.empty(N)
            origin = np.zeros(thetas_in.shape[1])
            for i in prange(N):
                _, t_max = get_polytope_bounds(A_in, b_in, origin, thetas_in[i])
                out[i] = t_max
            return out
            
        R_vec = compute_bounds_parallel(thetas, self.A, self.b)
        return R_vec

    def f_cartesian_polytope(self):
        @njit
        def density_cartesian(x):
            return 0.0 # Pure Uniform Polytope Sampling!
        return density_cartesian

    def get_samples(self, batch_size=3256, burn_in_samples=5000, max_anchors=50):
        target_density = self.f_cartesian_polytope()
        
        # Pass A and b down into the universal sampler so it triggers slice_step_polytope
        return self._sampling_universal(
            density_cartesian=target_density,
            A=self.A, 
            b=self.b,
            batch_size=batch_size,
            burn_in_samples=burn_in_samples,
            max_anchors=max_anchors
        )