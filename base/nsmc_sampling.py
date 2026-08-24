from ..methods.sampling import sampling
from ..methods.utility import utilities
from ..methods.convergence import convergence

class nsmc_sampling(sampling, utilities, convergence):
    """
    Parent class to generalize the Warped Hybrid Slice Sampling (WHSS) MCMC engine.
    It inherits the general sampling loop, utility functions, and convergence diagnostics.
    
    Parameters:
        d: dimension of the space
        a: length of the bounding box (retained for legacy plotting/bounding)
        k: required number of MCMC samples
    """
    def __init__(self, d, a, k):
        self.d = d
        self.a = a
        self.k = k