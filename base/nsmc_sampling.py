from ..methods.sampling import sampling
from ..methods.importance import importance_sampling
from ..methods.utility import utilities

class nsmc_sampling(sampling, importance_sampling, utilities):
    """
    Parent class to generalise the nsmc_sampling. It has the general functions used for any sampling density.
    Parameters:
        d: dimsion of the cube
        a: length of the cube.
        k: required number of accepted samples
    """
    def __init__(self,d,a,k):
        self.d=d
        self.a=a
        self.k=k