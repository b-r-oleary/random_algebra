from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
import numpy as np

class vector(multi_rv_frozen):
    """
    this is a method for making multivariate distributions from
    single variable distributions
    """
    def __init__(self, dists):
        self.dists = np.array(dists)

        self.args = tuple(self.dists)
        self.dist = self
        
    def logpdf(self, x):
        return np.array([
            dist.logpdf(x) for dist in self.dists
        ]).T
        
    def pdf(self, x):
        return np.array([
            dist.pdf(x) for dist in self.dists
        ]).T

    def cdf(self, x):
        return np.array([
            dist.cdf(x) for dist in self.dists
        ]).T
    
    def rvs(self, x):
        return np.array([
            dist.rvs(x) for dist in self.dists
        ]).T

    def __getitem__(self, i):
        return self.dists[i]

    def __add__(self, other):
        if isinstance(other, vector):
            return vector(self.dists + other.dists)
        elif isinstance(other, (rv_frozen, float, int)):
            return vector(self.dists + other)
        else:
            raise NotImplementedError()

    # def __mul__(self)