from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
import numpy as np

class vector(multi_rv_frozen):
    """
    this is a method for making multivariate distributions from
    single variable distributions
    """
    def __init__(self, dists):
        self.dists = dists
        
    def logpdf(self, x):
        return np.array([
            dist.logpdf(x) for dist in self.dists
        ]).T
        
    def pdf(self, x):
        return np.array([
            dist.pdf(x) for dist in self.dists
        ]).T
    
    def rvs(self, x):
        return np.array([
            dist.rvs(x) for dist in self.dists
        ]).T