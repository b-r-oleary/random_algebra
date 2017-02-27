from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from scipy.integrate import quad
import numpy as np

class rv_scale(rv_continuous):
    
    def __init__(self, dist, factor,
                 *args, **kwargs):
        
        self.dist0 = dist
        self.__factor = float(factor)
        
        rv_continuous.__init__(self, *args, **kwargs)
        
        bounds = self.__factor * np.array([self.a, self.b])
        self.a, self.b = min(bounds), max(bounds)
        
    def _pdf(self, x, *args):
        return self.dist0.pdf(x/self.__factor, *args)/np.abs(self.__factor)
    
    def _cdf(self, x, *args):
        output = self.dist0.cdf(x/self.__factor, *args)
        if np.sign(self.__factor) == -1:
            return 1 - output
        else:
            return output
    
    def rvs(self, *args, **kwargs):
        return self.__factor * self.dist0.rvs(*args, **kwargs)
    
    def _stats(self, moments='mv', *args):
        all_moments = 'mvsk'
        mu, mu2, g1, g2 = self.dist0.stats(*args, moments=all_moments)
        moments = [(mu,  1, 'm'),
                   (mu2, 2, 'v'),
                   (g1,  3, 's'),
                   (g2,  4, 'k')]
        output = []
        for moment, power, char in moments:
            if moment is not None:
                moment = moment * self.__factor**power
            output.append(
                moment
            )
        return tuple(output)
    
    
class rv_offset(rv_continuous):
    
    def __init__(self, dist, offset,
                 *args, **kwargs):
        
        self.dist0 = dist
        self.__offset = float(offset)
        
        rv_continuous.__init__(self, *args, **kwargs)
        
        bounds = np.array([self.a, self.b]) + self.__offset
        self.a, self.b = min(bounds), max(bounds)
        
    def _pdf(self, x, *args):
        return self.dist0.pdf(x - self.__offset, *args)
    
    def _cdf(self, x, *args):
        return self.dist0.cdf(x - self.__offset, *args)
    
    def rvs(self, *args, **kwargs):
        return self.__offset + self.dist0.rvs(*args, **kwargs)
    
    def _stats(self, moments='mv', *args):
        all_moments = 'mvsk'
        mu, mu2, g1, g2 = self.dist0.stats(*args, moments=all_moments)
        moments = [(mu,  1, 'm'),
                   (mu2, 0, 'v'),
                   (g1,  0, 's'),
                   (g2,  0, 'k')]
        output = []
        for moment, factor, char in moments:
            if moment is not None:
                moment = moment + self.__offset * factor
            output.append(
                moment
            )
        return tuple(output)
    
    
class rv_sum(rv_continuous):
    """
    
    this is an implementation of a distribution for the difference between two independent random variables
    
    dist0, dist1 are scipy.stats distributions for these random variables and this object represents
    'dist0' - 'dist1'
    
    """
    
    def __init__(self, dist0, dist1, *args, **kwargs):
        
        self.dist0 = dist0
        self.dist1 = dist1
        
        pdf_integrand = lambda y, z: self.dist0.pdf(z - y) * self.dist1.pdf(y)
        cdf_integrand = lambda y, z: self.dist0.cdf(z - y) * self.dist1.pdf(y)
        
        self.__pdf = np.vectorize(
                        lambda z: quad(pdf_integrand, self.dist1.a, self.dist1.b, args=(z,))[0]
                    )
        self.__cdf = np.vectorize(
                        lambda z: quad(cdf_integrand, self.dist1.a, self.dist1.b, args=(z,))[0]
                    )
        
        rv_continuous.__init__(self, *args, **kwargs)
    
    def _pdf(self, x):
        return self.__pdf(x)
    
    def _cdf(self, x):
        return self.__cdf(x)
    
    def rvs(self, *args, **kwargs):
        return self.dist0.rvs(*args, **kwargs) + self.dist1.rvs(*args, **kwargs)
    
    def _stats(self, moments='mv'):
        all_moments = 'mvsk'
        mu_a, mu2_a, g1_a, g2_a = self.dist0.stats(moments=all_moments)
        mu_b, mu2_b, g1_b, g2_b = self.dist1.stats(moments=all_moments)
        
        mu  = mu_a + mu_b
        mu2 = mu2_a + mu2_b
        g1, g2 = None, None
        
        return mu, mu2, g1, g2