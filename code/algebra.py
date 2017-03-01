from __future__ import division
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from scipy.special import binom as binom_coef
from scipy.integrate import quad
import numpy as np

class scale_gen(rv_continuous):
    """
    scale a random variable by a constant value
    """
    
    def _pdf(self, x, dist, factor):
        dist, factor = extract_first(dist, factor)
        return dist.pdf(x/factor)/np.abs(factor)
    
    def _cdf(self, x, dist, factor):
        dist, factor = extract_first(dist, factor)
        output = dist.cdf(x/factor)
        if np.sign(factor) == -1:
            return 1 - output
        else:
            return output
    
    def _rvs(self, dist, factor):
        dist, factor = extract_first(dist, factor)
        return dist.rvs(size=self._size) * factor
    
    def _munp(self, n, dist, factor):
        dist, factor = extract_first(dist, factor)
        return dist.moment(n) * (factor ** n)
    
    def _entropy(self, dist, factor):
        dist, factor = extract_first(dist, factor)
        return dist.entropy() + np.log(factor)
    
    def _argcheck(self, dist, factor):
        dist, factor = extract_first(dist, factor)
        conditions = [
            isinstance(dist, rv_frozen),
            isinstance(factor, (int, float)),
            factor != 0,
        ]
        return all(conditions)
    
scale = scale_gen(name="scale")


class offset_gen(rv_continuous):
    """
    add an offset to a random variable
    """
    
    def _pdf(self, x, dist, offset):
        dist, offset = extract_first(dist, offset)
        return dist.pdf(x - offset)
    
    def _cdf(self, x, dist, offset):
        dist, offset = extract_first(dist, offset)
        return dist.cdf(x - offset)
    
    def _rvs(self, dist, offset):
        dist, offset = extract_first(dist, offset)
        return dist.rvs(size=self._size) + offset
    
    def _munp(self, n, dist, offset):
        dist, offset = extract_first(dist, offset)
        moment = 0
        for k in range(n + 1):
            m_k = dist.moment(k)
            moment += binom_coef(n, k) * m_k * (offset ** (n - k))
        return moment
    
    def _entropy(self, dist, offset):
        dist, offset = extract_first(dist, offset)
        return dist.entropy()
    
    def _argcheck(self, dist, offset):
        dist, offset = extract_first(dist, offset)
        conditions = [
            isinstance(dist, rv_frozen),
            isinstance(offset, (int, float)),
        ]
        return all(conditions)
    
offset = offset_gen(name="offset")

class add_gen(rv_continuous):
    """
    add two random variables
    """
    
    def _pdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        pdf_integrand = lambda y, z: dist0.pdf(z - y) * dist1.pdf(y)
        pdf = np.vectorize(
                lambda z: quad(pdf_integrand, dist1.a, dist1.b, args=(z,))[0]
              )
        return pdf(x)
    
    def _cdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        cdf_integrand = lambda y, z: dist0.cdf(z - y) * dist1.pdf(y)
        cdf = np.vectorize(
                lambda z: quad(cdf_integrand, dist1.a, dist1.b, args=(z,))[0]
              )
        return cdf(x)
    
    def _rvs(self, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        return dist0.rvs(size=self._size) + dist1.rvs(size=self._size)
    
    def _munp(self, n, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        moment = 0
        for k in range(n + 1):
            x = dist0.moment(k)
            y = dist1.moment(n - k)
            moment += binom_coef(n, k) * x * y
        return moment
    
    def _argcheck(self, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        conditions = [
            isinstance(dist0, rv_frozen),
            isinstance(dist1, rv_frozen)
        ]
        return all(conditions)
    
add = add_gen(name="add")


class posterior_gen(rv_continuous):
    """
    generate the posterior distribution given a likelihood distribution and prior distribution
    """
    
    def _pdf(self, x, likelihood, prior):
        likelihood, prior = extract_first(likelihood, prior)
        output = likelihood.pdf(x) * prior.pdf(x) / self.__get_norm(likelihood, prior)
        return output
    
    def _rvs(self, likelihood, prior, factor=30):
        likelihood, prior = extract_first(likelihood, prior)
        
        size = tuple(self._size)
        N = np.prod(size)
        
        items = prior.rvs(size= N * factor)
        prob  = likelihood.pdf(items)
        prob  = prob/sum(prob)
        
        return np.random.choice(items, N, p=prob).reshape(size)
    
    def _munp(self, n, likelihood, prior):
        likelihood, prior = extract_first(likelihood, prior)
        self.__get_norm(likelihood, prior)
        return quad(lambda x: x**n * self._pdf(x, likelihood, prior), self.a, self.b)[0]
    
    def _argcheck(self, likelihood, prior):
        likelihood, prior = extract_first(likelihood, prior)
        conditions = [
            isinstance(likelihood, rv_frozen),
            isinstance(prior,      rv_frozen),
        ]
        return all(conditions)
    
    def __get_norm(self, likelihood, prior):
        object_hash = hash(likelihood) * hash(prior)
        if hasattr(self, "__norm"):
            if object_hash == self.__object_hash:
                return self.__norm
            
        self.a = max([likelihood.a, prior.a])
        self.b = min([likelihood.b, prior.b])
        
        self.__norm = quad(lambda x: likelihood.pdf(x) * prior.pdf(x), self.a, self.b)[0]
        self.__object_hash = object_hash
        return self.__norm
    
posterior = posterior_gen(name="posterior")


def extract_first(*args):
    """
    since scipy.stats distributions are fed an array of inputs,
    but we dont want to work with an array of distribution inputs,
    lets bypass the array manipulation by just grabbing the first entry
    """
    outputs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            output = np.atleast_1d(arg)[0]
        else:
            output = arg
        outputs.append(output)
    return tuple(outputs)