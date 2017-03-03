from __future__ import division
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from scipy.special import binom as binom_coef
from scipy.integrate import quad as quad0
from scipy.stats import norm, beta, cauchy, chi2, uniform
import numpy as np


quad = lambda f, a, b, **kwargs: quad0(f, a, b, limit=200, **kwargs)


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

# wrap the add method to capture some special cases for speed:
def scale_special_cases(scale):

    def modified_scale(dist, factor):
        name = dist.get_name()
        if name == "norm":
            return norm(dist.mean() * factor, dist.std() * factor)
        else:
            return scale(dist, factor)

    return modified_scale
    
scale = scale_special_cases(
            scale_gen(name="scale")
        )


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

# wrap the add method to capture some special cases for speed:
def offset_special_cases(offset):

    def modified_offset(dist, value):
        name = dist.get_name()
        if name == "norm":
            return norm(dist.mean() + value, dist.std())
        else:
            return offset(dist, value)

    return modified_offset
    
offset = offset_special_cases(
            offset_gen(name="offset")
         )

class add_gen(rv_continuous):
    """
    add two random variables
    """
    
    def _pdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        pdf_integrand = lambda y, z: dist0.pdf(z - y) * dist1.pdf(y)
        pdf = np.vectorize(
                lambda z: quad(pdf_integrand, -np.inf, np.inf, args=(z,))[0]
              )
        return pdf(x)
    
    def _cdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        cdf_integrand = lambda y, z: dist0.cdf(z - y) * dist1.pdf(y)
        cdf = np.vectorize(
                lambda z: quad(cdf_integrand, -np.inf, np.inf, args=(z,))[0]
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

# wrap the add method to capture some special cases for speed:
def add_special_cases(add):

    def modified_add(dist0, dist1):
        name0 = dist0.get_name()
        name1 = dist1.get_name()
        if name0 == name1:
            if name0 == "norm":
                mean0, var0 = dist0.stats()
                mean1, var1 = dist1.stats()
                return norm(mean0 + mean1, np.sqrt(var0 + var1))
            if name0 == "cauchy":
                loc0, loc1 = dist0.median(), dist1.median()
                scale0 = dist0.ppf(.75) - loc0
                scale1 = dist1.ppf(.75) - loc1
                return cauchy(loc0 + loc1, scale0 + scale1)
            if name0 == "chi2":
                df0, = dist0.args
                df1, = dist1.args
                return chi2(df0 + df1)

        return add(dist0, dist1)

    return modified_add

add = add_special_cases(
            add_gen(name="add")
      )


class multiply_gen(rv_continuous):
    """
    multiply two random variables
    """
    
    def _pdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        pdf_integrand = lambda y, z: dist0.pdf(z / y) * dist1.pdf(y) / np.abs(y)
        pdf = np.vectorize(
                lambda z: quad(pdf_integrand, -np.inf, np.inf, args=(z,))[0]
              )
        return pdf(x)
    
    def _cdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        cdf_integrand = lambda y, z: dist0.cdf(z / y) * dist1.pdf(y)
        cdf = np.vectorize(
                lambda z: quad(cdf_integrand, -np.inf, np.inf, args=(z,))[0]
              )
        return cdf(x)
    
    def _rvs(self, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        return dist0.rvs(size=self._size) * dist1.rvs(size=self._size)
    
    def _munp(self, n, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        return dist0.moment(n) * dist1.moment(n)
    
    def _argcheck(self, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        conditions = [
            isinstance(dist0, rv_frozen),
            isinstance(dist1, rv_frozen)
        ]
        return all(conditions)
    
multiply = multiply_gen(name="multiply")


class inverse_gen(rv_continuous):
    """
    find the multiplicative inverse of a random variable
    """
    
    def _pdf(self, x, dist):
        dist, = extract_first(dist)
        return dist.pdf(1 / x)/ (x**2)
    
    def _cdf(self, x, dist):
        dist, = extract_first(dist)
        return 1 - dist.cdf(1 / x)
    
    def _rvs(self, dist):
        dist, = extract_first(dist)
        return 1 / dist.rvs(size=self._size)
    
    def _munp(self, n, dist):
        dist, = extract_first(dist)
        return quad(lambda x: dist.pdf(x) * (1.0 / x)**n, -np.inf, np.inf)[0]
    
    def _argcheck(self, dist):
        dist, = extract_first(dist)
        conditions = [
            isinstance(dist, rv_frozen),
        ]
        return all(conditions)

inverse = inverse_gen(name="inverse")

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
        
        items = prior.rvs(size=(N * factor,))
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

# class either_gen(rv_continuous):

#     def _pdf(self, x, dist0, dist1):
#         dist0, dist1 = extract_first(dist0, dist1)
#         return (dist0.pdf(x) + dist1.pdf(x)) / 2.0

#     def _cdf(self, x, dist0, dist1):
#         dist0, dist1 = extract_first(dist0, dist1)
#         return (dist0.cdf(x) + dist1.cdf(x)) / 2.0

#     def _rvs(self, likelihood, prior, factor=30):
#         dist0, dist1 = extract_first(dist0, dist1)
        
#         dist = np.random.choice([dist0, dist1])
#         return dist.
    
#     def _munp(self, n, likelihood, prior):
#         likelihood, prior = extract_first(likelihood, prior)
#         self.__get_norm(likelihood, prior)
#         return quad(lambda x: x**n * self._pdf(x, likelihood, prior), self.a, self.b)[0]
    
#     def _argcheck(self, likelihood, prior):
#         likelihood, prior = extract_first(likelihood, prior)
#         conditions = [
#             isinstance(likelihood, rv_frozen),
#             isinstance(prior,      rv_frozen),
#         ]
#         return all(conditions)

# wrap the add method to capture some special cases for speed:
def posterior_special_cases(posterior):

    def modified_posterior(dist0, dist1):
        name0 = dist0.get_name()
        name1 = dist1.get_name()
        if name0 == name1:
            if name0 == "norm":
                mean0, var0 = dist0.stats()
                mean1, var1 = dist1.stats()

                var = 1 / (1 / var0 + 1 / var1)
                mean = var * ( mean0 / var0 + mean1 / var1)

                return norm(mean, np.sqrt(var))

            if name0 == "beta":
                a0, b0 = dist0.args
                a1, b1 = dist1.args
                return beta(a0 + a1 - 1, b0 + b1 - 1)

            if name0 == "uniform":
                mean0, width0  = dist0.mean(), dist0.std() * np.sqrt(3)
                mean1, width1  = dist1.mean(), dist1.std() * np.sqrt(3)

                lower0, upper0 = mean0 - width0, mean0 + width0
                lower1, upper1 = mean1 - width1, mean1 + width1

                lower = max([lower0, lower1])
                upper = min([upper0, upper1])

                if lower >= upper:
                    raise NotImplementedError('i have not yet handled this case')

                return uniform(lower, upper - lower)

        return posterior(dist0, dist1)

    return modified_posterior

posterior = posterior_special_cases(
                posterior_gen(name="posterior")
            )

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