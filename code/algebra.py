from __future__ import division
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from scipy.special import binom as binom_coef
from scipy.special import factorial
from scipy.integrate import quad as quad0
from scipy.stats import norm, beta, cauchy, chi2, uniform, lognorm, exponnorm, foldnorm, loggamma, reciprocal
from summation import infinite_sum
import numpy as np

quad = lambda f, a, b, **kwargs: quad0(f, a, b, limit=200, **kwargs)


def push_bounds_to_dist(method):

    def modified_method(*args, **kwargs):
        output = method(*args, **kwargs)
        output.dist.a, output.dist.b = output.a, output.b
        return output

    return modified_method


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
        if name == "scale":
            d, f = dist.args
            return modified_scale(d, f * factor)
        if name == "offset":
            d, o = dist.args
            return offset(
                        modified_scale(
                            d,
                            factor),
                        o * factor)
        else:
            return scale(dist, factor)

    @push_bounds_to_dist
    def modified_bounds(dist, factor):
        output = modified_scale(dist, factor)
        bounds = factor * np.array([dist.a, dist.b])
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds
    
scale = scale_special_cases(
            scale_gen(name="scale")
        )

class abs_gen(rv_continuous):
    """
    scale a random variable by a constant value
    """
    
    def _pdf(self, x, dist):
        dist, = extract_first(dist)
        return (x >= 0) * (dist.pdf(x) + dist.pdf(-x))
    
    def _cdf(self, x, dist):
        dist, = extract_first(dist)
        return (x >= 0) * (dist.cdf(np.abs(x)) - dist.cdf(-np.abs(x)))
    
    def _rvs(self, dist):
        dist, = extract_first(dist)
        return np.abs(dist.rvs(size=self._size))
    
    def _munp(self, n, dist):
        dist, = extract_first(dist)
        return quad(lambda x: self._pdf(x, dist), self.a, self.b)[0]
    
    def _argcheck(self, dist):
        dist, = extract_first(dist)
        conditions = [
            isinstance(dist, rv_frozen),
        ]
        return all(conditions)

def abs_special_cases(abs_val):

    def modified_abs(dist):
        name = dist.get_name()
        if name == "norm":
            m, s = dist.mean(), dist.std()
            return foldnorm(m/s, 0, s)

        return abs_val(dist)

    @push_bounds_to_dist
    def modified_bounds(dist):
        output = modified_abs(dist)
        bounds = np.abs(np.array([dist.a, dist.b]))
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

abs_val = abs_special_cases(
                abs_gen(name="abs")
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
        elif name == "offset":
            d, o = dist.args
            return modified_offset(d, o + value)
        else:
            return offset(dist, value)

    @push_bounds_to_dist
    def modified_bounds(dist, value):
        output = modified_offset(dist, value)
        bounds = np.array([dist.a, dist.b]) + value
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds
    
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
        else:
            items = resolve(["norm", "expon"], [dist0, dist1])
            if items is not None:
                d0, d1 = items
                n_mean, n_std = d0.mean(), d0.std()
                e_mean, e_std = d1.mean(), d1.std()

                m = n_mean + (e_mean - e_std)
                s = n_std
                k = e_std / n_std
                return exponnorm(k, m, s)

        return add(dist0, dist1)

    @push_bounds_to_dist
    def modified_bounds(dist0, dist1):
        output = modified_add(dist0, dist1)
        bounds = np.array([dist0.a + dist1.a, dist0.b + dist1.b])
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

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
                lambda z: quad(pdf_integrand, dist1.a, dist1.b, args=(z,))[0]
              )
        return pdf(x)
    
    def _cdf(self, x, dist0, dist1):
        dist0, dist1 = extract_first(dist0, dist1)
        cdf_integrand = lambda y, z: dist0.cdf(z / y) * dist1.pdf(y)
        cdf = np.vectorize(
                lambda z: quad(cdf_integrand, dist1.a, dist1.b, args=(z,))[0]
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

def multiply_special_cases(multiply):

    def modified_multiply(dist0, dist1):
        name0, name1 = dist0.get_name(), dist1.get_name()
        if name0 == name1:
            if name0 == "lognorm":
                if len(dist0.args) == 1 and len(dist1.args) == 1:
                    s0, s1 = dist0.args[0], dist1.args[0]
                    return lognorm(np.sqrt(s0**2 + s1**2))

        return multiply(dist0, dist1)

    @push_bounds_to_dist
    def modified_bounds(dist0, dist1):
        output = modified_multiply(dist0, dist1)
        bounds = np.array([dist0.a * dist1.a, 
                           dist0.b * dist1.b,
                           dist0.a * dist1.b,
                           dist0.b * dist1.a])
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds
    
multiply = multiply_special_cases(
                multiply_gen(name="multiply")
           )


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
        return quad(lambda x: dist.pdf(x) * (1.0 / x)**n, self.a, self.b)[0]
    
    def _argcheck(self, dist):
        dist, = extract_first(dist)
        conditions = [
            isinstance(dist, rv_frozen),
        ]
        return all(conditions)

def inverse_special_cases(inverse):

    def modified_inverse(dist):
        name = dist.get_name()
        if name == "reciprocal":
            a, b = dist.a, dist.b
            return reciprocal(1.0/b, 1.0/a)
        return inverse(dist)

    @push_bounds_to_dist
    def modified_bounds(dist):
        output = modified_multiply(dist)
        bounds = [dist.a, dist.b]
        if 0 in bounds:
            if mean(bounds) > 0:
                new_bounds = [1 / np.max(bounds), np.inf]
            else:
                new_bounds = [-np.inf, 1 / np.min(bounds)]
        else:
            new_bounds = [1 / np.min(bounds), 1 / np.max(bounds)]

        output.a, output.b = np.min(new_bounds), np.max(new_bounds)
        return output

    return modified_bounds

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

    @push_bounds_to_dist
    def modified_bounds(dist0, dist1):
        output = modified_posterior(dist0, dist1)
        bounds = np.max([dist0.a, dist1.a]), np.min([dist0.b, dist1.b])
        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

posterior = posterior_special_cases(
                posterior_gen(name="posterior")
            )


class power_gen(rv_continuous):
    
    def _pdf(self, x, dist, k):
        k, dist = extract_first(k, dist)
        
        pdf = lambda y: dist.pdf(np.sign(y) * np.abs(y)**(1 / k)) / (k * np.abs(y)**(1 - 1 / k)) 
        
        if k % 2 == 0:
            return (x > 0) * (pdf(np.abs(x)) + pdf(-np.abs(x)))
        if k % 2 == 1:
            return pdf(x)

    def _cdf(self, x, dist, k):
        k, dist = extract_first(k, dist)
        
        if k % 2 == 0:
            return (x > 0) * ( dist.cdf(  np.abs(x) ** (1 / k)) - 
                               dist.cdf(- np.abs(x) ** (1 / k)))
        if k % 2 == 1:
            return dist.cdf(np.sign(x) * np.abs(x) ** (1 / k))
        
    def _munp(self, n, dist, k):
        k, dist = extract_first(k, dist)
        return dist.moment(n * k)
    
    def _rvs(self, dist, k):
        k, dist = extract_first(k, dist)
        return dist.rvs(self._size) ** k
    
    def _argcheck(self, dist, k):
        k, dist = extract_first(k, dist)
        conditions = [
            isinstance(dist, rv_frozen),
            isinstance(k,    int),
        ]
        return all(conditions)

def power_special_cases(power):

    def modified_power(dist, k):
        name = dist.get_name()

        if isinstance(k, int) and k > 0:
            if name == 'lognorm':
                return reduce(multiply, [dist] * k)

        return power(dist, k)

    @push_bounds_to_dist
    def modified_bounds(dist, k):
        output = modified_power(dist, k)

        bounds = [dist.a**k, dist.b**k]
        if k % 2 == 0:
            if np.sign(dist.a) != np.sign(dist.b):
                bounds.append(0)

        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

power = power_special_cases(
            power_gen(name="power") 
        )

class exp_gen(rv_continuous):
    
    def _pdf(self, x, base, dist):
        base, dist = extract_first(base, dist)
        return (x > 0) * (dist.pdf(np.log(x) / np.log(base))) / ( np.log(base) * x )

    def _cdf(self, x, base, dist):
        base, dist = extract_first(base, dist)
        return (x > 0) * dist.cdf(np.log(x) / np.log(base))
        
    def _munp(self, n, base, dist):
        base, dist = extract_first(base, dist)
        return quad(lambda x: x**n * self._pdf(x, base, dist), self.a, self.b)[0]
    
    def _rvs(self, base, dist):
        base, dist = extract_first(base, dist)
        return base ** dist.rvs(self._size)
    
    def _argcheck(self, base, dist):
        base, dist = extract_first(base, dist)
        conditions = [
            isinstance(dist, rv_frozen),
            isinstance(base, (float, int)),
            base > 0,
        ]
        return all(conditions)

def exp_special_cases(exp):

    def modified_exp(base, dist):
        name = dist.get_name()
        if name == "norm":
            mean, std = dist.mean(), dist.std()
            if mean == 0:
                return lognorm(std * np.log(base))
        return exp(base, dist)

    @push_bounds_to_dist
    def modified_bounds(base, dist):
        output = modified_exp(base, dist)

        bounds = [base**dist.a, base**dist.b]

        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

exp = exp_special_cases(
         exp_gen(name="exp")
      )


class log_gen(rv_continuous):
    
    def _pdf(self, x, dist, base):
        dist, base = extract_first(dist, base)
        return dist.pdf(base**x) * np.log(base) * (base**x)

    def _cdf(self, x, dist, base):
        dist, base = extract_first(dist, base)
        return dist.cdf(base**x)
        
    def _munp(self, n, dist, base):
        dist, base = extract_first(dist, base)
        return quad(lambda x: (np.log(x) / np.log(base))**n * dist.pdf(x), self.a, self.b)[0]
    
    def _rvs(self, dist, base):
        dist, base = extract_first(dist, base)
        return np.log(dist.rvs(size=self._size)) / np.log(base)
    
    def _argcheck(self, dist, base):
        dist, base = extract_first(dist, base)
        conditions = [
            isinstance(dist, rv_frozen),
            isinstance(base, (float, int)),
        ]
        return all(conditions)

def log_special_cases(log):

    def modified_log(dist, base=np.exp(1)):
        name = dist.get_name()
        if name == 'lognorm':
            args = dist.args
            if len(args) == 1:
                return norm(0, args[0] / np.log(base))

        if name == 'gamma':
            mean, std = dist.mean(), dist.std()
            if mean == std**2:
                return loggamma(mean)

        return log(dist, base)

    @push_bounds_to_dist
    def modified_bounds(dist, base=np.exp(1)):
        output = modified_log(dist, base=base)

        bounds = np.log([dist.a, dist.b]) / np.log(base)

        output.a, output.b = np.min(bounds), np.max(bounds)
        return output

    return modified_bounds

log = log_special_cases(
         log_gen(name="log")
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

def resolve(names, items):
    """
    this method returns None
    if the list **items** do not contain names in **names**,
    but otherwise outputs items in the same order as names
    """
    if len(names) != len(items):
        raise IOError()
        
    item_names = [item.get_name() for item in items]
    
    if sorted(names) != sorted(item_names):
        return None
    
    output = []
    for name in names:
        index = item_names.index(name)
        output.append(items.pop(index))
        item_names.pop(index)
        
    return tuple(output)
