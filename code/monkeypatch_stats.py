from __future__ import division, print_function

from .algebra import rv_continuous, rv_frozen,\
					           scale, offset, add, multiply, inverse,\
                     posterior, power, exp, log, abs_val,\
                     logit, expit

from .algebra_with_dist_arrays import combination, either, order_statistic,\
                                      min_statistic, max_statistic, median, mean, argmax

from .plots import plot_dist

from scipy.stats import norm, uniform, loggamma, skew
from scipy.optimize import root
from scipy.special import polygamma
from sklearn.mixture import GaussianMixture

import numpy as np

## for some reason calling `moment` is much faster
## than calling `mean` and `std`, so i am overiding those methods:
def _mean(self):
    return self.moment(1)

def _std(self):
    return np.sqrt(self.moment(2) - self.moment(1)**2)

def get_loggamma_c(skewness):
    """
    this method provides the shape parameter c for a loggamma distribution
    given the skewness
    """
    r = root(
          lambda c: (polygamma(2, c) / np.power(polygamma(1, c), 1.5) + np.abs(skewness)), 1
            )
    if r.message == 'The solution converged.':
        return r.x[0]
    else:
        return r.nan

def get_normal_approx(self):
    """
    a method to be appended to scipy.stats distributions
    to quickly extract a normal distribution approximation from
    a potentially complex distribution that may be slow to compute
    """
    return norm(self.mean(), self.std())

def approx(self, type="norm", from_samples=False, samples=25000, n_components=2):

    types = ["norm", "uniform", "loggamma", "gaussian_mixture"]

    if type not in types:
        raise NotImplementedError("allowed types are " + str(types))

    if type == "gaussian_mixture":
        if n_components <= 1:
            return self.approx(type="norm", 
                               from_samples=from_samples, 
                               samples=samples)
        from_samples=True

    if from_samples:
        rv = self.rvs(samples)
        mean, std, skewness = np.mean(rv), np.std(rv), skew(rv)
    else:
        mean, std, skewness = self.mean(), self.std(), self.stats(moments="s")

    if type == "norm":
        return norm(mean, std)
    elif type == "uniform":
        return uniform(mean - np.sqrt(3) * std, np.sqrt(12) * std)
    elif type == "loggamma":
        if skewness == 0:
            return self.approx(type="norm", 
                               from_samples=from_samples, 
                               samples=samples)

        c = get_loggamma_c(skewness)

        a = loggamma(c)
        a = scale(
                offset(a, - a.mean()),
                -np.sign(skewness) / a.std())
        a = offset(
                  scale(a, std),
                  mean)
        return a
    elif type == "gaussian_mixture":
        gm = GaussianMixture(n_components=n_components)
        gm.fit(np.atleast_2d(rv).T)

        mean    = gm.means_[:, 0]
        var     = gm.covariances_[:,0, 0]
        weights = gm.weights_

        return combination([norm(m, np.sqrt(v))
                            for m, v in zip(mean, var)],
                            list(weights))

    else:
        raise NotImplementedError('approximation not defined for input type')

def get_name(self):
    name = self.dist.__class__.__name__
    if name.endswith('_gen'):
        name = name[:-4]
    return name

def plot(self, **kwargs):
    return plot_dist(self, **kwargs)

# add the addition and comparison operators
def __add__(self, other):
    if isinstance(other, rv_frozen):
        return add(self, other)
    elif isinstance(other, (int, float)):
        if other == 0:
            return self
        elif other in [np.inf, -np.inf, np.nan]:
            return other
        else:
            return offset(self, float(other))
    else:
        raise NotImplementedError()
    
def __mul__(self, other):
    if isinstance(other, bool):
        return self
    elif isinstance(other, (int, float)):
        if other == 1:
            return self
        elif other in [0]:
            return other
        elif not np.isfinite(other):
            return np.nan
        else:
            return scale(self, float(other))
    elif isinstance(other, rv_frozen):
        return multiply(self, other)
    else:
        raise NotImplementedError()
        
def __radd__(self, other):
    return self.__add__(other)

def __rmul__(self, other):
    return self.__mul__(other)

def __div__(self, other):
    if isinstance(other, (int, float)):
        if other == 0:
            other = np.nan
        else:
            other = 1/float(other)
        return self.__mul__(other)
    elif isinstance(other, rv_frozen):
        return multiply(self, inverse(other))
    else:
        raise NotImplementedError()

def __rdiv__(self, other):
    inverse_self = inverse(self)
    return inverse_self.__mul__(other)

def __truediv__(self, other):
    return self.__div__(other)

def __rtruediv__(self, other):
    return self.__rdiv__(other)

def __sub__(self, other):
    return self + (-1) * other

def __rsub__(self, other):
    return (-1) * self + other

def __lt__(self, other):
    if isinstance(other, (rv_frozen, int, float)):
      return (self - other).cdf(0)
    else:
      raise NotImplementedError()

def __le__(self, other):
    return self.__lt__(other)

def __gt__(self, other):
    return 1 - self.__lt__(other)

def __ge__(self, other):
    return self.__gt__(other)

def __and__(self, other):
    return posterior(self, other)

def __or__(self, other):
    return either(self, other)

def __pow__(self, n):
    if isinstance(n, (int, float)):
        return power(self, n)
    else:
        raise NotImplementedError()

def __rpow__(self, base):
    if isinstance(base, (int, float)):
        return exp(base, self)
    else:
        raise NotImplementedError()

def __neg__(self):
    return scale(self, -1)

def __pos__(self):
    return self

def __exp__(self):
    return exp(np.exp(1), self)

def __log__(self):
    return log(self)

def __abs__(self):
    return abs_val(self)

def __str__(self):
    name = self.get_name()

    args = ["[{l}]".format(l=',\n '.join(str(item) for item in arg))
            if isinstance(arg, (tuple, list)) else arg
            for arg in self.args]

    if any([isinstance(arg, (rv_frozen, list, tuple)) for arg in self.args]):

        string_args = "(\n" + indent(
                        ',\n'.join([str(arg) for arg in args])
                      ) + '\n)'
    else:
        string_args = str(tuple(args))
    return "{name}{args}".format(name=name,
                                 args=string_args)

def __repr__(self):
    return str(self)
    
def indent(text, prefix="   "):
    return '\n'.join([
            prefix + line for line in text.split('\n')
        ])

objects = [rv_frozen]

methods = dict(mean=_mean,
               std=_std,
               get_normal_approx=get_normal_approx,
               approx=approx,
               get_name=get_name,
               plot=plot,
               exp=__exp__,
               log=__log__,
               __neg__=__neg__,
               __pos__=__pos__,
               __abs__=__abs__,
               __add__=__add__,
               __mul__=__mul__,
               __div__=__div__,
               __radd__=__radd__,
               __rmul__=__rmul__,
               __rdiv__=__rdiv__,
               __truediv__=__truediv__,
               __rtruediv__=__rtruediv__,
               __sub__=__sub__,
               __rsub__=__rsub__,
               __le__=__le__,
               __lt__=__lt__,
               __gt__=__gt__,
               __ge__=__ge__,
               __and__=__and__,
               __or__=__or__,
               __pow__=__pow__,
               __rpow__=__rpow__,
               __str__=__str__,
               __repr__=__repr__)
    
for obj in objects:
    for name, method in methods.items():
        setattr(obj, name, method)