from __future__ import division, print_function
from .algebra import rv_continuous, rv_frozen,\
					           scale, offset, add, multiply, inverse, posterior
from .algebra_with_dist_arrays import combination, either, order_statistic,\
                                      min_statistic, max_statistic, median, mean

from scipy.stats import norm, uniform
import numpy as np

## for some reason calling `moment` is much faster
## than calling `mean` and `std`, so i am overiding those methods:
def _mean(self):
    return self.moment(1)

def _std(self):
    return np.sqrt(self.moment(2) - self.moment(1)**2)

def get_normal_approx(self):
    """
    a method to be appended to scipy.stats distributions
    to quickly extract a normal distribution approximation from
    a potentially complex distribution that may be slow to compute
    """
    return norm(self.mean(), self.std())

def approx(self, type="norm", from_samples=False, samples=10000):
    if from_samples:
        rv = self.rvs(samples)
        mean, std = np.mean(rv), np.std(rv)
    else:
        mean, std = self.mean(), self.std()

    if type == "norm":
        return norm(mean, std)
    elif type == "uniform":
        return uniform(mean - np.sqrt(3) * std, np.sqrt(12) * std)


def get_name(self):
    name = self.dist.__class__.__name__
    if name.endswith('_gen'):
        name = name[:-4]
    return name

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

def __sub__(self, other):
    return self + (-1) * other

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

def __str__(self):
    name = self.get_name()
    if any([isinstance(arg, rv_frozen) for arg in self.args]):
        string_args = "(\n" + indent(
                        ',\n'.join([str(arg) for arg in self.args])
                      ) + '\n)'
    else:
        string_args = str(self.args)
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
               __add__=__add__,
               __mul__=__mul__,
               __div__=__div__,
               __radd__=__radd__,
               __rmul__=__rmul__,
               __rdiv__=__rdiv__,
               __sub__=__sub__,
               __le__=__le__,
               __lt__=__lt__,
               __gt__=__gt__,
               __ge__=__ge__,
               __and__=__and__,
               __or__=__or__,
               __str__=__str__,
               __repr__=__repr__)
    
for obj in objects:
    for name, method in methods.items():
        setattr(obj, name, method)