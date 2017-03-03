from __future__ import division
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from .algebra import add, scale
import numpy as np


class combination_gen(rv_continuous):
    """
    return a combination of input distributions weighted by input probabilities
    """

    def pdf(self, x, dists, ps):
    	assert self.argcheck(dists, ps)
    	ps = np.array(ps) / sum(ps)
        return np.sum([p * dist.pdf(x) for p, dist in zip(ps, dists)], axis=0)

    def cdf(self, x, dists, ps):
    	assert self.argcheck(dists, ps)
    	ps = np.array(ps) / sum(ps)
        return np.sum([p * dist.cdf(x) for p, dist in zip(ps, dists)], axis=0)

    def rvs(self, dists, ps, random_state=0, size=None):
    	assert self.argcheck(dists, ps)
    	ps = np.array(ps) / sum(ps)
        _size = size
        if size is None:
            size = 1

    	N = np.prod(size)

        ns = np.random.multinomial(N, ps)
        rv = np.random.permutation(
                np.concatenate([
                dist.rvs(n) for dist, n in zip(dists, ns)
                ])
             ).reshape(size)

        if _size is None:
            return rv[0]
        else:
            return rv
    
    def moment(self, n, dists, ps):
    	assert self.argcheck(dists, ps)
    	ps = np.array(ps) / sum(ps)
    	return np.sum([p * dist.moment(n) for p, dist in zip(ps, dists)])
    
    def argcheck(self, dists, ps):
        conditions = [
            all([isinstance(dist, rv_frozen) for dist in dists]),
            all([isinstance(p, (int, float)) for p in ps]),
            len(dists) == len(ps),
            all([p >= 0 for p in ps])
        ]
        return all(conditions)

combination = combination_gen(name="combination")

def either(dist0, dist1):
    return combination([dist0, dist1], [.5, .5])

class order_statistic_gen(rv_continuous):
    """
    return a combination of input distributions weighted by input probabilities
    """

    def pdf(self, x, k, dists):
        assert self.argcheck(k, dists)
        terms = []
        for perm in permutations(dists):
            terms.append(lambda y: (
                np.prod([dist.cdf(y) for dist in perm[:k]])
               *perm[k].pdf(y)
               *np.prod([1 - dist.cdf(y) for dist in perm[k+1:]])
            ))
        return sum([term(x) for term in terms])

    def rvs(self, k, dists, random_state=0, size=None):
        assert self.argcheck(k, dists)

        _size = size
        if size is None:
            size = 1

        N = np.prod(size)
        
        rv = np.sort(
                np.array([
                    dist.rvs(N) for dist in dists
                ]), axis=0
             )[k,:].reshape(size)

        if _size is None:
            return rv[0]
        else:
            return rv
    
    def argcheck(self, k, dists):
        conditions = [
            all([isinstance(dist, rv_frozen) for dist in dists]),
            k in range(len(dists))
        ]
        return all(conditions)

order_statistic = order_statistic_gen(name="order_statistic")

def min_statistic(dists):
    return order_statistic(0, dists)

def max_statistic(dists):
    return order_statistic(len(dists) - 1, dists)

def median(dists):
    if len(dists) % 2 == 0:
        return scale(
                    add(
                        order_statistic(len(dists) // 2 - 1, dists),
                        order_statistic(len(dists) // 2    , dists),
                        ), .5
                    )
    else:
        return order_statistic(len(dists) // 2, dists)
    
def mean(dists):
    return scale(
               reduce(add, dists), 1 / float(len(dists))
           )