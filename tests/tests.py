import sys
import unittest
from itertools import product
sys.path.append('../../')

import random_algebra
from scipy.stats import norm, beta, pareto, loggamma


class TestRVFrozenProperties(unittest.TestCase):

    precision = 1E-6

    def test_string_method(self):

        dists = [(norm(), "norm()"),
                 (2 * beta(3, 5), ("scale(\n"
                                   "   beta(3, 5),\n"
                                   "   2.0\n"
                                   ")"))]

        for dist, str_dist in dists:
            self.assertEqual(str(dist),
                             str_dist)

    def test_approx(self):

        dists = [beta(3, 2),
                 norm(),
                 pareto(5),
                 loggamma(5)]

        types = ["norm"]

        for dist, type in product(dists, types):

            appr = dist.approx(type=type)
            mean_diff = dist.mean() - appr.mean()
            std_diff  = dist.std() - appr.std()

            self.assertTrue(abs(mean_diff) < self.precision)
            self.assertTrue(abs(std_diff)  < self.precision)

if __name__ == '__main__':
    unittest.main()