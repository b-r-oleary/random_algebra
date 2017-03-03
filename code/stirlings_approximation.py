"""
here are a few methods that facilitate factorial calculations
when n is large enough such that factorials are no longer practical to calculate
"""

import numpy as np
from scipy.special import factorial

def log_factorial(n, threshold=25):
    "stirlings approximation to log(factorial(n)) when n > threshold"
    if n < threshold:
        return np.log(factorial(n))
    else:
        return (n * np.log(n) - n + np.log(2 * np.pi * n) / 2)

def log_binom(n, k, threshold=25):
    "stirlings approximation to log(binomial(n, k)) when n, k, n - k > threshold"
    return (  log_factorial(n, threshold=threshold) 
            - log_factorial(k, threshold=threshold) 
            - log_factorial(n - k, threshold=threshold))

def log_multinomial(array, threshold=25):
    "stirlings approximation to the log of the multinomial coefficient when items in array > threshold"
    return (  log_factorial(sum(array), threshold=threshold) 
            - sum([log_factorial(i, threshold=threshold) for i in array]))