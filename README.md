
# `random_algebra`

### Random Variable Algebra With SciPy Probability Distributions

This is intended to be a small library that extends `scipy.stats` that facilitates some simple algebra with random variables (without having to resort to a monte-carlo approach) using numerical integration from `scipy.optimize`.

With the simple statement,

```python
import random_algebra
```

we monkeypatch the `scipy.statst.rv_continuous` class that is inherited by all continuous probability distributions. 


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font_scale=1.3)

import random_algebra
from random_algebra.code import test_dist

from scipy.stats import beta, uniform, norm
```

Then, simple linear functions of the distribution objects result in new probability distribution objects. For example, here `c` is a random variable that is distribute like the difference between two beta distributed variables:


```python
a = beta(4,1)
b = beta(10,4)
c = a - b
```

Inequalities operated on these objects result in probabilites that the inequality is true. Additionally, since these objects can quickly become complex, and hence computationally intensive, we can export normal approximations to these objects at any time.


```python
print("P(a > b) = %.2f numerical calculation" % (a > b))
print("P(a > b) = %.2f normal approximation" % ((a - b).get_normal_approx() > 0))
```

    P(a > b) = 0.70 numerical calculation
    P(a > b) = 0.67 normal approximation


I have included a simple plot function to compare a probability distribution with an empirically generated histogram of samples from that distribution, and the normal approximation to that distribution.


```python
plot_types = ["pdf", "cdf"]

for i, plot_type in enumerate(plot_types):
    plt.subplot(len(plot_types), 1, i + 1)
    
    test_dist(c,
              lower=-1, upper=1,
              n=35, samples=20000,
              type=plot_type)
    
    plt.xlabel('a - b')
    plt.ylabel(plot_type + '(a - b)')
    
plt.tight_layout()
```


![png](./images/output_7_0.png)


Here is another example in which case we are computing the difference between two uniform distributions of different widths.


```python
d = 2 * uniform() - uniform()

test_dist(d, lower=-2.5, upper=2.5, n=35)
plt.ylabel("pdf(d)")
plt.xlabel('d')
```




    <matplotlib.text.Text at 0x7f3af5a873d0>




![png](./images/output_9_1.png)

