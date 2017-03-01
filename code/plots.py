import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="white", font_scale=1.3)

def plot_dist( dist, 
                lower=None, upper=None, 
                n=35, samples=10000,
                legend_loc=(1.5, .7),
                type="pdf"):
    """
    given an input distribution object, plot
    the exact distribution, normal approximation to
    the distribution, and an empirical distribution
    from a histogram of randomly generated variables

    *inputs*
    dist (rv_frozen) scipy.stats frozen distribution
    lower
    """
    samples = dist.rvs(size=(samples,))

    s_range = max(samples) - min(samples)

    if lower is None:
        lower = min(samples) - s_range/10.0
    if upper is None:
        upper = max(samples) + s_range/10.0

    if type not in ["pdf", "cdf"]:
        raise NotImplementedError()

    x = np.linspace(lower, upper, n)
    
    if type == "pdf":
        p = dist.pdf(x)
        p_a = dist.get_normal_approx().pdf(x)
        hist_kws=dict(cumulative=False)
    elif type == "cdf":
        p = dist.cdf(x)
        p_a = dist.get_normal_approx().cdf(x)
        hist_kws=dict(cumulative=True)

    sns.distplot(samples, hist_kws=hist_kws,
                 kde=False, norm_hist=True, label="random samples")
    plt.plot(x, p, label="exact " + type)
    plt.plot(x, p_a, label="normal approximation " + type)
    plt.legend(bbox_to_anchor=legend_loc)
    plt.xlim([lower, upper])