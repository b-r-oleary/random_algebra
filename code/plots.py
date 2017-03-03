import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="white", font_scale=1.3)

def plot_dist(dists, labels=None, label="", **kwargs):
    if isinstance(dists, (tuple, list)):
        return _plot_dists(dists, labels=labels, **kwargs)
    else:
        return _plot_dist( dists, label =label , **kwargs)

def _plot_dist(dist, 
                lower=None, upper=None, 
                n=35, samples=10000,
                legend_loc=(1.5, .7),
                type="pdf",
                show_normal_approx=True,
                show_random_samples=True,
                show_dist=True,
                label=""):
    """
    given an input distribution object, plot
    the exact distribution, normal approximation to
    the distribution, and an empirical distribution
    from a histogram of randomly generated variables

    *inputs*
    dist (rv_frozen) scipy.stats frozen distribution
    lower
    """
    if not show_random_samples:
        samples = 300

    samples = dist.rvs(size=(samples,))

    s_range = max(samples) - min(samples)

    if lower is None:
        lower = min(samples) - s_range/10.0
    if upper is None:
        upper = max(samples) + s_range/10.0

    if type not in ["pdf", "cdf"]:
        raise NotImplementedError()

    if show_random_samples:
        if type == "pdf":
            hist_kws=dict(cumulative=False)
        elif type == "cdf":
            hist_kws=dict(cumulative=True)

        sns.distplot(samples, hist_kws=hist_kws,
                     kde=False, norm_hist=True, label="random samples " + label)

    x = np.linspace(lower, upper, n)

    if show_dist:
        if type == "pdf":
            p = dist.pdf(x)
        elif type == "cdf":
            p = dist.cdf(x)

        plt.plot(x, p, label="exact " + type + " " + label)

    if show_normal_approx:
        if type == "pdf":
            p_a = dist.get_normal_approx().pdf(x)
        elif type == "cdf":
            p_a = dist.get_normal_approx().cdf(x)

        plt.plot(x, p_a, label="normal approximation " + type + " " + label)

    plt.legend(bbox_to_anchor=legend_loc)
    plt.xlim([lower, upper])


def _plot_dists(dists, 
                lower=None, upper=None,
                show_normal_approx=False,
                show_random_samples=False,
                show_dist=True,
                labels=None,
               **kwargs):

    uppers = []
    lowers = []

    if labels is None:
        labels = [""] * len(dists)
    else:
        if len(labels) != len(dists):
            raise IOError("number of labels must match number of distributions.")

    for dist, label in zip(dists, labels):
        plot_dist(dist,
                  lower=lower, upper=upper,
                  show_normal_approx=show_normal_approx,
                  show_random_samples=show_random_samples,
                  show_dist=show_dist,
                  label=label,
                  **kwargs)

        lo, up = plt.gca().get_xlim()

        lowers.append(lo)
        uppers.append(up)

    if lower is None:
        lo = min(lowers)
    if upper is None:
        up = max(uppers)

    plt.gca().set_xlim([lo, up])
