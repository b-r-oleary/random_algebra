import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
sns.set(style="white", font_scale=1.3)

from tempfile import NamedTemporaryFile
from matplotlib import animation

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
                show_approx=True,
                show_random_samples=True,
                show_dist=True,
                label="",
                approx_options=None,
                color=None):
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

    label = str(label)

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
                     kde=False, norm_hist=True, label="random samples " + label,
                     color=color)

    x = np.linspace(lower, upper, n)

    if show_dist:
        if type == "pdf":
            p = dist.pdf(x)
        elif type == "cdf":
            p = dist.cdf(x)

        plt.plot(x, p, label="exact " + type + " " + label, color=color)

    if show_approx:
        if approx_options is None:
            approx_options = dict()
        if type == "pdf":
            p_a = dist.approx(**approx_options).pdf(x)
        elif type == "cdf":
            p_a = dist.approx(**approx_options).cdf(x)

        plt.plot(x, p_a, '--', label="approx " + type + " " + label, color=color)

    plt.legend(bbox_to_anchor=legend_loc)
    plt.xlim([lower, upper])


def _plot_dists(dists, 
                lower=None, upper=None,
                show_approx=False,
                show_random_samples=False,
                show_dist=True,
                labels=None,
                colors=None,
               **kwargs):

    if colors is None:
        colors = sns.color_palette(n_colors=len(dists))

    uppers = []
    lowers = []

    if labels is None:
        labels = [""] * len(dists)
    else:
        if len(labels) != len(dists):
            raise IOError("number of labels must match number of distributions.")

    for dist, label, color in zip(dists, labels, colors):
        plot_dist(dist,
                  lower=lower, upper=upper,
                  show_approx=show_approx,
                  show_random_samples=show_random_samples,
                  show_dist=show_dist,
                  label=label,
                  color=color,
                  **kwargs)

        lo, up = plt.gca().get_xlim()

        lowers.append(lo)
        uppers.append(up)

    if lower is None:
        lo = min(lowers)
    if upper is None:
        up = max(uppers)

    plt.gca().set_xlim([lo, up])


IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""

def anim_to_gif(anim):
    data="0"
    with NamedTemporaryFile(suffix='.gif') as f:
        anim.save(f.name, writer='imagemagick', fps=10);
        data = open(f.name, "rb").read()
        data = data.encode("base64")
    return IMG_TAG.format(data)

def display_animation(anim):
    from IPython.display import HTML
    plt.close(anim._fig)
    return HTML(anim_to_gif(anim))