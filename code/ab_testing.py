from . import argmax, plot_dist
from scipy.stats import binom, beta
import numpy as np
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd


class Generator(object):
    """
    this is a data acquisition object:
    
    using `get_samples` you can request acquiring N data points
    from a certain source, and by using `get_likelihood`, you
    can get acquire the data and then return a 1D likelihood function
    for a model parameter given the data.
    """
    def __init__(self, generator, likelihood):
        """
        *inputs*
        generator (function) takes number of samples N as an input, returns acquired data
        likelihood (function) takes the acquired data as input and returns a likelihood function
                for a model parameter given the data.
        """
        self.generator  = generator
        self.likelihood = likelihood
        
    def get_samples(self, N):
        "acquire N samples of data"
        return self.generator(N)
    
    def get_likelihood(self, N):
        "acquire N samples of data and return the likelihood function for the model parameter given the data"
        values = self.get_samples(N)
        return self.likelihood(N, values)
            

class BinomBetaTestGenerator(Generator):
    """
    this is an object used for testing purposes which simulates a beta-binomial process
    """
    
    def __init__(self, p):
        """
        *inputs*
        p (float 0-1) the probability of a positive outcome for the binomial process
        """
        
        generator  = lambda N: binom(N, p).rvs()
        likelihood = lambda N, values: beta(values + 1, N - values + 1)
        
        Generator.__init__(self, generator, likelihood)


class Variant(object):
    """
    this is a variant in an A/B test
    
    this object houses a probability distribution for a model parameter
    that describes the success of this variant, and has a connection
    to a data acquiring Generator object to acquire data on demand.
    """
    
    def __init__(self, generator, prior=None, name=None):
        """
        *inputs*
        generator (Generator) this is a data acquiring object
        prior (scipy.stats distribution) a prior distribution for the model parameter
        """
        self.name = name
        self.generator = generator
        self.dist = prior
        
    def add_evidence(self, N):
        """
        acquire N data points and update the probability distribution
        for the model parameter
        """
        likelihood = self.generator.get_likelihood(N)

        if self.dist is None:
            self.dist = likelihood
        else:
            self.dist = likelihood & self.dist
            
    def __str__(self):
        return "<" + self.__class__.__name__ + ": " + str(self.dist) + ">"
    
    def __repr__(self):
        return str(self)
    
            
class BinomBetaTestVariant(Variant):
    """
    this is an object used for testing purposes which simulates a variant with a beta-binomial process
    """
    
    def __init__(self, p, prior=beta(1,1), **kwargs):
        """
        *inputs*
        p (float 0-1) the probability of a positive outcome for the binomial process
        prior (scipy.stats distribution) a prior distribution for the model parameter
        """
        
        generator = BinomBetaTestGenerator(p)
        
        Variant.__init__(self, generator, prior=prior, **kwargs)

        
class MultiArmedBandit(object):
    """
    this is an object used to house a multi-armed bandit A/B experiment.
    
    given a series of variants, data is acquired proportionally to the
    probability that a given variant has a model parameter larger than
    all of the other variants in steps.
    """
    
    def __init__(self, arms=None, name="multi-armed bandit"):
        """
        *inputs*
        arms (list of Variant objects) these are the variants that are to be compared in the experiment
        """
        if isinstance(arms, Variant):
            arms = [arms]
            
        if arms is None:
            arms = []
        
        if not(all([isinstance(arm, Variant) for arm in arms])):
            raise IOError('each arm input must be a Variant')
            
        for i, arm in enumerate(arms):
            if arm.name is None:
                arm.name = "arm:{i}".format(i=i)
        
        self.arms = arms
        self.p = None
        self.name = name
        self.animation = None
        
        self.round = 0
        self.history = dict(round   = [],
                            variant = [],
                            samples = [],
                            probability = [])
        
        if len(arms) > 0:
            self.evaluate_p()
            
    def update_history(self, allocated_samples=None):
        """
        update the history dictionary
        """
        if allocated_samples is None:
            allocated_samples = np.array([0] * len(self.arms))
            
        self.history["round"]       += [self.round] * len(self.arms)
        self.history["variant"]     += [arm.name for arm in self.arms]
        self.history["samples"]     += list(allocated_samples)
        self.history["probability"] += list(self.p)
        
        self.round += 1
        
    def add_arm(self, arm):
        """
        add an arm to the experiment while the experiment is running
        """
        if not(isinstance(arm, Variant)):
            raise IOError("all arms must be variants")
            
        if arm.name is None:
            arm.name = "arm:{i}".format(i=len(self.arms))
        
        self.arms.append(arm)
        self.evaluate_p()
        self.update_history()
        
    def evaluate_p(self):
        """
        evaluate the probability that the model parameter for each variant is larger
        than that for all of the other variants
        """
        self.p = argmax([
                        arm.dist for arm in self.arms
                 ]).pmf(range(len(self.arms)))
        return self.p
        
    def step(self, samples):
        """
        perform a step in the experiment, randomly allocating a number
        of samples *samples*
        """
        if not(isinstance(samples, int)):
            raise IOError("samples must be an integer")
        
        allocation = np.random.multinomial(samples, self.p)
        
        for arm, n_samples in zip(self.arms, allocation):
            arm.add_evidence(n_samples)
            
        self.evaluate_p()
        self.update_history(allocation)
        
    def run(self, steps, samples_per_step):
        """
        run an experiment consisting of a number of steps *steps* with a number of samples per step *samples_per_step*
        """
        if not(isinstance(steps, int)):
            raise IOError('steps must be an integer input')
            
        for step in range(steps):
            self.step(samples_per_step)
            
    def get_history(self):
        return pd.DataFrame(self.history)
    
    def plot(self, type=None, **kwargs):
        """
        plot the experiment history
        
        *inputs*
        type (str) should be either None, "samples", or "probability"
        **kwargs -> forwarded to self._plot
        """
        if type is None:
            plt.subplot(3,1,1)
            self.plot("samples", **kwargs)
            plt.subplot(3,1,2)
            self.plot("probability", **kwargs)
            plt.subplot(3,1,3)
            self.plot_distributions(**kwargs)
            plt.tight_layout()
        if type == "samples":
            return self.plot_samples(**kwargs)
        if type == "probability":
            return self.plot_probability(**kwargs)
            
        
    def plot_samples(self, kind='area', stacked=True, alpha=.75, legend_loc=(1.25, 1), **kwargs):
        return self._plot(parameter="samples", title="allocation of samples",
                         kind=kind, stacked=stacked, alpha=alpha, legend_loc=legend_loc, **kwargs)
    
    def plot_probability(self, kind='area', stacked=True, alpha=.75, legend_loc=(1.25, 1), **kwargs):
        return self._plot(parameter="probability", title="probability that each variant is best",
                         kind=kind, stacked=stacked, alpha=alpha, legend_loc=legend_loc, **kwargs)
    
    def _plot(self, parameter="samples", 
              kind='bar', stacked=True, alpha=.75, legend_loc=(1.25, 1),
              title = "allocation of samples",
              **kwargs):
        
        h = self.get_history()
        h.pivot('round', 'variant')[parameter].plot(kind=kind, stacked=stacked, alpha=alpha, ax=plt.gca(), **kwargs)
        plt.legend(bbox_to_anchor=legend_loc)
        plt.ylabel(parameter)
        plt.title(self.name + ': ' + title)
        if parameter == "probability":
            plt.ylim([0, 1])
            
    def plot_distributions(self, legend_loc=(1.25, 1),
                           title="probability distributions for the metric for each arm",
                           **kwargs):
        plot_dist([arm.dist for arm in self.arms],
                  labels=[arm.name for arm in self.arms],
                  legend_loc=legend_loc,
                  **kwargs)
        plt.xlabel('metric')
        plt.ylabel('probability')
        plt.title(self.name + ': ' + title)
        
    def run_with_animation(self, steps, samples_per_step,
                            dpi=100, size_inches=(7, 6.5),
                            xlim=[0, 1], ylim=[0, 15], n=250,
                            ylabel='probability', xlabel='metric of success',
                            title="distributions for metrics of success over time",
                            interval=1, blit=True):
        """
        run the experiment while creating a gif animation
        """
        
        fig = plt.figure()
        fig.set_dpi(dpi)
        fig.set_size_inches(*size_inches)

        ax = plt.axes(xlim=xlim, ylim=ylim)
        x = np.linspace(xlim[0], xlim[1], n)

        lines = [ax.plot([], [])[0] 
                 for i in range(len(self.arms))]

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(self.name + '\n' + title)
        ax.legend([arm.name for arm in self.arms])

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            for arm, line in zip(self.arms, lines):
                y = arm.dist.pdf(x)
                line.set_data(x, y)
            self.step(samples_per_step)
            return lines

        self.animation = animation.FuncAnimation(
                                    fig, animate, 
                                    init_func=init, 
                                    frames=steps, 
                                    interval=interval, 
                                    blit=blit)
        
        return self.animation
