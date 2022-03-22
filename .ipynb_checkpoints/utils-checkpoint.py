import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empiricaldist import Pmf

# distributions, &c.
from scipy.stats import binom, gaussian_kde

def normalize(joint):
    """ Normalize a joint distribution """
    prob_data = joint.to_numpy().sum()
    joint /= prob_data
    return prob_data

def plot_contour(joint, **options):
    """ Plot a joint distribution """
    low = joint.to_numpy().min()
    high = joint.to_numpy().max()
    levels = np.linspace(low, high, 6)
    levels = levels[1:]
    
    cs = plt.contour(joint.columns, joint.index, joint, levels=levels, linewidths=1)
    #ax.set_xlabel(options.get('xlabel'))
    #ax.set_ylabel(options.get('ylabel'))
    #ax.legend()
    #return ax
    
def marginal(joint, axis):
    """
    Compute the marginal distribution from a joint one
    
    axis=0: return the distribution of the first variable
    axis=1: return the distribution of the second variable
    """
    return Pmf(joint.sum(axis=axis))

def make_uniform(qs, name=None, **options):
    """Make a Pmf that represents a uniform distribution.
    qs: quantities
    name: string name for the quantities
    options: passed to Pmf
    returns: Pmf
    """
    pmf = Pmf(1.0, qs, **options)
    pmf.normalize()
    if name:
        pmf.index.name = name
    return pmf
    
def make_joint(s1, s2):
    """Compute the outer product of two Series.
    First Series goes across the columns;
    second goes down the rows.
    s1: Series
    s2: Series
    return: DataFrame
    """
    X, Y = np.meshgrid(s1, s2)
    return pd.DataFrame(X*Y, columns=s1.index, index=s2.index) 

def make_binomial(n, p):
    """Make a binomial distribution.
    n: number of trials
    p: probability of success
    returns: Pmf representing the distribution of k
    """
    ks = np.arange(n+1)
    ps = binom.pmf(ks, n, p)
    return Pmf(ps, ks)

def make_mixture(pmf, pmf_seq):
    """Make a mixture of distributions.
    pmf: mapping from each hypothesis to its probability
         (or it can be a sequence of probabilities)
    pmf_seq: sequence of Pmfs, each representing
             a conditional distribution for one hypothesis
    returns: Pmf representing the mixture
    """
    df = pd.DataFrame(pmf_seq).fillna(0).transpose()
    df *= np.array(pmf)
    total = df.sum(axis=1)
    return Pmf(total)

def pmf_from_dist(dist, qs):
    """Make a discrete approximation.
    dist: SciPy distribution object
    qs: quantities
    returns: Pmf
    """
    ps = dist.pdf(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf

def kde_from_sample(sample, qs, **options):
    """Make a kernel density estimate from a sample
    
    sample: sequence of values
    qs: quantities where we should evaluate the KDE
    
    returns: normalized Pmf
    """
    kde = gaussian_kde(sample)
    ps = kde(qs)
    pmf = Pmf(ps, qs, **options)
    pmf.normalize()
    return pmf