import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empiricaldist import Pmf

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
    
    