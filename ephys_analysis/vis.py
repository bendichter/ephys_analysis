import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from seaborn import despine

import numpy as np


def plot_mean_resultant_length(mrls, ax=None, color='k', nbins=10, xmin=0.01,
                               xmax=1.0, **kwargs):
    """Plot a histogram of mean resultant lengths

    Parameters
    ----------
    mrls: iterable
        list of mean resultant lengths
    ax: plt.axes
    color: colorspec
    nbins: int
    xmin: double
    xmax: double
    kwargs: dict
        goes to ax.plot
    """
    if ax is None:
        fig, ax = plt.subplots()

    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 2 * nbins)[1:-1:2]

    n, _ = np.histogram(mrls, bins)

    ax.plot(xx, n, color=color, **kwargs)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([.01, .1, 1])
    despine(ax=ax)


def plot_phase_pref(phases, ax=None, color='r', nbins=20, **kwargs):
    """Plot a histogram of phase preferences

    Parameters
    ----------
    phases: iterable
        list of phase preferences
    ax: ax.axes
    color: colorspec
    nbins: int
    kwargs: dict
        goes to plt.plot

    Returns
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    phase_bins = np.linspace(-np.pi, np.pi, nbins)
    xx = np.linspace(-np.pi, np.pi, nbins * 2)[1:-1:2]

    n, _ = np.histogram(phases, phase_bins)

    ax.plot(xx, n, color=color, **kwargs)
    despine(ax=ax)


def plot_1d_place_fields(xx, place_fields, h=0, ax=None):
    if ax is None:
        ax = plt.gca()
    if np.any(place_fields):
        for i_field in range(int(max(place_fields))):
            show_field = np.zeros(place_fields.shape) * np.nan
            show_field[place_fields == i_field + 1] = h
            ax.plot(xx, show_field)
