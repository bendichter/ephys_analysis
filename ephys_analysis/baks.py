from scipy.special import gamma
import numpy as np


def baks(spike_times, time, a=4, b=0.8):
    """Bayesian Adaptive Kernel Smoother (BAKS)
    adapted from https://github.com/nurahmadi/BAKS

    Parameters
    ----------
    spike_times: np.ndarray
        event spike times
    time: np.ndarray
        time at which the firing rate is estimated
    a: float
        shape parameter (alpha)
    b: float
        scale parameter (beta)


    Returns
    -------

    firing_rate: np.ndarray
    h: float

    """

    n = len(spike_times)

    b = n**(b)

    sumnum = 0
    sumdenom = 0
    for i in range(n):
        numerator = (((time - spike_times[i])**2)/2 + 1./b)**(-a)
        denominator = (((time - spike_times[i])**2)/2 + 1./b)**(-a - 0.5)
        sumnum += numerator
        sumdenom += denominator
    h = (gamma(a)/gamma(a+0.5))*(sumnum/sumdenom)

    firing_rate = np.zeros(time.shape)
    for j in range(n):
        k = (1./(np.sqrt(2*np.pi)*h))*np.exp(-((time - spike_times[j])**2)/(2*h**2))
        firing_rate = firing_rate + k

    return firing_rate, h
