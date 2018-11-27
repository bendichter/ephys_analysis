from scipy.signal import hilbert, butter, filtfilt
import numpy as np

from .utils import rms, threshcross, isin_time_windows, listdict2dictlist
from . import circstats


def parse_passband(passband):
    """

    Parameters
    ----------
    passband: np.array | str
        (low, high) of bandpass filter or one of the following canonical bands:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)

    """
    if passband == 'delta':
        passband = np.array([0, 4])
    elif passband == 'theta':
        passband = np.array([4, 10])
    elif passband == 'spindles':
        passband = np.array([10, 20])
    elif passband == 'gamma':
        passband = np.array([30, 80])
    elif passband == 'ripples':
        passband = np.array([100, 250])

    return passband


def filter_lfp(lfp, sampling_rate=1250.0, passband='theta', order=4, filter='butter',
               ripple=20):
    """Apply a passband filter a signal. Butter is implemented but other
    filters are not.

    Parameters
    ----------
    lfp: np.array
        (ntt,)
    sampling_rate: float, optional
        sampling rate of LFP (default=1250.0)
    passband: np.array | str
        (low, high) of bandpass filter or the name of a canonical band:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)

    order: int
        number of cycles (default=4)
    filter: str
        choose filter: {'butter'},'cheby2', 'fir1'
    ripple: double
        attenuation factor used for cheby2 filter

    Returns
    -------
    filt: np.array
        (ntt,)

    """

    passband = parse_passband(passband)

    if filter == 'butter':
        b, a = butter(order, passband / (sampling_rate / 2), 'bandpass')
        filt = filtfilt(b, a, lfp)
        return filt
    else:
        NotImplementedError('filter type not implemented')


def calc_power_windows(filt, passband, power_thresh, sampling_rate=1250,
                  starting_time=0.0):
    """Calculate the windows of time where the power of the signal is above a
    theshold.

    Parameters
    ----------
    filt: np.array
        Filtered lfp signal. Usually, this is the output of filter_lfp.
    passband: tuple | str
        (low, high) of bandpass filter or one of the following canonical bands:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)
    power_thresh: power threshold in z-score
    sampling_rate: double
    in Hz. default = 1250 Hz
    starting_time: double
        in seconds. default = 0.0

    Returns
    -------

    """

    passband = parse_passband(passband)

    power = rms(filt, np.ceil(sampling_rate / passband[0]))
    thresh = np.mean(power) + np.std(power) * power_thresh
    windows = threshcross(power, thresh, 'both') / sampling_rate + starting_time

    # only keep windows of certain width
    min_width = 2 / passband[1]  # set the minimum width to two cycles
    windows = windows[(np.diff(windows) > min_width).ravel()]

    return windows


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def hilbert_lfp(filt):
    """Calculate the phase and amplitude of a filtered signal. By default, this function
    uses a bridge to octave because octave is much faster at hilbert transforms.

    Parameters
    ----------
    filt : np.array
        Filtered lfp signal. Usually, this is the output of filter_lfp.

    Returns
    -------
    phase : np.ndarray
    amplitude : np.ndarray

    """
    # hilbert runs way faster on a power of 2
    hilb = hilbert(filt, next_power_of_2(len(filt)))
    hilb = hilb[:len(filt)]

    amp = np.abs(hilb)
    phase = np.mod(np.angle(hilb), 2*np.pi)

    return phase, amp


def do_circstats(phases):
    """Apply circular statistics to phase data

    Parameters
    ----------
    phases: np.array

    Returns
    -------

    """

    return {'mean_resultant_length': circstats.resvec(phases),
            'mean_phase': circstats.mean(phases),
            'kappa': circstats.kappa(phases),
            'p': circstats.p(phases)}


def phase_modulation(lfp, spikes, passband, power_thresh, sampling_rate=1250.0,
                     starting_time=0.0):
    """Calculate circular statistics for the lfp phases of spike times for a
    list of cells

    Parameters
    ----------
    lfp: np.array
    spikes: list(np.array)
        list (k,) of list of spike times in seconds
    passband: np.array | str
        (low, high) of bandpass filter or one of the following canonical bands:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)
    power_thresh: float
    sampling_rate: float, optional, default = 1250.0
    starting_time: float, optional, default = 0.0

    Returns
    -------
    stats: dict
        dict{stat_name: list (k,),
             ...}

    """
    filt = filter_lfp(lfp, passband, sampling_rate)
    filt_phase, _ = hilbert_lfp(filt)
    windows = calc_power_windows(filt, passband, power_thresh, sampling_rate, starting_time)

    stats = []
    for ispikes in spikes:
        in_windows = isin_time_windows(ispikes, windows)
        phases = filt_phase[(ispikes[in_windows] * sampling_rate).astype('int')]
        stats.append(do_circstats(phases))

    stats = listdict2dictlist(stats)

    return stats