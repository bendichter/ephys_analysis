from scipy.signal import hilbert, butter, filtfilt
import numpy as np

from snippets import circstats

from .utils import rms, threshcross, isin_time_windows


def parse_passband(passband):
    if passband == 'delta':
        passband = [0, 4]
    elif passband == 'theta':
        passband = [4, 10]
    elif passband == 'spindles':
        passband = [10, 20]
    elif passband == 'gamma':
        passband = [30, 80]
    elif passband == 'ripples':
        passband = [100, 250]

    return passband


def filter_lfp(lfp, passband, sampling_rate=1250, order=4, filter='butter',
               ripple=20):
    """Apply a passband filter a signal

    Parameters
    ----------
    lfp: np.array
        (n,)
    passband: np.array | str
        (low, high) of bandpass filter or one of the following canonical bands:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)
    sampling_rate: double
    order: double
        number of cycles (default=4)
    filter: str
        choose filter: {'butter'},'cheby2', 'fir1'
    ripple: double
        attenuation factor used for cheby2 filter

    Returns
    -------
    filt: np.array
        (n,)

    """

    passband = parse_passband(passband)

    if filter == 'butter':
        b, a = butter(order, passband / (sampling_rate / 2), 'bandpass')
    elif filter == 'fir1':
        raise NotImplementedError('fir1 not implemented')
    elif filter == 'cheby2':
        raise NotImplementedError('cheby2 not implemented')

    filt = filtfilt(b, a, lfp)

    return filt


def power_windows(filt, passband, power_thresh, sampling_rate=1250,
                  starting_time=0.0):
    """Calculate the windows of time where the power of the signal is above a
    theshold.

    Parameters
    ----------
    filt: np.array
        Filtered lfp signal. Usually, this is the output of filter_lfp.
    passband: np.array | str
        [low high] of bandpass filter or one of the following canonical bands:
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


def hilbert_lfp(filt, use_octave=True):
    """Calculate the phase and amplitude of a filtered signal. By default, this function
    uses a bridge to octave because octave is much faster at hilbert transforms.

    Parameters
    ----------
    filt: np.array
        Filtered lfp signal. Usually, this is the output of filter_lfp.
    use_octave: bool
        Whether to use octave for computing the hilbert transform. It's much
        faster than python for this function in my experience. default=true.

    Returns
    -------

    """
    if use_octave:
        import oct2py
        oc = oct2py.Oct2Py()
        oc.eval('pkg load signal')
        hilb = oc.hilbert(filt)
    else:
        hilb = hilbert(filt)

    amp = np.abs(hilb)
    phase = np.mod(np.angle(hilb))

    return phase, amp


def do_circstats(phases):
    """Apply circular statistics to phase data

    Parameters
    ----------
    phases: np.array

    Returns
    -------

    """

    def circle_p(phases):
        n = len(phases)
        R = circstats.resvec(phases) * n

        # Zar, Biostatistical Analysis, p. 617
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

        return p

    return {'mean_resultant_length': circstats.resvec(phases),
            'mean_phase': circstats.mean(phases),
            'kappa': circstats.kappa(phases),
            'p': circle_p(phases)}


def phase_modulation(lfp, spikes, passband, power_thresh, sampling_rate=1250.0,
                     starting_time=0.0, use_octave=True):
    """

    Parameters
    ----------
    lfp: np.array
    spikes: list(np.array)
        list (k,) of list of spike times in seconds
    passband: np.array | str
        [low high] of bandpass filter or one of the following canonical bands:
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)
    power_thresh: float
    sampling_rate: float, optional, default = 1250.0
    starting_time: float, optional, default = 0.0
    use_octave: bool, optional, default=True
        Whether to use octave for computing the hilbert transform. It's much
        faster than python for this function in my experience.

    Returns
    -------
    stats: dict
        dict{stat_name: list (k,),
             ...}

    """
    filt = filter_lfp(lfp, passband, sampling_rate)
    filt_phase, _ = hilbert_lfp(filt, use_octave=use_octave)
    windows = power_windows(filt, passband, power_thresh, sampling_rate, starting_time)

    stats = []
    for ispikes in spikes:
        in_windows = isin_time_windows(ispikes, windows)
        phases = filt_phase[(ispikes[in_windows] * sampling_rate).astype('int')]
        stats.append(do_circstats(phases))

    # list of dicts -> dict of lists
    stats = {key: [x[key] for x in stats] for key in stats[0].keys()}

    return stats
