from .baks import baks
from .utils import find_nearest
import numpy as np
from .FMA import map_stats2
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


def apply_gaussian_filter(spikes, tt, d, param=0.15, mode='constant'):
    """

    Parameters
    ----------
    spikes: array-like
    tt: array-like
    d: array-like
    param: float
    mode: str (optional)
        passed to scipy.ndimage.filters.gaussian_filter

    Returns
    -------

    """
    spike_pos_inds = find_nearest(spikes, tt)
    spike_bins = np.zeros(d.shape)
    spike_bins[spike_pos_inds] = 1
    diff_d = np.diff(d[:2])
    diff_tt = np.diff(tt[:2])

    return gaussian_filter(spike_bins, param / diff_d, mode=mode) / diff_tt


def get_pop_stats(nwb, d, tt, all_fr, param, filt_type='gaussian', samp_size=5000, threshold=0.1, min_peak=0.2,
                  min_size=0.15):
    """Randomly select a subsample of neurons and gather the stats for imposed and detected firing rate into one DataFrame

    Parameters
    ----------
    nwb: pynwb.NWBFile
    tt: time counter
    d: space counter
    all_fr: array-like
    param: float
        If filt_type is 'baks', param is alpha
        If filt_type is 'gaussian', param is sd of filter
    filt_type: str
        'gaussian' or 'baks'
    samp_size: int (optional)
        number of neurons used to calculate population statistics. Default = 5000
    threshold: float
        in Hz. delineates place field boundaries
    min_peak: float
        in Hz.
    min_size: float
        in meters.

    Returns
    -------
    df: pd.DataFrame

    """

    min_size = int(min_size / max(d) * len(d))

    df = pd.DataFrame()
    for cell_id in np.random.permutation(71999)[:samp_size]:

        spikes = nwb.units.get_unit_spike_times(cell_id)
        if not len(spikes):
            continue

        if filt_type == 'baks':
            detected_firing_rate = baks(spikes, tt, a=param)[0]
        elif filt_type == 'gaussian':
            detected_firing_rate = apply_gaussian_filter(spikes, tt, d, param)
        else:
            ValueError('filter_type' + filt_type + ' not recognized.')
        imposed_stats = map_stats2(all_fr[cell_id], threshold=threshold, min_size=min_size, min_peak=min_peak)
        detected_stats = map_stats2(detected_firing_rate, threshold=threshold, min_size=min_size, min_peak=min_peak)

        stats = {'imposed_' + key: val for key, val in imposed_stats.items()}
        stats.update(**{'detected_' + key: val for key, val in detected_stats.items()})
        stats.update(firing_rate=all_fr[cell_id])
        stats.update(detected_firing_rate=detected_firing_rate)

        df = df.append(pd.DataFrame([stats]))
    return df


def get_field_widths(df):

    x_ = []
    y_ = []
    for x, y in df[['imposed_sizes', 'detected_sizes']].values:
        for xx, yy in zip(sorted(x), sorted(y)):
            x_.append(xx)
            y_.append(yy)

    return x_, y_


def scan_params(nwb, d, tt, all_fr, params, filt_type, samp_size=5000):
    """

    Parameters
    ----------
    nwb: pynwb.NWBFile
    d: array-like
    tt: array-like
    all_fr: np.array
    params: int
    filt_type: str
    samp_size: int

    Returns
    -------

    dict

    """
    nfields_mse = []
    filt_mse = []
    field_widths_mse = []

    for param in tqdm(params):
        df = get_pop_stats(nwb, d, tt, all_fr, param, filt_type=filt_type, samp_size=samp_size)

        xx = [len(x) for x in df['imposed_sizes'].values]
        yy = [len(x) for x in df['detected_sizes'].values]
        nfields_mse.append(mean_squared_error(xx, yy))

        filt_mse.append(np.mean([mean_squared_error(x, y)
                                 for x, y in zip(df['firing_rate'], df['detected_firing_rate'])]))

        field_widths_mse.append(mean_squared_error(*get_field_widths(df)))

    return {'nfields_mse': nfields_mse, 'filt_mse': filt_mse, 'field_widths_mse': field_widths_mse}
