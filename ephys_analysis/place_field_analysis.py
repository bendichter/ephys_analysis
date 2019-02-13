from .baks import baks
from .utils import find_nearest
import numpy as np
from .FMA import map_stats2
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


def get_1d_neuron_stats(d, tt, fr, spikes, filt_type, imposed_threshold=1,
                        inferred_threshold=1, min_peak=2.0, param=None):
    """Compute the place coding statistics for a single neuron

    Parameters
    ----------
    d: array-like
        space counter in meters
    tt: array-like
        time counter in seconds
    fr: array-like
        firing rate in Hz
    spikes: array-like
        spikes in seconds
    filt_type: str
        'gaussian' or 'baks'
    imposed_threshold: float
    inferred_threshold: float
    min_peak: float
    param: float (optional)
        If filt_type is 'baks', param is alpha, and default is 4
        If filt_type is 'gaussian', param is sd of filter and default is .15 (15 cm)

    Returns
    -------
    out: dict

    """
    if filt_type == 'baks':
        if param is None:
            param = 4  # a
        filtered_firing_rate = baks(spikes, tt, a=param)[0]
    elif filt_type == 'gaussian':
        if param is None:
            param = 0.15  # cm, sd of gaussian
        spike_pos_inds = find_nearest(spikes, tt)
        spike_bins = np.zeros(d.shape)
        spike_bins[spike_pos_inds] = 1
        diff_d = np.diff(d[:2])
        diff_tt = np.diff(tt[:2])
        filtered_firing_rate = gaussian_filter(spike_bins, param / diff_d, mode='constant') / diff_tt
    else:
        ValueError('filter_type' + filt_type + ' not recognized.')

    imposed_stats = map_stats2(fr, threshold=imposed_threshold, min_peak=min_peak, min_size=int(.2 / max(d) * len(d)))
    inferred_stats = map_stats2(filtered_firing_rate, threshold=inferred_threshold,
                                min_size=int(.2 / max(d) * len(d)), min_peak=min_peak)
    out = {**{'imposed_' + key: val for key, val in imposed_stats.items()},
           **{'inferred_' + key: val for key, val in inferred_stats.items()},
           'fr': fr, 'inf_fr': filtered_firing_rate}

    return out


def get_pop_stats(nwb, all_fr, param, tt, d, filt_type='gaussian', samp_size=5000):
    """Randomly select a subsample of neurons and gather the stats into one DataFrame

    Parameters
    ----------
    nwb: pynwb.NWBFile
    all_fr: array-like
    param: float
        If filt_type is 'baks', param is alpha
        If filt_type is 'gaussian', param is sd of filter
    tt: time counter
    d: space counter
    filt_type: str
        'gaussian' or 'baks'
    samp_size: int (optional)
        number of neurons used to calculate population statistics. Default = 5000

    Returns
    -------
    df: pd.DataFrame

    """
    df = pd.DataFrame()
    for cell_id in np.random.permutation(71999)[:samp_size]:

        spikes = nwb.units.get_unit_spike_times(cell_id)
        if not len(spikes):
            continue

        fr = all_fr[cell_id]

        stats = get_1d_neuron_stats(d, tt, fr, spikes, filt_type=filt_type, param=param)
        df = df.append(pd.DataFrame([stats]))
    return df


def get_field_widths(df):
    x_ = []
    y_ = []
    for x, y in df[['imposed_sizes', 'inferred_sizes']].values:
        for xx, yy in zip(sorted(x), sorted(y)):
            x_.append(xx)
            y_.append(yy)

    return x_, y_


def scan_params(nwb, d, tt, all_fr, params, filt_type, samp_size=5000):
    nfields_mse = []
    filt_mse = []
    field_widths_mse = []

    for param in tqdm(params):
        df = get_pop_stats(nwb, d, tt, all_fr, param, filt_type=filt_type, samp_size=samp_size)

        xx = [len(x) for x in df['imposed_sizes'].values]
        yy = [len(x) for x in df['inferred_sizes'].values]
        nfields_mse.append(mean_squared_error(xx, yy))

        filt_mse.append(np.mean([mean_squared_error(x, y) for x, y in zip(df['fr'], df['inf_fr'])]))

        field_widths_mse.append(mean_squared_error(*get_field_widths(df)))

    return {'nfields_mse': nfields_mse, 'filt_mse': filt_mse, 'field_widths_mse': field_widths_mse}
