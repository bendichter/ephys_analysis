# Code to replicate Senzai2017Neuron figures
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from .utils import isin_single_interval, smooth, find_nearest

from h5py import File

from .analysis import phase_modulation
from .vis import plot_mean_resultant_length, plot_phase_pref
from .data_io import get_cell_data


def gather_fig2b(nwbfile):

    #nwbfile = '/Users/bendichter/Desktop/Buzsaki/SenzaiBuzsaki2017/YutaMouse41-150903/YutaMouse41-150903_1.nwb'

    spikes, cell_types = get_cell_data(nwbfile)

    with File(nwbfile, 'r') as f:
        lfp = f['processing']['shared']['LFP']['LFP']['data'][:]
        sampling_rate = f['processing']['shared']['LFP']['LFP']['starting_time'].attrs['rate']

    return spikes, cell_types, lfp, sampling_rate
    

def compute_fig2b(lfp, spikes, sampling_rate):
    theta_results = phase_modulation(lfp, spikes, 'theta', power_thresh=0.5,
                                     sampling_rate=sampling_rate,
                                     desc='theta phase modulation')

    gamma_results = phase_modulation(lfp, spikes, 'gamma', power_thresh=1.0,
                                     sampling_rate=sampling_rate,
                                     desc='gamma phase modulation')

    return theta_results, gamma_results


def plot_fig2b(gamma_results, theta_results, cell_types):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axc = 0
    for results in (gamma_results, theta_results):
        for stat, plotter in (('mean_phase', plot_phase_pref),
                              ('mean_resultant_length', plot_mean_resultant_length)):
            for cell_type, color in (('granule cell', 'orange'),
                                     ('mossy cell', 'purple')):
                #sig = results['p'] < .05) & results['kappa'] >= 1.0
                #plotter(results[stat][(cell_types == cell_type) & sig],
                #        color=color, ax=axs[axc])
                plotter(results[stat][cell_types == cell_type], color=color,
                        ax=axs[axc])

            axc += 1

    axs[0].set_ylabel('neurons')
    axs[0].set_xlabel('phase')
    axs[1].set_xlabel('mean length')
    axs[0].legend(('granule cells', 'mossy cells'))


def linearize_trial(norm_trial_pos, diameter):
    decision_ind = np.where(-norm_trial_pos[:, 0] > .95)[0][0]
    x_center = -1 - norm_trial_pos[:decision_ind, 0]
    x_arm = np.arctan2(np.abs(norm_trial_pos[decision_ind:, 1]),
                       -norm_trial_pos[decision_ind:, 0])
    full_run = np.hstack((x_center, x_arm + x_center[-1])) * diameter / 2
    return full_run


def compute_lin_pos(trials_df, pos, pos_tt, direction,
                    diameter, running_speed=.03):
    """Compute linearized position across trials. Only include points where
    the direction is the chosen direction and animal is running

    Parameters
    ----------
    trials_df:pd.DataFrame
    pos: np.ndarray
    pos_tt: np.ndarray
    direction: str
    diameter: float
        in meters
    running_speed: float
        in m/s. Default = 3 cm/s

    Returns
    -------

    """

    trials_df = trials_df

    df = trials_df
    df = df[df[direction] == 1]

    running = np.zeros(len(pos), dtype='bool')
    lin_pos = np.zeros(len(pos)) * np.nan

    for i, row in list(df.iterrows()):
        trial_inds = isin_single_interval(pos_tt, [row['start'], row['end']],
                                          inclusive_left=True,
                                          inclusive_right=False)
        trial_pos = pos[trial_inds, :]
        linearized_pos = linearize_trial(trial_pos, diameter)
        lin_pos[trial_inds] = linearized_pos

        speed = np.diff(linearized_pos) / np.diff(pos_tt[trial_inds])
        speed = np.hstack((0, np.abs(smooth(speed, 40))))

        if running_speed:
            running[trial_inds] = speed > running_speed
    if running_speed:
        lin_pos[~running] = np.nan

    return lin_pos


def compute_linear_firing_rate(trials_df, pos, pos_tt, spikes, direction,
                               gaussian_sd=0.0557, diameter=0.65,
                               spatial_bin_len=0.0168, running_speed=0.03):
    """The occupancy and number of spikes, speed-gated, binned, and smoothed
    over position

    Parameters
    ----------
    trials_df: pd.DataFrame
        trials info
    pos: np.ndarray
        normalized x,y position for theta (aka eight) maze
    pos_tt: np.ndarray
        sample times in seconds
    spikes: np.ndarray
        for a single cell in seconds
    direction: str
    gaussian_sd: float
        in meters. Default = 5.57 cm
    diameter: float
        in meters. Default = 65 cm
    spatial_bin_len: float
        in meters. Detault = 1.68 cm
    running_speed: float
        in m/s. Default = 3 cm/s


    Returns
    -------
    xx: np.ndarray
        center of position bins in meters
    occupancy: np.ndarray
        time in each spatial bin in seconds, during appropriate trials and
        while running
    filtered_n_spikes: np.ndarray
        number of spikes in each spatial bin,  during appropriate trials, while
        running, and processed with a Gaussian filter

    """
    arm_len = diameter * np.pi / 2
    # include an extra bin to catch boundary errors
    spatial_bins = np.arange(-diameter, arm_len + spatial_bin_len,
                             spatial_bin_len)

    sampling_rate = len(pos_tt) / (np.max(pos_tt) - np.min(pos_tt))

    lin_pos = compute_lin_pos(trials_df, pos, pos_tt, direction, diameter,
                              running_speed)

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)

    finite_lin_pos = lin_pos[np.isfinite(lin_pos)]

    pos_on_spikes = lin_pos[spike_pos_inds]
    finite_pos_on_spikes = pos_on_spikes[np.isfinite(pos_on_spikes)]

    occupancy = np.histogram(finite_lin_pos, bins=spatial_bins)[0][:-2] / sampling_rate
    n_spikes = np.histogram(finite_pos_on_spikes, bins=spatial_bins)[0][:-2]

    filtered_n_spikes = gaussian_filter(n_spikes, gaussian_sd / spatial_bin_len)
    xx = spatial_bins[:-3] + (spatial_bins[1] - spatial_bins[0]) / 2

    return xx, occupancy, filtered_n_spikes


def compute_linear_place_fields(firing_rate, min_window_size=5,
                                min_firing_rate=1., thresh=0.5):
    """Find consecutive bins where all are >= 50% of local max firing rate and
    the local max in > 1 Hz

    Parameters
    ----------
    firing_rate: array-like(dtype=float)
    min_window_size: int
    min_firing_rate: float
    thresh: float

    Returns
    -------
    np.ndarray(dtype=bool)

    """

    is_place_field = np.zeros(len(firing_rate), dtype='bool')
    for start in range(len(firing_rate) - min_window_size):
        for fin in range(start + min_window_size, len(firing_rate)):
            window = firing_rate[start:fin]
            mm = max(window)
            if mm > min_firing_rate and all(window > thresh * mm):
                is_place_field[start:fin] = True
            else:
                break

    return is_place_field


def info_per_spike(occupancy, filtered_n_spikes):
    if all(filtered_n_spikes == 0):
        return 0
    eps = 2.22044604925e-16
    p_i = occupancy / np.sum(occupancy)
    lam_i = filtered_n_spikes / occupancy
    lam = np.mean(lam_i)
    return np.sum(p_i * lam_i / lam * np.log2(lam_i / lam + eps))


def info_per_sec(occupancy, filtered_n_spikes):
    if all(filtered_n_spikes == 0):
        return 0
    eps = 2.22044604925e-16
    p_i = occupancy / np.sum(occupancy)
    lam_i = filtered_n_spikes / occupancy
    lam = np.mean(lam_i)
    return np.sum(p_i * lam_i * np.log2(lam_i / lam + eps))
