# Code to replicate Senzai2017Neuron figures
import numpy as np
import matplotlib.pyplot as plt
import scipy

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


def linearize_trial(norm_trial_pos):
    decision_ind = np.where(-norm_trial_pos[:, 0] > .95)[0][0]
    x_center = -1 - norm_trial_pos[:decision_ind, 0]
    x_arm = np.arctan2(np.abs(norm_trial_pos[decision_ind:, 1]), -norm_trial_pos[decision_ind:, 0])
    full_run = np.hstack((x_center, x_arm + x_center[-1])) * .65 / 2
    return full_run


def compute_linear_place_fields(trials_df, pos, pos_tt, spikes, direction,
                                spatial_bins=np.arange(-.65, 1.02, .0168)):

    sampling_rate = len(pos_tt) / (np.max(pos_tt) - np.min(pos_tt))

    trials_df = trials_df

    df = trials_df
    df = df[df[direction] == 1]
    linearized_trials = list()
    speeds = list()

    running = np.zeros(len(pos), dtype='bool')
    lin_pos = np.zeros(len(pos)) * np.nan

    for i, row in list(df.iterrows()):
        trial_inds = isin_single_interval(pos_tt, [row['start'], row['end']],
                                          inclusive_left=True, inclusive_right=False)
        trial_pos = pos[trial_inds, :]
        linearized_pos = linearize_trial(trial_pos)
        linearized_trials.append(linearized_pos)
        lin_pos[trial_inds] = linearized_pos

        speed = np.diff(linearized_pos) / np.diff(pos_tt[trial_inds])
        speed = np.hstack((0, np.abs(smooth(speed, 40))))
        speeds.append(speed)

        running[trial_inds] = speed > .03

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)
    # keep only spike_pos_inds where running for that ind is True
    spike_pos_inds = spike_pos_inds[running[spike_pos_inds]]

    Oc = np.histogram(lin_pos[running], spatial_bins)[0]
    Spk = np.histogram(lin_pos[spike_pos_inds], spatial_bins)[0]
    FR = Spk / Oc * sampling_rate
    xx = spatial_bins[:-1] + (spatial_bins[1] - spatial_bins[0]) / 2

    return FR, xx



