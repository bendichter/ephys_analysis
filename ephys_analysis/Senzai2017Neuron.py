# Code to replicate Senzai2017Neuron figures
import numpy as np
import matplotlib.pyplot as plt
import scipy

from h5py import File

from .analysis import phase_modulation
from .vis import plot_mean_resultant_length, plot_phase_pref
from .utils import threshcross
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


def gen_fig2b(gamma_results, theta_results, cell_types):
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


def get_position(fname):

    aa = scipy.io.loadmat(fname)
    pos = aa['twhl_linearized'][:, 1:]
    pos_scale = 65 / 2
    pos = pos * pos_scale  # in cm
    tt = aa['twhl_linearized'][:, 0]

    return pos, tt


def compute_circular_T_place_fields(position, tt, spikes, pos_bin_length, speed_thresh):
    # 1.86 cm spatial bins
    # speed > 3 cm/s
    # gaussian kernel: 5.57 cm
    # place field: continuous 8.3 cm stretch (5 pixels)
        # peak FR > 1 Hz
        # there is a second condition that I don't understand:
        # For the circular T-maze, a place field was defined as a continuous region of at least 8.3 cm (5 pixels) where the firing rate was above 50% of the local maximum firing rate and the peak firing rate of the area was > 1 Hz
    #T-maze: 65 cm central arm, 102 cm left side arm, 102 cm right side arm)
    speed = np.sqrt(np.sum(np.diff(position, axis=0) ** 2, 1)) / np.diff(tt)
    speed = scipy.signal.medfilt(speed, 11)
    speed[np.isnan(speed)] = 0
    speed_windows = threshcross(speed, 3, 'both')

    xbins = np.linspace(0, 65, 36)
    ybins = np.linspace(-102, 102, 110)






    pass



