# Code to replicate Senzai2017Neuron figures
import numpy as np
import matplotlib.pyplot as plt

from h5py import File

from .analysis import phase_modulation
from .vis import plot_mean_resultant_length, plot_phase_pref
from .utils import listdict2dictlist

import pdb


def gather_fig2b(nwbfile, cell_types=('granule cell', 'mossy cell')):

    nwbfile = '/Users/bendichter/Desktop/Buzsaki/SenzaiBuzsaki2017/YutaMouse41-150903/YutaMouse41-150903_1.nwb'
    f = File(nwbfile, 'r')

    # gather spikes across tasks
    cell_dict = {}
    for exp in f['processing'].keys():
        if not (exp == 'shared'):
            spikes = f['processing'][exp]['spikes']
            for cell in spikes:
                if spikes[cell]['unit_description'].value in cell_types:
                    if cell not in cell_dict:
                        cell_dict[cell] = {}
                        cell_dict[cell]['spikes'] = []
                    cell_dict[cell]['spikes'] += list(spikes[cell]['times'][:])
                    cell_dict[cell]['type'] = spikes[cell]['unit_description'].value

    spikes, cell_types = zip(*[(np.array(value['spikes']), value['type']) for value in cell_dict.values()])
    cell_types = np.array(cell_types)

    lfp = f['processing']['shared']['LFP']['LFP']['data'][:]
    sampling_rate = f['processing']['shared']['LFP']['LFP']['starting_time'].attrs['rate']

    return spikes, cell_types, lfp, sampling_rate
    

def compute_fig2b(lfp, spikes, sampling_rate):
    theta_results = phase_modulation(lfp, spikes, 'theta', power_thresh=0.5,
                                     sampling_rate=sampling_rate, desc='theta phase modulation')

    gamma_results = phase_modulation(lfp, spikes, 'gamma', power_thresh=1.0,
                                     sampling_rate=sampling_rate, desc='gamma phase modulation')

    return theta_results, gamma_results


def gen_fig2b(theta_results, gamma_results, cell_types):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axc = 0
    for results in (theta_results, gamma_results):
        for stat, plotter in (('mean_phase', plot_phase_pref),
                              ('mean_resultant_length', plot_mean_resultant_length)):
            for cell_type, color in (('granule cell', 'orange'),
                                     ('mossy cell', 'purple')):
                #sig = (results['p'][cell_types == cell_type] < .05) & \
                #      (results['kappa'][cell_types == cell_type] >= 1.0)
                plotter(results[stat][cell_types == cell_type], color=color, ax=axs[axc])

            axc += 1

    axs[0].set_ylabel('neurons')
    axs[0].set_xlabel('phase')
    axs[1].set_xlabel('mean length')
    axs[0].legend(('granule cells', 'mossy cells'))

