# Code to replicate Senzai2017Neuron figures

import matplotlib.pyplot as plt

from h5py import File

from .lfp import phase_modulation
from .vis import plot_mean_resultant_length, plot_phase_pref


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

    spikes, cell_types = zip(*[(value['spikes'], value['type']) for value in cell_dict.values()])

    lfp = f['processing']['shared']['LFP']['LFP']['data'][:]
    sampling_rate = f['processing']['shared']['LFP']['LFP']['starting_time'].attrs['rate']

    return spikes, cell_types, lfp, sampling_rate
    

def compute_fig2b(lfp, spikes, sampling_rate):
    theta_results = phase_modulation(lfp, spikes, 'theta', power_thresh=0.5,
                                     sampling_rate=sampling_rate)
    theta_results = [x for x in theta_results if (x['p'] < 0.05) and (x['k'] > 1.0)]

    gamma_results = phase_modulation(lfp, spikes, 'gamma', power_thresh=1.0,
                                     sampling_rate=sampling_rate)
    gamma_results = [x for x in gamma_results if (x['p'] < 0.05) and (x['k'] > 1.0)]

    return theta_results, gamma_results


def gen_fig2b(theta_results, gamma_results, cell_types):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axc = 0
    for results in (theta_results, gamma_results):
        for stat, plotter in (('mean_phase', plot_phase_pref),
                              ('mean_resultant_length', plot_mean_resultant_length)):
            for cell_type, color in (('granule cell', 'orange'),
                                     ('mossy cell', 'purple')):
                plotter(results[stat][cell_types == cell_type], color=color, ax=axs[axc])

            axc += 1