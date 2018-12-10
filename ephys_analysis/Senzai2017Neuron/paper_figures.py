# Code to replicate Senzai2017Neuron figures
import matplotlib.pyplot as plt

from ephys_analysis.band_analysis import phase_modulation
from ephys_analysis.vis import plot_mean_resultant_length, plot_phase_pref


def compute_fig2b(lfp, spikes, sampling_rate):
    theta_results = phase_modulation(lfp, spikes, 'theta', power_thresh=0.5,
                                     sampling_rate=sampling_rate)

    gamma_results = phase_modulation(lfp, spikes, 'gamma', power_thresh=1.0,
                                     sampling_rate=sampling_rate)

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


