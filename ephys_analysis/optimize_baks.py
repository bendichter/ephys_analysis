from neuroh5.io import NeuroH5CellAttrGen
from pynwb import NWBHDF5IO
from nested.optimize_utils import *
from collections import defaultdict
import click
from ephys_analysis.utils import find_nearest
from ephys_analysis.FMA import map_stats2
from ephys_analysis.vis import plot_1d_place_fields
from ephys_analysis.baks import baks
import math


context = Context()


def get_position(nwb):
    ts = nwb.modules['behavior'].data_interfaces['Position'].spatial_series['Position']
    data = ts.data * ts.conversion
    tt = np.arange(len(data)) / ts.rate
    return tt, data


def compute_distance_travelled(pos):
    distance = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(pos, axis=0)**2., axis=1))), 0, 0.)
    return distance


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_baks_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, debug):
    """
    Execute with default parameters on one process:
    python optimize_baks.py --config-file-path=$PATH_TO_CONFIG_FILE  --plot

    Execute with default parameters with multiple processes:
    mpirun -n 7 python -m mpi4py.futures optimize_baks.py --config-file-path=$PATH_TO_CONFIG_FILE --plot

    Execute parallel optimization:
    mpirun -n 7 python -m mpi4py.futures -m nested.optimize optimize_baks.py \
        --config-file-path=$PATH_TO_CONFIG_FILE --plot --pop-size=200 --path-lenght=3 --max-iter=50 --disp

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str (path)
    :param label: str
    :param verbose: int
    :param plot: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0
    if debug:
        config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                                    export_file_path=export_file_path, label=label, disp=context.disp, verbose=verbose,
                                    plot=plot, debug=debug, **kwargs)
    else:
        from nested.parallel import MPIFuturesInterface
        context.interface = MPIFuturesInterface()
        context.interface.start(disp=True)
        context.interface.ensure_controller()
        config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                                    export_file_path=export_file_path, label=label, disp=context.disp, verbose=verbose,
                                    plot=plot, debug=debug, is_controller=True, **kwargs)
        context.interface.apply(config_optimize_interactive, __file__, config_file_path=config_file_path,
                                output_dir=output_dir, export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, verbose=verbose, plot=plot, debug=debug, **kwargs)

        features = {}
        args = context.interface.execute(get_args_static_distribute_cells)
        group_size = len(args[0])
        sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                    [[context.plot] * group_size]
        primitives = context.interface.map(compute_features_spatial_firing_rates, *sequences)
        new_features = context.interface.execute(filter_features_spatial_firing_rates, primitives, features,
                                                 context.export, context.plot)
        features.update(new_features)
        features, objectives = context.interface.execute(get_objectives, features, context.export)

        sys.stdout.flush()
        print 'features:'
        pprint.pprint({key: val for (key, val) in features.iteritems() if key in context.feature_names})
        print 'objectives'
        pprint.pprint({key: val for (key, val) in objectives.iteritems() if key in context.objective_names})
        sys.stdout.flush()

        context.interface.stop()


def config_worker():
    init_context()


def init_context():
    if 'plot' not in context():
        context.plot = False
    pop_names = ['MPP', 'LPP']
    nwb_spikes_file = NWBHDF5IO(context.nwb_spikes_file_path, 'r')
    nwb_spikes = nwb_spikes_file.read()

    t, pos = get_position(nwb_spikes)
    d = compute_distance_travelled(pos)
    if context.plot:
        fig, axes = plt.subplots()
        axes.plot(t, d)
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Distance (m)')
        axes.set_title('Spatial trajectory')
        clean_axes(axes)
        fig.show()

    start_time = time.time()
    spike_trains = defaultdict(dict)
    nwb_gids = nwb_spikes.units.id.data[:]
    nwb_cell_types = nwb_spikes.units['cell_type'][:]
    nwb_spike_trains = nwb_spikes.units['spike_times'][:]
    for i in xrange(len(nwb_gids)):
        gid = nwb_gids[i]
        pop_name = nwb_cell_types[i]
        spike_trains[pop_name][gid] = nwb_spike_trains[i]
    del nwb_spikes, nwb_gids, nwb_cell_types, nwb_spike_trains
    nwb_spikes_file.close()

    count = sum([len(spike_trains[pop_name]) for pop_name in spike_trains])
    if context.verbose > 1:
        print 'optimize_baks: pid: %i; loading spikes for %i gids from cell populations: %s took %.1f s' % \
              (os.getpid(), count, ', '.join(str(pop_name) for pop_name in spike_trains), time.time() - start_time)
        sys.stdout.flush()

    if 'block_size' not in context():
        context.block_size = context.num_workers
    gid_block_size = int(math.ceil(float(count) / context.block_size))

    imposed_rates = defaultdict(dict)
    start_time = time.time()
    for pop_name in pop_names:
        count = 0
        cell_attr_gen = NeuroH5CellAttrGen(context.neuroh5_rates_file_path, pop_name, comm=context.comm,
                                           namespace=context.neuroh5_rates_namespace)
        for gid, attr_dict in cell_attr_gen:
            if gid is not None:
                imposed_rates[pop_name][gid] = attr_dict['rate']
                # spike_trains[pop_name][gid] = attr_dict['spiketrain'] / 1000.  # convert to s
                count += 1
        if context.verbose > 1:
            print 'optimize_baks: pid: %i; loading imposed rates for %i gids from cell population: %s took %.1f s' % \
                  (os.getpid(), count, pop_name, time.time() - start_time)
            sys.stdout.flush()

    if context.plot:
        t_bins = np.linspace(0., max(t), 100)
        d_bins = np.linspace(0., max(d), 100)
        fig, axes = plt.subplots(2,2)
        for i, pop_name in enumerate(['MPP', 'LPP']):
            axes[0,i].set_title('Imposed rates: %s' % pop_name)
            axes[0,i].set_xlabel('Distance (m)')
            axes[0,i].set_ylabel('Firing rate (Hz)')
            axes[1,i].set_title('Binned spike counts: %s' % pop_name)
            axes[1,i].set_xlabel('Distance (m)')
            axes[1,i].set_ylabel('Count')
            count = 0
            for gid in imposed_rates[pop_name]:
                rate = imposed_rates[pop_name][gid]
                if np.max(rate) > 5.:
                    hist, edges = np.histogram(spike_trains[pop_name][gid], bins=t_bins)
                    axes[0,i].plot(d, rate)
                    axes[1,i].plot(d_bins[1:], hist)
                    count += 1
                if count > 20:
                    break
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    plotted = {pop_name: False for pop_name in pop_names}
    min_field_len = int(context.min_field_width / max(d) * len(d))
    context.update(locals())


def update_baks_params(x, local_context):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    context.update(param_array_to_dict(x, context.param_names))


def get_args_static_distribute_cells():
    """
    Distribute ranges of cells across workers.
    :return: list of lists
    """
    pop_names_list = []
    gid_lists = []
    for pop_name in context.pop_names:
        count = 0
        gids = context.spike_trains[pop_name].keys()
        while count < len(gids):
            pop_names_list.append(pop_name)
            gid_lists.append(gids[count:count+context.gid_block_size])
            count += context.gid_block_size
    return [pop_names_list, gid_lists]


def compute_features_spatial_firing_rates(x, pop_name, gids, export=False, plot=False):
    """

    :param x: array
    :param pop_name: str
    :param gids: list of int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_source_contexts(x, context)

    inferred_firing_rates = dict()
    num_fields_residuals = []
    mean_field_width_residuals = []
    rate_residuals = []
    if plot and not context.plotted[pop_name]:
        fig, axes = plt.subplots(3, 4, sharex=True)
        for j in xrange(4):
            axes[2][j].set_xlabel('Distance (m)')
        for i in xrange(3):
            axes[i][0].set_ylabel('Firing rate (Hz)')
    count = 0
    for gid in gids:
        imposed_rate = context.imposed_rates[pop_name][gid]
        spike_train = context.spike_trains[pop_name][gid]
        binned_spike_indexes = find_nearest(spike_train, context.t)

        if len(spike_train) > 0:
            inferred_firing_rate = baks(spike_train, context.t, a=context.alpha, b=context.beta)[0]
        else:
            inferred_firing_rate = np.zeros(len(context.t))
        if np.max(inferred_firing_rate) > 5.:
            if plot and not context.plotted[pop_name] and count < 12:
                row = count / 4
                col = count % 4
                imposed_fields = map_stats2(imposed_rate,
                                            threshold=context.field_rate_threshold, min_size=context.min_field_len,
                                            min_peak=context.min_field_amp)['fields']
                inferred_fields = map_stats2(inferred_firing_rate, threshold=context.field_rate_threshold,
                                             min_size=context.min_field_len,
                                             min_peak=context.min_field_amp)['fields']
                axes[row][col].plot(context.d, inferred_firing_rate, label='Inferred rate')
                axes[row][col].plot(context.d, imposed_rate, label='Imposed rate')
                axes[row][col].plot(context.d[binned_spike_indexes], np.ones(len(binned_spike_indexes)), 'k.',
                                    label='Spikes')
                plot_1d_place_fields(context.d, imposed_fields, ax=axes[row][col])
                plot_1d_place_fields(context.d, inferred_fields, h=2, ax=axes[row][col])
                axes[row][col].set_title('cell_id: %i' % gid)
                count += 1
        inferred_firing_rates[gid] = inferred_firing_rate
    if plot and not context.plotted[pop_name]:
        axes[0][0].legend(loc='best')
        clean_axes(axes)
        fig.suptitle(pop_name)
        fig.tight_layout()
        fig.show()
        context.plotted[pop_name] = True

    for gid in gids:
        imposed_rate = context.imposed_rates[pop_name][gid]
        imposed_field_stats = map_stats2(imposed_rate, threshold=context.field_rate_threshold,
                                         min_size=context.min_field_len, min_peak=context.min_field_amp)
        inferred_field_stats = map_stats2(inferred_firing_rates[gid], threshold=context.field_rate_threshold,
                                          min_size=context.min_field_len, min_peak=context.min_field_amp)
        imposed_num_fields = len(imposed_field_stats['sizes'])
        inferred_num_fields = len(inferred_field_stats['sizes'])
        num_fields_residuals.append(abs(imposed_num_fields - inferred_num_fields))

        if imposed_num_fields == 0 and inferred_num_fields == 0:
            mean_field_width_residuals.append(np.nan)
        else:
            if imposed_num_fields > 0:
                mean_imposed_field_width = np.mean(imposed_field_stats['sizes']) * max(context.d)
            else:
                mean_imposed_field_width = 0.
            if inferred_num_fields > 0:
                mean_inferred_field_width = np.mean(inferred_field_stats['sizes']) * max(context.d)
            else:
                mean_inferred_field_width = 0.
            mean_field_width_residuals.append(abs(mean_imposed_field_width - mean_inferred_field_width))
        rate_residuals.append(np.mean(np.abs(np.subtract(imposed_rate, inferred_firing_rates[gid]))))

    if context.verbose > 1:
        print 'Process: %i: computing spatial_firing_rate features for population: %s cell_ids: %i:%i with x: %s ' \
              'took %.1f s' % (os.getpid(), pop_name, gids[0], gids[-1], ', '.join('%.2f' % i for i in x),
                               time.time()-start_time)
        sys.stdout.flush()

    return dict(num_fields_residuals=num_fields_residuals, mean_field_width_residuals=mean_field_width_residuals,
                rate_residuals=rate_residuals)


def filter_features_spatial_firing_rates(primitives, current_features, export=False, plot=False):
    """

    :param primitives: list of dict
    :param current_features: dict
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    features = {}
    num_fields_residuals = []
    mean_field_width_residuals = []
    rate_residuals = []
    for this_feature_dict in primitives:
        num_fields_residuals.extend(this_feature_dict['num_fields_residuals'])
        mean_field_width_residuals.extend(this_feature_dict['mean_field_width_residuals'])
        rate_residuals.extend(this_feature_dict['rate_residuals'])

    features['num_fields_residuals'] = np.mean(num_fields_residuals)
    features['num_fields_error'] = np.mean(np.square(np.divide(num_fields_residuals,
                                                               context.target_range['num_fields'])))
    features['field_width_residuals'] = np.nanmean(mean_field_width_residuals)
    features['field_width_error'] = np.nanmean(np.square(np.divide(mean_field_width_residuals,
                                                               context.target_range['field_width'])))
    features['firing_rate_residuals'] = np.mean(np.square(np.divide(rate_residuals,
                                                                    context.target_range['firing_rate'])))

    if context.disp:
        print 'Process: %i: filtering spatial_firing_rate features for %i cell_ids took %.1f s' % \
              (os.getpid(), len(num_fields_residuals), time.time()-start_time)
        sys.stdout.flush()
    return features


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}
    for objective_name in context.objective_names:
        objectives[objective_name] = features[objective_name]

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)