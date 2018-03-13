from pynwb import NWBHDF5IO, load_namespaces
from pynwb.ecephys import FilteredEphys, ElectricalSeries
from analysis import filter_lfp, hilbert_lfp

ns_path = 'general.namespace.yaml'
load_namespaces(ns_path)

# compute filter phase and amplitude
fpath = '/Users/bendichter/dev/to_nwb/yuta_data.nwb'
filter_results = {}
with NWBHDF5IO(fpath, 'a') as io:
    nwbfile = io.read()
    lfp_data = nwbfile.acquisition['lfp'].data[:]
    lfp_fs = nwbfile.acquisition['lfp'].rate
    lfp_electrodes = nwbfile.acquisition['lfp'].electrodes

    lfp_module = nwbfile.create_processing_module('lfp', source='source',
                                                  description='description')

    for filt in ('theta', 'gamma'):
        filter_results['filter'] = {}
        filt_data = filter_lfp(lfp_data, filt, lfp_fs)
        filter_results['filter']['phase'], filter_results['filter']['amp'] = hilbert_lfp(filt_data)
        es = ElectricalSeries(filt + ' filtered LFP', source='source',
                              data=filt_data, electrodes=lfp_electrodes,
                              rate=lfp_fs, starting_time=0.0)

        lfp_module.add_container(FilteredEphys(source='source',
                                               electrical_series=es,
                                               name=filt + ' filtered LFP'))
    io.write(nwbfile)
