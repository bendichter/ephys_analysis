import numpy as np

from h5py import File


def filter_by_cell_type(spikes_module, cell_types):
    """

    Parameters
    ----------
    spikes_module: h5py._hl.group.Group
    cell_types: list
        list of cell type names

    Returns
    -------
    np.array of cell_index numbers

    """

    values = spikes_module['Cell Types']['values'][:]
    indices = spikes_module['Cell Types']['indices']
    cell_index = spikes_module['Cell Types']['cell_index']

    cell_types = [x.encode() for x in cell_types]

    out = []
    for ind, cell_ind in zip(indices, cell_index):
        if values[ind] in cell_types:
            out.append(cell_ind)

    return np.array(out)


def get_cell_type(spikes_module, input_cell_index):
    """

    Parameters
    ----------
    spikes_module: h5py._hl.group.Group
    input_cell_index

    Returns
    -------

    """

    values = spikes_module['Cell Types']['values'][:]
    indices = spikes_module['Cell Types']['indices']
    cell_index = spikes_module['Cell Types']['cell_index']

    ii = [np.where(cell_index[:] == x)[0][0] for x in input_cell_index]

    return values[indices[ii]]


def get_lfp(nwbfile):
    """

    Parameters
    ----------
    nwbfile: pynwb.file.NWBFile

    Returns
    -------
    data: np.ndarray
    fs: float

    """
    data = nwbfile.acquisition['lfp'].data[:]
    fs = nwbfile.acquisition['lfp'].rate

    return data, fs


def get_cell_data(nwbfile, cell_types=('granule cell', 'mossy cell')):
    """

    Parameters
    ----------
    nwbfile: str
    cell_types: tuple of strings

    Returns
    -------
    spikes, cell_types
        spikes: list of np.arrays of spike times
        cell_types: np.array of strings of cell types

    """
    # gather spikes across tasks
    with File(nwbfile, 'r') as f:
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

        spikes, cell_types = zip(*[(np.array(value['spikes']), value['type'])
                                   for value in cell_dict.values()])
        cell_types = np.array(cell_types)

    return spikes, cell_types
