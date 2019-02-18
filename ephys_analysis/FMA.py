"""
Adapted from FMAToolbox: https://github.com/buzsakilab/buzcode/tree/master/externalPackages/FMAToolbox

by Ben Dichter
"""
import numpy as np
from scipy.ndimage import label


def find_field(firing_rate, threshold):
    """Copied FMAToolbox

    Parameters
    ----------
    firing_rate: np.ndarray
    threshold: float

    Returns
    -------

    """
    mm = np.max(firing_rate)

    labeled_image, num_labels = label(firing_rate > threshold)
    for i in range(1, num_labels + 1):
        image_label = labeled_image == i
        if mm in firing_rate[image_label]:
            return image_label, image_label


def find_field2(firing_rate, thresh):
    """Only works for 1D

    Parameters
    ----------
    firing_rate: np.array
    thresh: float

    Returns
    -------

    """
    firing_rate = np.array(firing_rate)
    imm = np.argmax(firing_rate)
    mm = np.max(firing_rate)
    # TODO: make more efficient by using argmax instead of where()[0]
    first = np.where(np.diff(firing_rate[:imm]) < 0)[0]
    if len(first) == 0:
        first = 0
    else:
        first = first[-1] + 2

    last = np.where(np.diff(firing_rate[imm:]) > 0)[0]

    if len(last) == 0:
        last = len(firing_rate)
    else:
        last = last[0] + imm + 1
    field_buffer = np.zeros(firing_rate.shape, dtype='bool')
    field_buffer[first:last] = True
    field = field_buffer & (firing_rate > thresh * mm)

    return field_buffer, field


def map_stats2(firing_rate, threshold=0.1, min_size=5, min_peak=1.0):
    """

    :param firing_rate: array
    :param threshold: float
    :param min_size: int
    :param min_peak: float
    :return:
    """
    firing_rate = firing_rate.copy()
    firing_rate = firing_rate - np.min(firing_rate)
    out = dict(sizes=list(), peaks=list(), means=list())
    out['fields'] = np.zeros(firing_rate.shape)
    field_counter = 1
    while True:
        peak = np.max(firing_rate)
        if peak < min_peak:
            break
        field_buffer, field = find_field(firing_rate, threshold)
        field_size = np.sum(field)
        if field_size > min_size and \
                (np.max(firing_rate[field]) > (2 * np.min(firing_rate[field_buffer]))):
            out['fields'][field] = field_counter
            out['sizes'].append(float(field_size) / len(firing_rate))
            out['peaks'].append(peak)
            out['means'].append(np.mean(firing_rate[field]))
            field_counter += 1
        firing_rate[field_buffer] = 0

    return out


def map_stats(firing_rate, threshold=0.2, min_size=None, min_peak=1.0):
    """

    Parameters
    ----------
    firing_rate: np.ndarray(dtype=float, shape=NxN)
        1 or 2-D map in Hz. Default = 0.2
    threshold: float
        Values above threshold * peak belong to the field
    min_size: int | None
        Fields smaller than this size are considered spurious and ignored
        (default = 100 for 2D, 10 for 1D)
    min_peak: float
        Peaks smaller than this size are considered spurious and ignored
        (default = 1.0)

    Returns
    -------

    """
    if min_size is None:
        ndim = len(firing_rate.shape)
        if ndim == 1:
            min_size = 10
        elif ndim == 2:
            min_size = 100
        else:
            raise ValueError('no default min size value for 3+D maps')

    firing_rate = firing_rate.copy()
    firing_rate = firing_rate - np.min(firing_rate)
    out = dict(fields=list(), sizes=list(), peaks=list(), means=list())
    while True:
        peak = np.max(firing_rate)

        if peak < min_peak:
            break

        field1 = find_field2(firing_rate, peak * threshold)
        size1 = np.sum(field1)
        # Does this field include two coalescent subfields? To answer this
        # this question, we simply re-run the same field-searching procedure on
        # the field we then either keep the original field or choose the
        # subfield if the latter is less than 1/2 the size of the former
        m = peak * threshold
        field2 = find_field2(firing_rate-m, (peak-m) * threshold)
        size2 = np.sum(field2)

        if size2 < 1/2 * size1:
            field = field2
        else:
            field = field1

        field_size = np.sum(field)
        if field_size > min_size:
            out['fields'].append(field)
            out['sizes'].append(field_size)
            out['peaks'].append(peak)
            out['means'].append(np.mean(firing_rate[field]))
        firing_rate[field] = 0
        if np.all(firing_rate == 0):
            break
    if out['fields']:
        out['fields'] = np.array([x * (i + 1) for i, x in enumerate(out['fields'])])
        out['fields'] = np.sum(out['fields'], 0)
    else:
        out['fields'] = np.zeros(firing_rate.shape)

    return out








