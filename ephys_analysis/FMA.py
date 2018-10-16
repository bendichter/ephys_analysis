import numpy as np
from scipy.ndimage import label

from .utils import find_nearest


def find_field(firing_rate, threshold):
    mm = np.max(firing_rate)

    labeled_image, num_labels = label(firing_rate > threshold)
    for i in range(1, num_labels + 1):
        image_label = labeled_image == i
        if mm in firing_rate[image_label]:
            return image_label


def map_stats(firing_rate, threshold=0.2, min_size=100, min_peak=1.0):

    firing_rate = firing_rate.copy()
    out = dict(fields=list(), sizes=list(), peaks=list(), means=list())
    while True:
        peak = np.max(firing_rate)

        if peak < min_peak:
            break

        field1 = find_field(firing_rate, peak * threshold)
        size1 = np.sum(field1)
        m = peak * threshold
        field2 = find_field(firing_rate-m, (peak-m) * threshold)
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


class Map:

    def __init__(self, tt, xx, t2, y=None, z=None, smooth=2, nbins=[50, 50],
                 min_time=0, mode='discard', max_distance=5, max_gap=0.1,
                 type='lll'):
        if isinstance(nbins, int):
            nbins = [nbins, nbins]

        if z is None:
            point_process = True
        else:
            point_process = False
        y = np.linspace(0, 1, nbins[1])
        dt = np.diff(tt)
        df = np.hstack((dt, dt[-1]))

        if point_process:
            n = find_nearest(z, tt)

        if y is None:
            self.x = np.linspace(0, 1, nbins[0])
            self.count = np.histogram(xx, self.x)








