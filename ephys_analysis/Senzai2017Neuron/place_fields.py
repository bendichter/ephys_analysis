import numpy as np
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from ephys_analysis.utils import isin_single_interval, smooth, find_nearest


def linearize_trial(norm_trial_pos, diameter):
    """Linearize a single trial position series from a theta maze

    Parameters
    ----------
    norm_trial_pos: np.ndarray
        Entire maze bounded by (0, 1, 0, 1)
    diameter: float
        Diameter of the maze in meters

    Returns
    -------
    np.ndarray

    """

    decision_ind = np.where(-norm_trial_pos[:, 0] > .95)[0][0]
    x_center = -1 - norm_trial_pos[:decision_ind, 0]
    x_arm = np.arctan2(np.abs(norm_trial_pos[decision_ind:, 1]),
                       -norm_trial_pos[decision_ind:, 0])
    full_run = np.hstack((x_center, x_arm + x_center[-1])) * diameter / 2 + 2 / np.pi

    return full_run


def linearize_session(pos, pos_tt, diameter=0.65, running_speed=.03, speed_smoother=40, trials=None):
    """Compute linearized position across trials. Only include points during the trials
     of interest and while animal is running

    Parameters
    ----------
    pos: np.ndarray
        normalized x, y position for theta (aka eight) maze
    pos_tt: np.ndarray
    diameter: float (optional)
        in meters. Default = 65 cm
    running_speed: float (optional)
        in m/s. Default = 3 cm/s
    speed_smoother: int
        number of time points to smooth speed over
    trials: np.ndarray (optional)
        [[start1, end1], [start2, end2], ...]

    Returns
    -------
    lin_pos: np.ndarray
        Linearized position. If not running, lin_pos is NaN

    """

    if trials is None:
        trials = [[np.min(pos_tt), np.max(pos_tt)]]

    lin_pos = np.zeros(len(pos)) * np.nan

    for i, trial in trials.iterrows():
        trial_inds = isin_single_interval(pos_tt, [trial['start_time'], trial['stop_time']],
                                          inclusive_left=True, inclusive_right=False)
        trial_pos = pos[trial_inds, :]
        linearized_pos = linearize_trial(trial_pos, diameter)
        if trial['condition'] == 'run_right':
            linearized_pos += 1 + 2 / np.pi
        lin_pos[trial_inds] = linearized_pos

        speed = np.diff(linearized_pos) / np.diff(pos_tt[trial_inds])
        speed = np.hstack((0, np.abs(smooth(speed, speed_smoother))))

        if running_speed:
            lin_pos[trial_inds][speed < running_speed] = np.nan

    return lin_pos


def compute_speed(pos, pos_tt, smooth_param=40):
    """Compute boolean of whether the speed of the animal was above a threshold
    for each time point

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds

    Returns
    -------
    running: np.ndarray(dtype=bool)

    """
    speed = np.hstack((0, np.sqrt(np.sum(np.diff(pos.T) ** 2, axis=0)) / np.diff(pos_tt)))
    return smooth(speed, smooth_param)


def compute_2d_occupancy(pos, pos_tt, edges, speed_thresh=0.03):
    """Computes occupancy per bin in seconds

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    edges: np.ndarray(dtype=float)
        edges of histogram in meters
    speed_thresh: float, optional
        in meters. Default = 3.0 cm/s

    Returns
    -------
    occupancy: np.ndarray(dtype=float)
        in seconds
    running: np.ndarray(dtype=bool)

    """

    sampling_period = (np.max(pos_tt) - np.min(pos_tt)) / len(pos_tt)
    is_running = compute_speed(pos, pos_tt) > speed_thresh
    run_pos = pos[is_running, :]
    occupancy = np.histogram2d(run_pos[:, 0],
                               run_pos[:, 1],
                               [edges, edges])[0] * sampling_period  # in seconds

    return occupancy, is_running


def compute_2d_n_spikes(pos, pos_tt, spikes, edges, speed_thresh=0.03):
    """Returns speed-gated occupancy and speed-gated and Gaussian-filtered
    firing rate

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    spikes: np.ndarray(dtype=float)
        in seconds
    edges: np.ndarray(dtype=float)
        edges of histogram in meters
    speed_thresh: float
        in meters. Default = 3.0 cm/s

    Returns
    -------
    """

    is_running = compute_speed(pos, pos_tt) > speed_thresh

    spike_pos_inds = find_nearest(spikes, pos_tt)
    spike_pos_inds = spike_pos_inds[is_running[spike_pos_inds]]
    pos_on_spikes = pos[spike_pos_inds, :]

    n_spikes = np.histogram2d(pos_on_spikes[:, 0],
                              pos_on_spikes[:, 1],
                              [edges, edges])[0]

    return n_spikes


def compute_2d_firing_rate(pos, pos_tt, spikes, pixel_width=0.0092,
                           speed_thresh=0.03, field_len=0.46,
                           gaussian_sd=0.0184):
    """Returns speed-gated occupancy and speed-gated and
    Gaussian-filtered firing rate

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    spikes: np.ndarray(dtype=float)
        in seconds
    pixel_width: float
        in meters. Default = 0.92 cm
    speed_thresh: float
        in meters. Default = 3.0 cm/s
    field_len: float
        in meters. Default = 46 cm
    gaussian_sd: float
        in meters. Default = 1.84 cm

    Returns
    -------

    occupancy: np.ndarray
        in seconds
    filtered_firing_rate: np.ndarray(shape=(cell, x, y), dtype=float)
        in Hz

    """

    edges = np.arange(0, field_len + pixel_width, pixel_width)

    occupancy, running = compute_2d_occupancy(pos, pos_tt, edges)

    n_spikes = compute_2d_n_spikes(pos, pos_tt, spikes, edges, speed_thresh)

    firing_rate = n_spikes / occupancy  # in Hz
    firing_rate[np.isnan(firing_rate)] = 0

    filtered_firing_rate = gaussian_filter(firing_rate, gaussian_sd / pixel_width)

    return occupancy, filtered_firing_rate, edges


def compute_2d_place_fields(firing_rate, min_firing_rate=1, thresh=0.2,
                            min_size=100):
    """Compute place fields

    Parameters
    ----------
    firing_rate: np.ndarray(NxN, dtype=float)
    min_firing_rate: float
        in Hz
    thresh: float
        % of local max
    min_size: float
        minimum size of place field in pixels

    Returns
    -------
    receptive_fields: np.ndarray(NxN, dtype=int)
        Each receptive field is labeled with a unique integer
    """

    local_maxima_inds = firing_rate == maximum_filter(firing_rate, 3)
    receptive_fields = np.zeros(firing_rate.shape, dtype=int)
    n_receptive_fields = 0
    firing_rate = firing_rate.copy()
    for local_max in np.flipud(np.sort(firing_rate[local_maxima_inds])):
        labeled_image, num_labels = label(firing_rate > max(local_max * thresh,
                                                            min_firing_rate))
        if not num_labels:  # nothing above min_firing_thresh
            return receptive_fields
        for i in range(1, num_labels + 1):
            image_label = labeled_image == i
            if local_max in firing_rate[image_label]:
                break
            if np.sum(image_label) >= min_size:
                n_receptive_fields += 1
                receptive_fields[image_label] = n_receptive_fields
                firing_rate[image_label] = 0

    return receptive_fields


def compute_1d_occupancy(pos, spatial_bins, sampling_rate):
    finite_lin_pos = pos[np.isfinite(pos)]

    occupancy = np.histogram(
        finite_lin_pos, bins=spatial_bins)[0][:-2] / sampling_rate

    return occupancy


def compute_linear_firing_rate(pos, pos_tt, spikes, gaussian_sd=0.0557,
                               spatial_bin_len=0.0168):
    """The occupancy and number of spikes, speed-gated, binned, and smoothed
    over position

    Parameters
    ----------
    pos: np.ndarray
        linearized position
    pos_tt: np.ndarray
        sample times in seconds
    spikes: np.ndarray
        for a single cell in seconds
    gaussian_sd: float (optional)
        in meters. Default = 5.57 cm
    spatial_bin_len: float (optional)
        in meters. Default = 1.68 cm


    Returns
    -------
    xx: np.ndarray
        center of position bins in meters
    occupancy: np.ndarray
        time in each spatial bin in seconds, during appropriate trials and
        while running
    filtered_n_spikes: np.ndarray
        number of spikes in each spatial bin,  during appropriate trials, while
        running, and processed with a Gaussian filter

    """

    spatial_bins = np.arange(np.min(pos), np.max(pos) + spatial_bin_len,
                             spatial_bin_len)

    sampling_rate = len(pos_tt) / (np.max(pos_tt) - np.min(pos_tt))

    occupancy = compute_1d_occupancy(pos, spatial_bins, sampling_rate)

    # find pos_tt bin associated with each spike
    spike_pos_inds = find_nearest(spikes, pos_tt)

    pos_on_spikes = pos[spike_pos_inds]
    finite_pos_on_spikes = pos_on_spikes[np.isfinite(pos_on_spikes)]

    n_spikes = np.histogram(finite_pos_on_spikes, bins=spatial_bins)[0][:-2]

    firing_rate = n_spikes / occupancy

    filtered_firing_rate = gaussian_filter(
        firing_rate, gaussian_sd / spatial_bin_len)
    xx = spatial_bins[:-3] + (spatial_bins[1] - spatial_bins[0]) / 2

    return xx, occupancy, filtered_firing_rate


def compute_linear_place_fields(firing_rate, min_window_size=5,
                                min_firing_rate=1., thresh=0.5):
    """Find consecutive bins where all are >= 50% of local max firing rate and
    the local max in > 1 Hz

    Parameters
    ----------
    firing_rate: array-like(dtype=float)
    min_window_size: int
    min_firing_rate: float
    thresh: float

    Returns
    -------
    np.ndarray(dtype=bool)

    """

    is_place_field = np.zeros(len(firing_rate), dtype='bool')
    for start in range(len(firing_rate) - min_window_size):
        for fin in range(start + min_window_size, len(firing_rate)):
            window = firing_rate[start:fin]
            mm = max(window)
            if mm > min_firing_rate and all(window > thresh * mm):
                is_place_field[start:fin] = True
            else:
                break

    return is_place_field


def info_per_spike(occupancy, firing_rate):
    if all(firing_rate == 0):
        return 0
    eps = 2.22044604925e-16

    # in case occupancy and firing_rate are 2D
    occupancy = occupancy.ravel()
    firing_rate = firing_rate.ravel()

    p_i = occupancy / np.sum(occupancy)
    lam_i = firing_rate
    lam = np.mean(lam_i)
    return np.sum(p_i * lam_i / lam * np.log2(lam_i / lam + eps))


def info_per_sec(occupancy, firing_rate):
    if all(firing_rate == 0):
        return 0
    eps = 2.22044604925e-16
    # in case occupancy and firing_rate are 2D
    occupancy = occupancy.ravel()
    firing_rate = firing_rate.ravel()

    p_i = occupancy / np.sum(occupancy)
    lam_i = firing_rate
    lam = np.mean(lam_i)
    return np.sum(p_i * lam_i * np.log2(lam_i / lam + eps))
