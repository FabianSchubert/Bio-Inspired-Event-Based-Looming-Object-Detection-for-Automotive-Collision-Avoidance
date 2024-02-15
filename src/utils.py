import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import collections

from itertools import product


class FixedDict(collections.abc.MutableMapping):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError(k)

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __str__(self):
        return self.__data.__str__()


def play_var_interactive(v, pop, var, n_x, n_y, wd, hd, fig):
    ax = [
        [fig.add_subplot(n_y, n_x, k * n_x + l + 1) for l in range(n_x)]
        for k in range(n_y)
    ]

    t_smp_init = 0
    t_max = np.inf

    imgs = []

    for k in range(n_y):
        imgs.append([])
        for l in range(n_x):
            var_name = f"{var+pop}_{k}_{l}"

            va = np.array(v[var_name])
            va = np.reshape(va, (va.shape[0], hd, wd))

            t_max = np.minimum(t_max, va.shape[0])

            _ax = ax[k][l]

            _ax.set_xlim([0, wd])
            _ax.set_ylim([0, hd])
            _ax.invert_yaxis()

            _ax.xaxis.set_tick_params(labelbottom=False)
            _ax.yaxis.set_tick_params(labelleft=False)

            _ax.set_xticks([])
            _ax.set_yticks([])

            img = ax[k][l].imshow(va[t_smp_init], vmin=va.min(), vmax=va.max())

            imgs[-1].append(img)

    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(
        ax=ax_slider, label="Time", valmin=0, valmax=t_max - 1, valinit=t_smp_init
    )

    def update(t_smp):
        for k in range(n_y):
            for l in range(n_x):
                var_name = f"{var+pop}_{k}_{l}"

                va = np.array(v[var_name])
                va = np.reshape(va, (va.shape[0], hd, wd))

                idx = int(t_smp)
                imgs[k][l].set_array(va[idx])

        fig.canvas.draw_idle()

    t_slider.on_changed(update)

    return ax, ax_slider, t_slider


def play_spikes_from_file(evts_file, w, h):
    evts = np.load(evts_file)
    ids = evts[:, :2].astype("int")
    ids_flat = ids[:, 1] * w + ids[:, 0]

    times = evts[:, 2]

    fig = plt.figure()

    ax, ax_slider, t_slider = play_spikes_interactive(times, [ids_flat], w, h, fig)

    return ax, ax_slider, t_slider


def play_spikes_interactive(t, ID, pop, n_x, n_y, wd, hd, fig, tol=1.0):
    ax = [
        [fig.add_subplot(n_y, n_x, k * n_x + l + 1) for l in range(n_x)]
        for k in range(n_y)
    ]

    lines = []

    t_smp_init = 0
    t_min = np.inf
    t_max = 0

    for k in range(n_y):
        lines.append([])
        for l in range(n_x):
            var_name = f"{pop}_{k}_{l}"

            t_min = np.minimum(t_min, t[var_name][0].min())
            t_max = np.maximum(t_max, t[var_name][0].max())

            _ax = ax[k][l]

            _ax.set_xlim([0, wd])
            _ax.set_ylim([0, hd])
            _ax.invert_yaxis()

            _ax.xaxis.set_tick_params(labelbottom=False)
            _ax.yaxis.set_tick_params(labelleft=False)

            _ax.set_xticks([])
            _ax.set_yticks([])

            idx_init = np.where(np.abs(t[var_name][0] - t_smp_init) <= tol)[0]
            ID_smp_init = ID[var_name][0][idx_init]
            (line,) = ax[k][l].plot(
                np.array(ID_smp_init) % wd, np.array(ID_smp_init) // wd, "."
            )
            lines[-1].append(line)

    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    t_slider = Slider(
        ax=ax_slider, label="Time", valmin=t_min, valmax=t_max, valinit=t_smp_init
    )

    def update(t_smp):
        for k in range(n_y):
            for l in range(n_x):
                var_name = f"{pop}_{k}_{l}"

                idx = np.where(np.abs(t[var_name][0] - t_smp) <= tol)[0]
                ID_smp = ID[var_name][0][idx]
                lines[k][l].set_data(np.array(ID_smp) % wd, np.array(ID_smp) // wd)
        fig.canvas.draw_idle()

    t_slider.on_changed(update)

    return ax, ax_slider, t_slider, lines


def scatter3D(t, ID, wd):
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.gca()(projection ="3d")
    x = np.array(ID) % wd
    y = np.array(ID) // wd
    plt.gca().scatter3D(t, x, y, ".", s=0.5)


def gauss_kern(x, sigma):
    return np.exp(-(x**2.0) / (2.0 * sigma**2.0)) / (sigma * np.sqrt(2.0 * np.pi))


def make_sdf(t, sigma, dt, t_window=None, kernel_func=gauss_kern):
    """
    make a spike density function
    Inputs:
    t - list of spike times in a list or np.array
    sigma - standard deviation of Gaussian kernel in ms
    dt - time step of the resulting SDF in ms
    """

    assert dt > 0.0
    assert sigma > 0.0

    n_sp = t.shape[0]

    if t_window is None:
        assert n_sp > 0, "Empty event array and no time window specified"

        t0 = t[0] - 2.0 * sigma
        t1 = t[-1] + 2.0 * sigma

    else:
        t0, t1 = t_window
        assert t1 >= t0

    T = t1 - t0

    nt_arr = int(T / dt) + 1  # at least one time point.
    t_arr = np.arange(nt_arr) * dt + t0

    sdf = np.zeros((nt_arr))

    for t_sp in t:
        sdf += kernel_func(t_arr - t_sp, sigma)

    sdf *= 1000.0  # kHz to Hz (assuming that spike times etc. are in millisecs)

    if n_sp > 0:
        sdf /= n_sp

    return t_arr, sdf


def iou(box1, box2):
    x11, x12 = box1["x"], (box1["x"] + box1["w"])
    x21, x22 = box2["x"], (box2["x"] + box2["w"])
    y11, y12 = box1["y"], (box1["y"] + box1["h"])
    y21, y22 = box2["y"], (box2["y"] + box2["h"])

    intersect = np.maximum(
        0.0, np.minimum(x12, x22) - np.maximum(x11, x21)
    ) * np.maximum(0.0, np.minimum(y12, y22) - np.maximum(y11, y21))
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersect

    # if multiple box pairs are processed, the dtype is np.ndarray,
    # and we filter out the cases where one would get division by zero
    if type(intersect) is np.ndarray:
        _iou = np.zeros(intersect.shape)
        filt = union != 0
        _iou[filt] = intersect[filt] / union[filt]
        return _iou
    # if it's just one pair, do the same but without the indexing.
    else:
        return intersect / union if union != 0.0 else 0.0


def crop_factor(box, crop):
    """
    Instead of intersection over union, this calculates
    the intersection between the box and the crop (also a box)
    over the area of the box. Therefore,
    it returns 1 whenever the box is fully covered by the crop,
    and a value between 0 and 1 if the crop partially or fully (=0)
    cuts parts of the box. A special case is if the box area is
    zero, for which this function always returns zero.
    """

    x11, x12 = box["x"], (box["x"] + box["w"])
    x21, x22 = crop["x"], (crop["x"] + crop["w"])
    y11, y12 = box["y"], (box["y"] + box["h"])
    y21, y22 = crop["y"], (crop["y"] + crop["h"])

    intersect = np.maximum(
        0.0, np.minimum(x12, x22) - np.maximum(x11, x21)
    ) * np.maximum(0.0, np.minimum(y12, y22) - np.maximum(y11, y21))

    area = box["w"] * box["h"]

    if type(area) is np.ndarray:
        _crop_factor = np.zeros(area.shape)
        filt = area != 0
        _crop_factor[filt] = intersect[filt] / area[filt]
        return _crop_factor
    else:
        return intersect / area if area != 0.0 else intersect


def compare_boxes_patches(
    box_sequ, w, h, n_subdiv_x, n_subdiv_y, half_step=True, comp_func=crop_factor
):
    """
    Returns a grid of lists of values which hold the
    the results of the provided comparator function(crop_fact by defualt),
    calculated between the provided box sequence as the first input to the function
    and the grid as the second input. The grid is specified by its width,
    height, the number of subdivisions in x and y. The latter defines the width
    and height of the tiles as e.g. w_tile = width / n_subdiv_x and is
    NOT affected by whether half_steps are used to add tiles in-between.
    So the total size of the grid is n_subdiv_y x n_subdiv_x if half_step is False,
    and (2 * n_subdiv_y - 1) x (2 * n_subdiv_y - 1) if half_step is True.
    """
    n_tiles_x = (2 * n_subdiv_x - 1) if half_step else n_subdiv_x
    n_tiles_y = (2 * n_subdiv_y - 1) if half_step else n_subdiv_y

    w_tile = w / n_subdiv_x
    h_tile = h / n_subdiv_y

    stride_x = w_tile * 0.5 if half_step else w_tile
    stride_y = h_tile * 0.5 if half_step else h_tile

    comp_grid = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]

    for i, j in product(range(n_tiles_y), range(n_tiles_x)):
        x0, y0 = stride_x * j, stride_y * i

        box_tile = np.array(
            [(x0, y0, w_tile, h_tile)],
            dtype=[("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4")],
        )

        comp_sequ = comp_func(box_sequ, box_tile)

        comp_grid[i][j] = list(comp_sequ)

    return comp_grid


def crop_factor_patches(box_sequ, w, h, n_subdiv_x, n_subdiv_y, half_step=True):
    """
    Returns a grid of lists of crop factors (see function), which
    contains the crop factor between the provided box sequence and
    the grid acting as the crops. The grid is specified by its width,
    height, the number of subdivisions in x and y. The latter defines the width
    and height of the tiles as e.g. w_tile = width / n_subdiv_x and is
    NOT affected by whether half_steps are used to add tiles in-between.
    So the total size of the grid is n_subdiv_y x n_subdiv_x if half_step is False,
    and (2 * n_subdiv_y - 1) x (2 * n_subdiv_y - 1) if half_step is True.
    """
    n_tiles_x = (2 * n_subdiv_x - 1) if half_step else n_subdiv_x
    n_tiles_y = (2 * n_subdiv_y - 1) if half_step else n_subdiv_y

    w_tile = w / n_subdiv_x
    h_tile = h / n_subdiv_y

    stride_x = w_tile * 0.5 if half_step else w_tile
    stride_y = h_tile * 0.5 if half_step else h_tile

    crop_factor_grid = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]

    for i, j in product(range(n_tiles_y), range(n_tiles_x)):
        x0, y0 = stride_x * j, stride_y * i

        box_tile = np.array(
            [(x0, y0, w_tile, h_tile)],
            dtype=[("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4")],
        )

        crop_factor_sequ = crop_factor(box_sequ, box_tile)

        crop_factor_grid[i][j] = list(crop_factor_sequ)

    return crop_factor_grid


def group_boxes_by_ts(boxes, presort=False):
    """
    group boxes with same time stamps. Returns a list
    with entries for each time stamp.
    By default, presort is False because it is assumed
    that the input boxes are sorted by their time stamps
    in ascending order. If True, they will be sorted.
    Likewise, the returned grouped boxes are ordered by their
    time stamp in ascending order"""

    if presort:
        t_idx = np.argsort(boxes["ts"])
        boxes = boxes[t_idx]

    boxes_ts_unique, indices = np.unique(boxes["ts"], return_index=True)
    indices = indices.astype("int")

    boxes_ts_grouped = []

    if len(indices) > 0:
        for k in range(len(indices) - 1):
            boxes_ts_grouped.append((boxes[indices[k] : indices[k + 1]]))
        boxes_ts_grouped.append(boxes[indices[-1] :])

    return boxes_ts_grouped
