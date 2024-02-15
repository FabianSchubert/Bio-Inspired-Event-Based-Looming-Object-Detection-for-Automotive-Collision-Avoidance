import numpy as np
from .prophesee_automotive_dataset_toolbox.src.io.psee_loader import PSEELoader


def get_atis_event_array(filename, t0_ms=0, t1_ms=None):
    video = PSEELoader(filename)
    t1 = video.total_time() if t1_ms is None else t1_ms * 1e3
    t0 = t0_ms * 1e3
    video.seek_time(t0)
    events = video.load_delta_t(t1 - t0)
    events["t"] = events["t"] / 1e3
    return events
    """
    events_float = np.ndarray((events.shape[0], 4))
    events_float[:, 0] = events["x"].astype(float)
    events_float[:, 1] = events["y"].astype(float)
    # time in milliseconds
    events_float[:, 2] = events["t"].astype(float)/1e3
    events_float[:, 3] = events["p"].astype(float) * 2 - 1.

    return events_float
    """

def spike_bitmask(events_ts, x, y, nx, ny, nt, dt):
    num_neurons = nx * ny
    bits = np.zeros((nt, 32 * int(np.ceil(num_neurons / 32))), dtype=np.uint8)
    neur_ids = y.astype("int") * nx + x.astype("int")
    events_step = (events_ts/dt).astype("int")
    bits[events_step, neur_ids] = 1

    return np.packbits(bits, axis=1, bitorder="little").flatten()


def polarity_bitmask(events_ts, pol, x, y, nx, ny, nt, dt):
    num_neurons = nx * ny
    bits = np.zeros((nt, 32 * int(np.ceil(num_neurons / 32))), dtype=np.uint8)
    neur_ids = y.astype("int") * nx + x.astype("int")
    events_step = (events_ts/dt).astype("int")
    bits[events_step, neur_ids] = (pol == 1).astype(np.uint8)

    return np.packbits(bits, axis=1, bitorder="little").flatten()


def filter_events_canvas(events, x, y, w, h):
    filt = (events["x"] >= x) * (events["x"] < (x + w)) * \
        (events["y"] >= y) * (events["y"] < (y + h))
    return events[filt]


def tiled_events(events, w, h, n_subdiv_w, n_subdiv_h, half_step=True):
    tiled_events = []

    w_tile, h_tile = (w / n_subdiv_w, h / n_subdiv_h)
    n_w, n_h = (2 * n_subdiv_w - 1,
                2 * n_subdiv_h - 1) if half_step else (n_subdiv_w, n_subdiv_h)
    stride_w, stride_h = (w_tile * 0.5,
                          h_tile * 0.5) if half_step else (w_tile, h_tile)

    for k in range(n_h):
        for l in range(n_w):
            x, y = int(l * stride_w), int(k * stride_h)
            _evts = filter_events_canvas(events, x, y, w_tile, h_tile)
            _evts["x"] -= x
            _evts["y"] -= y
            tiled_events.append((k, l, _evts))

    return tiled_events


def convert_spike_id_events_to_spike_coord_events(
        spike_t, spike_id, spike_pol,
        width, height):
    """
    Spike ID = y * width + x 
    Ordering of each event: (t, x, y, p)
    """

    datatype = [('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]

    spk_lst = []
    assert spike_t.shape[0] == spike_id.shape[0] == spike_pol.shape[0], \
        "length of arrays does not match"
    assert spike_id.max() < width * height, \
        "spike ids exceed grid size"

    for k in range(spike_t.shape[0]):
        x = int(spike_id[k] % height)
        y = int(spike_id[k] // width)
        spk_lst.append((int(spike_t[k]), x, y, int(spike_pol[k])))

    return np.array(spk_lst, dtype=datatype)
