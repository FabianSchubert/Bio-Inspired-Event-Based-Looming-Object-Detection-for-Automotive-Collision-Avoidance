import numpy as np
import h5py
import hdf5plugin  # need to import this to prevent error when loading h5py files
import os

from numba import jit, uint32, uint16

# ("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")]

from .settings import base_fold_raw_data, base_fold_input_data

from src.config import EVENTS_DTYPE

# downsample as command line argument
from argparse import ArgumentParser

SEQUENCE_LENGTH = 4.0  # in seconds
SEQUENCE_LENGTH_MS = SEQUENCE_LENGTH * 1000.0  # in milliseconds
DT_MS = 10.0  # 100 fps

COLL_TYPE = "none_with_traffic"
DIAMETER_OBJECT = np.nan
VELOCITY = np.nan
T_END = SEQUENCE_LENGTH_MS

METADATA = {
    "coll_type": COLL_TYPE,
    "t_end": T_END,
    "dt": DT_MS,
    "vel": VELOCITY,
    "diameter_object": DIAMETER_OBJECT,
}


# use numba jit, otherwise this is way too slow
@jit(nopython=True)
def downsample_events(
    t,
    x,
    y,
    p,
    dt_ms=DT_MS,
    width=640,
    height=480,
    threshold=1,  # deviation from zero of sum of polarities required in dt_ms to trigger a downsampled event
):
    # (assumes t is already in milliseconds!)

    assert threshold > 0, "Threshold must be greater than 0"

    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)
    t_new = np.zeros_like(t)
    p_new = np.zeros_like(p)

    n_events = 0

    # 32 bit int might be a bit overkill for keeping track of the sum of polarities within the time window,
    # but 16 bit int might be too small
    sum_tmp = np.zeros((height * width), dtype=np.int32)
    # occupied = np.zeros((height * width), dtype=np.int32)

    t_ds = ((t // dt_ms) * dt_ms).astype(np.uint32)

    # not necessary if t is already sorted
    # sortidx = np.argsort(t_ds)
    # t_ds = t_ds[sortidx]
    # x = x[sortidx]
    # y = y[sortidx]
    # p = p[sortidx]

    _t = t_ds[0]

    for i in range(len(t_ds)):

        if t_ds[i] != _t:
            idx = np.where(np.abs(sum_tmp) >= threshold)[0]
            # idx = np.where(occupied)[0]
            x_new[n_events : n_events + len(idx)] = (idx % width).astype(np.uint16)
            y_new[n_events : n_events + len(idx)] = (idx // width).astype(np.uint16)
            t_new[n_events : n_events + len(idx)] = _t
            p_new[n_events : n_events + len(idx)] = (sum_tmp[idx] > 0).astype(np.uint16)
            # p_new[n_events : n_events + len(idx)] = (occupied[idx] > 0).astype(np.uint16)
            n_events += len(idx)
            sum_tmp = np.zeros((height * width), dtype=np.int32)
            # occupied = np.zeros((height * width), dtype=np.int32)
            _t = t_ds[i]

        # if occupied[int(y[i]) * width + int(x[i])] == 0:
        # occupied[int(y[i]) * width + int(x[i])] = int(p[i]) * 2 - 1

        sum_tmp[int(y[i]) * width + int(x[i])] += int(p[i]) * 2 - 1

    t_new = t_new[:n_events].copy()
    x_new = x_new[:n_events].copy()
    y_new = y_new[:n_events].copy()
    p_new = p_new[:n_events].copy()

    return t_new, x_new, y_new, p_new


def gen_sequences(downsample=False):

    scenes = os.listdir(base_fold_raw_data)

    metadata_scenes = {}

    file_path_sq_id = []

    scene_sq_id = []

    for scene in scenes:
        file_path = os.path.join(base_fold_raw_data, scene, "events/left/events.h5")
        with h5py.File(file_path, "r") as f:
            metadata_scenes[scene] = {}
            ms_t = f["ms_to_idx"]
            metadata_scenes[scene]["sequence_length"] = len(ms_t) / 1000.0
            metadata_scenes[scene]["num_sequences"] = int(
                len(ms_t) / SEQUENCE_LENGTH_MS
            )

            file_path_sq_id.extend(
                [file_path] * metadata_scenes[scene]["num_sequences"]
            )
            scene_sq_id.extend(range(metadata_scenes[scene]["num_sequences"]))

    N_SQ = len(file_path_sq_id)

    print(f"Raw scenes give {N_SQ} sequences of {SEQUENCE_LENGTH} seconds each")

    sq_by_id = dict(
        zip(
            range(N_SQ),
            [
                {"scene_sq_id": id, "file_path": file_path}
                for id, file_path in zip(scene_sq_id, file_path_sq_id)
            ],
        )
    )

    prev_path = None

    for i, sq in sq_by_id.items():
        print(f"Processing sequence {i + 1}/{N_SQ}")
        scene_sq_id = sq["scene_sq_id"]
        file_path = sq["file_path"]
        if file_path != prev_path:
            file = h5py.File(file_path, "r")
            prev_path = file_path

            ms_to_idx = file["ms_to_idx"]
            x = file["events/x"]
            y = file["events/y"]
            t = file["events/t"]
            p = file["events/p"]

        start_idx = ms_to_idx[int(scene_sq_id * SEQUENCE_LENGTH_MS)]
        end_idx = ms_to_idx[int((scene_sq_id + 1) * SEQUENCE_LENGTH_MS)]

        _x = x[start_idx:end_idx]
        _y = y[start_idx:end_idx]
        _t = t[start_idx:end_idx]
        _t = ((_t - _t[0]) / 1000.0).astype(
            np.uint32
        )  # microseconds to milliseconds and shift to start at 0
        _p = p[start_idx:end_idx]

        # optional: remove "duplicate" events in a millisecond bin (by downsampling to 1ms)

        n_before = len(_t)

        if downsample:
            _t, _x, _y, _p = downsample_events(_t, _x, _y, _p, threshold=2)
            n_after = len(_t)
            # downsample to DT_MS
            # _t = ((_t // DT_MS) * DT_MS).astype(np.uint32)
            # calculate a unique id for each event
            # _unique_id = _t * 640 * 480 * 2 + _y * 640 * 2 + _x * 2 + _p
            # find unique events. This returns the index of the *first* appearance of each unique event,
            # which, in this case, would be the first event in a given millisecond at a given location (ignoring polarity).
            # _, idx = np.unique(_unique_id, return_index=True)
            # _x_unique = _x[idx]
            # _y_unique = _y[idx]
            # _t_unique = _t[idx]
            # _p_unique = _p[idx]

            # n_after = len(_t_unique)
            # import matplotlib.pyplot as plt
            # plt.ion()
            # plt.figure()
            # plt.plot(_x_ds[:100000], _y_ds[:100000], ".", markersize=1)
            # plt.gca().invert_yaxis()
            # plt.gca().set_aspect("equal")

            # plt.figure()
            # plt.plot(_x_unique[:100000], _y_unique[:100000], ".", markersize=1)
            # plt.gca().invert_yaxis()
            # plt.gca().set_aspect("equal")

            print(f"Downsampled from {n_before} to {n_after} events")

        # construct a structurred array events
        # can we avoid zip?
        events = np.empty(len(_t), dtype=EVENTS_DTYPE)
        events["t"] = _t
        events["x"] = _x
        events["y"] = _y
        events["p"] = _p

        # save the data

        output_fold = os.path.join(base_fold_input_data, f"example_{i}/")

        if not os.path.exists(output_fold):
            os.makedirs(output_fold)

        np.save(os.path.join(output_fold, "events.npy"), events)
        np.savez(os.path.join(output_fold, "sim_data.npz"), **METADATA)
        print(f"Saved sequence {i + 1}/{N_SQ} to {output_fold}", end="\n\n")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--downsample", action="store_true")
    args = parser.parse_args()

    print(f"Downsample: {args.downsample}")

    gen_sequences(downsample=args.downsample)
