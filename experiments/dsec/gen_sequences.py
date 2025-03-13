import numpy as np
import h5py
import hdf5plugin  # need to import this to prevent error when loading h5py files
import os

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
            # downsample to DT_MS
            _t = ((_t // DT_MS) * DT_MS).astype(np.uint32)
            # calculate a unique id for each event
            _unique_id = _t * 640 * 480 + _y * 640 + _x
            # find unique events. This returns the index of the *first* appearance of each unique event,
            # which, in this case, would be the first event in a given millisecond at a given location (ignoring polarity).
            _, idx = np.unique(_unique_id, return_index=True)
            _x = _x[idx]
            _y = _y[idx]
            _t = _t[idx]
            _p = _p[idx]

            n_after = len(_t)
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
