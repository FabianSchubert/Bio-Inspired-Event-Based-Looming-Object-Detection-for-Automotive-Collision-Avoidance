import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch


def load_metadata(fold, t_steps):

    fold = Path(fold)
    # get all subfolders
    samples = [f for f in fold.iterdir() if f.is_dir()]
    # sort samples by number
    samples.sort(key=lambda x: int(x.stem.split("_")[1]))

    labels = []
    metadata = []

    samples_filtered = []

    for sample in samples:
        file_path = sample / "sim_data.npz"

        if not file_path.exists():
            print(f"File {file_path} does not exist, skipping")
            continue

        metadat = np.load(file_path, allow_pickle=True)

        sample_steps = int(metadat["t_end"] // metadat["dt"])

        if sample_steps < t_steps:
            continue

        samples_filtered.append(sample)

        labels.append(int(not metadat["coll_type"][()].startswith("none")))

        metadata.append(
            {
                "coll_type": metadat["coll_type"][()],
                "t_end_ms": metadat["t_end"][()],
                "dt": metadat["dt"][()],
                "vel": metadat["vel"][()],
                "diam": metadat["diameter_object"][()],
            }
        )

    return samples_filtered, labels, metadata


def load_raw_data(samples: list[Path]):

    labels = []
    metadata = []
    data = []

    mem_gb = 0

    for sample in samples:

        # in case it is a string
        sample = Path(sample)

        assert sample.exists(), f"Folder {sample} does not exist"

        events = np.load(sample / "events.npy")
        metadat = np.load(sample / "sim_data.npz", allow_pickle=True)

        labels.append(int(not metadat["coll_type"][()].startswith("none")))

        data.append(events)

        mem_gb += events.nbytes / 1e9
        print(f"Total estimate of event data size: {mem_gb:.2f} GB", end="\r")

        metadata.append(
            {
                "coll_type": metadat["coll_type"][()],
                "t_end_ms": metadat["t_end"][()],
                "dt": metadat["dt"][()],
                "vel": metadat["vel"][()],
                "diam": metadat["diameter_object"][()],
            }
        )

    return data, labels, metadata


def convert_to_tens(
    data,
    labels,
    metadata,
    t_steps,
    width,
    height,
    subdiv_width,
    subdiv_height,
    n_discard_top,
    n_discard_bottom,
    n_discard_left,
    n_discard_right,
):
    X = []

    idx_drop = []
    for i, events in tqdm(enumerate(data), disable=True):
        print(f"Converting sample {i+1}/{len(data)}")
        dt = metadata[i]["dt"]
        t_end = metadata[i]["t_end_ms"]
        nt = int(t_end / dt)
        if nt < t_steps:
            idx_drop.append(i)
            print(f"Sample {i} has too few frames, dropping it.")
            continue
        _dat = torch.zeros(t_steps, subdiv_height, subdiv_width).type(torch.int8)

        # if it is any other than none_with_crossing, start at idx so that we end with the end of the sequence
        # if it is none_with_crossing, we center around the middle of the sequence, because the object is at the center
        # of the frame roughly at the middle of the sequence

        if metadata[i]["coll_type"] != "none_with_crossing":
            t_offset = nt - t_steps
        else:
            t_offset = (nt - t_steps) // 2
        for t in tqdm(range(t_steps), leave=False, disable=True):
            _evts = events[
                (events["t"] >= (t + t_offset) * dt)
                & (events["t"] < (t + 1 + t_offset) * dt)
                & (events["y"] >= n_discard_top * subdiv_height // 2)
                & (events["y"] < height - n_discard_bottom * subdiv_height // 2)
                & (events["x"] >= n_discard_left * subdiv_width // 2)
                & (events["x"] < width - n_discard_right * subdiv_width // 2)
            ]
            _dat[
                t,
                _evts["y"].astype(np.int16) - n_discard_top * subdiv_height // 2,
                _evts["x"].astype(np.int16) - n_discard_left * subdiv_width // 2,
            ] = torch.tensor(2 * _evts["p"].astype(int) - 1).type(torch.int8)
        X.append(_dat)

    X = torch.stack(X)

    labels = torch.tensor(
        [labels[i] for i in range(len(labels)) if i not in idx_drop]
    ).type(torch.int8)
    metadata = [metadata[i] for i in range(len(metadata)) if i not in idx_drop]

    return X, labels, metadata


def gen_x_sequ(X, Y, t_steps_to_coll_max=None):
    _t_steps = X.shape[1]
    X_1 = X[:, 1:, ...]
    X_0 = X[:, :-1, ...]
    X_seq = torch.stack([X_0, X_1], dim=2)
    X_seq = X_seq.view(-1, X_seq.shape[-3], X_seq.shape[-2], X_seq.shape[-1])

    Y_expand = Y.unsqueeze(1).repeat(1, _t_steps - 1)  # .flatten()
    if t_steps_to_coll_max is not None:
        Y_expand[:, :-t_steps_to_coll_max] = 0
    Y_expand = Y_expand.flatten()

    return X_seq, Y_expand
