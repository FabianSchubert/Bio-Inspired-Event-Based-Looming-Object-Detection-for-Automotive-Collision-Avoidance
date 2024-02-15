import numpy as np
import os
from glob import glob

from config import BOXES_DTYPE


def load_file(evt_file: str, box_file: (str | None) = None) -> list[tuple]:
    if box_file is None:
        box_file = evt_file.split("td.npy")[0] + "bbox.npy"

    events = np.load(evt_file, allow_pickle=True)
    boxes = np.load(box_file, allow_pickle=True)

    return list(zip(events, boxes))


def load_folders(folders: (list[str] | str)) -> list[tuple]:
    if isinstance(folders, str):
        folders = [folders]

    data = []
    for fold in folders:
        evt_files = glob(os.path.join(fold, "*td.npy"))
        for evt_file in evt_files:
            data.extend(load_file(evt_file))

    return data


def save_data(folder: str, filename_base: str, data: list[tuple]) -> None:
    filename_events = filename_base + "_td.npy"
    filename_boxes = filename_base + "_bbox.npy"

    events = np.array([dat[0] for dat in data], dtype=object)
    boxes = np.array([dat[1] for dat in data], dtype=BOXES_DTYPE)

    np.save(os.path.join(folder, filename_events), events, allow_pickle=True)
    np.save(os.path.join(folder, filename_boxes), boxes, allow_pickle=True)
