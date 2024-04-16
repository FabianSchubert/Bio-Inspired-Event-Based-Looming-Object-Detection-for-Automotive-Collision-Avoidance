import numpy as np
import os
from glob import glob

from src.config import BOXES_DTYPE


def load_file(
    evt_file: str,
    box_file: (str | None) = None,
    load_events: bool = True,
    load_boxes: bool = True,
    return_filenames: bool = False,
) -> list[tuple]:
    if box_file is None:
        box_file = evt_file.split("td.npy")[0] + "bbox.npy"

    if load_events:
        events = np.load(evt_file, allow_pickle=True)
    if load_boxes:
        boxes = np.load(box_file, allow_pickle=True)

    if load_events and load_boxes:
        data = list(zip(events, boxes))
    elif load_events:
        data = events
    elif load_boxes:
        data = boxes
    else:
        data = []

    if return_filenames:
        return data, evt_file, box_file
    else:
        return data


def load_folders(
    folders: (list[str] | str),
    load_events: bool = True,
    load_boxes: bool = True,
    return_filenames: bool = False,
) -> list[tuple]:
    if isinstance(folders, str):
        folders = [folders]

    data = []
    for fold in folders:
        evt_files = glob(os.path.join(fold, "*td.npy"))
        for evt_file in evt_files:
            data.append(
                load_file(
                    evt_file,
                    load_events=load_events,
                    load_boxes=load_boxes,
                    return_filenames=return_filenames,
                )
            )

    return data


def save_data(folder: str, filename_base: str, data: list[tuple]) -> None:
    filename_events = filename_base + "_td.npy"
    filename_boxes = filename_base + "_bbox.npy"

    events = np.array([dat[0] for dat in data], dtype=object)
    boxes = np.array([dat[1] for dat in data], dtype=BOXES_DTYPE)

    np.save(os.path.join(folder, filename_events), events, allow_pickle=True)
    np.save(os.path.join(folder, filename_boxes), boxes, allow_pickle=True)
