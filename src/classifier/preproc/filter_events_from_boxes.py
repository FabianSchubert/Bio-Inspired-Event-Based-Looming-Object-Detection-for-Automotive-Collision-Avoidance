import numpy as np

from glob import glob
import os

from prophesee_dataset_toolbox.io.psee_loader import PSEELoader
from src.utils import compare_boxes_patches, iou, crop_factor, group_boxes_by_ts

from src.config import BOXES_DTYPE

from typing import Callable

import threading


def th_func_boxes_default(x: np.ndarray) -> np.ndarray:
    return x >= 0.75


def extract_negative_events(
    evt_file: str,
    box_file: str | None = None,
    delta_t: float | int = 100000,
    compare_func_boxes: Callable = crop_factor,
    threshold_func_boxes: Callable[np.ndarray, np.ndarray] = th_func_boxes_default,
    n_subdiv_x: int = 4,
    n_subdiv_y: int = 4,
    half_stride: bool = True,
    min_event_count: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extracts events from all tiles
    that do NOT fulfill the condition specified by the compare function
    and the threshold condition. The compare function should return
    a scalar for pairs of boxes. The default is the crop factor between
    the boxes from the file and each tile acting as the crop.

    We do not have information about potential objects in tiles at times
    where no box annotations are present, so this is why we restrain it to cases where we
    know for sure which tiles do and do not contain cars or pedestrians.
    """
    video = PSEELoader(evt_file)

    # infer file name for boxes from event file
    if not box_file:
        box_file = evt_file.split("td.dat")[0] + "bbox.npy"

    boxes = np.load(box_file)

    height, width = video.get_size()

    tile_width, tile_height = int(width / n_subdiv_x), int(height / n_subdiv_y)

    stride_x = int(tile_width / 2.0) if half_stride else tile_width
    stride_y = int(tile_height / 2.0) if half_stride else tile_height

    delta_t = int(delta_t)

    # only use boxes that start after delta_t, otherwise
    # we might not get the full time span of events.
    boxes = boxes[boxes["ts"] >= delta_t]

    # the returned boxes are simply the dimensions and positions
    # of the selected tiles (see below)
    boxes_extract = np.array([], dtype=BOXES_DTYPE)

    # find groups of annotations at points in time.
    boxes_grouped = group_boxes_by_ts(boxes)

    events_list = []

    # boxes require a tracking id, so just counting from 0
    tr_id = 0

    for bx_sq in boxes_grouped:
        t = int(bx_sq["ts"][0])

        # calculate comparison function over grid and boxes in group
        compare_grid = np.array(
            compare_boxes_patches(
                bx_sq,
                width,
                height,
                n_subdiv_x,
                n_subdiv_y,
                half_step=half_stride,
                comp_func=compare_func_boxes,
            )
        )

        # which tiles are not fulfilling the threshold condition with any of the boxes?
        empty_box_idx = np.where(threshold_func_boxes(compare_grid).sum(axis=2) == 0)

        # if we find any..
        if empty_box_idx[0].shape[0] > 0:
            # load events for the time window.
            video.seek_time(t - delta_t)
            events = video.load_delta_t(delta_t)
            for idx_y, idx_x in zip(*empty_box_idx):
                x0 = idx_x * stride_x
                y0 = idx_y * stride_y

                x1 = x0 + tile_width
                y1 = y0 + tile_height

                # get the tile events...
                events_filter = events[
                    (events["x"] >= x0)
                    & (events["x"] < x1)
                    & (events["y"] >= y0)
                    & (events["y"] < y1)
                ].copy()  # copy to prevent overwriting the original (not 100% sure if this is necessary).

                if len(events_filter) >= min_event_count:
                    # and shift them to zero in time and space.
                    events_filter["x"] -= x0
                    events_filter["y"] -= y0
                    events_filter["t"] -= t - delta_t

                    # construct the box from tile data, using class_id = 2 (0: cars, 1: pedestrians)
                    # the origin is set to zero as we do the same for the events.
                    box = np.array(
                        [(t, 0, 0, tile_width, tile_height, 2, 1.0, tr_id)],
                        dtype=boxes.dtype,
                    )

                    tr_id += 1

                    events_list.append(events_filter)
                    boxes_extract = np.append(boxes_extract, box)

    # see the return statement of extract_events_from_boxes for an
    # explanation of this weird thing about appending a None.
    return boxes_extract, np.array(events_list + [None], dtype=object)[:-1]


def extract_events_from_boxes(
    evt_file: str,
    box_file: str | None = None,
    delta_t: float | int = 100000,  # microsecs!!!
    compare_func_boxes: Callable = crop_factor,
    threshold_func_boxes: Callable[np.ndarray, np.ndarray] = th_func_boxes_default,
    n_subdiv_x: int = 4,
    n_subdiv_y: int = 4,
    half_stride: bool = True,
    min_event_count: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function extracts events from all tiles
    that do fulfill the condition specified by the compare function
    and the threshold condition. The compare function should return
    a scalar for pairs of boxes. The default is the crop factor between
    the boxes from the file and each tile acting as the crop.

    We do not have information about potential objects in tiles at times
    where no box annotations are present, so this is why we restrain it to cases where we
    know for sure which tiles do and do not contain cars or pedestrians.
    """
    video = PSEELoader(evt_file)

    # infer the box_file from the event file if not provided
    if not box_file:
        box_file = evt_file.split("td.dat")[0] + "bbox.npy"

    boxes = np.load(box_file)

    height, width = video.get_size()

    tile_width, tile_height = int(width / n_subdiv_x), int(height / n_subdiv_y)

    stride_x = int(tile_width / 2.0) if half_stride else tile_width
    stride_y = int(tile_height / 2.0) if half_stride else tile_height

    delta_t = int(delta_t)

    # do not use boxes that appear before delta_t
    # (so that we are sure that we extract events from t - delta_t to t)
    boxes = boxes[boxes["ts"] >= delta_t]

    # calculate the IOU between all boxes and all tiles.
    compare_grid = np.array(
        compare_boxes_patches(
            boxes,
            width,
            height,
            n_subdiv_x,
            n_subdiv_y,
            half_step=half_stride,
            comp_func=compare_func_boxes,
        )
    )

    # find the combinations where the compare function fulfills the threshold function
    overlap_idx = np.where(threshold_func_boxes(compare_grid))

    # get the corresponding boxes
    boxes_extract = np.array(boxes[overlap_idx[2]])

    events_list = []
    boxes_list = []
    # extract the events fitting into the selected
    # tiles and time windows.

    unique_ts = np.unique(boxes_extract["ts"])

    for ts in unique_ts:
        video.seek_time(ts - delta_t)
        events = video.load_delta_t(delta_t)

        box_ids = np.where(boxes_extract["ts"] == ts)[0]
        for i in box_ids:
            x0 = int(overlap_idx[1][i] * stride_x)
            y0 = int(overlap_idx[0][i] * stride_y)

            x1 = x0 + tile_width
            y1 = y0 + tile_height

            events_filter = events[
                (events["x"] >= x0)
                & (events["x"] < x1)
                & (events["y"] >= y0)
                & (events["y"] < y1)
            ].copy()

            if len(events_filter) >= min_event_count:
                events_filter["x"] -= x0
                events_filter["y"] -= y0
                events_filter["t"] -= int(ts) - delta_t

                box_filter = boxes_extract[i].copy()
                # shift the box by the tile origin.
                # Note: the actual timing of the box is retained,
                # but this is usually not used for the classification.
                box_filter["x"] -= x0
                box_filter["y"] -= y0

                events_list.append(events_filter)
                boxes_list.append(box_filter)

    # super annyoing: if there is only one element in the events list,
    # numpy discards the dtype of the element and just reduces it to tuples...
    # so we append a none object, pack the whole thing into an object array, and then
    # discard this last element again...
    return (
        np.array(boxes_list, dtype=BOXES_DTYPE),
        np.array(events_list + [None], dtype=object)[:-1],
    )


def process_files(evt_files: list[str], output_path: str, task_number: int, **args):
    for i, evt_fl in enumerate(evt_files):
        print(
            f"\x1b[2K\rprocessing file {i+1}/{len(evt_files)} in thread {task_number}",
            end="\r",
        )
        _bx, _evt = extract_events_from_boxes(evt_fl, **args)
        _bx_neg, _evt_neg = extract_negative_events(evt_fl, **args)

        _bx_comb = np.append(_bx, _bx_neg)
        _evt_comb = np.append(_evt, _evt_neg)

        file_base = os.path.basename(evt_fl.split("_td.dat")[0])

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(
            os.path.join(output_path, file_base + "_bbox.npy"),
            _bx_comb,
            allow_pickle=True,
        )
        np.save(
            os.path.join(output_path, file_base + "_td.npy"),
            _evt_comb,
            allow_pickle=True,
        )

        del _bx, _evt, _bx_neg, _evt_neg, _bx_comb, _evt_comb


def process_folder(
    input_path: str, output_path: str, num_threads: int = 1, **args
) -> None:
    evt_files = glob(os.path.join(input_path, "*_td.dat"))

    evt_files_split = np.array_split(evt_files, num_threads)

    threads = []
    for k in range(num_threads):
        _thread = threading.Thread(
            target=process_files,
            args=(list(evt_files_split[k]), output_path, k),
            kwargs=args,
        )
        threads.append(_thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    """
    for i, evt_fl in enumerate(evt_files):
        print(f"\x1b[2K\rprocessing file {i+1}/{len(evt_files)}", end="\r")
        _bx, _evt = extract_events_from_boxes(evt_fl, **args)
        _bx_neg, _evt_neg = extract_negative_events(evt_fl, **args)

        _bx_comb = np.append(_bx, _bx_neg)
        _evt_comb = np.append(_evt, _evt_neg)

        file_base = os.path.basename(evt_fl.split("_td.dat")[0])

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(
            os.path.join(output_path, file_base + "_bbox.npy"),
            _bx_comb,
            allow_pickle=True,
        )
        np.save(
            os.path.join(output_path, file_base + "_td.npy"),
            _evt_comb,
            allow_pickle=True,
        )

        del _bx, _evt, _bx_neg, _evt_neg, _bx_comb, _evt_comb
    """
