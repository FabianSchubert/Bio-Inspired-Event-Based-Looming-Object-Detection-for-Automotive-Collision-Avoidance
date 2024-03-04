import numpy as np
import os
from scipy.stats import linregress
from .sort import Sort, KalmanBoxTracker

from src.utils import group_boxes_by_ts

DTYPE_BOXES = [
    ("ts", "<u8"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("w", "<f4"),
    ("h", "<f4"),
    ("class_id", "u1"),
    ("confidence", "<f4"),
    ("track_id", "<u4"),
]


def track_boxes(
    boxes_grouped, t_sequ_microsecs=60e6, max_age=3, min_hits=3, iou_threshold=0.3
):
    """
    assign proper tracking IDs to boxes using SORT.
    returns a list of structured arrays, each holding
    the boxes with the new IDs in the corresponding frame.
    The "frame rate" in the input data is either 1Hz, 2Hz, or 4Hz,
    which is inferred from the time stamps of the boxes provided.
    The time stamps in the output boxes are still in microseconds,
    but snapped to the temporal grid given by the frame rate.
    E.g., if the original box appeared at 15099999, this would be
    simply 15000000 in the output.
    """
    times = np.array([bx_gr["ts"][0] for bx_gr in boxes_grouped])

    # determine if the sequence uses 1s, 0.5s, or 0.25s spacings between box labels
    if np.all(np.unique(times // 1e6, return_counts=True)[1] == 1):
        delta_t = 1e6
    elif np.all(np.unique(times // 5e6, return_counts=True)[1] == 1):
        delta_t = 0.5e6
    else:
        delta_t = 0.25e6

    frames_with_boxes = (times // delta_t).astype("int")
    num_frames = int(t_sequ_microsecs / delta_t)

    classes = set([])

    for bx_group in boxes_grouped:
        cl = set(np.unique(bx_group["class_id"]))
        classes.update(cl)

    classes = list(classes)

    boxes_grouped_track_ids = [
        np.array([], dtype=DTYPE_BOXES) for _ in range(num_frames)
    ]

    # run the tracking for each class separately to prevent mixing of boxes.
    for i, cl in enumerate(classes):
        mot_tracker = Sort(
            max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold
        )

        frm_id = 0

        for f in range(num_frames):
            if f == frames_with_boxes[frm_id]:
                bx_group = boxes_grouped[frm_id]
                bx_group_class = bx_group[np.where(bx_group["class_id"] == cl)[0]]

                sort_boxes = np.empty((len(bx_group_class), 5))
                sort_boxes[:, 0] = bx_group_class["x"]
                sort_boxes[:, 1] = bx_group_class["y"]
                sort_boxes[:, 2] = bx_group_class["x"] + bx_group_class["w"]
                sort_boxes[:, 3] = bx_group_class["y"] + bx_group_class["h"]
                sort_boxes[:, 4] = np.ones(len(bx_group_class))

                if frm_id < len(frames_with_boxes) - 1:
                    frm_id += 1
            else:
                sort_boxes = np.empty((0, 5))

            mot_tracker.update(sort_boxes)

            trackers = [
                list(tr.get_state()[0]) + [tr.id] for tr in mot_tracker.trackers
            ]

            new_boxes = np.array(
                [
                    (
                        f * delta_t,
                        tr[0],
                        tr[1],
                        tr[2] - tr[0],
                        tr[3] - tr[1],
                        cl,
                        1.0,
                        tr[4],
                    )
                    for tr in trackers
                ],
                dtype=DTYPE_BOXES,
            )

            boxes_grouped_track_ids[f] = np.append(
                boxes_grouped_track_ids[f], new_boxes
            )

    return boxes_grouped_track_ids


def get_box_sizes_array(boxes_grouped):
    max_box_number = 0
    for bx_gr in boxes_grouped:
        for tr_id in bx_gr["track_id"]:
            max_box_number = max(max_box_number, tr_id)

    sizes = np.zeros((len(boxes_grouped), max_box_number + 1)) * np.nan

    for k, bx_gr in enumerate(boxes_grouped):
        for bx in bx_gr:
            sizes[k, bx["track_id"]] = bx["w"] * bx["h"]

    return sizes


def find_arg_interval(arr, left, right):
    return np.where((arr >= left) & (arr <= right))[0]


def running_lin_regress(x, times, time_window):
    assert x.ndim == 2
    assert times.ndim == 1
    assert x.shape[0] == times.shape[0]

    n_times = x.shape[0]
    n_features = x.shape[1]

    fit_slope = np.zeros((n_times, n_features))
    fit_intercept = np.zeros((n_times, n_features))
    fit_r_squ = np.zeros((n_times, n_features))

    for t in range(n_times):
        for k in range(n_features):
            ind_time_window = find_arg_interval(times, times[t] - time_window, times[t])
            nan_filter = np.isfinite(x[ind_time_window, k])

            x_filter = x[ind_time_window, k][nan_filter]
            t_filter = times[ind_time_window][nan_filter]

            if x_filter.shape[0] > 1:
                fit = linregress(t_filter, x_filter)
                fit_slope[t, k] = fit.slope
                fit_intercept[t, k] = fit.intercept
                fit_r_squ[t, k] = fit.rvalue**2.0

    return fit_slope, fit_intercept, fit_r_squ


def detect_looming_sequences(
    boxes,
    t_sequ_microsecs=60e6,
    tracking_params={"max_age": 3, "min_hits": 3, "iou_threshold": 0.3},
    t_filter_microsecs=3e6,  # minimum length of looming sequences
    box_size_th_end=2500,
):  # minimum size at the end of a looming sequence
    """
    detect looming sequences by processing box data:
    - group boxes by timestamp
    - get proper tracking ids using SORT
    - use tracked boxes to filter out sequences of boxes that
      increase in size.
    """
    boxes_grouped = group_boxes_by_ts(boxes)

    boxes_grouped_track_ids = track_boxes(
        boxes_grouped, t_sequ_microsecs=t_sequ_microsecs, **tracking_params
    )

    box_sizes_array = get_box_sizes_array(boxes_grouped_track_ids)

    n_times, n_boxes = box_sizes_array.shape

    times = np.linspace(0.0, t_sequ_microsecs, n_times, endpoint=False)

    sequences = []
    box_sequences = []

    for k in range(n_boxes):
        state = 0
        for t in range(n_times):
            if state == 0 and not (np.isnan(box_sizes_array[t, k])):
                t0 = t
                state = 1
            elif state == 1 and (
                np.isnan(box_sizes_array[t, k])
                or (box_sizes_array[t, k] <= box_sizes_array[t - 1, k])
            ):
                t1 = t - 1
                state = 0
                if (box_sizes_array[t1, k] >= box_size_th_end) and (
                    (times[t1] - times[t0]) >= t_filter_microsecs
                ):
                    sequences.append((times[t0], times[t1], k))
                    bx_sequence = np.array(
                        [
                            boxes_grouped_track_ids[t_idx][
                                np.where(
                                    boxes_grouped_track_ids[t_idx]["track_id"] == k
                                )[0][0]
                            ]
                            for t_idx in range(t0, t1 + 1)
                        ],
                        dtype=DTYPE_BOXES,
                    )
                    box_sequences.append(bx_sequence)

    return sequences, box_sequences, boxes_grouped_track_ids


if __name__ == "__main__":
    BASE = "/mnt/data0/prophesee_data/ATIS_Automotive_Detection_Dataset/"

    FOLDERS = (
        [f"train_{k}" for k in "abcdef"]
        + [f"val_{k}" for k in "ab"]
        + [f"test_{k}" for k in "ab"]
    )

    FOLDERS = [os.path.join(BASE, f) for f in FOLDERS]

    # get the full paths of all bbox files in all folders
    FILES = [
        os.path.join(fold, fn)
        for fold in FOLDERS
        for fn in os.listdir(fold)
        if fn.endswith("bbox.npy")
    ]

    NUM_FILES = len(FILES)

    # just to keep track of the number of sequences per class
    looming_sq = {"car": 0, "ped": 0}

    for k, file in enumerate(FILES):
        print(f"processing file {k+1}/{NUM_FILES}", end="\r")
        boxes = np.load(file)
        sequences, box_sequences, grouped_tracked_boxes = detect_looming_sequences(
            boxes
        )

        for bx_sq in box_sequences:
            if bx_sq["class_id"][0] == 0:
                looming_sq["car"] += 1
            else:
                looming_sq["ped"] += 1

        # only save if there are some actual looming sequences in the bbox file.
        if len(box_sequences) > 0:
            file_sequences = file[:-4] + "_looming_sequences.npy"
            # appending the None and then discarding it seems to be the only way
            # to prevent numpy from reducing the array to the type object completely
            # if the length of box_sequences is 1.
            np.save(
                file_sequences,
                np.array(box_sequences + [None], dtype=object)[:-1],
                allow_pickle=True,
            )
        del sequences
        del box_sequences
        del grouped_tracked_boxes

        # set static variable count back to zero, otherwise it will keep
        # ramping up the tracking ids across the files
        KalmanBoxTracker.count = 0

    print(f"number of sequences by class: {looming_sq}")
