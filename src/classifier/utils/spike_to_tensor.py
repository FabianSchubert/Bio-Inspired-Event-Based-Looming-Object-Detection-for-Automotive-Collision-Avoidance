from src.viz import gen_evt_hist
import numpy as np

import torch
from torch.utils.data import Dataset


def gen_bin_event_tensor(
    events: np.ndarray,
    width: int,
    height: int,
    t0_microsecs: float = 0.0,
    t1_microsecs: float = 100000.0,
    dt_microsecs: float = 1000.0,
) -> torch.Tensor:
    nt = int((t1_microsecs - t0_microsecs) / dt_microsecs)
    _events = events.copy()
    _events["t"] = (_events["t"] - t0_microsecs) // dt_microsecs
    _events = _events[_events["t"] < nt]
    _events_pos = _events[_events["p"] == 1]
    _events_neg = _events[_events["p"] == 0]

    _events_pos = np.array(
        [
            _events_pos["t"].astype("int"),
            _events_pos["y"].astype("int"),
            _events_pos["x"].astype("int"),
        ]
    ).T

    _events_neg = np.array(
        [
            _events_neg["t"].astype("int"),
            _events_neg["y"].astype("int"),
            _events_neg["x"].astype("int"),
        ]
    ).T

    hist_pos = np.histogramdd(_events_pos, bins=(nt, height, width))[0]
    hist_neg = np.histogramdd(_events_neg, bins=(nt, height, width))[0]

    hist = hist_pos - hist_neg
    hist = np.clip(hist, -1, 1)

    return torch.tensor(hist, dtype=torch.float32)


# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(
            box1[..., 1], box2[..., 1]
        )

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


def gen_evt_tensor(
    event_sequence: list,
    width: int,
    height: int,
    t0_microsecs: float = 0.0,
    t1_microsecs: float = 100000.0,
    dt_microsecs: float = 1000.0,
) -> torch.Tensor:
    nt = int((t1_microsecs - t0_microsecs) / dt_microsecs)
    data = np.empty((len(event_sequence), nt, height, width))
    for i, evts in enumerate(event_sequence):
        for j in range(nt):
            t0 = t0_microsecs + j * dt_microsecs
            t1 = t0 + dt_microsecs
            data[i, j] = gen_evt_hist(evts, t0, t1, width, height)

    return torch.tensor(data, dtype=torch.float32)


def gen_target_tensor_yolo(
    boxes: list,
    anchors: list,
    image_size: tuple[int, int],
    grid_sizes: list[int],
    ignore_iou_thresh: float = 0.5,
) -> torch.Tensor:
    anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # 3 scales of anchors
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3

    targets = [torch.zeros((num_anchors_per_scale, s, s, 6)) for s in grid_sizes]

    for box in boxes:
        # just calculate the iou between the box and anchors,
        # assuming they are centered at the same point
        iou_anchors = iou(box[2:4], anchors, is_pred=False)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height, class_label = box
        has_anchor = [False, False, False]
        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // num_anchors_per_scale
            anchor_on_scale = anchor_idx % num_anchors_per_scale

            s = grid_sizes[scale_idx]
            i, j = int(s * y), int(s * x)
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if not anchor_taken and not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                x_cell, y_cell = s * x - j, s * y - i

                width_cell, height_cell = (width * s, height * s)

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                # Assigning the box coordinates to the target
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                # Assigning the class label to the target
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                # Set the anchor box as assigned for the scale
                has_anchor[scale_idx] = True

            # If the anchor box is already assigned, check if the
            # IoU is greater than the threshold
            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                # Set the probability to -1 to ignore the anchor box
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1

    return tuple(targets)
