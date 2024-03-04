import esim_py
import numpy as np
import sys
import os
from typing import Union

from config import EVENTS_DTYPE

p = {
    "contrast_threshold_pos": 0.8,
    "contrast_threshold_neg": 0.8,
    "refractory_period": 0.0,
    "log_eps": 1e-3,
    "use_log_img": True,
}


def vid2e(
    scene_path: str, event_path: Union[None, str] = None, esim_params: dict = p
) -> None:

    frame_path = os.path.join(scene_path, "frames/")

    if not event_path:
        event_path = os.path.join(scene_path, "events.npy")

    # constructor
    esim = esim_py.EventSimulator(*esim_params.values())

    events_from_images = esim.generateFromFolder(
        frame_path, os.path.join(scene_path, "timestamps.txt")
    )

    events_converted = np.array(
        list(
            zip(
                events_from_images[:, 2] * 1000.0,
                events_from_images[:, 0],
                events_from_images[:, 1],
                (events_from_images[:, 3].astype("int") + 1) // 2,
            )
        ),
        dtype=EVENTS_DTYPE,
    )

    np.save(event_path, events_converted)


if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]
    EVENT_PATH = sys.argv[2] if len(sys.argv) == 3 else None

    vid2e(SCENE_PATH, EVENT_PATH)
