import esim_py
from src.synth_data.vid2e import vid2e

from itertools import product

import os

P_VID2E = {
    "contrast_threshold_pos": 0.3,
    "contrast_threshold_neg": 0.3,
    "refractory_period": 0.0,
    "log_eps": 1e-3,
    "use_log_img": True
}

VEL_MPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

OBJECTS = [
    "disc_bright",
    "disc_dark",
    "square_bright",
    "square_dark",
]

BACKGROUNDS = ["gray_bg"]

base_fold = os.path.join(os.path.dirname(__file__), "../../data/synthetic/")

for vel, obj, bg in product(VEL_MPS, OBJECTS, BACKGROUNDS):
    data_fold = f"{obj}_on_{bg}_{vel}_mps/"
    print(f"Processing {data_fold}...")
    data_fold = os.path.join(base_fold, data_fold)

    vid2e(data_fold, esim_params=P_VID2E)

