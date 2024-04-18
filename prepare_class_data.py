from src.classifier.preproc.filter_events_from_boxes import process_folder
from src.classifier.preproc.balance_data import (
    balance_undersample,
    balance_oversample,
    lim_samples_rnd,
)

from src.classifier.data_io import load_folders, save_data
from src.utils import crop_factor

import src.config as config

import os

FOLD_TRAIN = "train_a"
FOLD_VAL = "val_a"
FOLD_TEST = "test_a"

FOLDERS = [FOLD_TRAIN, FOLD_VAL, FOLD_TEST]

DELTA_T_MICROSECS = 100000  # microseconds
TH_CROP = 0.75

threshold_func_compare = lambda x: (x >= TH_CROP)

PREPROCESS = True

MIN_EVENT_COUNT = 50

NUM_THREADS = 5

N_SUBDIV = 2
HALF_STEP_TILES = True

for fld in FOLDERS:
    box_events_folder = os.path.join(config.PATH_PROCESSED_DATA, "box_events", f"{N_SUBDIV}_tiles", fld)

    if PREPROCESS:
        print(f"processing raw data from {fld}...")
        process_folder(
            os.path.join(config.PATH_RAW_DATA, fld),
            box_events_folder,
            num_threads=NUM_THREADS,
            delta_t=DELTA_T_MICROSECS,
            n_subdiv_x=N_SUBDIV,
            n_subdiv_y=N_SUBDIV,
            half_stride=HALF_STEP_TILES,
            min_event_count=MIN_EVENT_COUNT,
            compare_func_boxes=crop_factor,
            threshold_func_boxes=threshold_func_compare,
        )
