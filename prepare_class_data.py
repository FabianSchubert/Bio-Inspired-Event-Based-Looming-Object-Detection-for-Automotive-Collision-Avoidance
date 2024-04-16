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

DELTA_T = 100000
TH_CROP = 0.75

threshold_func_compare = lambda x: (x >= TH_CROP)

N_SAMPLES = 2500

PREPROCESS = False

MIN_EVENT_COUNT = 50

NUM_THREADS = 5

N_SUBDIV_X = 2
N_SUBDIV_Y = 2
HALF_STEP_TILES = True

for fld in FOLDERS:
    box_events_folder = os.path.join(config.PATH_PROCESSED_DATA, "box_events", fld)

    if PREPROCESS:
        print(f"processing raw data from {fld}...")
        process_folder(
            os.path.join(config.PATH_RAW_DATA, fld),
            box_events_folder,
            num_threads=NUM_THREADS,
            delta_t=DELTA_T,
            n_subdiv_x=N_SUBDIV_X,
            n_subdiv_y=N_SUBDIV_Y,
            half_stride=HALF_STEP_TILES,
            min_event_count=MIN_EVENT_COUNT,
            compare_func_boxes=crop_factor,
            threshold_func_boxes=threshold_func_compare,
        )

    __import__("ipdb").set_trace()

    print("loading in processed data...")
    data = load_folders(
        box_events_folder, load_events=False, load_boxes=True, return_filenames=True
    )

    __import__("ipdb").set_trace()
    print("balancing and pruning data...")
    data_bal_prune = lim_samples_rnd(
        balance_oversample(balance_undersample(data, boxes_only=True)), N_SAMPLES
    )

    balanced_pruned_data_folder = os.path.join(
        config.PATH_PROCESSED_DATA, "balanced_pruned"
    )

    # we store everything in one file, so just call it after the source folder
    filename = fld
    print("saving balanced and pruned data...")
    save_data(balanced_pruned_data_folder, filename, data_bal_prune)
