from src.classifier.preproc.filter_events_from_boxes import process_folder, process_files_full_image

import numpy as np

import src.config as config

import os

FOLD_TRAIN = "train_a"
FOLD_VAL = "val_a"
FOLD_TEST = "test_a"

FOLDERS = [FOLD_TRAIN, FOLD_VAL, FOLD_TEST]

DELTA_T_MICROSECS = 100000  # microseconds
DELTA_T_BIN = 1000

BIN_TENSOR = False

SAVE_BOXES_AS_TXT = True

NUM_THREADS = 5

for i, fld in enumerate(FOLDERS):
    print(f"processing raw data from {fld}...")
    '''
    process_folder(
        os.path.join(config.PATH_RAW_DATA, fld),
        os.path.join(config.PATH_PROCESSED_DATA, "event_slices_boxes", fld),
        num_threads=NUM_THREADS,
        target_func=process_files_full_image,
        delta_t=DELTA_T_MICROSECS,
        delta_t_bin=DELTA_T_BIN,
        convert_events_to_bin_tensor=BIN_TENSOR,
        save_boxes_as_txt=SAVE_BOXES_AS_TXT,
    )'''

    labels = os.listdir(os.path.join(config.PATH_PROCESSED_DATA, "event_slices_boxes", fld, "labels/"))
    # sort the labels
    labels = sorted(labels)

    events = os.listdir(os.path.join(config.PATH_PROCESSED_DATA, "event_slices_boxes", fld, "events/"))
    # sort the events
    events = sorted(events)

    # check if the file ids are the same
    assert len(labels) == len(events)
    for label, event in zip(labels, events):
        label = label.split("_bbox")[0]
        event = event.split("_td")[0]
        assert label == event

    with open(os.path.join(config.PATH_PROCESSED_DATA, "event_slices_boxes", fld, "labels.csv"), "w") as f:
        for event, label in zip(events, labels):
            f.write(f"{event} {label}\n")


