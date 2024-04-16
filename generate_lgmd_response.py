from src.looming_sim.lgmd.simulator_LGMD import LGMD_model
from src.looming_sim.default_settings import params as p
from src.looming_sim.format_spike_data import (
    convert_spike_id_events_to_spike_coord_events,
)
from prophesee_toolbox.io.dat_events_tools import (
    write_event_buffer,
    write_header,
)

import numpy as np

import os

p["REC_SPIKES"] = ["P", "S"]
p["REC_NEURONS"] = []

"""
This script does not tile the input as it is more efficient to
store responses to the entire scene first and then extract the
tiled variants afterwards when preprocessing the data for
training.
"""
p["N_SUBDIV_X"] = 1
p["N_SUBDIV_Y"] = 1
p["HALF_STEP_TILES"] = False

N_TILES_X = (2 * p["N_SUBDIV_X"] - 1) if p["HALF_STEP_TILES"] else p["N_SUBDIV_X"]
N_TILES_Y = (2 * p["N_SUBDIV_Y"] - 1) if p["HALF_STEP_TILES"] else p["N_SUBDIV_Y"]

BASE_FOLD_DATA = "/mnt/data0/prophesee_data/ATIS_Automotive_Detection_Dataset"
TRAIN_FOLDERS = [f"train_{k}" for k in "abcdef"]
VAL_FOLDERS = [f"val_{k}" for k in "ab"]
TEST_FOLDERS = [f"test_{k}" for k in "ab"]

BASE_FOLD_NEW_DATA = (
    "/mnt/data0/prophesee_data/ATIS_Automotive_Detection_Dataset/lgmd_responses"
)

T_START_MS, T_END_MS = 0.0, 60000.0

network = LGMD_model(p)

file_list = []


def write_evts_to_file(fn, events):
    """
    save numpy events in the ATIS dataset format
    """
    fl = write_header(fn)
    write_event_buffer(fl, events)
    fl.close()


for folder in TRAIN_FOLDERS + VAL_FOLDERS + TEST_FOLDERS:
    full_path_input_folder = os.path.join(BASE_FOLD_DATA, folder)
    full_path_new_data_folder_p = os.path.join(BASE_FOLD_NEW_DATA, folder, "P")
    full_path_new_data_folder_s = os.path.join(BASE_FOLD_NEW_DATA, folder, "S")

    if not os.path.exists(full_path_new_data_folder_p):
        os.makedirs(full_path_new_data_folder_p)

    if not os.path.exists(full_path_new_data_folder_s):
        os.makedirs(full_path_new_data_folder_s)

    for filename in os.listdir(full_path_input_folder):
        if filename.endswith("td.dat"):
            input_event_file = os.path.join(full_path_input_folder, filename)

            spike_t, spike_ID, _, _ = network.run_model(
                T_START_MS,
                T_END_MS - T_START_MS,
                event_data=input_event_file,
                rec_neurons=p["REC_NEURONS"],
                rec_synapses=[],
            )

            p_events = convert_spike_id_events_to_spike_coord_events(
                spike_t["P_0_0"],
                spike_ID["P_0_0"],
                np.ones(spike_t["P_0_0"].shape[0]),
                p["INPUT_WIDTH"],
                p["INPUT_HEIGHT"],
            )

            s_events = convert_spike_id_events_to_spike_coord_events(
                spike_t["S_0_0"],
                spike_ID["S_0_0"],
                np.ones(spike_t["S_0_0"].shape[0]),
                network.S_width,
                network.S_height,
            )

            write_evts_to_file(
                os.path.join(full_path_new_data_folder_p, filename), p_events
            )

            write_evts_to_file(
                os.path.join(full_path_new_data_folder_s, filename), s_events
            )
