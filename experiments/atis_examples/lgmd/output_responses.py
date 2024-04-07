from src.looming_sim.simulator_LGMD import LGMD_model

from .network_settings import params

from src.viz import play_event_anim, play_var_anim

from src.utils import convert_spk_id_to_evt_array

from itertools import product

import numpy as np

import os


REC_DT = 10.0

REC_NEURONS = [("S", "V"), ("LGMD", "V"), ("P", "V"), ("P", "VI")]

base_fold = os.path.join(os.path.dirname(__file__), "../../../data/atis_examples/")

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/atis_examples/",
)

input_data_files = os.listdir(base_fold)
n_examples = len(input_data_files)

lgmd_network = LGMD_model(params)

for k in range(n_examples):
    results_fold = os.path.join(base_fold_results, f"example_{k}/lgmd/")

    if not os.path.exists(results_fold):
        os.makedirs(results_fold)

    evt_file = os.path.join(base_fold, f"events_test_{k}.npy")
    evts = np.load(evt_file)

    t_end = evts["t"][-1]

    lgmd_network.load_input_data_from_file(evt_file)
    lgmd_network.push_input_data_to_device()

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = (
        lgmd_network.run_model(0.0, t_end, rec_neurons=REC_NEURONS, rec_timestep=REC_DT)
    )

    v_p = []
    v_s = []
    v_out = []
    sp_p = []
    sp_s = []
    sp_out = []

    for i in range(lgmd_network.n_tiles_y):
        v_p.append([])
        v_s.append([])
        v_out.append([])
        sp_p.append([])
        sp_s.append([])
        sp_out.append([])

        for j in range(lgmd_network.n_tiles_x):
            v_s[-1].append(
                np.reshape(
                    rec_vars_n[f"VS_{i}_{j}"],
                    (-1, lgmd_network.S_height, lgmd_network.S_width),
                )
            )
            v_out[-1].append(
                rec_vars_n[f"VLGMD_{i}_{j}"].flatten(),
            )
            '''
            evts_s[-1].append(
                convert_spk_id_to_evt_array(
                    spike_ID[f"S_{i}_{j}"],
                    spike_t[f"S_{i}_{j}"],
                    lgmd_network.S_width,
                    lgmd_network.S_height,
                )
            )'''
            sp_out[-1].append(
                convert_spk_id_to_evt_array(
                    spike_ID[f"LGMD_{i}_{j}"],
                    spike_t[f"LGMD_{i}_{j}"],
                    1,
                    1,
                )
            )
            sp_p[-1].append(
                convert_spk_id_to_evt_array(
                    spike_ID[f"P_{i}_{j}"],
                    spike_t[f"P_{i}_{j}"],
                    lgmd_network.tile_width,
                    lgmd_network.tile_height,
                )
            )

            sp_s[-1].append(
                convert_spk_id_to_evt_array(
                    spike_ID[f"S_{i}_{j}"],
                    spike_t[f"S_{i}_{j}"],
                    lgmd_network.S_width,
                    lgmd_network.S_height,
                )
            )

    if (lgmd_network.n_tiles_x == 1) and (lgmd_network.n_tiles_y == 1):
        sp_p = np.array(sp_p, dtype=sp_p[0][0].dtype)
        sp_s = np.array(sp_s, dtype=sp_p[0][0].dtype)
    else:
        sp_p = np.array(sp_p, dtype=object)
        sp_s = np.array(sp_s, dtype=object)

    np.savez(
        os.path.join(results_fold, "results.npz"),
        v_s=v_s,
        v_out=v_out,
        sp_p=sp_p,
        sp_s=sp_s,
        sp_out=sp_out,
        rec_n_t=rec_n_t,
    )
