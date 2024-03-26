from src.lgmd_sim.simulator_LGMD import LGMD_model

from .network_settings import params

from src.viz import play_event_anim, play_var_anim

from src.utils import convert_spk_id_to_evt_array

from itertools import product

import numpy as np

import os


REC_DT = 10.0

REC_NEURONS = [("S", "V"), ("P", "V")]

base_fold = os.path.join(os.path.dirname(__file__), "../../../data/atis_examples/")

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/atis_examples/s_cell_spike_responses/x_y_smooth/",
)

lgmd_network = LGMD_model(params)

for k in range(7):
    results_fold = os.path.join(base_fold_results, f"example_{k}")

    if not os.path.exists(results_fold):
        os.makedirs(results_fold)

    evt_file = os.path.join(base_fold, f"events_test_{k}.npy")
    evts = np.load(evt_file)

    t_end = evts["t"][-1]

    lgmd_network.load_input_data_from_file(evt_file)
    lgmd_network.push_input_data_to_device()

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = (
        lgmd_network.run_model(
            0.0, t_end, rec_neurons=REC_NEURONS, rec_timestep=REC_DT
        )
    )

    vp = []
    vs = []
    evts_s = []
    for i in range(lgmd_network.n_tiles_y):
        vp.append([])
        vs.append([])
        evts_s.append([])
        for j in range(lgmd_network.n_tiles_x):
            vp[-1].append(
                np.reshape(
                    rec_vars_n[f"VP_{i}_{j}"],
                    (-1, lgmd_network.tile_height, lgmd_network.tile_width),
                )
            )
            vs[-1].append(
                np.reshape(
                    rec_vars_n[f"VS_{i}_{j}"],
                    (-1, lgmd_network.S_height, lgmd_network.S_width),
                )
            )
            evts_s[-1].append(
                convert_spk_id_to_evt_array(
                    spike_ID[f"S_{i}_{j}"],
                    spike_t[f"S_{i}_{j}"],
                    lgmd_network.S_width,
                    lgmd_network.S_height,
                )
            )

    np.savez(
        os.path.join(results_fold, "results.npz"),
        vp=vp,
        vs=vs,
        evts_s=evts_s,
        rec_n_t=rec_n_t,
    )