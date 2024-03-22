from src.lgmd_sim.simulator_LGMD import LGMD_model

from .network_settings import params

from src.viz import play_event_anim, play_var_anim

from src.utils import convert_spk_id_to_evt_array

from itertools import product

import numpy as np

import os

import matplotlib.pyplot as plt


VEL_MPS = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

OBJECTS = [
    "disc_bright",
    "disc_dark",
    "square_bright",
    "square_dark",
    "disc_rand_struct_bright",
]

BACKGROUNDS = [
    "gray_bg",
    "cloudy_bg"
]

REC_DT = 10.0

REC_NEURONS = [
    ("S", "V"),
    ("OUT", "S_left"),
    ("OUT", "S_right"),
    ("OUT", "V_est"),
    ("S", "Vx"),
    ("S", "Vy"),
    ("S", "Vt"),
]

base_fold = os.path.join(os.path.dirname(__file__), "../../../data/synthetic/")

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/synth_looming/s_cell_spike_responses/x_y_smooth/",
)

lgmd_network = LGMD_model(params)

for vel, obj, bg in product(VEL_MPS, OBJECTS, BACKGROUNDS):
    data_fold = f"{obj}_on_{bg}_{vel}_mps/"

    results_fold = os.path.join(base_fold_results, data_fold)

    if not os.path.exists(results_fold):
        os.makedirs(results_fold)

    data_fold = os.path.join(base_fold, data_fold)

    lgmd_network.load_input_data_from_file(os.path.join(data_fold, "events.npy"))
    lgmd_network.push_input_data_to_device()

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = (
        lgmd_network.run_model(
            0.0, 10000.0, rec_neurons=REC_NEURONS, rec_timestep=REC_DT
        )
    )

    vs = np.reshape(
        rec_vars_n["VS_0_0"], (-1, lgmd_network.S_height, lgmd_network.S_width)
    )
    vxs = np.reshape(
        rec_vars_n["VxS_0_0"], (-1, lgmd_network.S_height, lgmd_network.S_width)
    )
    vys = np.reshape(
        rec_vars_n["VyS_0_0"], (-1, lgmd_network.S_height, lgmd_network.S_width)
    )
    vts = np.reshape(
        rec_vars_n["VtS_0_0"], (-1, lgmd_network.S_height, lgmd_network.S_width)
    )

    vout = rec_vars_n["V_estOUT_0_0"].flatten()

    sleftout = rec_vars_n["S_leftOUT_0_0"].flatten()
    srightout = rec_vars_n["S_rightOUT_0_0"].flatten()

    # evts_p = convert_spk_id_to_evt_array(
    #    spike_ID["P_0_0"],
    #    spike_t["P_0_0"],
    #    lgmd_network.tile_width,
    #    lgmd_network.tile_height,
    # )

    # evts_s = convert_spk_id_to_evt_array(
    #    spike_ID["S_0_0"],
    #    spike_t["S_0_0"],
    #    lgmd_network.S_width,
    #    lgmd_network.S_height,
    # )

    #'''
    np.savez(
        os.path.join(results_fold, "results.npz"),
        vs=vs,
        vout=vout,
        sleftout=sleftout,
        srightout=srightout,
        rec_n_t=rec_n_t,
    )# '''

    """

    play_var_anim(
        vs,
        0.0,
        10000.0,
        REC_DT,
        100.0,
        -np.maximum(0.0, vs.max()),
        np.maximum(0.0, vs.max()),
        os.path.join(data_fold, "vs_anim/"),
    )

    import pdb

    pdb.set_trace()
    # """
