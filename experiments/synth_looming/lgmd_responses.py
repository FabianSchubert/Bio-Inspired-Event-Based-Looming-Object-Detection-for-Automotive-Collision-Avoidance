from src.lgmd_sim.simulator_LGMD import LGMD_model

from src.default_settings import params as default_params

from src.synth_data.viz import plot_event_slice, play_event_anim, play_var_anim

from config import EVENTS_DTYPE

from itertools import product

import numpy as np

import os


def convert_spk_id_to_evt_array(
    spike_id: np.ndarray,
    spike_t: np.ndarray,
    width: int,
    height: int,
    spike_pol: None | np.ndarray = None,
) -> np.ndarray:
    if spike_id.shape[0] > 0:
        assert (
            spike_id.max() < width * height
        ), "largest spike id does not fit into width & height dimensions"

    assert spike_id.shape[0] == spike_t.shape[0], "array sizes do not match"
    if not spike_pol:
        spike_pol = np.ones((spike_id.shape[0]))

    x = spike_id % width
    y = spike_id // width

    evt_array = np.array(list(zip(spike_t, x, y, spike_pol)), dtype=EVENTS_DTYPE)

    return evt_array


params = dict(default_params)

params["N_SUBDIV_X"] = 1
params["N_SUBDIV_Y"] = 1

params["INPUT_EVENT_CURRENT"] = 15.0

VEL_MPS = [
    0.5,
    # 1.0,
    # 1.5,
    # 2.0,
    # 2.5,
    # 3.0,
]

OBJECTS = [
    "disc_bright",
    # "disc_dark",
    # "square_bright",
    # "square_dark",
    # "disc_rand_struct_bright",
]

BACKGROUNDS = ["gray_bg"]

REC_DT = 10.0

base_fold = os.path.join(os.path.dirname(__file__), "../../data/synthetic/")

base_fold_results = os.path.join(
    os.path.dirname(__file__), "../../data/experiments/synth_looming/"
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
            0.0, 10000.0, rec_neurons=params["REC_NEURONS"], rec_timestep=REC_DT
        )
    )

    vp = np.reshape(
        rec_vars_n["VP_0_0"], (-1, lgmd_network.tile_height, lgmd_network.tile_width)
    )

    vs = np.reshape(
        rec_vars_n["VS_0_0"], (-1, lgmd_network.S_height, lgmd_network.S_width)
    )

    evts_p = convert_spk_id_to_evt_array(
        spike_ID["P_0_0"],
        spike_t["P_0_0"],
        lgmd_network.tile_width,
        lgmd_network.tile_height,
    )

    evts_s = convert_spk_id_to_evt_array(
        spike_ID["S_0_0"],
        spike_t["S_0_0"],
        lgmd_network.S_width,
        lgmd_network.S_height,
    )

    np.savez(
        os.path.join(results_fold, "results.npz"),
        vp=vp,
        vs=vs,
        evts_p=evts_p,
        evts_s=evts_s,
        rec_n_t=rec_n_t,
    )

    # """

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

    play_event_anim(
        evts_s, 0.0, 10000, 50.0, lgmd_network.S_width, lgmd_network.S_height
    )

    import pdb

    pdb.set_trace()
    # """
