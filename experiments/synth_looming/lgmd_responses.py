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
    # 0.5,
    # 1.0,
    # 1.5,
    2.0,
    # 2.5,
    # 3.0,
]

OBJECTS = [
    "disc_bright",
    # "disc_dark",
    # "square_bright",
    # "square_dark",
]

BACKGROUNDS = ["gray_bg"]

base_fold = os.path.join(os.path.dirname(__file__), "../../data/synthetic/")

lgmd_network = LGMD_model(params)

for vel, obj, bg in product(VEL_MPS, OBJECTS, BACKGROUNDS):
    data_fold = f"{obj}_on_{bg}_{vel}_mps/"
    data_fold = os.path.join(base_fold, data_fold)

    lgmd_network.load_input_data_from_file(os.path.join(data_fold, "events.npy"))
    lgmd_network.push_input_data_to_device()

    spike_t, spike_ID, rec_vars_n, rec_vars_s = lgmd_network.run_model(
        0.0, 10000.0, rec_neurons=params["REC_NEURONS"]
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

    play_var_anim(vs, 0.0, 10000., params["DT_MS"], 50., vs.min(), vs.max(), os.path.join(data_fold, "vs_anim/"))

    import pdb

    pdb.set_trace()
