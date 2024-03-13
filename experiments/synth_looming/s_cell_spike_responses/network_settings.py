import numpy as np
import os
from skimage.io import imread

DT = 1.0

kernel_g = (
    imread(os.path.join(os.path.dirname(__file__), "kernel_g.png")).mean(axis=2) / 255.0
)

KERNEL_HEIGHT, KERNEL_WIDTH = kernel_g.shape

params = {
    # general settings
    "NAME": "s cell spike response",
    "OUT_DIR": ".",
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "N_SUBDIV_X": 1,
    "N_SUBDIV_Y": 1,
    "HALF_STEP_TILES": True,
    # S kernel settings
    "KERNEL_WIDTH": KERNEL_WIDTH,  # inhibition kernel width; needs to be ODD
    "KERNEL_HEIGHT": KERNEL_HEIGHT,  # inhibition kernel height; needs to be ODD
    "KERNEL_G": -kernel_g.flatten(),
    "SCALE_KERNEL_G": 1.0,
    "KERNEL_D": np.ones((25), dtype="int"),
    "SCALE_KERNEL_D": 200.0,
    # P to S settings
    "TAU_SYN_IN_S_I": 150.0,
    "TAU_SYN_IN_S_E": 150.0,
    "W_IN_S_E": 0.25,
    # S to LGMD settings
    "W_S_LGMD": 0.02,  # 0.04,
    # P to LGMD settings
    "W_IN_LGMD": -0.5,  # -0.04,
    "TAU_IN_LGMD": 50.0,
    "THRESH_IN_LGMD": 200.0,
    "SYN_DELAY_IN_LGMD": int(50),
    # P settings
    "MAX_INPUT_SPIKES": 100000000,
    "INPUT_EVENT_CURRENT": 15.,
    "TAU_MEM_P": 50.0,
    "V_THRESH_P": 0.1,
    "V_RESET_P": 0.0,
    # S settings
    "TAU_MEM_S": 50.0,
    "V_THRESH_S": 5.0,
    "V_RESET_S": 0.0,
    # LGMD settings
    "TAU_MEM_LGMD": 20.0,
    "V_THRESH_LGMD": 1.0,
    "V_RESET_LGMD": 0.0,
    # simulation settings
    "DT_MS": 1.0,
    "TRIAL_MS": 10000.0,
    "T_START_MS": 0.0,
    "TIMING": False,
    "MODEL_SEED": None,
    "N_BATCH": 1,
    "CUDA_VISIBLE_DEVICES": False,
    # recording settings
    "REC_SPIKES": ["P", "S", "LGMD"],
    "REC_NEURONS": [("P", "V"), ("S", "V"), ("LGMD", "V")],
    "REC_SYNAPSES": [],  # ("in_S_I","in_syn")],
}

params["SPK_REC_STEPS"] = int(params["TRIAL_MS"] / params["DT_MS"])
params["NT_MAX"] = int(params["TRIAL_MS"] / params["DT_MS"])
