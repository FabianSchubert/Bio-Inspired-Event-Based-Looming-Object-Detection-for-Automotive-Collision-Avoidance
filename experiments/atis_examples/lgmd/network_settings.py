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
    "N_SUBDIV_X": 2,
    "N_SUBDIV_Y": 2,
    "HALF_STEP_TILES": True,
    "SPK_REC_STEPS": 100,
    # S kernel settings
    "KERNEL_WIDTH": KERNEL_WIDTH,  # inhibition kernel width; needs to be ODD
    "KERNEL_HEIGHT": KERNEL_HEIGHT,  # inhibition kernel height; needs to be ODD
    "KERNEL_G": -kernel_g.flatten(),
    "SCALE_KERNEL_G": 0.1,
    "KERNEL_D": np.ones((25), dtype="int"),
    "SCALE_KERNEL_D": 500.0,
    # P to S settings
    "TAU_SYN_IN_S_I": 150.0,
    "TAU_SYN_IN_S_E": 150.0,
    "W_IN_S_E": 0.25,
    # S to LGMD settings
    "W_S_LGMD": 1.0,
    # P to LGMD settings
    "W_IN_LGMD": -0.003,
    "TAU_IN_LGMD": 250.0,
    "THRESH_IN_LGMD": 300.0,
    "SYN_DELAY_IN_LGMD": int(0),
    # P settings
    "TAU_MEM_P": 50.0,
    "TAU_I_P": 20.0,
    "V_THRESH_P": 0.01,
    "V_RESET_P": 0.0,
    "INPUT_EVENT_CURRENT": 1.0,
    "MAX_INPUT_SPIKES": 100000000,
    # S settings
    "TAU_MEM_S": 50.0,
    "V_THRESH_S": 1.0,
    "V_RESET_S": 0.0,
    # LGMD settings
    "TAU_MEM_LGMD": 150.0,
    "V_THRESH_LGMD": 100000.0,
    "V_RESET_LGMD": 0.0,
    "SCALE_I_IN_LGMD": 500.,
    # simulation settings
    "DT_MS": 1.0,
    "T_START_MS": 0.0,
    "TIMING": False,
    "MODEL_SEED": None,
    "N_BATCH": 1,
    "CUDA_VISIBLE_DEVICES": False,
    # recording settings
    "REC_SPIKES": ["P", "S", "LGMD"],
    "REC_SYNAPSES": [],  # ("in_S_I","in_syn")],
}

params["NT_MAX"] = int(10000.0 / params["DT_MS"])
