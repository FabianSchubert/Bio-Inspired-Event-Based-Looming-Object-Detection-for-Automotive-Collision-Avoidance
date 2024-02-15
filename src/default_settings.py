import numpy as np
from .utils import FixedDict

# define a few constant parameters based on the Blanchard et al. 2000 paper
BLANCHARD_DT = 1000 / 16  # time step (corresponds to 16 frames/s)
BLANCHARD_p_P = 0.4  # decay factor P neurons
BLANCHARD_p_S = 0.02  # decay factor S neurons
BLANCHARD_p_LGMD = 0.4  # decay factor LGMD
BLANCHARD_p_I = 0.6  # decay factor I neurons
BLANCHARD_p_E = 0.8  # decay factor E neurons
BLANCHARD_p_F = 0.3  # decay factor F neuron

N_SUBDIV_X = 4
N_SUBDIV_Y = 4

circle_kernel = 0.5 * np.array(
    [
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]
)

# circle_kernel *= 5./circle_kernel.sum()

DT = 1.0

# pixels per millisecond
v_crit = 0.025
delay_steps = int(3.0 / (v_crit * DT))


# all parameters of the model (baseline values)
p = {
    "NAME": "test",
    "OUT_DIR": ".",
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "N_SUBDIV_X": 4,
    "N_SUBDIV_Y": 4,
    "HALF_STEP_TILES": True,
    "KERNEL_WIDTH": 5,  # inhibition kernel width; needs to be ODD
    "KERNEL_HEIGHT": 5,  # inhibition kernel height; needs to be ODD
    # -0.2*
    "KERNEL_G": -1
    * np.array(
        [
            0,
            0,
            0.2,
            0,
            0,
            0,
            0.32,
            0.4,
            0.32,
            0,
            0.2,
            0.4,
            0,
            0.4,
            0.2,
            0,
            0.32,
            0.4,
            0.32,
            0,
            0,
            0,
            0.2,
            0,
            0,
        ]
    ),
    "SCALE_KERNEL_G": 1.0,
    "KERNEL_D": int(BLANCHARD_DT)
    * np.array(
        [0, 0, 2, 0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 1, 0, 0, 0, 2, 0, 0],
        dtype=int,
    ),
    "SCALE_KERNEL_D": 1.0,
    "TAU_SYN_IN_S_I": -BLANCHARD_DT / np.log(BLANCHARD_p_I),
    "W_IN_S_E": 1,
    "TAU_SYN_IN_S_E": -BLANCHARD_DT / np.log(BLANCHARD_p_E),
    "W_S_LGMD": 0.02,  # 0.04,
    "W_IN_LGMD": -0.5,  # -0.04,
    "TAU_IN_LGMD": -BLANCHARD_DT / np.log(BLANCHARD_p_F),
    # this is the Blanchard et al. threshold as F to LGMD weight is 1
    "THRESH_IN_LGMD": 200.0,
    # buffer size for input spike times (all input neurons, one input sequence)
    "MAX_INPUT_SPIKES": 100000000,
    "TAU_MEM_P": -BLANCHARD_DT / np.log(BLANCHARD_p_P),
    "V_THRESH_P": 0.1,
    "V_RESET_P": 0.1,
    "TAU_MEM_S": -BLANCHARD_DT / np.log(BLANCHARD_p_S),
    "V_THRESH_S": 0.5,  # S Theta in Blanchard et al. 2000
    "V_RESET_S": 0.5,  # S "Reset alpha" in Blanchard et al. 2000
    "TAU_MEM_LGMD": -BLANCHARD_DT / np.log(BLANCHARD_p_LGMD),
    "V_THRESH_LGMD": 0.25,
    "V_RESET_LGMD": 0.25,
    "DT_MS": 1.0,
    "TRIAL_MS": 10000.0,
    "T_START_MS": 0.0,
    "TIMING": False,
    "MODEL_SEED": None,
    "N_BATCH": 1,
    "WRITE_TO_DISK": False,
    "REC_SPIKES": ["P", "S", "LGMD"],
    "REC_NEURONS": [("P", "V"), ("S", "V"), ("LGMD", "V")],
    "REC_SYNAPSES": [],  # ("in_S_I","in_syn")],
    "CUDA_VISIBLE_DEVICES": False,
    "INPUT_EVENT_FILE": "events.npy",
    "INPUT_EVENT_CURRENT": 5,
    "SYN_DELAY_LGMD": int(BLANCHARD_DT),
}

p["SPK_REC_STEPS"] = int(p["TRIAL_MS"] / p["DT_MS"])
p["NT_MAX"] = int(60000.0 / p["DT_MS"])

# this fixes the keys in the dictionary (to prevent accidental typos to go unnoticed)
params = FixedDict(p)
