import numpy as np
from .utils import FixedDict
import os


def gen_gamma_kern(sigm: float, theta: float) -> np.ndarray:
    l_int = int(2.0 * sigm)

    x, y = np.meshgrid(np.arange(-l_int, l_int + 1), np.arange(-l_int, l_int + 1))

    x_p = x * np.cos(theta) + y * np.sin(theta)
    y_p = -x * np.sin(theta) + y * np.cos(theta)

    weight_kern = np.exp(-(x_p**2.0 + y_p**2.0) / (2.0 * sigm**2.0))
    weight_kern /= weight_kern.sum()

    kern = x_p * weight_kern

    return kern


def gen_gauss_kern(sigm: float) -> np.ndarray:
    l_int = int(2.0 * sigm)

    x, y = np.meshgrid(np.arange(-l_int, l_int + 1), np.arange(-l_int, l_int + 1))

    kern = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigm**2.0))

    return kern / kern.sum()


N_SUBDIV_X = 4
N_SUBDIV_Y = 4


DT = 1.0

T_DEL_KERNEL = 200.0
SIGM_KERNEL = 10.0

kernel_s_x = gen_gamma_kern(SIGM_KERNEL, 0.0) / T_DEL_KERNEL
kernel_s_y = gen_gamma_kern(SIGM_KERNEL, np.pi / 2.0) / T_DEL_KERNEL
kernel_s_norm = gen_gauss_kern(SIGM_KERNEL)

kernel_height, kernel_width = kernel_s_x.shape

# all parameters of the model (baseline values)
p = {
    "NAME": "test",
    "OUT_DIR": ".",
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "N_SUBDIV_X": 4,
    "N_SUBDIV_Y": 4,
    "HALF_STEP_TILES": True,
    "KERNEL_WIDTH": kernel_width,  # inhibition kernel width; needs to be ODD
    "KERNEL_HEIGHT": kernel_height,  # inhibition kernel height; needs to be ODD
    "KERNEL_SX": kernel_s_x.flatten(),
    "KERNEL_SY": kernel_s_y.flatten(),
    "KERNEL_SNORM": kernel_s_norm.flatten(),
    # -0.2*
    "KERNEL_D": T_DEL_KERNEL
    * np.ones((kernel_width * kernel_height)),  # kernel_d.flatten(),
    "W_IN_S_E": 0.25,
    "W_S_LGMD": 0.02,  # 0.04,
    "W_IN_LGMD": -0.5,  # -0.04,
    "TAU_IN_LGMD": 50.0,
    # this is the Blanchard et al. threshold as F to LGMD weight is 1
    "THRESH_IN_LGMD": 200.0,
    # buffer size for input spike times (all input neurons, one input sequence)
    "MAX_INPUT_SPIKES": 100000000,
    "TAU_MEM_P": 50.0,
    "V_THRESH_P": 0.1,
    "V_RESET_P": 0.0,
    "TAU_MEM_S": 50.0,
    "V_THRESH_S": 5000.0,  # S Theta in Blanchard et al. 2000
    "V_RESET_S": 0.0,  # S "Reset alpha" in Blanchard et al. 2000
    "S_REG_NORM": 1e-2,
    "TAU_MEM_LGMD": 20.0,
    "V_THRESH_LGMD": 1.0,
    "V_RESET_LGMD": 0.0,
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
    "SYN_DELAY_LGMD": int(50),
}

p["SPK_REC_STEPS"] = int(p["TRIAL_MS"] / p["DT_MS"])
p["NT_MAX"] = int(60000.0 / p["DT_MS"])

# this fixes the keys in the dictionary (to prevent accidental typos to go unnoticed)
params = FixedDict(p)
