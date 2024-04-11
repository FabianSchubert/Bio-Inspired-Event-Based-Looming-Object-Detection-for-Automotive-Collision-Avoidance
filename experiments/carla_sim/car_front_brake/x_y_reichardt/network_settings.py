import numpy as np
from src.utils import FixedDict

kern_w = 11

x, y = np.meshgrid(np.arange(kern_w)-kern_w//2, np.arange(kern_w)-kern_w//2)
kernel_s_norm = np.exp(-(x**2. + y**2.)/(2.*(kern_w/(5.))**2.))

DT_MS = 1.0

params = {
    "NAME": "X_Y_model",
    "CUDA_VISIBLE_DEVICES": False,
    "DT_MS": DT_MS,
    "TIMING": False,
    "N_BATCH": 1,
    "MODEL_SEED": None,
    "REC_SPIKES": ["P", "S", "OUT"],
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "N_SUBDIV_X": 4,
    "N_SUBDIV_Y": 4,
    "HALF_STEP_TILES": True,
    "SPK_REC_STEPS": 100,
    #
    "TAU_MEM_P": 50.0,
    "TAU_I_P": 20.0,
    "V_THRESH_P": 0.01,
    "V_RESET_P": 0.0,
    "INPUT_EVENT_CURRENT": 1.0,
    #
    "P_S_NORM_KERNEL": kernel_s_norm,
    "P_S_PS_TAU": 2.5,
    "P_S_T_DELAY_STEPS": int(25./DT_MS),
    #
    "TAU_MEM_S": 100.0,
    "S_POS_NORM_REG": 1e1,
    "VEL_NORM_S": 1e0,
    "THRESHOLD_S": 1e-7,
    #
    "TAU_MEM_OUT": 100.0,
    "G_FILT_BIAS_OUT": 1e-8,
    "G_FILT_SCALE_OUT": 2.5e-8,
}

params["NT_MAX"] = int(10000./params["DT_MS"])

params = FixedDict(params)
