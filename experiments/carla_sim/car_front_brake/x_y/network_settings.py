import numpy as np
from src.utils import FixedDict

kern_w = 15

x, y = np.meshgrid(np.arange(kern_w)-kern_w//2, np.arange(kern_w)-kern_w//2)
kernel_s_t = np.exp(-(x**2. + y**2.)/(2.*(kern_w/(5.))**2.))

params = {
    "NAME": "X_Y_model",
    "CUDA_VISIBLE_DEVICES": False,
    "DT_MS": 1.0,
    "TIMING": False,
    "N_BATCH": 1,
    "MODEL_SEED": None,
    "REC_SPIKES": ["P", "S", "OUT"],
    "INPUT_WIDTH": 304,
    "INPUT_HEIGHT": 240,
    "N_SUBDIV_X": 2,
    "N_SUBDIV_Y": 2,
    "HALF_STEP_TILES": True,
    "SPK_REC_STEPS": 100,
    #
    "TAU_MEM_P": 50.0,
    "TAU_I_P": 20.0,
    "V_THRESH_P": 0.01,
    "V_RESET_P": 0.0,
    "INPUT_EVENT_CURRENT": 1.0,
    #
    "P_S_T_KERNEL": kernel_s_t,
    "P_S_PS_TAU": 50.0,
    #
    "TAU_MEM_S": 100.0,
    "TAU_FILT_S": 100.0,
    "B_REG_S": 1e-8,
    "S_POS_NORM_REG": 1e0,
    #
    "TAU_MEM_OUT": 100.0,
    "G_FILT_BIAS_OUT": -1e2,
    "G_FILT_SCALE_OUT": 1e-3,
}

params["NT_MAX"] = int(10000./params["DT_MS"])

params = FixedDict(params)
