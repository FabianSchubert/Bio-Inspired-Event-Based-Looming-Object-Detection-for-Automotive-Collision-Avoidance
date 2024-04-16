import numpy as np
from src.utils import FixedDict

kern_w = 9

x, y = np.meshgrid(np.arange(kern_w)-kern_w//2, np.arange(kern_w)-kern_w//2)
kernel_s_t = np.exp(-(x**2. + y**2.)/(2.*(kern_w/(5.))**2.))

params = {
    "NAME": "EMD_model",
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
    "P_S_T_KERNEL": kernel_s_t,
    #
    "TAU_MEM_S": 200.0,
    "TAU_IN_S": 150.0,
    "V_NORM_S": 1.0,
    "POS_NORM_REG_S": 1e2,
    #
    "G_FILT_BIAS_OUT": 0.,#1e-8,
    "G_FILT_SCALE_OUT": 1e-8,
    "OUTPUT_SCALE": 1.0,
}

params["NT_MAX"] = int(20000./params["DT_MS"])

params = FixedDict(params)
