import numpy as np
import os


kernel_s_t = np.loadtxt(
    os.path.join(os.path.dirname(__file__), "kernel_g.csv"), delimiter=","
)

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
    "N_SUBDIV_X": 4,
    "N_SUBDIV_Y": 4,
    "HALF_STEP_TILES": True,
    #
    "TAU_MEM_P": 50.0,
    "V_THRESH_P": 0.1,
    "V_RESET_P": 0.0,
    "INPUT_EVENT_CURRENT": 5.0,
    #
    "P_S_T_KERNEL": kernel_s_t,
    #
    "TAU_MEM_S": 50.0,
    "TAU_FILT_S": 10.0,
    "B_REG_S": 1e-2,
    #
    "TAU_MEM_OUT": 50.0,
    "G_FILT_BIAS_OUT": 1.0,
    "G_FILT_SCALE_OUT": 1.0,
}
