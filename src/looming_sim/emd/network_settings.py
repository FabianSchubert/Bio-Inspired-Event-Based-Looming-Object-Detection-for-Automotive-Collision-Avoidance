import numpy as np

kern_w = 13

sigm_kernel = 0.3

x, y = np.meshgrid(np.linspace(-1.,1.,kern_w), np.linspace(-1.,1.,kern_w))
kernel_p_s = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigm_kernel**2.0))
kernel_p_s /= np.sum(kernel_p_s)

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
    "N_SUBDIV_X": 1,
    "N_SUBDIV_Y": 1,
    "HALF_STEP_TILES": True,
    "SPK_REC_STEPS": 100,
    #
    "TAU_MEM_P": 20.0,
    "TAU_I_P": 5.0,
    "V_THRESH_P": 0.5,
    "T_REFRAC_P": 50.0,
    #
    "P_S_KERNEL": kernel_p_s,
    #
    "TAU_V_S": 50.0,
    "TAU_PX_S": 1.0,
    "V_REG_S": 1e-2,
    #
    "SIGM_POS_WEIGHTS_X": 0.25,
    "SIGM_POS_WEIGHTS_Y": 0.25,
    #
    "OUTPUT_SCALE": 500.0,
    "R_REG": 1e-2,
    "TAU_MEM_OUT": 50.0,
    "TAU_R_OUT": 20.0,
    "FILT_SCALE_OUT": 0.05,
    "FILT_BIAS_OUT": 0.1,
}
