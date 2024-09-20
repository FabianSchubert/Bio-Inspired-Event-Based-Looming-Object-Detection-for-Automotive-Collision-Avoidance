import numpy as np

half_kern_w = 2

sigm_kernel = 0.3

x, y = np.meshgrid(
    np.arange(-half_kern_w, half_kern_w + 1), np.arange(-half_kern_w, half_kern_w + 1)
)
kernel_norm = np.exp(-(x**2.0 + y**2.0) / (2.0 * (sigm_kernel * half_kern_w) ** 2.0))
kernel_norm /= np.sum(kernel_norm)

kernel_x = x * kernel_norm
kernel_y = y * kernel_norm

params = {
    "NAME": "EMD_model",
    "CUDA_VISIBLE_DEVICES": False,
    "DT_MS": 1.0,
    "TIMING": False,
    "N_BATCH": 1,
    "MODEL_SEED": None,
    "REC_SPIKES": ["P", "S", "OUT"],
    "INPUT_WIDTH": 640,  # 304,
    "INPUT_HEIGHT": 480,  # 240,
    "N_SUBDIV_X": 2,
    "N_SUBDIV_Y": 2,
    "HALF_STEP_TILES": True,
    "SPK_REC_STEPS": 100,
    #
    "NORM_KERNEL": kernel_norm,
    "X_KERNEL": kernel_x,
    "Y_KERNEL": kernel_y,
    #
    "V_REG_S": 0.1,
    #
    "SIGM_POS_WEIGHTS_X": 0.25,
    "SIGM_POS_WEIGHTS_Y": 0.25,
    #
    "OUTPUT_SCALE": 1.77,
    "TAU_MEM_OUT": 50.0,
    "FILT_SCALE_OUT": 1.56,
    "FILT_BIAS_OUT": -2.53,
}
