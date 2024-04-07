import numpy as np
import matplotlib.pyplot as plt

from src.viz import play_event_anim, play_var_anim

res = np.load(
    "data/experiments/atis_examples/example_1/x_y/results.npz", allow_pickle=True
)

v_s = res["v_s"]
vsp = np.maximum(0.0, v_s)

sp_p = res["sp_p"]

play_event_anim(
    sp_p[0, 0],
    0.0,
    sp_p[0, 0]["t"].max(),
    50.0,
    sp_p[0, 0]["x"].max(),
    sp_p[0, 0]["y"].max(),
)

play_var_anim(v_s[0, 0], 0.0, v_s.shape[2], 1.0, 1.0, v_s[0, 0].min(), v_s[0, 0].max())
plt.plot(v_s[0,0].mean(axis=(1,2)))

#play_var_anim(vsp[0, 0], 0.0, vsp.shape[2], 1.0, 1.0, vsp[0, 0].min(), vsp[0, 0].max())
#plt.plot(vsp[0,0].mean(axis=(1,2)))

plt.show()
