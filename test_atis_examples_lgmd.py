import numpy as np
import matplotlib.pyplot as plt

from src.viz import play_event_anim, play_var_anim

res = np.load(
    "data/experiments/atis_examples/example_4/lgmd/results.npz", allow_pickle=True
)

v_s = res["v_s"]
vsp = np.maximum(0.0, v_s)

sp_p = res["sp_p"]
sp_s = res["sp_s"]

y, x = 1, 1

play_event_anim(
    sp_p[y, x],
    0.0,
    sp_p[y, x]["t"].max(),
    50.0,
    sp_p[y, x]["x"].max(),
    sp_p[y, x]["y"].max(),
)

play_event_anim(
    sp_s[y, x],
    0.0,
    sp_s[y, x]["t"].max(),
    50.0,
    sp_s[y, x]["x"].max(),
    sp_s[y, x]["y"].max(),
)

play_var_anim(v_s[y, x], 0.0, v_s.shape[2], 1.0, 1.0, v_s[y, x].min(), v_s[y, x].max())

def s(x):
    return (np.tanh(2.*x)+1.)/2.

v_l = np.maximum(0.0, v_s[y, x,:, :, :v_s.shape[4]//2]).mean(axis=(1,2))
v_r = np.maximum(0.0, v_s[y, x,:, :, v_s.shape[4]//2:]).mean(axis=(1,2))

sc = 0.5e-9
b = 0.5e-9

filt = s((v_l - b)/sc) * s((v_r - b)/sc)

plt.plot(filt)
#plt.plot(filt * (v_l + v_r) * 0.5)
#plt.plot((v_l + v_r) * 0.5)
#plt.plot(v_l, '--')
#plt.plot(v_r, '--')

#plt.plot(v_s[y, x].mean(axis=(1,2)))

#play_var_anim(vsp[0, 0], 0.0, vsp.shape[2], 1.0, 1.0, vsp[0, 0].min(), vsp[0, 0].max())
#plt.plot(vsp[0,0].mean(axis=(1,2)))

plt.show()
