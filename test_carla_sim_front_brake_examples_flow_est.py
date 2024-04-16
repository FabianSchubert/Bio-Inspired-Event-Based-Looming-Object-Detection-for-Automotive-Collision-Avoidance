import numpy as np
import matplotlib.pyplot as plt

from src.viz import play_event_anim, play_var_anim

res = np.load(
    "data/experiments/carla_sim/car_front_brake/two_wheel/example_0/flow_est/results.npz",
    allow_pickle=True,
)

v_s = res["v_s"]
vsp = np.maximum(0.0, v_s)

sp_p = res["sp_p"]

y, x = 1, 1


play_event_anim(
    sp_p[y, x],
    0.0,
    sp_p[y, x]["t"].max(),
    50.0,
    sp_p[y, x]["x"].max(),
    sp_p[y, x]["y"].max(),
    save_frames="data/experiments/carla_sim/car_front_brake/two_wheel/example_0/flow_est/sp_p/",
    save_video="data/experiments/carla_sim/car_front_brake/two_wheel/example_0/flow_est/sp_p/center_tile.mp4",
)

play_var_anim(
    v_s[y, x],
    0.0,
    v_s.shape[2],
    1.0,
    1.0,
    v_s[y, x].min(),
    v_s[y, x].max(),
    #save_frames="data/experiments/carla_sim/car_front_brake/trucks/example_0/flow_est/v_s/",
    #save_video="data/experiments/carla_sim/car_front_brake/trucks/example_0/flow_est/v_s/center_tile.mp4",
)

# def s(x):
#    return (np.tanh(2.*x)+1.)/2.


def s(x):
    return 1.0 - np.exp(-np.maximum(0.0, x))


# v_l = np.maximum(0.0, v_s[y, x,:, :, :v_s.shape[4]//2]).mean(axis=(1,2))
# v_r = np.maximum(0.0, v_s[y, x,:, :, v_s.shape[4]//2:]).mean(axis=(1,2))

v_l = v_s[y, x, :, :, : v_s.shape[4] // 2].mean(axis=(1, 2))
v_r = v_s[y, x, :, :, v_s.shape[4] // 2 :].mean(axis=(1, 2))

# th_v = 1e-7
# v_l = ((v_s[y, x,:, :, :v_s.shape[4]//2] > th_v)*v_s[y, x,:, :, :v_s.shape[4]//2]).mean(axis=(1,2))
# v_r = ((v_s[y, x,:, :, v_s.shape[4]//2:] > th_v)*v_s[y, x,:, :, v_s.shape[4]//2:]).mean(axis=(1,2))

sc = 1e-8
b = 2.5e-8


filt = s((v_l - b) / sc) * s((v_r - b) / sc)

plt.plot(filt * (v_l + v_r) * 0.5)
plt.plot((v_l + v_r) * 0.5)
plt.plot(v_l, "--")
plt.plot(v_r, "--")

plt.ylim(top=8e-9)

# plt.plot(v_s[y, x].mean(axis=(1,2)))

# play_var_anim(vsp[0, 0], 0.0, vsp.shape[2], 1.0, 1.0, vsp[0, 0].min(), vsp[0, 0].max())
# plt.plot(vsp[0,0].mean(axis=(1,2)))

plt.show()
