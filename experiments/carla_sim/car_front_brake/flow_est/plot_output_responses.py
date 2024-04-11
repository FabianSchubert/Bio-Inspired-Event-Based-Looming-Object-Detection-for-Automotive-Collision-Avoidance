import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

from itertools import product

# plt.style.use(
#    "https://github.com/FabianSchubert/mpl_style/raw/main/custom_style.mplstyle"
# )

base_fold = os.path.join(
    os.path.dirname(__file__),
    "../../../../data/experiments/carla_sim/car_front_brake/",
)

base_fold_event_data = os.path.join(
    os.path.dirname(__file__),
    "../../../../data/carla_sim/car_front_brake/",
)

clm = {
    "t": "t",
    "exid": "Example ID",
    "tly": "Tile ID y",
    "tlx": "Tile ID x",
    "vs": "V S",
    "vout": "V Out",
}

event_metadata = np.load(os.path.join(base_fold_event_data, "sim_data.npz"))
collision_time = event_metadata["collision_time"]

data = pd.DataFrame(columns=clm.values())

for k, res_fold in enumerate(os.listdir(base_fold)):
    res = np.load(os.path.join(base_fold, res_fold, "flow_est/results.npz"))

    v_s = res["v_s"]
    v_out = res["v_out"]
    t_ax = res["rec_n_t"]

    tiles_y = len(v_out)
    tiles_x = len(v_out[0])

    for i in range(tiles_y):
        for j in range(tiles_x):
            _data_tile = pd.DataFrame(
                {
                    clm["vs"]: v_s[i][j].mean(axis=(1, 2)),
                    clm["vout"]: v_out[i][j],
                    "t": t_ax,
                    clm["exid"]: k,
                    clm["tly"]: i,
                    clm["tlx"]: j,
                }
            )

            data = pd.concat([data, _data_tile], ignore_index=True)

data["t"] = data["t"].astype("float")

n_experiments = len(os.listdir(base_fold))

figs = []
axs = []

for k in range(n_experiments):
    fig, ax = plt.subplots(
        tiles_y,
        tiles_x,
        figsize=(8, 8),
    )
    figs.append(fig)
    axs.append(ax)

    for i in range(tiles_y):
        for j in range(tiles_x):
            if tiles_y * tiles_x > 1:
                _ax = ax.flatten()[i * tiles_x + j]
            else:
                _ax = ax

            sns.lineplot(
                data=data[
                    (data[clm["exid"]] == k)
                    & (data[clm["tly"]] == i)
                    & (data[clm["tlx"]] == j)
                ],
                x="t",
                y="V Out",
                ax=_ax,
            )

            _ax.axvline(collision_time, color="r", linestyle="--")

            _ax.set_ylim([-1e-2, 1e-1])

            if i != (tiles_y - 1):
                _ax.set_xlabel("")
                _ax.xaxis.set_tick_params(labelbottom=False)
                _ax.set_xticks([])
            if j != 0:
                _ax.set_ylabel("")
                _ax.yaxis.set_tick_params(labelleft=False)
                _ax.set_yticks([])

    # _ax.set_xlim([0.0, 1000.0])

    fig.suptitle(f"Example {k}")

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(top=0.95)

    fig.savefig(
        os.path.join(os.path.dirname(__file__), f"results/responses_ex{k}.png"),
        dpi=500,
    )

plt.show()
