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

clm = {
    "t": "t",
    "exid": "Example ID",
    "tly": "Tile ID y",
    "tlx": "Tile ID x",
    "vs": "V S",
    "vout": "V Out",
}

data = pd.DataFrame(columns=clm.values())

res_folds = os.listdir(base_fold)
res_folds.sort()

for k, res_fold in enumerate(res_folds):
    res = np.load(os.path.join(base_fold, res_fold, "x_y_reichardt/results.npz"))

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

fig = []
ax = []

for k in range(n_experiments):
    _fig, _ax = plt.subplots(
        tiles_y,
        tiles_x,
        figsize=(8, 8),
    )
    fig.append(_fig)
    ax.append(_ax)

    for i in range(tiles_y):
        for j in range(tiles_x):
            sns.lineplot(
                data=data[
                    (data[clm["exid"]] == k)
                    & (data[clm["tly"]] == i)
                    & (data[clm["tlx"]] == j)
                ],
                x="t",
                y="V S",
                ax=_ax[i, j],
            )

            _ax[i, j].set_ylim([-5e-8, 5e-8])

            if i != (tiles_y - 1):
                _ax[i, j].set_xlabel("")
                _ax[i, j].xaxis.set_tick_params(labelbottom=False)
                _ax[i, j].set_xticks([])
            if j != 0:
                _ax[i, j].set_ylabel("")
                _ax[i, j].yaxis.set_tick_params(labelleft=False)
                _ax[i, j].set_yticks([])

    # _ax.set_xlim([0.0, 1000.0])

    _fig.suptitle(f"Example {k}")

    _fig.tight_layout(pad=0.1)

    _fig.savefig(
        os.path.join(os.path.dirname(__file__), f"results/responses_ex{k}.png"),
        dpi=500,
    )

plt.show()
