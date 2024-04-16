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
    "t": "t [ms]",
    "exid": "Example ID",
    "vcl": "Vehicle Class",
    "tly": "Tile ID y",
    "tlx": "Tile ID x",
    "vs": "V S",
    "vout": "V Out",
}

vehicle_classes = ["cars", "two_wheel", "trucks"]

data = pd.DataFrame(columns=clm.values())

for vehicle_class in vehicle_classes:

    vehicle_class_fold = os.path.join(base_fold, vehicle_class + "/")

    n_examples = len([evtfl for evtfl in os.listdir(vehicle_class_fold) if evtfl.startswith("example")])

    for k in range(n_examples):
        res_fold = os.path.join(vehicle_class_fold, f"example_{k}")

        res = np.load(os.path.join(res_fold, "lgmd/results.npz"))

        metadata = np.load(os.path.join(base_fold_event_data, vehicle_class, f"example_{k}/sim_data.npz"), allow_pickle=True)
        collision_time = metadata["collision_time"][()]

        if collision_time is not None:
            collision_time = int(collision_time)
        else:
             print("Collision time is None, skipping")
             continue         

        v_s = res["v_s"]
        v_out = res["v_out"]
        t_ax = res["rec_n_t"] - collision_time

        tiles_y = len(v_out)
        tiles_x = len(v_out[0])

        for i in range(tiles_y):
            for j in range(tiles_x):
                _data_tile = pd.DataFrame(
                    {
                        clm["vs"]: v_s[i][j].mean(axis=(1, 2)),
                        clm["vout"]: v_out[i][j],
                        clm["t"]: t_ax,
                        clm["exid"]: k,
                        clm["vcl"]: vehicle_class,
                        clm["tly"]: i,
                        clm["tlx"]: j,
                    }
                )

                data = pd.concat([data, _data_tile], ignore_index=True)

data[clm["t"]] = data[clm["t"]].astype("float")

n_experiments = len(os.listdir(base_fold))

figs = []
axs = []

for vehicle_class in vehicle_classes:
    #n_experiments = data[data[clm["vcl"]] == vehicle_class][clm["exid"]].nunique()

    tiles_y = data[data[clm["vcl"]] == vehicle_class][clm["tly"]].nunique()
    tile_y_center = tiles_y // 2
    tiles_x = data[data[clm["vcl"]] == vehicle_class][clm["tlx"]].nunique()

    fig, ax = plt.subplots(
        1,#tiles_y,
        tiles_x,
        figsize=(8, 3),
    )
    figs.append(fig)
    axs.append(ax)

    #for i in range(tiles_y):
    for j in range(tiles_x):
        if tiles_x > 1:
            _ax = ax[j]
        else:
            _ax = ax

        sns.lineplot(
            data=data[
                (data[clm["vcl"]] == vehicle_class)
                & (data[clm["tly"]] == tile_y_center)
                & (data[clm["tlx"]] == j)
            ],
            x="t [ms]",
            y="V Out",
            hue="Example ID",
            ax=_ax,
        )

        _ax.axvline(0., color="r", linestyle="--")

        _ax.set_ylim([-0.2e-2, 3e-2])
        _ax.set_xlim([-3000,500])

        if i != (tiles_y - 1):
            _ax.set_xlabel("")
            _ax.xaxis.set_tick_params(labelbottom=False)
            _ax.set_xticks([])
        if j != 0:
            _ax.set_ylabel("")
            _ax.yaxis.set_tick_params(labelleft=False)
            _ax.set_yticks([])

    # _ax.set_xlim([0.0, 1000.0])

    fig.suptitle(vehicle_class.replace("_", " "))

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(top=0.9)

    fig.savefig(
        os.path.join(os.path.dirname(__file__), f"results/{vehicle_class}.png"),
        dpi=500,
    )

plt.show()
