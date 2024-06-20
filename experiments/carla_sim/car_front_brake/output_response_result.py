import numpy as np

#from .settings import base_fold_results, base_fold_input_data

from .settings import (
    base_fold_input_data_turn_front_brake as base_fold_input_data,
    base_fold_results_turn_front_brake as base_fold_results,
)

import os

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps

import seaborn as sns

plt.style.use(
    "https://raw.githubusercontent.com/FabianSchubert/mpl_style/main/custom_style.mplstyle"
)

plt.rcParams["figure.dpi"] = 150

models = ["LGMD", "EMD"]

vehicle_classes = ["cars", "two_wheel", "trucks"]

# n_subdiv = [2, 3, 4]


DT_BIN_MS = 100.0

CLM = {
    "tlx": "Tile ID X",
    "tly": "Tile ID Y",
    "v_out": "V Out",
    "t": "t [ms]",
    "is_correct": "Collision",
    "example_id": "Example ID",
}


def get_output_responses(
    vehicle_classes,
    model,
    n_tile,
    n_tiles_discard_top=0,
    n_tiles_discard_bottom=0,
    t_max=8000.0,
):
    if not isinstance(vehicle_classes, list):
        vehicle_classes = [vehicle_classes]

    df_response = pd.DataFrame(columns=CLM.values())

    for vehicle_class in vehicle_classes:
        base_fold_sim_data = os.path.join(base_fold_input_data, vehicle_class)

        base_fold_results_vehicle = os.path.join(
            base_fold_results, vehicle_class, model, f"{n_tile}_tiles/"
        )

        n_examples = len(os.listdir(base_fold_results_vehicle))

        for i in range(n_examples):
            results_fold = os.path.join(base_fold_results_vehicle, f"example_{i}")
            sim_data_fold = os.path.join(base_fold_sim_data, f"example_{i}")

            sim_data = np.load(
                os.path.join(sim_data_fold, "sim_data.npz"), allow_pickle=True
            )

            collision_time = sim_data["collision_time"][()]

            if collision_time is None:
                print(
                    f"Collision time is None for {vehicle_class}, {model}, {n_tile} tiles, example {i}"
                )
                continue

            for is_baseline in [True, False]:
                if is_baseline:
                    results_data = np.load(
                        os.path.join(results_fold, "results_baseline.npz"),
                        allow_pickle=True,
                    )
                else:
                    results_data = np.load(
                        os.path.join(results_fold, "results.npz"), allow_pickle=True
                    )

                rec_t = results_data["rec_n_t"] - collision_time

                v_out = results_data["v_out"]

                v_out_filt = v_out[
                    n_tiles_discard_top : v_out.shape[0] - n_tiles_discard_bottom,
                    :,
                    (rec_t <= 0.0) * (rec_t >= -t_max),
                ]

                for id_y in range(v_out_filt.shape[0]):
                    for id_x in range(v_out_filt.shape[1]):
                        new_data = pd.DataFrame(
                            {
                                CLM["tlx"]: id_x,
                                CLM["tly"]: id_y,
                                CLM["v_out"]: v_out_filt[id_y, id_x],
                                CLM["t"]: (
                                    rec_t[(rec_t <= 0.0) * (rec_t >= -t_max)]
                                    / DT_BIN_MS
                                ).astype(int)
                                * DT_BIN_MS,
                                CLM["is_correct"]: int(not is_baseline),
                                CLM["example_id"]: i,
                            }
                        )

                        df_response = pd.concat(
                            [df_response, new_data], ignore_index=True
                        )
    df_response = (
        df_response.groupby([k for k in CLM.values() if k != "V Out"])
        .agg("mean")
        .reset_index()
    )

    return df_response

N_SUBDIV = 4
N_TILES_DISCARD_TOP = 3
N_TILES_DISCARD_BOTTOM = 2

T_MAX = 8000.0

MODEL = "EMD"
VEHICLE_CLASSES = vehicle_classes

ylim = {"LGMD": 20., "EMD": 0.75e-7}

df = get_output_responses(
    VEHICLE_CLASSES,
    MODEL,
    N_SUBDIV,
    n_tiles_discard_top=N_TILES_DISCARD_TOP,
    n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    t_max=T_MAX,
)

n_tiles_y = N_SUBDIV * 2 - 1 - N_TILES_DISCARD_TOP - N_TILES_DISCARD_BOTTOM
n_tiles_x = N_SUBDIV * 2 - 1

fig, ax = plt.subplots(n_tiles_y, n_tiles_x, figsize=(3 * n_tiles_x, 2 * n_tiles_y))

for i in range(n_tiles_y):
    for j in range(n_tiles_x):
        df_filt = df[
            (df[CLM["tlx"]] == j) & (df[CLM["tly"]] == i)# & (df[CLM["is_correct"]] == 1)
        ]

        _ax = ax.flatten()[i * n_tiles_x + j]

        sns.lineplot(df_filt, x="t [ms]", y="V Out", hue="Collision", ax=_ax)

        if i != n_tiles_y - 1:
            _ax.set_xlabel("")
        if j != 0:
            _ax.set_ylabel("")

        _ax.set_ylim([-0.1 * ylim[MODEL], ylim[MODEL]])


fig.tight_layout(pad=0.1)

plt.show()

import ipdb

ipdb.set_trace()
