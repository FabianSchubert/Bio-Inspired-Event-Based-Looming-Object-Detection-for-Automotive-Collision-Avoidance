import numpy as np

from .settings import base_fold_results, base_fold_input_data

import os

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps

import seaborn as sns

from sklearn.metrics import precision_recall_curve

plt.style.use(
    "https://raw.githubusercontent.com/FabianSchubert/mpl_style/main/custom_style.mplstyle"
)

models = ["LGMD", "EMD"]

vehicle_classes = ["cars", "two_wheel", "trucks"]

n_tiles = [2, 3, 4]

MIN_MIN_REACTION_TIME_MS = 500.0
MAX_MIN_REACTION_TIME_MS = 2000.0

MAX_REACTION_TIME_MS = 8000.0

N_SUBDIV = 3
N_TILES_DISCARD_TOP = 2
N_TILES_DISCARD_BOTTOM = 1


def calc_pr_curve(
    vehicle_classes,
    model,
    n_tile,
    min_reaction_time_before_collision_ms,
    max_reaction_time_before_collision_ms,
    return_f1=False,
    n_tiles_discard_top=0,
    n_tiles_discard_bottom=0,
):
    if not isinstance(vehicle_classes, list):
        vehicle_classes = [vehicle_classes]

    df_stats_response = pd.DataFrame(
        columns=[
            "vehicle_class",
            "model",
            "n_tiles",
            "example",
            "correct response",
            "score",
        ]
    )

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
                    (rec_t <= -min_reaction_time_before_collision_ms)
                    * (rec_t >= -max_reaction_time_before_collision_ms),
                ]

                v_out_filt_flat = v_out_filt.flatten()

                """
                # for each threshold value in V_OUT_THRESHOLD_EMD, calculate
                # if v_out_filt_flat is anywhere above the threshold and store
                # the results in a boolean array
                if model == "EMD":
                    th_arr = V_OUT_THRESHOLD_EMD
                else:
                    th_arr = V_OUT_THRESHOLD_LGMD

                above_threshold = np.array(
                    [np.any(v_out_filt_flat >= threshold) for threshold in th_arr]
                )

                if is_baseline:
                    response_type = [
                        "FP" if resp else "TN" for resp in above_threshold
                    ]
                else:
                    response_type = [
                        "TP" if resp else "FN" for resp in above_threshold
                    ]

                new_data = pd.DataFrame(
                    {
                        "vehicle_class": vehicle_class,
                        "model": model,
                        "n_tiles": n_tile,
                        "example": i,
                        "threshold": th_arr,
                        "response type": response_type,
                    }
                )"""

                new_data = pd.DataFrame(
                    {
                        "vehicle_class": vehicle_class,
                        "model": model,
                        "n_tiles": n_tile,
                        "example": i,
                        "correct response": int(not is_baseline),
                        "score": np.max(v_out_filt_flat),
                    },
                    index=[0],
                )

                df_stats_response = pd.concat(
                    [
                        df_stats_response,
                        new_data,
                    ],
                    ignore_index=True,
                )

    df_stats_response["n_tiles"] = df_stats_response["n_tiles"].astype(int)
    df_stats_response["example"] = df_stats_response["example"].astype(int)
    df_stats_response["correct response"] = df_stats_response[
        "correct response"
    ].astype(int)

    pr, rc, th = precision_recall_curve(
        df_stats_response["correct response"], df_stats_response["score"]
    )

    if return_f1:
        f1 = 2 * pr * rc / (pr + rc)

        return pr, rc, th, f1

    else:
        return pr, rc, th


fig, ax = plt.subplots(2, 3, figsize=(9, 5))

vhc = "cars"


for min_rt in np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, 20):
    col = colormaps["winter"](
        (min_rt - MIN_MIN_REACTION_TIME_MS)
        / (MAX_MIN_REACTION_TIME_MS - MIN_MIN_REACTION_TIME_MS)
    )

    pr, rc, th, f1 = calc_pr_curve(
        vhc,  # vehicle_classes,
        "EMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        return_f1=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    ax[0, 0].plot(th, rc[:-1], "-", c=col)
    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 0].set_ylabel("Recall")
    ax[0, 0].set_xlabel("Threshold")
    ax[0, 0].set_title("EMD")

    ax[0, 1].plot(th, pr[:-1], "-", c=col)
    ax[0, 1].set_ylim(-0.1, 1.1)
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_xlabel("Threshold")
    ax[0, 1].set_title("EMD")

    ax[0, 2].plot(th, f1[:-1], "-", c=col)
    ax[0, 2].set_ylim(-0.1, 1.1)
    ax[0, 2].set_ylabel("F1 Score")
    ax[0, 2].set_xlabel("Threshold")
    ax[0, 2].set_title("EMD")

    pr, rc, th, f1 = calc_pr_curve(
        vhc,
        "LGMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        return_f1=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    ax[1, 0].plot(th, rc[:-1], "-", c=col)
    ax[1, 0].set_ylim(-0.1, 1.1)
    ax[1, 0].set_ylabel("Recall")
    ax[1, 0].set_xlabel("Threshold")
    ax[1, 0].set_title("LGMD")

    ax[1, 1].plot(th, pr[:-1], "-", c=col)
    ax[1, 1].set_ylim(-0.1, 1.1)
    ax[1, 1].set_ylabel("Precision")
    ax[1, 1].set_xlabel("Threshold")
    ax[1, 1].set_title("LGMD")

    ax[1, 2].plot(th, f1[:-1], "-", c=col)
    ax[1, 2].set_ylim(-0.1, 1.1)
    ax[1, 2].set_ylabel("F1 Score")
    ax[1, 2].set_xlabel("Threshold")
    ax[1, 2].set_title("LGMD")

fig.tight_layout()

plt.show()


import ipdb

ipdb.set_trace()
