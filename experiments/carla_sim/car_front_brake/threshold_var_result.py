import numpy as np

from .settings import base_fold_results, base_fold_input_data

from src.config import ROOT_DIR

import os

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps

import seaborn as sns

from sklearn.metrics import precision_recall_curve

from src.classifier.utils.spike_to_img import gen_img_arr

import torch
from torchvision.models import resnet18
from torch.nn import Linear, Conv2d

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

WIDTH, HEIGHT = 304, 240

rsn = resnet18(pretrained=False)
rsn.fc = Linear(512, 3)
rsn.conv1 = Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rsn.to(device)

rsn.load_state_dict(torch.load(os.path.join(ROOT_DIR, "../resnet18_atis.pt")))


def calc_pr_curve_with_classifier(
    vehicle_classes,
    model,
    n_tile,
    min_reaction_time_before_collision_ms,
    max_reaction_time_before_collision_ms,
    th_range,
    delta_t_ms=100.0,
    return_f1=False,
    n_tiles_discard_top=0,
    n_tiles_discard_bottom=0,
):
    if not isinstance(vehicle_classes, list):
        vehicle_classes = [vehicle_classes]

    responses = []

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

                rec_t_original = results_data["rec_n_t"]
                rec_t = results_data["rec_n_t"] - collision_time

                v_out = results_data["v_out"]
                v_out_filt = v_out[
                    n_tiles_discard_top : v_out.shape[0] - n_tiles_discard_bottom,
                    :,
                    (rec_t <= -min_reaction_time_before_collision_ms)
                    * (rec_t >= -max_reaction_time_before_collision_ms),
                ]

                responses.append(
                    {
                        "vehicle_class": vehicle_class,
                        "model": model,
                        "n_tiles": n_tile,
                        "example": i,
                        "correct_response": int(not is_baseline),
                        "response": v_out_filt,
                        "t": rec_t[
                            (rec_t <= -min_reaction_time_before_collision_ms)
                            * (rec_t >= -max_reaction_time_before_collision_ms)
                        ],
                        "t_original": rec_t_original[
                            (rec_t <= -min_reaction_time_before_collision_ms)
                            * (rec_t >= -max_reaction_time_before_collision_ms)
                        ],
                    }
                )

    pr = []
    rc = []
    acc = []

    true_label = [sample["correct_response"] for sample in responses]

    for i, th in enumerate(th_range):
        predicted_labels = []

        for response in responses:
            v_out_bin = 1.0 * (response["response"] >= th)
            if np.sum(v_out_bin) == 0:
                predicted_labels.append(0)
            else:
                pos_resp_pos = np.where(v_out_bin)
                idx_earliest = np.argmin(pos_resp_pos[2])
                idx_y = pos_resp_pos[0][idx_earliest] + n_tiles_discard_top
                idx_x = pos_resp_pos[1][idx_earliest]
                idx_t = pos_resp_pos[2][idx_earliest]

                t_ms = response["t_original"][idx_t]

                # load in original event data
                evt_data_fold = os.path.join(
                    base_fold_input_data,
                    response["vehicle_class"],
                    f"example_{response['example']}",
                )

                evt_data = np.load(
                    os.path.join(evt_data_fold, "events.npy"), allow_pickle=True
                )

                xleft = idx_x * WIDTH // n_tile
                xright = (idx_x + 1) * WIDTH // n_tile
                ytop = idx_y * HEIGHT // n_tile
                ybottom = (idx_y + 1) * HEIGHT // n_tile

                evt_data_filt = evt_data[
                    (evt_data["t"] >= t_ms - delta_t_ms)
                    * (evt_data["t"] <= t_ms)
                    * (evt_data["x"] >= xleft)
                    * (evt_data["x"] <= xright)
                    * (evt_data["y"] >= ytop)
                    * (evt_data["y"] <= ybottom)
                ].copy()
                evt_data_filt["x"] -= xleft
                evt_data_filt["y"] -= ytop
                evt_data_filt["t"] -= int(t_ms - delta_t_ms)

                img_arr = gen_img_arr(evt_data_filt, WIDTH // n_tile, HEIGHT // n_tile, dt_microsecs=delta_t_ms)

                import pdb; pdb.set_trace()



def calc_pr_curve(
    vehicle_classes,
    model,
    n_tile,
    min_reaction_time_before_collision_ms,
    max_reaction_time_before_collision_ms,
    return_f1=False,
    return_accuracy=False,
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

    results = pr, rc, th

    if return_accuracy:
        acc = []
        for _th in th:
            acc.append(
                np.mean(
                    df_stats_response["correct response"]
                    == (df_stats_response["score"] >= _th)
                )
            )
        results.append(acc)

    if return_f1:
        f1 = 2 * pr * rc / (pr + rc)
        results.append(f1)

    return results


fig, ax = plt.subplots(2, 3, figsize=(9, 5))

vhc = "cars"

calc_pr_curve_with_classifier(
    vhc,
    "EMD",
    N_SUBDIV,
    MIN_MIN_REACTION_TIME_MS,
    MAX_REACTION_TIME_MS,
    np.linspace(0, 8e-8, 100),
)


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
