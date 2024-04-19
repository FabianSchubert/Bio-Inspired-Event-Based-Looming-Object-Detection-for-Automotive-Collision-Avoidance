import numpy as np

from .settings import base_fold_results, base_fold_input_data

from src.config import ROOT_DIR

import os

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps

import seaborn as sns

from sklearn.metrics import precision_recall_curve

from src.viz import gen_evt_hist

import torch
from torchvision.models import resnet18
from torch.nn import Linear, Conv2d

from tqdm import tqdm

from time import time_ns

plt.style.use(
    "https://raw.githubusercontent.com/FabianSchubert/mpl_style/main/custom_style.mplstyle"
)

models = ["LGMD", "EMD"]

vehicle_classes = ["cars", "two_wheel", "trucks"]

n_tiles = [2, 3, 4]

MIN_MIN_REACTION_TIME_MS = 500.0
MAX_MIN_REACTION_TIME_MS = 2000.0

MAX_REACTION_TIME_MS = 8000.0

N_SUBDIV = 2
N_TILES_DISCARD_TOP = 1
N_TILES_DISCARD_BOTTOM = 1

WIDTH, HEIGHT = 304, 240

rsn = resnet18(pretrained=False)
rsn.fc = Linear(512, 3)
rsn.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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
    refr_time_classifier_ms=100.0,
    return_f1=False,
    return_accuracy=False,
    n_tiles_discard_top=0,
    n_tiles_discard_bottom=0,
    measure_class_time=True,
):
    if not isinstance(vehicle_classes, list):
        vehicle_classes = [vehicle_classes]

    responses = []

    if measure_class_time:
        timing_class = np.array([])

    for vehicle_class in vehicle_classes:
        base_fold_sim_data = os.path.join(base_fold_input_data, vehicle_class)

        base_fold_results_vehicle = os.path.join(
            base_fold_results, vehicle_class, model, f"{n_tile}_tiles/"
        )

        #n_examples = len(os.listdir(base_fold_results_vehicle))
        n_examples = 3

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
    f1 = []

    true_label = [sample["correct_response"] for sample in responses]

    for i, th in tqdm(enumerate(th_range)):
        predicted_labels = []

        for response in tqdm(responses):
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

                if measure_class_time:
                        t0 = time_ns()

                img_arr = gen_evt_hist(
                    evt_data_filt, 0.0, delta_t_ms, WIDTH // n_tile, HEIGHT // n_tile
                )

                rsn.eval()
                with torch.no_grad():
                    img_torch = (
                        torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)
                    )
                    
                    pred = int(np.argmax(rsn(img_torch.float()).cpu().numpy()[0]) != 2)
                if measure_class_time:
                    t1 = time_ns()
                    timing_class = np.append(timing_class, t1 - t0)

                predicted_labels.append(pred)
        
        if measure_class_time:
            print(f"Mean classification time: {np.mean(timing_class) * 1e-6} +- {np.std(timing_class) / np.sqrt(timing_class.shape[0]) * 1e-6} ms")

        tp = np.sum(
            np.logical_and(np.array(predicted_labels) == 1, np.array(true_label) == 1)
        )
        fp = np.sum(
            np.logical_and(np.array(predicted_labels) == 1, np.array(true_label) == 0)
        )
        fn = np.sum(
            np.logical_and(np.array(predicted_labels) == 0, np.array(true_label) == 1)
        )
        tn = np.sum(
            np.logical_and(np.array(predicted_labels) == 0, np.array(true_label) == 0)
        )

        pr.append(tp / (tp + fp))
        rc.append(tp / (tp + fn))
        acc.append((tp + tn) / (tp + tn + fp + fn))
        f1.append(2 * pr[-1] * rc[-1] / (pr[-1] + rc[-1]))

    results = [pr, rc, th_range]
    if return_accuracy:
        results.append(acc)
    if return_f1:
        results.append(f1)

    return results


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

    results = [pr, rc, th]

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




vhc = vehicle_classes

n_sweep = 20

max_acc = {
    "EMD": [[], []],
    "LGMD": [[], []],
}

fig, ax = plt.subplots(2, 3, figsize=(9, 5))

for min_rt in np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep):
    col = colormaps["winter"](
        (min_rt - MIN_MIN_REACTION_TIME_MS)
        / (MAX_MIN_REACTION_TIME_MS - MIN_MIN_REACTION_TIME_MS)
    )

    pr, rc, th, acc, f1 = calc_pr_curve(
        vhc,  # vehicle_classes,
        "EMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        return_f1=True,
        return_accuracy=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    idx_max_acc = np.argmax(acc)
    max_acc["EMD"][0].append(th[idx_max_acc])
    max_acc["EMD"][1].append(acc[idx_max_acc])

    ax[0, 0].plot(th, rc[:-1], "-", c=col)
    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 0].set_ylabel("Recall")
    ax[0, 0].set_xlabel("Threshold")
    ax[0, 0].set_title("EMD", loc="left")

    ax[0, 1].plot(th, pr[:-1], "-", c=col)
    ax[0, 1].set_ylim(-0.1, 1.1)
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_xlabel("Threshold")
    # ax[0, 1].set_title("EMD")

    # ax[0, 2].plot(th, f1[:-1], "-", c=col)
    # ax[0, 2].set_ylim(-0.1, 1.1)
    # ax[0, 2].set_ylabel("F1 Score")
    # ax[0, 2].set_xlabel("Threshold")
    # ax[0, 2].set_title("EMD")

    ax[0, 2].plot(th, acc, "-", c=col)
    ax[0, 2].set_ylim(-0.1, 1.1)
    ax[0, 2].set_ylabel("Accuracy")
    ax[0, 2].set_xlabel("Threshold")
    # ax[0, 2].set_title("EMD")

    pr, rc, th, acc, f1 = calc_pr_curve(
        vhc,
        "LGMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        return_f1=True,
        return_accuracy=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    idx_max_acc = np.argmax(acc)
    max_acc["LGMD"][0].append(th[idx_max_acc])
    max_acc["LGMD"][1].append(acc[idx_max_acc])

    ax[1, 0].plot(th, rc[:-1], "-", c=col)
    ax[1, 0].set_ylim(-0.1, 1.1)
    ax[1, 0].set_ylabel("Recall")
    ax[1, 0].set_xlabel("Threshold")
    ax[1, 0].set_title("LGMD", loc="left")

    ax[1, 1].plot(th, pr[:-1], "-", c=col)
    ax[1, 1].set_ylim(-0.1, 1.1)
    ax[1, 1].set_ylabel("Precision")
    ax[1, 1].set_xlabel("Threshold")
    # ax[1, 1].set_title("LGMD")

    ax[1, 2].plot(th, acc, "-", c=col)
    ax[1, 2].set_ylim(-0.1, 1.1)
    ax[1, 2].set_ylabel("Accuracy")
    ax[1, 2].set_xlabel("Threshold")
    # ax[1, 2].set_title("LGMD")

fig.tight_layout()

fig_max_acc, ax_max_acc = plt.subplots(1, 1, figsize=(5, 4))

ax_max_acc.plot(
    np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep),
    max_acc["EMD"][1],
    "-",
    label="EMD",
)

ax_max_acc.plot(
    np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep),
    max_acc["LGMD"][1],
    "-",
    label="LGMD",
)

ax_max_acc.set_xlabel("Min reaction time before collision [ms]")
ax_max_acc.set_ylabel("Achievable Accuracy")
ax_max_acc.set_title("Max Accuracy Achievable")
ax_max_acc.legend()

ax_max_acc.set_ylim(-0.1, 1.1)

fig_max_acc.tight_layout()

plt.show()


#### with classifier

max_acc = {
    "EMD": [[], []],
    "LGMD": [[], []],
}

n_sweep_th = 15

fig, ax = plt.subplots(2, 3, figsize=(9, 5))

for min_rt in np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep):
    col = colormaps["winter"](
        (min_rt - MIN_MIN_REACTION_TIME_MS)
        / (MAX_MIN_REACTION_TIME_MS - MIN_MIN_REACTION_TIME_MS)
    )



    pr, rc, th, acc, f1 = calc_pr_curve_with_classifier(
        vhc,  # vehicle_classes,
        "EMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        np.linspace(0.0, 7e-8, n_sweep_th),
        return_f1=True,
        return_accuracy=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    idx_max_acc = np.argmax(acc)
    max_acc["EMD"][0].append(th[idx_max_acc])
    max_acc["EMD"][1].append(acc[idx_max_acc])

    ax[0, 0].plot(th, rc, "-", c=col)
    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 0].set_ylabel("Recall")
    ax[0, 0].set_xlabel("Threshold")
    ax[0, 0].set_title("EMD", loc="left")

    ax[0, 1].plot(th, pr, "-", c=col)
    ax[0, 1].set_ylim(-0.1, 1.1)
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_xlabel("Threshold")
    # ax[0, 1].set_title("EMD")

    # ax[0, 2].plot(th, f1[:-1], "-", c=col)
    # ax[0, 2].set_ylim(-0.1, 1.1)
    # ax[0, 2].set_ylabel("F1 Score")
    # ax[0, 2].set_xlabel("Threshold")
    # ax[0, 2].set_title("EMD")

    ax[0, 2].plot(th, acc, "-", c=col)
    ax[0, 2].set_ylim(-0.1, 1.1)
    ax[0, 2].set_ylabel("Accuracy")
    ax[0, 2].set_xlabel("Threshold")
    # ax[0, 2].set_title("EMD")

    pr, rc, th, acc, f1 = calc_pr_curve_with_classifier(
        vhc,  # vehicle_classes,
        "LGMD",
        N_SUBDIV,
        min_rt,
        MAX_REACTION_TIME_MS,
        np.linspace(0.0, 35.0, n_sweep_th),
        return_f1=True,
        return_accuracy=True,
        n_tiles_discard_top=N_TILES_DISCARD_TOP,
        n_tiles_discard_bottom=N_TILES_DISCARD_BOTTOM,
    )

    idx_max_acc = np.argmax(acc)
    max_acc["LGMD"][0].append(th[idx_max_acc])
    max_acc["LGMD"][1].append(acc[idx_max_acc])

    ax[1, 0].plot(th, rc, "-", c=col)
    ax[1, 0].set_ylim(-0.1, 1.1)
    ax[1, 0].set_ylabel("Recall")
    ax[1, 0].set_xlabel("Threshold")
    ax[1, 0].set_title("LGMD", loc="left")

    ax[1, 1].plot(th, pr, "-", c=col)
    ax[1, 1].set_ylim(-0.1, 1.1)
    ax[1, 1].set_ylabel("Precision")
    ax[1, 1].set_xlabel("Threshold")
    # ax[1, 1].set_title("LGMD")

    ax[1, 2].plot(th, acc, "-", c=col)
    ax[1, 2].set_ylim(-0.1, 1.1)
    ax[1, 2].set_ylabel("Accuracy")
    ax[1, 2].set_xlabel("Threshold")
    # ax[1, 2].set_title("LGMD")

fig.tight_layout()

fig_max_acc, ax_max_acc = plt.subplots(1, 1, figsize=(5, 4))

ax_max_acc.plot(
    np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep),
    max_acc["EMD"][1],
    "-",
    label="EMD",
)

ax_max_acc.plot(
    np.linspace(MIN_MIN_REACTION_TIME_MS, MAX_MIN_REACTION_TIME_MS, n_sweep),
    max_acc["LGMD"][1],
    "-",
    label="LGMD",
)

ax_max_acc.set_xlabel("Min reaction time before collision [ms]")
ax_max_acc.set_ylabel("Achievable Accuracy")
ax_max_acc.set_title("Max Accuracy Achievable")
ax_max_acc.legend()

ax_max_acc.set_ylim(-0.1, 1.1)

fig_max_acc.tight_layout()

plt.show()


import ipdb

ipdb.set_trace()
