import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

plt.style.use(
    "https://github.com/FabianSchubert/mpl_style/raw/main/custom_style.mplstyle"
)

import os

base_fold = os.path.join(
    os.path.dirname(__file__), "../../data/experiments/synth_looming/"
)

clm = {"t": "t", "count": "Event Count", "vel": "Velocity", "obj": "Object"}

evt_count = pd.DataFrame(columns=clm.values())

for res_fold in os.listdir(os.path.join(base_fold, "results/")):

    vel = float(res_fold.split("_mps")[0][-3:])

    obj = res_fold.split("_on_")[0]

    print(vel)

    res = np.load(os.path.join(base_fold, "results", res_fold, "results.npz"))

    vs = res["vs"]
    s_spikes = res["evts_s"]

    df_s = pd.DataFrame(s_spikes)

    _evt_count = df_s.groupby("t").sum()
    _evt_count = _evt_count.sort_index()
    _evt_count = _evt_count[_evt_count["p"] < 50000.0]
    _evt_count[clm["t"]] = _evt_count.index
    _evt_count[clm["count"]] = _evt_count["p"]
    _evt_count[clm["vel"]] = vel
    _evt_count[clm["obj"]] = obj
    _evt_count.drop(labels=["x", "y", "p"], axis=1, inplace=True)

    evt_count = pd.concat([evt_count, _evt_count], ignore_index=True)

    # ax.plot(evt_count.index, evt_count["p"])

objects = evt_count[clm["obj"]].unique()

evt_count["t"] = evt_count["t"].astype("float")
evt_count[clm["count"]] = evt_count[clm["count"]].astype("float")
evt_count["t"] = 10.0 * (evt_count["t"] // 10.0)

evt_grouped_avg = evt_count.groupby(["t", clm["vel"], clm["obj"]]).mean()

evt_count_avg = evt_grouped_avg.reset_index()
evt_count_avg["Object Screen Size"] = 1.0 / (
    evt_count_avg[clm["vel"]] * (10000.0 - evt_count_avg["t"])
)
evt_count_avg["Screen Exp. Rate"] = 1.0 / (
    evt_count_avg[clm["vel"]] * (10000.0 - evt_count_avg["t"]) ** 2.0
)

evt_count_avg["Event Count / Velocity"] = (
    evt_count_avg[clm["count"]] / evt_count_avg[clm["vel"]]
)


peak_count = evt_count_avg.loc[
    evt_count_avg.groupby([clm["vel"], clm["obj"]]).idxmax()[clm["count"]]
]

fig, ax = plt.subplots(2, 2, figsize=(7, 6))

for k, obj in enumerate(objects):
    _ax = ax[k // 2, k % 2]

    sns.lineplot(
        data=evt_count_avg[evt_count_avg[clm["obj"]] == obj],
        x="Object Screen Size",
        y="Event Count / Velocity",
        hue=clm["vel"],
        ax=ax[k // 2, k % 2],
    )

    _ax.set_title(obj)

fig.tight_layout(pad=0.1)

fig.savefig(os.path.join(base_fold, "s_responses.png"), dpi=500)

plt.show()
