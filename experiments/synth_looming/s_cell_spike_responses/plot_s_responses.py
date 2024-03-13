import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

from itertools import product

#plt.style.use(
#    "https://github.com/FabianSchubert/mpl_style/raw/main/custom_style.mplstyle"
#)

base_fold = os.path.join(
    os.path.dirname(__file__),
    "../../../data/experiments/synth_looming/s_cell_spike_responses",
)

clm = {
    "t": "t",
    "count": "Event Count",
    "vel": "Velocity",
    "obj": "Object",
    "bg": "Background",
}

OBJ_D = 1.5
WIDTH = 304.
VIEW_ANGLE = 45.

FOC_L = WIDTH / (2.*np.tan(VIEW_ANGLE * np.pi / 360.))

evt_count = pd.DataFrame(columns=clm.values())

for res_fold in os.listdir(base_fold):
    vel = float(res_fold.split("_mps")[0][-3:])

    obj = res_fold.split("_on_")[0]

    bg = res_fold.split("_on_")[1].split("_bg_")[0]

    print(f"loading {obj} on {bg} at {vel} mps")

    res = np.load(os.path.join(base_fold, res_fold, "results.npz"))

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
    _evt_count[clm["bg"]] = bg
    _evt_count.drop(labels=["x", "y", "p"], axis=1, inplace=True)

    evt_count = pd.concat([evt_count, _evt_count], ignore_index=True)


objects = evt_count[clm["obj"]].unique()
backgrounds = evt_count[clm["bg"]].unique()

evt_count["t"] = evt_count["t"].astype("float")
evt_count[clm["count"]] = evt_count[clm["count"]].astype("float")
evt_count["t"] = 10.0 * (evt_count["t"] // 10.0)

evt_grouped_avg = evt_count.groupby(["t", clm["vel"], clm["obj"], clm["bg"]]).mean()

evt_count_avg = evt_grouped_avg.reset_index()
evt_count_avg["Object Screen Size"] = (OBJ_D * FOC_L) / (
    evt_count_avg[clm["vel"]] * (10. - evt_count_avg["t"]/1000.)
)
evt_count_avg["Screen Exp. Rate"] = (OBJ_D * FOC_L) / (
    evt_count_avg[clm["vel"]] * (1.0 - evt_count_avg["t"]/1000.) ** 2.0
)

evt_count_avg["Event Count / Velocity"] = (
    evt_count_avg[clm["count"]] / evt_count_avg[clm["vel"]]
)


peak_count = evt_count_avg.loc[
    evt_count_avg.groupby([clm["vel"], clm["obj"]]).idxmax()[clm["count"]]
]

prod_obj_bg = list(product(objects, backgrounds))

fig, ax = plt.subplots(int(np.ceil(len(prod_obj_bg) / 2)), 2, figsize=(7, 3 * int(np.ceil(len(prod_obj_bg) / 2))))


for k, (obj, bg) in enumerate(prod_obj_bg):
    _ax = ax[k // 2, k % 2]

    sns.lineplot(
        data=evt_count_avg[
            (evt_count_avg[clm["obj"]] == obj) & (evt_count_avg[clm["bg"]] == bg)
        ],
        x="t",
        y="Event Count / Velocity",
        hue=clm["vel"],
        ax=ax[k // 2, k % 2],
    )

    _ax.set_title(f"{obj} on {bg}")

fig.tight_layout(pad=0.1)

fig.savefig(os.path.join(os.path.dirname(__file__), "results/s_responses.png"), dpi=500)

plt.show()
