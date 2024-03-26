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
    "../../../data/experiments/synth_looming/s_cell_spike_responses/x_y_smooth/",
)

clm = {
    "t": "t",
    "vout": "Output Voltage",
    "vs": "S Voltage",
    # "count": "Event Count",
    "vel": "Velocity",
    "obj": "Object",
    "bg": "Background",
}

OBJ_D = 1.5
WIDTH = 304.0
VIEW_ANGLE = 45.0

FOC_L = WIDTH / (2.0 * np.tan(VIEW_ANGLE * np.pi / 360.0))

data = pd.DataFrame(columns=clm.values())

for res_fold in os.listdir(base_fold):
    vel = float(res_fold.split("_mps")[0][-3:])

    obj = res_fold.split("_on_")[0]

    bg = res_fold.split("_on_")[1].split("_bg_")[0]

    print(f"loading {obj} on {bg} at {vel} mps")

    res = np.load(os.path.join(base_fold, res_fold, "results.npz"))

    vs = res["vs"]
    vout = res["vout"]
    t = res["rec_n_t"]

    _data = pd.DataFrame()

    #_data[clm["vs"]] = np.percentile(vs, 99., axis=(1,2))
    _data[clm["vs"]] = vs.mean(axis=(1,2))#np.tanh(((vs > 0.0005).sum(axis=(1,2)) - 5000.)/500.)
    #_data[clm["vs"]] = ((vs > 0.0005) * vs).sum(axis=(1,2)) /(1. + (vs > 0.0005).sum(axis=(1,2)))
    #_data[clm["vs"]] = ((np.abs(vs) > 0.0001) * vs).sum(axis=(1,2))/(10000. + (np.abs(vs) > 0.0001).sum(axis=(1,2)))
    #vth = (vs > 0.00033)
    #_data[clm["vs"]] = (vth.sum(axis=(1,2)) > 10000.) # * (vth * vs).sum(axis=(1,2)) / vth.sum(axis=(1,2))
    #_data[clm["vs"]] = np.tanh(((vs > 0.00033).sum(axis=(1,2)) - 5000.)/1000.)
    #_data[clm["vs"]] = vout
    _data[clm["vout"]] = vout
    _data[clm["t"]] = t

    _data[clm["vel"]] = vel
    _data[clm["obj"]] = obj
    _data[clm["bg"]] = bg

    data = pd.concat([data, _data], ignore_index=True)


objects = data[clm["obj"]].unique()
backgrounds = data[clm["bg"]].unique()

data["t"] = data["t"].astype("float")

data["Object Screen Size"] = (OBJ_D * FOC_L) / (
    data[clm["vel"]] * (10.0 - data["t"] / 1000.0)
)
data["Screen Exp. Rate"] = (OBJ_D * FOC_L) / (
    data[clm["vel"]] * (10.0 - data["t"] / 1000.0) ** 2.0
)

data["Output Voltage / Velocity"] = data[clm["vout"]] / data[clm["vel"]]

peak_vout = data.loc[data.groupby([clm["vel"], clm["obj"]]).idxmax()[clm["vout"]]]

prod_obj_bg = list(product(objects, backgrounds))

fig, ax = plt.subplots(
    int(np.ceil(len(prod_obj_bg) / 2)),
    2,
    figsize=(7, 3 * int(np.ceil(len(prod_obj_bg) / 2))),
)


for k, (obj, bg) in enumerate(prod_obj_bg):
    _ax = ax[k // 2, k % 2]

    sns.lineplot(
        data=data[
            (
                (data[clm["obj"]] == obj)
                & (data[clm["bg"]] == bg)
                & (data["t"] > 2000.0)
            )
        ],
        x="t",
        y="S Voltage",
        hue=clm["vel"],
        ax=ax[k // 2, k % 2],
    )

    #_ax.set_xlim([0.0, 1000.0])

    _ax.set_title(f"{obj} on {bg}")

fig.tight_layout(pad=0.1)

fig.savefig(
    os.path.join(os.path.dirname(__file__), "results/x_y_smooth/s_responses.png"),
    dpi=500,
)

plt.show()
