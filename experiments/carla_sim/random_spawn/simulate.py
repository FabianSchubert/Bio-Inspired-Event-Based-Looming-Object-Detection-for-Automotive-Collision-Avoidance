import os

import numpy as np

# from src.looming_sim.lgmd.simulator_LGMD import run_LGMD_sim
from src.looming_sim.emd.simulator_EMD import run_EMD_sim

# from src.looming_sim.lgmd.network_settings import params as params_lgmd
from src.looming_sim.emd.network_settings import params as params_emd

from .settings import (
    base_fold_results,
    base_fold_input_data,
    base_fold_input_data_noisy,
    base_fold_results_noisy,
)

DATA_TYPE = ["noisy"]

run_sim = {"EMD": run_EMD_sim}

# params_lgmd = params_lgmd.copy()
params_emd = params_emd.copy()

# params_lgmd["DT_MS"] = 10.0
params_emd["DT_MS"] = 10.0

params = {"EMD": params_emd}

n_subdiv = [2]

# get the samples that were used in the optimisation process
samples_pytorch = np.load(os.path.join(base_fold_results, "../samples.npz"))

# exclude training and validation samples
# exclude_examples = list(samples_pytorch["train"]) + list(
#    samples_pytorch["val"]
# )

examples = list(samples_pytorch["test"])

# use all other samples.
# examples = [ex for ex in os.listdir(base_fold_input_data) if ex not in exclude_examples]

for data_type in DATA_TYPE:
    if data_type == "noisy":
        bf = base_fold_input_data_noisy
        bf_r = base_fold_results_noisy
    else:
        bf = base_fold_input_data
        bf_r = base_fold_results

    for ex in examples:
        evt_file = os.path.join(bf, ex, "events.npy")
        for sim_name, sim in run_sim.items():
            for n_subd in n_subdiv:
                results_fold = os.path.join(
                    bf_r, sim_name, str(n_subd * 2 - 1) + "_tiles"
                )

                sim(
                    evt_file,
                    results_fold,
                    results_filename=ex + ".npz",
                    custom_params={"N_SUBDIV_X": n_subd, "N_SUBDIV_Y": n_subd},
                    measure_sim_speed=False,
                    p=params[sim_name],
                )
        sim = None
