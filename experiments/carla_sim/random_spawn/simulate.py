import os

from src.looming_sim.lgmd.simulator_LGMD import run_LGMD_sim
from src.looming_sim.emd.simulator_EMD import run_EMD_sim

from src.looming_sim.lgmd.network_settings import params as params_lgmd
from src.looming_sim.emd.network_settings import params as params_emd

from .settings import base_fold_results, base_fold_input_data


# from .settings import base_fold_input_data, base_fold_results, base_fold_input_data_turn, base_fold_results_turn

# from .settings import base_fold_input_data_front_brake as base_fold_input_data, base_fold_results_front_brake as base_fold_results


#run_sim = {"LGMD": run_LGMD_sim, "EMD": run_EMD_sim}
run_sim = {"EMD": run_EMD_sim}

params_lgmd = params_lgmd.copy()
params_emd = params_emd.copy()

params_lgmd["DT_MS"] = 10.0
params_emd["DT_MS"] = 10.0

params = {"LGMD": params_lgmd, "EMD": params_emd}

# run_sim = {"EMD": run_EMD_sim}

#coll_type = ["cars", "pedestrians", "none"]

n_subdiv = [2]

examples = os.listdir(base_fold_input_data)

for ex in examples:
    evt_file = os.path.join(base_fold_input_data, ex, "events.npy")
    for sim_name, sim in run_sim.items():
        for n_subd in n_subdiv:
            results_fold = os.path.join(
                base_fold_results, sim_name, str(n_subd * 2 - 1) + "_tiles"
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
