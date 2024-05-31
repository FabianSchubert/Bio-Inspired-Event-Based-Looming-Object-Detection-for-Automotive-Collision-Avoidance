import os

from src.looming_sim.lgmd.simulator_LGMD import run_LGMD_sim
from src.looming_sim.emd.simulator_EMD import run_EMD_sim

#from .settings import base_fold_input_data, base_fold_results, base_fold_input_data_turn, base_fold_results_turn

#from .settings import base_fold_input_data_front_brake as base_fold_input_data, base_fold_results_front_brake as base_fold_results
from .settings import base_fold_input_data_turn_front_brake as base_fold_input_data, base_fold_results_turn_front_brake as base_fold_results

run_sim = {"LGMD": run_LGMD_sim, "EMD": run_EMD_sim}
#run_sim = {"EMD": run_EMD_sim}

vehicle_classes = ["cars", "two_wheel", "trucks"]

n_tiles = [2, 3, 4]

MEASURE_SIM_SPEED = False

for vehicle_class in vehicle_classes:
    vehicle_class_fold = os.path.join(base_fold_input_data, vehicle_class + "/")

    n_examples = len(
        [
            evtfl
            for evtfl in os.listdir(vehicle_class_fold)
            if evtfl.startswith("example")
        ]
    )

    for model in run_sim.keys():
        _sim = run_sim[model]
        for k in range(n_examples):
            for n_tile in n_tiles:
                results_fold = os.path.join(
                    base_fold_results,
                    vehicle_class,
                    model,
                    f"{n_tile}_tiles",
                    f"example_{k}",
                )

                evt_file = os.path.join(vehicle_class_fold, f"example_{k}/events.npy")

                _sim(
                    evt_file,
                    results_fold,
                    results_filename="results.npz",
                    custom_params={"N_SUBDIV_X": n_tile, "N_SUBDIV_Y": n_tile},
                    measure_sim_speed=MEASURE_SIM_SPEED,
                )

                evt_file = os.path.join(
                    vehicle_class_fold, f"example_{k}/events_baseline.npy"
                )

                _sim(
                    evt_file,
                    results_fold,
                    results_filename="results_baseline.npz",
                    custom_params={"N_SUBDIV_X": n_tile, "N_SUBDIV_Y": n_tile},
                    measure_sim_speed=MEASURE_SIM_SPEED,
                )

        _sim = None
