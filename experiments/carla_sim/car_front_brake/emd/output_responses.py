from src.looming_sim.emd.simulator_EMD import FlowEst_model

from .network_settings import params

from src.utils import convert_spk_id_to_evt_array

import numpy as np

import os


REC_DT = 10.0

REC_NEURONS = [("S", "V"), ("P", "V"), ("OUT", "V")]

base_fold = os.path.join(
    os.path.dirname(__file__), "../../../../data/carla_sim/car_front_brake/"
)

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../../data/experiments/carla_sim/car_front_brake/",
)

vehicle_classes = ["cars", "two_wheel", "trucks"]

network = FlowEst_model(params)

for vehicle_class in vehicle_classes:
    vehicle_class_fold = os.path.join(base_fold, vehicle_class + "/")

    n_examples = len(
        [
            evtfl
            for evtfl in os.listdir(vehicle_class_fold)
            if evtfl.startswith("example")
        ]
    )

    for k in range(n_examples):
        evt_file = os.path.join(vehicle_class_fold, f"example_{k}/events.npy")
        evts = np.load(evt_file)

        t_end = evts["t"][-1]

        network.load_input_data_from_file(evt_file)
        network.push_input_data_to_device()

        spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = network.run_model(
            0.0, t_end, rec_neurons=REC_NEURONS, rec_timestep=REC_DT
        )

        v_s = []
        v_out = []
        sp_p = []
        for i in range(network.n_tiles_y):
            v_s.append([])
            v_out.append([])
            sp_p.append([])
            for j in range(network.n_tiles_x):
                v_s[-1].append(
                    np.reshape(
                        rec_vars_n[f"VS_{i}_{j}"],
                        (-1, network.S_height, network.S_width),
                    )
                )
                v_out[-1].append(rec_vars_n[f"VOUT_{i}_{j}"].flatten())

                sp_p[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"P_{i}_{j}"],
                        spike_t[f"P_{i}_{j}"],
                        network.tile_width,
                        network.tile_height,
                    )
                )

        if (network.n_tiles_x == 1) and (network.n_tiles_y == 1):
            sp_p = np.array(sp_p, dtype=sp_p[0][0].dtype)
        else:
            sp_p = np.array(sp_p, dtype=object)

        results_fold = os.path.join(
            base_fold_results, vehicle_class, f"example_{k}/flow_est/"
        )

        if not os.path.exists(results_fold):
            os.makedirs(results_fold)

        np.savez(
            os.path.join(results_fold, "results.npz"),
            v_s=v_s,
            v_out=v_out,
            rec_n_t=rec_n_t,
            sp_p=sp_p,
        )
