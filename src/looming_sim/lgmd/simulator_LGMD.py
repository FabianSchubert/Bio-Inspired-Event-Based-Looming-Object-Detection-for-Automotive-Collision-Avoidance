import numpy as np

import os

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_lgmd import (
    lgmd_neuron,
    threshold_exp_curr,
)

from .network_settings import params

from ..models_generic import (
    bitmask_array_current_source,
    p_neuron,
    lif_neuron,
    one_one_with_boundary,
)

from ..simulator_base import Base_model

from src.utils import convert_spk_id_to_evt_array


class LGMD_model(Base_model):
    def define_network(self, p):
        P_params = {
            "tau_m": p["TAU_MEM_P"],
            "tau_i": p["TAU_I_P"],
            "V_thresh": p["V_THRESH_P"],
            "V_reset": p["V_RESET_P"],
        }
        self.P_inivars = {"V": 0.0, "VI": 0.0}

        # input current sources (spike source array of DVS events)
        input_params = {"unit_amplitude": p["INPUT_EVENT_CURRENT"]}

        self.input_inivars = {"nt": self.nt_max, "pop_size": self.n_input}

        # S neurons
        half_wd = (p["KERNEL_WIDTH"] - 1) // 2  # kernel widths need to be odd
        half_ht = (p["KERNEL_HEIGHT"] - 1) // 2  # kernel heights need to be odd
        self.S_width = self.tile_width - 2 * half_wd
        self.S_height = self.tile_height - 2 * half_ht
        self.n_S = (self.S_width) * (self.S_height)
        S_params = {
            "tau_m": p["TAU_MEM_S"],
            "V_thresh": p["V_THRESH_S"],
            "V_reset": p["V_RESET_S"],
        }
        self.S_inivars = {"V": 0.0, "VI": 0.0}

        # LGMD neuron
        LGMD_params = {
            "tau_m": p["TAU_MEM_LGMD"],
            "V_thresh": p["V_THRESH_LGMD"],
            "V_reset": p["V_RESET_LGMD"],
            "scale_i_in": p["SCALE_I_IN_LGMD"],
        }
        self.LGMD_inivars = {"V": 0.0, "VI": 0.0}

        """
        ---------------------------------------------------------------------------
        synapse populations
        ----------------------------------------------------------------------------
        """

        # excitatory input to S
        iniconn_params = {
            "in_ht": self.tile_height,
            "in_wd": self.tile_width,
            "out_yoff": p["KERNEL_HEIGHT"] // 2,
            "out_xoff": p["KERNEL_WIDTH"] // 2,
        }

        # inhibitory input to S
        self.in_S_I_inivars = {
            "g": p["KERNEL_G"] * p["SCALE_KERNEL_G"],
            "d": (p["KERNEL_D"] * p["SCALE_KERNEL_D"]).astype("int"),
        }

        self.I_kernel_params = {
            "conv_kh": p["KERNEL_HEIGHT"],
            "conv_kw": p["KERNEL_WIDTH"],
            "conv_ih": self.tile_height,
            "conv_iw": self.tile_width,
            "conv_ic": 1,
            "conv_oh": self.S_height,
            "conv_ow": self.S_width,
            "conv_oc": 1,
        }

        self.in_S_I_iniconn = genn_model.init_toeplitz_connectivity(
            "Conv2D", self.I_kernel_params
        )

        # input to LGMD inhibition (subsuming F neuron)
        in_LGMD_params = {
            "tau": p["TAU_IN_LGMD"],
            "threshold": p["THRESH_IN_LGMD"],
        }

        self.P = []
        self.input = []
        self.S = []
        self.LGMD = []

        self.in_S_E = []
        self.in_S_I = []
        self.S_LGMD = []
        self.in_LGMD = []

        for i in range(self.n_tiles_y):
            for j in range(self.n_tiles_x):
                self.P.append(
                    self.model.add_neuron_population(
                        f"P_{i}_{j}", self.n_input, p_neuron, P_params, self.P_inivars
                    )
                )

                self.input.append(
                    self.model.add_current_source(
                        f"input_{i}_{j}",
                        bitmask_array_current_source,
                        self.P[-1],
                        input_params,
                        self.input_inivars,
                    )
                )

                self.input[-1].set_extra_global_param(
                    "spikeBitmask",
                    np.packbits(
                        np.zeros(
                            (self.nt_max, 32 * int(np.ceil(self.n_input / 32))),
                            dtype=np.uint8,
                        ),
                        axis=1,
                        bitorder="little",
                    )
                    .flatten()
                    .view(dtype=np.uint32),
                )

                self.input[-1].set_extra_global_param(
                    "polarityBitmask",
                    np.packbits(
                        np.zeros(
                            (self.nt_max, 32 * int(np.ceil(self.n_input / 32))),
                            dtype=np.uint8,
                        ),
                        axis=1,
                        bitorder="little",
                    )
                    .flatten()
                    .view(dtype=np.uint32),
                )

                self.S.append(
                    self.model.add_neuron_population(
                        f"S_{i}_{j}", self.n_S, lif_neuron, S_params, self.S_inivars
                    )
                )

                # neuron populations

                self.LGMD.append(
                    self.model.add_neuron_population(
                        f"LGMD_{i}_{j}", 1, lgmd_neuron, LGMD_params, self.LGMD_inivars
                    )
                )

                self.in_S_E.append(
                    self.model.add_synapse_population(
                        f"in_S_E_{i}_{j}",
                        "SPARSE_GLOBALG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        {"g": p["W_IN_S_E"]},
                        {},
                        {},
                        "ExpCurr",
                        {"tau": p["TAU_SYN_IN_S_E"]},
                        {},
                        genn_model.init_connectivity(
                            one_one_with_boundary, iniconn_params
                        ),
                    )
                )

                self.in_S_I.append(
                    self.model.add_synapse_population(
                        f"in_S_I_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulseDendriticDelay",
                        {},
                        self.in_S_I_inivars,
                        {},
                        {},
                        "ExpCurr",
                        {"tau": p["TAU_SYN_IN_S_I"]},
                        {},
                        self.in_S_I_iniconn,
                    )
                )

                self.in_S_I[-1].pop.set_max_dendritic_delay_timesteps(
                    int(self.in_S_I_inivars["d"].max() + 1)
                )

                self.S_LGMD.append(
                    self.model.add_synapse_population(
                        f"S_LGMD_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.LGMD[-1],
                        "StaticPulse",
                        {},
                        {"g": p["W_S_LGMD"]},
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.in_LGMD.append(
                    self.model.add_synapse_population(
                        f"in_LGMD_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        int(p["SYN_DELAY_IN_LGMD"]),
                        self.P[-1],
                        self.LGMD[-1],
                        "StaticPulse",
                        {},
                        {"g": p["W_IN_LGMD"]},
                        {},
                        {},
                        threshold_exp_curr,
                        in_LGMD_params,
                        {},
                    )
                )

                self.in_LGMD[-1].ps_target_var = "Isyn_i"


def run_LGMD_sim(
    evt_file,
    save_fold,
    t_end=None,
    p=params.copy(),
    results_filename="results.npz",
    custom_params={},
    measure_sim_speed=False,
):
    print("Running LGMD simulation")
    p["REC_SPIKES"] = ["P", "S", "LGMD"]

    evts = np.load(evt_file)

    if t_end is None:
        t_end = evts["t"][-1]

    p["NT_MAX"] = int(t_end / p["DT_MS"]) + 1

    p.update(custom_params)

    network = LGMD_model(p)

    network.load_input_data_from_file(evt_file)
    network.push_input_data_to_device()

    if not measure_sim_speed:
        rec_neurons = [("S", "V"), ("P", "V"), ("LGMD", "V")]
    else:
        rec_neurons = []
    rec_dt = 10.0

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = network.run_model(
        0.0, t_end, rec_neurons=rec_neurons, rec_timestep=rec_dt, measure_sim_speed=measure_sim_speed
    )

    if not measure_sim_speed:
        v_s = []
        v_out = []
        sp_p = []
        sp_s = []
        sp_out = []
        for i in range(network.n_tiles_y):
            v_s.append([])
            v_out.append([])
            sp_p.append([])
            sp_s.append([])
            sp_out.append([])
            for j in range(network.n_tiles_x):
                v_s[-1].append(
                    np.reshape(
                        rec_vars_n[f"VS_{i}_{j}"],
                        (-1, network.S_height, network.S_width),
                    )
                )
                v_out[-1].append(rec_vars_n[f"VLGMD_{i}_{j}"].flatten())

                sp_p[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"P_{i}_{j}"],
                        spike_t[f"P_{i}_{j}"],
                        network.tile_width,
                        network.tile_height,
                    )
                )
                sp_p[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"S_{i}_{j}"],
                        spike_t[f"S_{i}_{j}"],
                        network.S_width,
                        network.S_height,
                    )
                )
                sp_out[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"LGMD_{i}_{j}"],
                        spike_t[f"LGMD_{i}_{j}"],
                        1,
                        1,
                    )
                )

        if (network.n_tiles_x == 1) and (network.n_tiles_y == 1):
            sp_p = np.array(sp_p, dtype=sp_p[0][0].dtype)
            sp_s = np.array(sp_s, dtype=sp_s[0][0].dtype)
            sp_out = np.array(sp_out, dtype=sp_out[0][0].dtype)
        else:
            sp_p = np.array(sp_p, dtype=object)
            sp_s = np.array(sp_s, dtype=object)
            sp_out = np.array(sp_out, dtype=object)

        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.savez(
            os.path.join(save_fold, results_filename),
            v_s=v_s,
            v_out=v_out,
            rec_n_t=rec_n_t,
            sp_p=sp_p,
            sp_s=sp_s,
            sp_out=sp_out,
        )

    network.end()

    network.free_input_egp()
    network.model.unload()

    network.model = None

    network = None

    spike_t = None
    spike_ID = None
    rec_vars_n = None
    rec_n_t = None
    rec_vars_s = None
    rec_s_t = None

    del network
