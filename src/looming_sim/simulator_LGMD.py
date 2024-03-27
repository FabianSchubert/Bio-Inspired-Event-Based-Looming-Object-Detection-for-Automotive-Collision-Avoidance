import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_lgmd import (
    lgmd_neuron,
    one_one_with_boundary,
    threshold_exp_curr,
)

from .models_generic import bitmask_array_current_source, lif_neuron

from .simulator_base import Base_model


class LGMD_model(Base_model):
    def define_network(self, p):
        P_params = {
            "tau_m": p["TAU_MEM_P"],
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
            "in_ht": p["INPUT_HEIGHT"],
            "in_wd": p["INPUT_WIDTH"],
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
            "conv_ih": p["INPUT_HEIGHT"],
            "conv_iw": p["INPUT_WIDTH"],
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
                        f"P_{i}_{j}", self.n_input, lif_neuron, P_params, self.P_inivars
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
