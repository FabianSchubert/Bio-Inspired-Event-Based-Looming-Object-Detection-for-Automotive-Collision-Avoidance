import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_x_y import (
    s_neuron,
    out_neuron,
    cont_wu,
)

from .models_generic import bitmask_array_current_source, lif_neuron

from .simulator_base import Base_model


class X_Y_model(Base_model):
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

        self.P_S_t_kernel = p["P_S_T_KERNEL"]
        self.kernel_height, self.kernel_width = self.P_S_t_kernel.shape

        assert self.kernel_height % 2 != 0, "kernel height must be uneven"
        assert self.kernel_width % 2 != 0, "kernel width must be uneven"

        # just to make sure it's normalised
        self.P_S_t_kernel /= self.P_S_t_kernel.sum()

        self.P_S_t_inivars = {"g": self.P_S_t_kernel.flatten()}

        # x derivative by shifting left and right.
        self.P_S_x_kernel = np.zeros(self.P_S_t_kernel.shape)
        self.P_S_x_kernel[:, 1:] += self.P_S_t_kernel[:, :-1] / 2.0
        self.P_S_x_kernel[:, :-1] -= self.P_S_t_kernel[:, 1:] / 2.0

        self.P_S_x_inivars = {"g": self.P_S_x_kernel.flatten()}

        # same for y
        self.P_S_y_kernel = np.zeros(self.P_S_t_kernel.shape)
        self.P_S_y_kernel[1:, :] += self.P_S_t_kernel[:-1, :] / 2.0
        self.P_S_y_kernel[:-1, :] -= self.P_S_t_kernel[1:, :] / 2.0

        self.P_S_y_inivars = {"g": self.P_S_y_kernel.flatten()}

        self.kernel_half_width = (self.kernel_width - 1) // 2
        self.kernel_half_height = (self.kernel_height - 1) // 2

        self.S_width = self.tile_width - 2 * self.kernel_half_width
        self.S_height = self.tile_height - 2 * self.kernel_half_height

        self.n_S = self.S_width * self.S_height

        self.S_OUT_left_weights = np.zeros((self.S_height, self.S_width, 1))
        self.S_OUT_left_weights[:, : self.S_width // 2] = 1.0 / (
            self.S_height * (self.S_width // 2)
        )
        self.S_OUT_left_weights = np.reshape(self.S_OUT_left_weights, (self.n_S, 1))

        self.S_OUT_right_weights = np.zeros((self.S_height, self.S_width, 1))
        self.S_OUT_right_weights[:, self.S_width // 2 :] = 1.0 / (
            self.S_height * (self.S_width - self.S_width // 2)
        )
        self.S_OUT_right_weights = np.reshape(self.S_OUT_right_weights, (self.n_S, 1))

        self.S_params = {
            "tau_m": p["TAU_MEM_S"],
            "tau_filt": p["TAU_FILT_S"],
        }
        self.S_inivars = {"Vx": 0.0, "Vy": 0.0, "Vt": 0.0, "Isyn_t_filt": 0.0, "V": 0.0}

        xs, ys = np.meshgrid(
            np.arange(self.S_width) - self.S_width // 2,
            np.arange(self.S_height) - self.S_height // 2,
        )
        dnorm = p["B_REG_S"] * (xs**2.0 + ys**2.0 + p["S_POS_NORM_REG"])

        self.OUT_params = {
            "tau_m": p["TAU_MEM_OUT"],
            "g_filt_bias": p["G_FILT_BIAS_OUT"],
            "g_filt_scale": p["G_FILT_SCALE_OUT"],
        }
        self.OUT_inivars = {"S_left": 0.0, "S_right": 0.0, "V": 0.0}

        """
        ---------------------------------------------------------------------------
        synapse populations
        ----------------------------------------------------------------------------
        """

        self.I_kernel_params = {
            "conv_kh": self.kernel_height,
            "conv_kw": self.kernel_width,
            "conv_ih": p["INPUT_HEIGHT"],
            "conv_iw": p["INPUT_WIDTH"],
            "conv_ic": 1,
            "conv_oh": self.S_height,
            "conv_ow": self.S_width,
            "conv_oc": 1,
        }

        self.P_S_iniconn = genn_model.init_toeplitz_connectivity(
            "Conv2D", self.I_kernel_params
        )

        self.P = []
        self.input = []
        self.S = []
        self.OUT = []

        self.P_S_x = []
        self.P_S_y = []
        self.P_S_t = []
        self.S_OUT_left = []
        self.S_OUT_right = []

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
                        f"S_{i}_{j}", self.n_S, s_neuron, self.S_params, self.S_inivars
                    )
                )

                self.S[-1].set_extra_global_param("x", xs.flatten())
                self.S[-1].set_extra_global_param("y", ys.flatten())
                self.S[-1].set_extra_global_param("dnorm", dnorm.flatten())

                # neuron populations

                self.OUT.append(
                    self.model.add_neuron_population(
                        f"OUT_{i}_{j}", 1, out_neuron, self.OUT_params, self.OUT_inivars
                    )
                )

                self.P_S_x.append(
                    self.model.add_synapse_population(
                        f"P_S_x_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.P_S_x_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_S_iniconn,
                    )
                )

                self.P_S_x[-1].ps_target_var = "Isyn_x"

                self.P_S_y.append(
                    self.model.add_synapse_population(
                        f"P_S_y_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.P_S_y_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_S_iniconn,
                    )
                )

                self.P_S_y[-1].ps_target_var = "Isyn_y"

                self.P_S_t.append(
                    self.model.add_synapse_population(
                        f"P_S_t_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.P_S_t_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_S_iniconn,
                    )
                )

                self.P_S_t[-1].ps_target_var = "Isyn_t"

                self.S_OUT_left.append(
                    self.model.add_synapse_population(
                        f"S_OUT_left_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.OUT[-1],
                        cont_wu,
                        {},
                        {"g": self.S_OUT_left_weights.flatten()},
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.S_OUT_left[-1].ps_target_var = "Isyn_left"

                self.S_OUT_right.append(
                    self.model.add_synapse_population(
                        f"S_OUT_right_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.OUT[-1],
                        cont_wu,
                        {},
                        {"g": self.S_OUT_right_weights.flatten()},
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.S_OUT_right[-1].ps_target_var = "Isyn_right"
