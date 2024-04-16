import numpy as np

import os

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_emd import (
    p_neuron,
    s_neuron,
    out_neuron,
    cont_wu,
)

from .network_settings import params

from ..models_generic import bitmask_array_current_source

from ..simulator_base import Base_model

from src.utils import convert_spk_id_to_evt_array


class EMD_model(Base_model):
    def define_network(self, p):
        P_params = {}
        self.P_inivars = {"V": 0.0}

        # input current sources (spike source array of DVS events)
        input_params = {"unit_amplitude": 1.0}

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
            "tau_in": p["TAU_IN_S"],
            "v_norm": p["V_NORM_S"],
        }

        self.S_inivars = {
            "dx": 0.0,
            "dy": 0.0,
            "dt": 0.0,
            "vt": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "V": 0.0,
        }

        xs, ys = np.meshgrid(
            np.arange(self.S_width) - self.S_width // 2,
            np.arange(self.S_height) - self.S_height // 2,
        )
        dnorm = np.sqrt(xs**2.0 + ys**2.0) + p["POS_NORM_REG_S"]
        xsnorm = xs / (dnorm**2.0)
        ysnorm = ys / (dnorm**2.0)

        self.OUT_params = {
            "g_filt_bias": p["G_FILT_BIAS_OUT"],
            "g_filt_scale": p["G_FILT_SCALE_OUT"],
            "output_scale": p["OUTPUT_SCALE"],
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
            "conv_ih": self.tile_height,
            "conv_iw": self.tile_width,
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
                        f"S_{i}_{j}", self.n_S, s_neuron, self.S_params, self.S_inivars
                    )
                )

                self.S[-1].set_extra_global_param("xnorm", xsnorm.flatten())
                self.S[-1].set_extra_global_param("ynorm", ysnorm.flatten())

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


def run_EMD_sim(
    evt_file,
    save_fold,
    t_end=None,
    p=params.copy(),
    results_filename="results.npz",
    custom_params={},
):
    p["REC_SPIKES"] = ["P", "S", "OUT"]

    evts = np.load(evt_file)

    if t_end is None:
        t_end = evts["t"][-1]

    p["NT_MAX"] = int(t_end / p["DT_MS"]) + 1

    p.update(custom_params)

    network = EMD_model(p)

    network.load_input_data_from_file(evt_file)
    network.push_input_data_to_device()

    rec_neurons = [("S", "V"), ("P", "V"), ("OUT", "V")]
    rec_dt = 10.0

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = network.run_model(
        0.0, t_end, rec_neurons=rec_neurons, rec_timestep=rec_dt
    )

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
            v_out[-1].append(rec_vars_n[f"VOUT_{i}_{j}"].flatten())

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
                    spike_ID[f"OUT_{i}_{j}"],
                    spike_t[f"OUT_{i}_{j}"],
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
