import numpy as np

import os

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_emd import (
    p_neuron,
    g_neuron,
    u_neuron,
    out_neuron,
    cont_wu,
    create_cont_wu,
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

        self.P_G_t_kernel = p["P_G_KERNEL"]
        self.kernel_p_g_height, self.kernel_p_g_width = self.P_G_t_kernel.shape

        assert self.kernel_p_g_height % 2 != 0, "p -> g kernel height must be uneven"
        assert self.kernel_p_g_width % 2 != 0, "p -> g kernel width must be uneven"

        # just to make sure it's normalised
        self.P_G_t_kernel /= self.P_G_t_kernel.sum()

        self.P_G_t_inivars = {"g": self.P_G_t_kernel.flatten()}

        # x derivative by shifting left and right.
        self.P_G_x_kernel = np.zeros(self.P_G_t_kernel.shape)
        self.P_G_x_kernel[:, 1:] += self.P_G_t_kernel[:, :-1] / 2.0
        self.P_G_x_kernel[:, :-1] -= self.P_G_t_kernel[:, 1:] / 2.0

        self.P_G_x_inivars = {"g": self.P_G_x_kernel.flatten()}

        # same for y
        self.P_G_y_kernel = np.zeros(self.P_G_t_kernel.shape)
        self.P_G_y_kernel[1:, :] += self.P_G_t_kernel[:-1, :] / 2.0
        self.P_G_y_kernel[:-1, :] -= self.P_G_t_kernel[1:, :] / 2.0

        self.P_G_y_inivars = {"g": self.P_G_y_kernel.flatten()}

        self.kernel_p_g_half_width = (self.kernel_p_g_width - 1) // 2
        self.kernel_p_g_half_height = (self.kernel_p_g_height - 1) // 2

        self.G_width = self.tile_width - 2 * self.kernel_p_g_half_width
        self.G_height = self.tile_height - 2 * self.kernel_p_g_half_height

        self.n_G = self.G_width * self.G_height

        ######################################################################
        self.G_U_kernel = p["G_U_KERNEL"]
        self.kernel_g_u_height, self.kernel_g_u_width = self.G_U_kernel.shape

        assert self.kernel_g_u_height % 2 != 0, "g -> u kernel height must be uneven"
        assert self.kernel_g_u_width % 2 != 0, "g -> u kernel width must be uneven"

        # normalisation is not strictly necessary
        self.G_U_kernel /= self.G_U_kernel.sum()

        self.G_U_inivars = {"g": self.G_U_kernel.flatten()}

        self.kernel_g_u_half_width = (self.kernel_g_u_width - 1) // 2
        self.kernel_g_u_half_height = (self.kernel_g_u_height - 1) // 2

        self.U_width = self.G_width - 2 * self.kernel_g_u_half_width
        self.U_height = self.G_height - 2 * self.kernel_g_u_half_height

        self.n_U = self.U_width * self.U_height

        ######################################################################
        self.U_OUT_left_weights = np.zeros((self.U_height, self.U_width, 1))
        self.U_OUT_left_weights[:, : self.U_width // 2] = 1.0 / (
            self.U_height * (self.U_width // 2)
        )
        self.U_OUT_left_weights = np.reshape(self.U_OUT_left_weights, (self.n_U, 1))

        self.U_OUT_right_weights = np.zeros((self.U_height, self.U_width, 1))
        self.U_OUT_right_weights[:, self.U_width // 2 :] = 1.0 / (
            self.U_height * (self.U_width - self.U_width // 2)
        )
        self.U_OUT_right_weights = np.reshape(self.U_OUT_right_weights, (self.n_U, 1))

        ######################################################################
        self.G_params = {
            "tau_in": p["TAU_IN_G"],
        }

        self.G_inivars = {
            "dx": 0.0,
            "dy": 0.0,
            "dt": 0.0,
            "vt": 0.0,
            "dxdx": 0.0,
            "dydy": 0.0,
            "dxdy": 0.0,
            "dxdt": 0.0,
            "dydt": 0.0,
        }

        self.U_params = {
            "reg": p["REG_U"],
            "tau_m": p["TAU_MEM_U"],
        }

        self.U_inivars = {
            "ux": 0.0,
            "uy": 0.0,
            "V": 0.0,
        }

        xu, yu = np.meshgrid(
            np.arange(self.U_width) - self.U_width // 2,
            np.arange(self.U_height) - self.U_height // 2,
        )
        dnorm = np.sqrt(xu**2.0 + yu**2.0) + p["POS_NORM_REG_U"]
        xunorm = xu / (dnorm**2.0)
        yunorm = yu / (dnorm**2.0)

        self.OUT_params = {
            "g_filt_bias": p["G_FILT_BIAS_OUT"],
            "g_filt_scale": p["G_FILT_SCALE_OUT"],
            "output_scale": p["OUTPUT_SCALE"],
        }
        self.OUT_inivars = {"U_left": 0.0, "U_right": 0.0, "V": 0.0}

        """
        ---------------------------------------------------------------------------
        synapse populations
        ----------------------------------------------------------------------------
        """

        self.P_G_kernel_params = {
            "conv_kh": self.kernel_p_g_height,
            "conv_kw": self.kernel_p_g_width,
            "conv_ih": self.tile_height,
            "conv_iw": self.tile_width,
            "conv_ic": 1,
            "conv_oh": self.G_height,
            "conv_ow": self.G_width,
            "conv_oc": 1,
        }

        self.P_G_iniconn = genn_model.init_toeplitz_connectivity(
            "Conv2D", self.P_G_kernel_params
        )

        self.G_U_kernel_params = {
            "conv_kh": self.kernel_g_u_height,
            "conv_kw": self.kernel_g_u_width,
            "conv_ih": self.G_height,
            "conv_iw": self.G_width,
            "conv_ic": 1,
            "conv_oh": self.U_height,
            "conv_ow": self.U_width,
            "conv_oc": 1,
        }

        self.G_U_iniconn = genn_model.init_toeplitz_connectivity(
            "Conv2D", self.G_U_kernel_params
        )

        self.P = []
        self.input = []
        self.G = []
        self.U = []
        self.OUT = []

        self.P_G_x = []
        self.P_G_y = []
        self.P_G_t = []
        self.G_U_dxdx = []
        self.G_U_dydy = []
        self.G_U_dxdy = []
        self.G_U_dxdt = []
        self.G_U_dydt = []
        self.U_OUT_left = []
        self.U_OUT_right = []

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

                self.G.append(
                    self.model.add_neuron_population(
                        f"G_{i}_{j}", self.n_G, g_neuron, self.G_params, self.G_inivars
                    )
                )

                self.U.append(
                    self.model.add_neuron_population(
                        f"U_{i}_{j}", self.n_U, u_neuron, self.U_params, self.U_inivars
                    )
                )

                self.U[-1].set_extra_global_param("xnorm", xunorm.flatten())
                self.U[-1].set_extra_global_param("ynorm", yunorm.flatten())

                self.OUT.append(
                    self.model.add_neuron_population(
                        f"OUT_{i}_{j}", 1, out_neuron, self.OUT_params, self.OUT_inivars
                    )
                )

                # synapses

                self.P_G_x.append(
                    self.model.add_synapse_population(
                        f"P_G_x_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.G[-1],
                        "StaticPulse",
                        {},
                        self.P_G_x_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_G_iniconn,
                    )
                )

                self.P_G_x[-1].ps_target_var = "Isyn_x"

                self.P_G_y.append(
                    self.model.add_synapse_population(
                        f"P_G_y_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.G[-1],
                        "StaticPulse",
                        {},
                        self.P_G_y_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_G_iniconn,
                    )
                )

                self.P_G_y[-1].ps_target_var = "Isyn_y"

                self.P_G_t.append(
                    self.model.add_synapse_population(
                        f"P_G_t_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.G[-1],
                        "StaticPulse",
                        {},
                        self.P_G_t_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.P_G_iniconn,
                    )
                )

                self.P_G_t[-1].ps_target_var = "Isyn_t"

                '''
                self.G_U_dxdx.append(
                    self.model.add_synapse_population(
                        f"G_U_dxdx_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.G[-1],
                        self.U[-1],
                        create_cont_wu("G_U_dxdx_cont_wu", "dxdx"),
                        {},
                        self.G_U_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.G_U_iniconn,
                    )
                )

                self.G_U_dxdx[-1].ps_target_var = "Isyn_dxdx"

                self.G_U_dydy.append(
                    self.model.add_synapse_population(
                        f"G_U_dydy_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.G[-1],
                        self.U[-1],
                        create_cont_wu("G_U_dydy_cont_wu", "dydy"),
                        {},
                        self.G_U_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.G_U_iniconn,
                    )
                )

                self.G_U_dydy[-1].ps_target_var = "Isyn_dydy"

                self.G_U_dxdy.append(
                    self.model.add_synapse_population(
                        f"G_U_dxdy_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.G[-1],
                        self.U[-1],
                        create_cont_wu("G_U_dxdy_cont_wu", "dxdy"),
                        {},
                        self.G_U_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.G_U_iniconn,
                    )
                )

                self.G_U_dxdy[-1].ps_target_var = "Isyn_dxdy"

                self.G_U_dxdt.append(
                    self.model.add_synapse_population(
                        f"G_U_dxdt_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.G[-1],
                        self.U[-1],
                        create_cont_wu("G_U_dxdt_cont_wu", "dxdt"),
                        {},
                        self.G_U_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.G_U_iniconn,
                    )
                )

                self.G_U_dxdt[-1].ps_target_var = "Isyn_dxdt"
                '''
                self.G_U_dydt.append(
                    self.model.add_synapse_population(
                        f"G_U_dydt_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.G[-1],
                        self.U[-1],
                        create_cont_wu("G_U_dydt_cont_wu", "dydt"),
                        {},
                        self.G_U_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.G_U_iniconn,
                    )
                )

                self.G_U_dydt[-1].ps_target_var = "Isyn_dydt"
                
                self.U_OUT_left.append(
                    self.model.add_synapse_population(
                        f"U_OUT_left_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.U[-1],
                        self.OUT[-1],
                        cont_wu,#create_cont_wu("U_OUT_left_cont_wu", "V"),
                        {},
                        {"g": self.U_OUT_left_weights.flatten()},
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )
                

                self.U_OUT_left[-1].ps_target_var = "Isyn_left"
                

                self.U_OUT_right.append(
                    self.model.add_synapse_population(
                        f"U_OUT_right_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.U[-1],
                        self.OUT[-1],
                        cont_wu,#create_cont_wu("U_OUT_right_cont_wu", "V"),
                        {},
                        {"g": self.U_OUT_right_weights.flatten()},
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.U_OUT_right[-1].ps_target_var = "Isyn_right"


def run_EMD_sim(
    evt_file,
    save_fold,
    t_end=None,
    p=params.copy(),
    results_filename="results.npz",
    custom_params={},
    measure_sim_speed=False,
):
    print("Running EMD simulation")
    p["REC_SPIKES"] = ["P", "G", "U", "OUT"]

    evts = np.load(evt_file)

    if t_end is None:
        t_end = evts["t"][-1]

    p["NT_MAX"] = int(t_end / p["DT_MS"]) + 1

    p.update(custom_params)

    network = EMD_model(p)

    network.load_input_data_from_file(evt_file)
    network.push_input_data_to_device()

    if not measure_sim_speed:
        rec_neurons = [("U", "ux"), ("U", "uy"), ("U", "V"), ("P", "V"), ("OUT", "V")]
    else:
        rec_neurons = []
    rec_dt = 10.0

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = network.run_model(
        0.0, t_end, rec_neurons=rec_neurons, rec_timestep=rec_dt, measure_sim_speed=measure_sim_speed
    )
    if not measure_sim_speed:
        ux = []
        uy = []
        v_u = []
        v_out = []
        v_p = []
        sp_p = []
        sp_g = []
        sp_u = []
        sp_out = []
        for i in range(network.n_tiles_y):
            ux.append([])
            uy.append([])
            v_u.append([])
            v_out.append([])
            sp_p.append([])
            sp_g.append([])
            sp_u.append([])
            sp_out.append([])
            for j in range(network.n_tiles_x):
                ux[-1].append(
                    np.reshape(
                        rec_vars_n[f"uxU_{i}_{j}"],
                        (-1, network.U_height, network.U_width),
                    )
                )
                uy[-1].append(
                    np.reshape(
                        rec_vars_n[f"uyU_{i}_{j}"],
                        (-1, network.U_height, network.U_width),
                    )
                )

                v_u[-1].append(
                    np.reshape(
                        rec_vars_n[f"VU_{i}_{j}"],
                        (-1, network.U_height, network.U_width),
                    )
                )

                v_out[-1].append(rec_vars_n[f"VOUT_{i}_{j}"].flatten())

                v_p[-1].append(
                    np.reshape(
                        rec_vars_n[f"VP_{i}_{j}"],
                        (-1, network.tile_height, network.tile_width),
                    )
                )

                sp_p[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"P_{i}_{j}"],
                        spike_t[f"P_{i}_{j}"],
                        network.tile_width,
                        network.tile_height,
                    )
                )
                sp_g[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"G_{i}_{j}"],
                        spike_t[f"G_{i}_{j}"],
                        network.G_width,
                        network.G_height,
                    )
                )

                sp_u[-1].append(
                    convert_spk_id_to_evt_array(
                        spike_ID[f"U_{i}_{j}"],
                        spike_t[f"U_{i}_{j}"],
                        network.U_width,
                        network.U_height,
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
            sp_g = np.array(sp_g, dtype=sp_g[0][0].dtype)
            sp_u = np.array(sp_u, dtype=sp_u[0][0].dtype)
            sp_out = np.array(sp_out, dtype=sp_out[0][0].dtype)
        else:
            sp_p = np.array(sp_p, dtype=object)
            sp_g = np.array(sp_g, dtype=object)
            sp_u = np.array(sp_u, dtype=object)
            sp_out = np.array(sp_out, dtype=object)

        if not os.path.exists(save_fold):
            os.makedirs(save_fold)

        np.savez(
            os.path.join(save_fold, results_filename),
            ux=ux,
            uy=uy,
            v_u=v_u,
            v_p=v_p,
            v_out=v_out,
            rec_n_t=rec_n_t,
            sp_p=sp_p,
            sp_g=sp_g,
            sp_u=sp_u,
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
