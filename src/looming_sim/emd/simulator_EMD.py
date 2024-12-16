import numpy as np

import os

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models_emd import (
    p_neuron,
    n_neuron,
    s_neuron,
    out_neuron,
    cont_wu,
    sparse_one_to_one_snippet_with_pad,
    create_cont_wu,
)

from .network_settings import params

from ..models_generic import bitmask_array_current_source

from ..simulator_base import Base_model

from src.utils import convert_spk_id_to_evt_array

cont_wu_xdxdt = create_cont_wu("cont_wu_xdxdt", "xdxdt")
cont_wu_xdx_squ = create_cont_wu("const_wu_xdx_squ", "xdx_squ")


def weights(x: np.ndarray, y: np.ndarray, sigma_x: float, sigma_y: float):
    return np.exp(-(x**2) / (2 * sigma_x**2)) * np.exp(-(y**2) / (2 * sigma_y**2))


class EMD_model(Base_model):
    def define_network(self, p):
        P_params = {}
        N_params = {}

        _p_vars = [v.name for v in p_neuron.get_vars()]
        self.P_inivars = dict(zip(_p_vars, [0.0] * len(_p_vars)))

        _n_vars = [v.name for v in n_neuron.get_vars()]
        self.N_inivars = dict(zip(_n_vars, [0.0] * len(_n_vars)))

        # input current sources (spike source array of DVS events)
        input_params = {"unit_amplitude": 1.0}

        self.input_inivars = {"nt": self.nt_max, "pop_size": self.n_input}

        self.norm_kernel = p["NORM_KERNEL"]
        self.kernel_height, self.kernel_width = self.norm_kernel.shape

        assert self.kernel_height % 2 != 0, "kernel height must be uneven"
        assert self.kernel_width % 2 != 0, "kernel width must be uneven"

        self.kernel_half_width = (self.kernel_width - 1) // 2
        self.kernel_half_height = (self.kernel_height - 1) // 2

        self.S_width = self.tile_width - 2 * self.kernel_half_width
        self.S_height = self.tile_height - 2 * self.kernel_half_height

        self.delta_x = 2.0 / (self.S_width - 1)

        # check it's normalised
        np.testing.assert_almost_equal(self.norm_kernel.sum(), 1.0, decimal=5)

        self.PN_S_norm_inivars = {"g": self.norm_kernel.flatten()}

        # x derivative by shifting left and right.
        self.x_kernel = p["X_KERNEL"]
        self.y_kernel = p["Y_KERNEL"]

        self.PN_S_x_inivars = {"g": self.x_kernel.flatten()}
        self.PN_S_y_inivars = {"g": self.y_kernel.flatten()}

        self.PN_S_one_to_one_inivars = {"g": 1.0}

        self.n_S = self.S_width * self.S_height

        xs, ys = np.meshgrid(
            np.linspace(-1.0, 1.0, self.S_width),
            np.linspace(
                -self.S_height / self.S_width,
                self.S_height / self.S_width,
                self.S_height,
            ),
        )
        pos_weights = weights(
            xs,
            ys,
            p["SIGM_POS_WEIGHTS_X"],
            p["SIGM_POS_WEIGHTS_Y"],
        )

        self.S_OUT_left_weights = np.zeros((self.S_height, self.S_width))
        self.S_OUT_left_weights[:, : self.S_width // 2] = 1.0
        self.S_OUT_left_weights *= pos_weights
        # self.S_OUT_left_weights /= np.sum(self.S_OUT_left_weights)

        self.S_OUT_right_weights = np.zeros((self.S_height, self.S_width))
        self.S_OUT_right_weights[:, self.S_width // 2 :] = 1.0
        self.S_OUT_right_weights *= pos_weights
        # self.S_OUT_right_weights /= np.sum(self.S_OUT_right_weights)

        self.sum_x_right_weights = (self.S_OUT_right_weights * xs).sum()

        self.S_OUT_avg_weights = pos_weights.copy()
        self.S_OUT_avg_weights /= np.sum(self.S_OUT_avg_weights)

        self.S_OUT_left_inivars = {"g": self.S_OUT_left_weights.flatten()}
        self.S_OUT_right_inivars = {"g": self.S_OUT_right_weights.flatten()}
        self.S_OUT_avg_inivars = {"g": self.S_OUT_avg_weights.flatten()}

        self.S_params = {
            "v_reg": p["V_REG_S"],
        }

        _s_vars = [v.name for v in s_neuron.get_vars()]
        self.S_inivars = dict(zip(_s_vars, [0.0] * len(_s_vars)))

        self.OUT_params = {
            "output_scale": p["OUTPUT_SCALE"],
            "tau_m": p["TAU_MEM_OUT"],
            # "tau_r": p["TAU_R_OUT"],
            "filt_scale": p["FILT_SCALE_OUT"],
            "filt_bias": p["FILT_BIAS_OUT"],
            "sum_x_right_weights": self.sum_x_right_weights,
        }

        _out_vars = [v.name for v in out_neuron.get_vars()]
        self.OUT_inivars = dict(zip(_out_vars, [0.0] * len(_out_vars)))

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

        self.PN_S_kernel_iniconn = genn_model.init_toeplitz_connectivity(
            "Conv2D", self.I_kernel_params
        )

        self.PN_S_one_to_one_iniconn = genn_model.init_connectivity(
            # "FixedProbability", {"prob": 0.01}
            sparse_one_to_one_snippet_with_pad,
            {
                "pad_x": self.kernel_half_width,
                "pad_y": self.kernel_half_height,
                "width_pre": self.tile_width,
                "height_pre": self.tile_height,
            },
        )

        self.P = []
        self.N = []
        self.P_input = []
        self.N_input = []
        self.S = []
        self.OUT = []

        self.P_S_x = []
        self.P_S_y = []
        self.P_S_norm = []
        self.P_S_one_to_one = []

        self.N_S_x = []
        self.N_S_y = []
        self.N_S_norm = []
        self.N_S_one_to_one = []

        self.S_OUT_v_proj_left = []
        self.S_OUT_v_proj_right = []
        self.S_OUT_v_avg_x = []

        for i in range(self.n_tiles_y):
            for j in range(self.n_tiles_x):

                self.P.append(
                    self.model.add_neuron_population(
                        f"P_{i}_{j}", self.n_input, p_neuron, P_params, self.P_inivars
                    )
                )

                self.N.append(
                    self.model.add_neuron_population(
                        f"N_{i}_{j}", self.n_input, n_neuron, N_params, self.N_inivars
                    )
                )

                self.P_input.append(
                    self.model.add_current_source(
                        f"P_input_{i}_{j}",
                        bitmask_array_current_source,
                        self.P[-1],
                        input_params,
                        self.input_inivars,
                    )
                )

                self.N_input.append(
                    self.model.add_current_source(
                        f"N_input_{i}_{j}",
                        bitmask_array_current_source,
                        self.N[-1],
                        input_params,
                        self.input_inivars,
                    )
                )

                self.P_input[-1].set_extra_global_param(
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

                self.P_input[-1].set_extra_global_param(
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

                self.N_input[-1].set_extra_global_param(
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

                self.N_input[-1].set_extra_global_param(
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
                        self.PN_S_x_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.P_S_x[-1].ps_target_var = "Isyn_p_x"

                self.P_S_y.append(
                    self.model.add_synapse_population(
                        f"P_S_y_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_y_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.P_S_y[-1].ps_target_var = "Isyn_p_y"

                self.P_S_norm.append(
                    self.model.add_synapse_population(
                        f"P_S_norm_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_norm_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.P_S_norm[-1].ps_target_var = "Isyn_p_norm"

                self.P_S_one_to_one.append(
                    self.model.add_synapse_population(
                        f"P_S_one_to_one_{i}_{j}",
                        "SPARSE_GLOBALG",
                        NO_DELAY,
                        self.P[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_one_to_one_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_one_to_one_iniconn,
                    )
                )

                self.P_S_one_to_one[-1].ps_target_var = "Isyn_p_one_to_one"

                self.N_S_x.append(
                    self.model.add_synapse_population(
                        f"P_N_x_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.N[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_x_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.N_S_x[-1].ps_target_var = "Isyn_n_x"

                self.N_S_y.append(
                    self.model.add_synapse_population(
                        f"N_S_y_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.N[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_y_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.N_S_y[-1].ps_target_var = "Isyn_n_y"

                self.N_S_norm.append(
                    self.model.add_synapse_population(
                        f"N_S_norm_{i}_{j}",
                        "TOEPLITZ_KERNELG",
                        NO_DELAY,
                        self.N[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_norm_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_kernel_iniconn,
                    )
                )

                self.N_S_norm[-1].ps_target_var = "Isyn_n_norm"

                self.N_S_one_to_one.append(
                    self.model.add_synapse_population(
                        f"N_S_one_to_one_{i}_{j}",
                        "SPARSE_GLOBALG",
                        NO_DELAY,
                        self.N[-1],
                        self.S[-1],
                        "StaticPulse",
                        {},
                        self.PN_S_one_to_one_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                        self.PN_S_one_to_one_iniconn,
                    )
                )

                self.N_S_one_to_one[-1].ps_target_var = "Isyn_n_one_to_one"

                self.S_OUT_v_proj_left.append(
                    self.model.add_synapse_population(
                        f"S_OUT_v_proj_left_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.OUT[-1],
                        create_cont_wu("cont_wu_v_proj_left", "v_proj"),
                        {},
                        self.S_OUT_left_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.S_OUT_v_proj_left[-1].ps_target_var = "Isyn_v_proj_left"

                self.S_OUT_v_proj_right.append(
                    self.model.add_synapse_population(
                        f"S_OUT_v_proj_right_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.OUT[-1],
                        create_cont_wu("cont_wu_v_proj_right", "v_proj"),
                        {},
                        self.S_OUT_right_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.S_OUT_v_proj_right[-1].ps_target_var = "Isyn_v_proj_right"

                self.S_OUT_v_avg_x.append(
                    self.model.add_synapse_population(
                        f"S_OUT_v_avg_x_{i}_{j}",
                        "DENSE_INDIVIDUALG",
                        NO_DELAY,
                        self.S[-1],
                        self.OUT[-1],
                        create_cont_wu("cont_wu_v_avg_x", "vx"),
                        {},
                        self.S_OUT_avg_inivars,
                        {},
                        {},
                        "DeltaCurr",
                        {},
                        {},
                    )
                )

                self.S_OUT_v_avg_x[-1].ps_target_var = "Isyn_v_avg_x"


def run_EMD_sim(
    evt_file,
    save_fold,
    t_end=None,
    p=params.copy(),
    results_filename="results.npz",
    custom_params={},
    measure_sim_speed=False,
    rec_neurons=[],
):
    rec_neurons = list(
        set([("OUT", "V"), ("OUT", "V_linear"), ("OUT", "r_left"), ("OUT", "r_right")])
        | set(rec_neurons)
    )

    print("Running EMD simulation")
    p["REC_SPIKES"] = ["P", "S", "OUT"]

    evts = np.load(evt_file)

    if len(evts) == 0:
        print("No events in file")
        return

    if t_end is None:
        t_end = evts["t"].max()

    p["NT_MAX"] = int(t_end / p["DT_MS"]) + 1

    p.update(custom_params)

    network = EMD_model(p)

    network.load_input_data_from_file(evt_file)
    network.push_input_data_to_device()

    if measure_sim_speed:
        rec_neurons = []

    rec_dt = 10.0

    spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t = network.run_model(
        0.0,
        t_end,
        rec_neurons=rec_neurons,
        rec_timestep=rec_dt,
        measure_sim_speed=measure_sim_speed,
    )

    if not measure_sim_speed:
        # v_s = []
        v_out = []
        v_out_linear = []
        r_left_out = []
        r_right_out = []
        sp_p = []
        sp_s = []
        sp_out = []
        for i in range(network.n_tiles_y):
            # v_s.append([])
            v_out.append([])
            v_out_linear.append([])
            r_left_out.append([])
            r_right_out.append([])
            sp_p.append([])
            sp_s.append([])
            sp_out.append([])
            for j in range(network.n_tiles_x):
                # v_s[-1].append(
                #    np.reshape(
                #        rec_vars_n[f"VS_{i}_{j}"],
                #        (-1, network.S_height, network.S_width),
                #    )
                # )
                v_out[-1].append(rec_vars_n[f"VOUT_{i}_{j}"].flatten())
                v_out_linear[-1].append(rec_vars_n[f"V_linearOUT_{i}_{j}"].flatten())
                r_left_out[-1].append(rec_vars_n[f"r_leftOUT_{i}_{j}"].flatten())
                r_right_out[-1].append(rec_vars_n[f"r_rightOUT_{i}_{j}"].flatten())

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
            # v_s=v_s,
            v_out=v_out,
            v_out_linear=v_out_linear,
            r_left_out=r_left_out,
            r_right_out=r_right_out,
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
