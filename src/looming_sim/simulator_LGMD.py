import numpy as np
import os
import sys

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from .models import (
    bitmask_array_current_source,
    lif_neuron,
    one_one_with_boundary,
    threshold_exp_curr,
)

from .format_spike_data import (
    get_atis_event_array,
    tiled_events,
    spike_bitmask,
    polarity_bitmask,
)

from itertools import product


class LGMD_model:
    def __init__(self, p):
        print("setting up model...")
        self.define_model(p)
        print("building model...")
        self.model.build()

        self.spk_rec_steps = int(max(1, p["SPK_REC_STEPS"]))

        self.model.load(num_recording_timesteps=self.spk_rec_steps)

    def define_model(self, p):
        """
        Method that defines the GeNN model of the neural network
        """

        kwargs = {}
        # Settings needed when running on HPC with multiple GPUs per node
        if p["CUDA_VISIBLE_DEVICES"]:
            from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL

            kwargs["selectGPUByDeviceID"] = True
            kwargs["deviceSelectMethod"] = DeviceSelect_MANUAL

        # create an empty model
        self.model = genn_model.GeNNModel(
            "float", p["NAME"], generateLineInfo=True, time_precision="double", **kwargs
        )
        self.model.dT = p["DT_MS"]
        self.model.timing_enabled = p["TIMING"]
        self.model.batch_size = p["N_BATCH"]
        if p["MODEL_SEED"] is not None:
            self.model._model.set_seed(p["MODEL_SEED"])

        # maximum number of time steps for the input bitmasks
        self.nt_max = p["NT_MAX"]

        # we need to fix that because of the spike buffers
        self.rec_spikes = p["REC_SPIKES"]

        """
        ---------------------------------------------------------------------------
        neuron populations
        Note that we are modelling the "E units" and "I units", as well as the "F unit" 
        as part of synapse models rather than as separate neuron populations. Hence, 
        neuron populations are: P, S, and LGMD
        ----------------------------------------------------------------------------
        """

        # input neurons (P)

        # ceil because we want to be sure to cover all x,y indices provided by
        # the input events.
        self.total_input_width = p["INPUT_WIDTH"]
        self.total_input_height = p["INPUT_HEIGHT"]
        self.n_subdiv_x = p["N_SUBDIV_X"]
        self.n_subdiv_y = p["N_SUBDIV_Y"]
        self.half_step_tiles = p["HALF_STEP_TILES"]

        self.tile_width = int(np.ceil(self.total_input_width / self.n_subdiv_x))
        self.tile_height = int(np.ceil(self.total_input_height / self.n_subdiv_y))

        self.n_tiles_x = (
            2 * self.n_subdiv_x - 1 if self.half_step_tiles else self.n_subdiv_x
        )
        self.n_tiles_y = (
            2 * self.n_subdiv_y - 1 if self.half_step_tiles else self.n_subdiv_y
        )

        self.n_input = self.tile_width * self.tile_height
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

        for k in range(self.n_tiles_y):
            for l in range(self.n_tiles_x):
                self.P.append(
                    self.model.add_neuron_population(
                        f"P_{k}_{l}", self.n_input, lif_neuron, P_params, self.P_inivars
                    )
                )

                self.input.append(
                    self.model.add_current_source(
                        f"input_{k}_{l}",
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
                        f"S_{k}_{l}", self.n_S, lif_neuron, S_params, self.S_inivars
                    )
                )

                # neuron populations

                self.LGMD.append(
                    self.model.add_neuron_population(
                        f"LGMD_{k}_{l}", 1, lif_neuron, LGMD_params, self.LGMD_inivars
                    )
                )

                self.in_S_E.append(
                    self.model.add_synapse_population(
                        f"in_S_E_{k}_{l}",
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
                        f"in_S_I_{k}_{l}",
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
                        f"S_LGMD_{k}_{l}",
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
                        f"in_LGMD_{k}_{l}",
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

                for pop in self.rec_spikes:
                    self.model.neuron_populations[
                        f"{pop}_{k}_{l}"
                    ].spike_recording_enabled = True

    def get_bitmasks_from_file(self, input_event_file):
        """
        Method for loading events from file, returning spike time and polarity bitmasks.
        """

        assert input_event_file.endswith(".npy") or input_event_file.endswith(".dat")

        if input_event_file.endswith(".npy"):
            evt = np.load(input_event_file)
        else:
            evt = get_atis_event_array(input_event_file)

        evts_tiled = tiled_events(
            evt,
            self.total_input_width,
            self.total_input_height,
            self.n_subdiv_x,
            self.n_subdiv_y,
            half_step=self.half_step_tiles,
        )

        x = [tl_data[-1]["x"] for tl_data in evts_tiled]
        y = [tl_data[-1]["y"] for tl_data in evts_tiled]
        st = [tl_data[-1]["t"] for tl_data in evts_tiled]
        pol = [tl_data[-1]["p"] for tl_data in evts_tiled]

        return self.generate_bitmasks(x, y, st, pol)

    def generate_bitmasks(self, x, y, st, pol):
        """
        Generate spike and polarity bitmasks for lists
        of x coord., y coord., spike time and polarity,
        corresponding to different neurons.
        """
        assert len(x) == len(y) == len(st) == len(pol)

        spike_bitmasks, pol_bitmasks = [], []

        for _x, _y, _st, _pol in zip(x, y, st, pol):
            assert np.amax(_x) < self.tile_width
            assert np.amax(_y) < self.tile_height
            spike_bitmasks.append(
                spike_bitmask(
                    _st,
                    _x,
                    _y,
                    self.tile_width,
                    self.tile_height,
                    self.nt_max,
                    self.model.dT,
                )
            )

            pol_bitmasks.append(
                polarity_bitmask(
                    _st,
                    _pol,
                    _x,
                    _y,
                    self.tile_width,
                    self.tile_height,
                    self.nt_max,
                    self.model.dT,
                )
            )

        return spike_bitmasks, pol_bitmasks

    def load_input_data_from_file(self, input_event_file):
        """
        Load events from input file and generate their
        bitmasks.
        """
        spk_bm, pol_bm = self.get_bitmasks_from_file(input_event_file)

        self.spk_bm = spk_bm
        self.pol_bm = pol_bm

    def push_input_data_to_device(self):
        """
        Push the input bitmask data from host memory
        to gpu memory.
        """
        for idx, (k, l) in enumerate(
            product(range(self.n_tiles_y), range(self.n_tiles_x))
        ):
            self.input[idx].extra_global_params["spikeBitmask"].view[:] = self.spk_bm[
                idx
            ].view(dtype=np.uint32)
            self.input[idx].push_extra_global_param_to_device("spikeBitmask")

            self.input[idx].extra_global_params["polarityBitmask"].view[:] = (
                self.pol_bm[idx].view(dtype=np.uint32)
            )
            self.input[idx].push_extra_global_param_to_device("polarityBitmask")

    def run_model(
        self,
        t_start_ms,
        trial_ms,
        event_data=None,
        rec_neurons=[],
        rec_synapses=[],
        save_dir=None,
        timing=True,
        rec_timestep=1.0,
    ):
        """
        Method to run the GeNN model. You may provide new event data by passing
        a file string to event_data. If event_data is None, the data that is
        currently in the device memory will be reused (you can start at any arbitrary
        time due to the bitmask format, which does not require resetting e.g. a spike index)
        """

        # set up recording if required
        spike_t = {}
        spike_ID = {}
        rec_vars_n = {}
        rec_vars_s = {}

        # if None, just reuse the data on the device
        if event_data:
            self.load_input_data_from_file(event_data)
            self.push_input_data_to_device()

        for idx, (k, l) in enumerate(
            product(range(self.n_tiles_y), range(self.n_tiles_x))
        ):
            for pop in self.rec_spikes:
                pop_name = f"{pop}_{k}_{l}"
                spike_t[pop_name] = np.empty((0), dtype="float64")
                spike_ID[pop_name] = np.empty((0), dtype="int64")

            for pop, var in rec_neurons:
                rec_vars_n[f"{var+pop}_{k}_{l}"] = []

            for pop, var in rec_synapses:
                rec_vars_s[f"{var+pop}_{k}_{l}"] = []

        rec_s_t = []
        rec_n_t = []

        # set time to t_start_ms and start simulation loop
        self.model.t = self.model.dT * int(t_start_ms / self.model.dT)
        self.model.timestep = int(t_start_ms / self.model.dT)

        # self.sync_input_startspike_with_time()

        # NOTE: should we reset variables of neurons and/or synapses before the run?

        t_next_rec = 0.0

        while self.model.t <= t_start_ms + trial_ms:
            self.model.step_time()

            if len(self.rec_spikes) > 0:
                if self.model.timestep % self.spk_rec_steps == 0:
                    self.model.pull_recording_buffers_from_device()

                    for k, l in product(range(self.n_tiles_y), range(self.n_tiles_x)):
                        for pop in self.rec_spikes:
                            pop_name = f"{pop}_{k}_{l}"
                            the_pop = self.model.neuron_populations[pop_name]
                            x = the_pop.spike_recording_data
                            if self.model.batch_size > 1:
                                for i in range(self.model.batch_size):
                                    spike_t[pop_name] = np.append(
                                        spike_t[pop_name], x[i][0] + i * trial_ms
                                    )
                                    spike_ID[pop_name] = np.append(
                                        spike_ID[pop_name], x[i][1]
                                    )
                            else:
                                spike_t[pop_name] = np.append(spike_t[pop_name], x[0])
                                spike_ID[pop_name] = np.append(spike_ID[pop_name], x[1])

            if self.model.t >= t_next_rec:
                t_next_rec = self.model.t + rec_timestep
                if len(rec_neurons) > 0:
                    for k, l in product(range(self.n_tiles_y), range(self.n_tiles_x)):
                        for pop, var in rec_neurons:
                            pop_name = f"{pop}_{k}_{l}"
                            the_pop = self.model.neuron_populations[pop_name]
                            the_pop.pull_var_from_device(var)
                            rec_vars_n[var + pop_name].append(
                                the_pop.vars[var].view.copy()
                            )
                    rec_n_t.append(self.model.t)

                if len(rec_synapses) > 0:
                    for k, l in product(range(self.n_tiles_y), range(self.n_tiles_x)):
                        for pop, var in rec_synapses:
                            pop_name = f"{pop}_{k}_{l}"
                            the_pop = self.model.synapse_populations[pop_name]
                            if var == "in_syn":
                                the_pop.pull_in_syn_from_device()
                                rec_vars_s[var + pop_name].append(the_pop.in_syn.copy())
                            else:
                                the_pop.pull_var_from_device(var)
                                rec_vars_s[var + pop_name].append(
                                    the_pop.vars[var].view.copy()
                                )
                    rec_s_t.append(self.model.t)

            if self.model.timestep % 1000 == 0:
                sys.stdout.write(f"t: {self.model.t}\r")
        # end of simulation loop

        # convert neuron and synapse recordings to 2d-arrays
        for key in rec_vars_n:
            rec_vars_n[key] = np.array(rec_vars_n[key])
        for key in rec_vars_s:
            rec_vars_s[key] = np.array(rec_vars_s[key])

        if save_dir:  # Saving results
            for k, l in product(range(self.n_tiles_y), range(self.n_tiles_x)):
                for pop in self.rec_spikes:
                    pop_name = f"{pop}_{k}_{l}"
                    np.save(
                        os.path.join(
                            save_dir, f"{self.model.model_name}_{pop_name}_spike_t"
                        ),
                        spike_t[pop_name],
                    )
                    np.save(
                        os.path.join(
                            save_dir, f"{self.model.model_name}_{pop_name}_spike_ID"
                        ),
                        spike_ID[pop_name],
                    )

                if len(rec_neurons) > 0:
                    np.save(
                        os.path.join(save_dir, self.model.model_name + "_n_t"), rec_n_t
                    )
                    for pop, var in rec_neurons:
                        pop_name = f"{pop}_{k}_{l}"
                        np.save(
                            os.path.join(
                                save_dir, f"{self.model.model_name}_{var+pop_name}"
                            ),
                            rec_vars_n[var + pop_name],
                        )

                if len(rec_synapses) > 0:
                    np.save(
                        os.path.join(save_dir, self.model.model_name + "_t"), rec_s_t
                    )
                    for pop, var in rec_synapses:
                        np.save(
                            os.path.join(
                                save_dir, f"{self.model.model_name}_{var+pop_name}"
                            ),
                            rec_vars_s[var + pop_name],
                        )
        if timing:
            print("Init: %f" % self.model.init_time)
            print("Init sparse: %f" % self.model.init_sparse_time)
            print("Neuron update: %f" % self.model.neuron_update_time)
            print("Presynaptic update: %f" % self.model.presynaptic_update_time)
            print("Synapse dynamics: %f" % self.model.synapse_dynamics_time)
        return (spike_t, spike_ID, rec_vars_n, rec_n_t, rec_vars_s, rec_s_t)
