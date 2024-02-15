import numpy as np

from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY_DUPLICATE
from typing import Optional, Sequence, Union
from ml_genn.neurons.neuron import Neuron
from ml_genn.neurons.input import Input
from src.classifier.utils import PreprocessedSpikesPol, batch_spikes_pol
from ml_genn.utils.data import calc_start_spikes
from ml_genn.utils.model import NeuronModel
from ml_genn.utils.value import InitValue, ValueDescriptor

from ml_genn.utils.decorators import network_default_params


class LeakyIntegrateFireSpikeInput(Neuron, Input):
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")
    tau_mem = ValueDescriptor("Alpha", lambda val, dt: np.exp(-dt / val))
    tau_refrac = ValueDescriptor("TauRefrac")

    @network_default_params
    def __init__(
        self,
        v_thresh: InitValue = 1.0,
        v_reset: InitValue = 0.0,
        v: InitValue = 0.0,
        tau_mem: InitValue = 20.0,
        tau_refrac: InitValue = None,
        relative_reset: bool = True,
        integrate_during_refrac: bool = True,
        scale_i: bool = False,
        unit_amplitude: InitValue = 1.0,
        max_spikes=1000000,
        softmax: Optional[bool] = None,
        readout=None,
        **kwargs,
    ):
        super(LeakyIntegrateFireSpikeInput, self).__init__(softmax, readout, **kwargs)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.tau_mem = tau_mem
        self.tau_refrac = tau_refrac
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac
        self.scale_i = scale_i
        self.unit_amplitude = unit_amplitude
        self.max_spikes = max_spikes

    def set_input(
        self,
        genn_pop,
        batch_size: int,
        shape,
        input: Union[PreprocessedSpikesPol, Sequence[PreprocessedSpikesPol]],
    ):
        # Batch spikes
        batched_spikes = batch_spikes_pol(input, batch_size)

        # Get view
        start_spikes_view = genn_pop.vars["StartSpike"].view
        end_spikes_view = genn_pop.vars["EndSpike"].view
        spike_times_view = genn_pop.extra_global_params["SpikeTimes"].view
        spike_polarity_view = genn_pop.extra_global_params["SpikePolarity"].view

        # Check that spike times will fit in view, copy them and push them
        num_spikes = len(batched_spikes.spike_times)
        assert num_spikes <= len(spike_times_view)
        spike_times_view[0:num_spikes] = batched_spikes.spike_times
        spike_polarity_view[0:num_spikes] = batched_spikes.pols
        genn_pop.push_extra_global_param_to_device("SpikeTimes")
        genn_pop.push_extra_global_param_to_device("SpikePolarity")

        # Calculate start and end spike indices
        end_spikes_view[:] = batched_spikes.end_spikes
        start_spikes_view[:] = calc_start_spikes(batched_spikes.end_spikes)
        genn_pop.push_var_to_device("StartSpike")
        genn_pop.push_var_to_device("EndSpike")

    def get_model(self, population, dt, batch_size):
        # Build basic model
        genn_model = {
            "var_name_types": [
                ("V", "scalar"),
                ("StartSpike", "unsigned int"),
                ("EndSpike", "unsigned int", VarAccess_READ_ONLY_DUPLICATE),
            ],
            "param_name_types": [
                ("Vthresh", "scalar"),
                ("Vreset", "scalar"),
                ("Alpha", "scalar"),
                ("unit_amplitude", "scalar"),
            ],
            "extra_global_params": [
                ("SpikeTimes", "scalar*"),
                ("SpikePolarity", "int8_t*"),
            ],
            "threshold_condition_code": "$(V) >= $(Vthresh)",
            "is_auto_refractory_required": False,
        }

        # Build reset code depending on whether
        # reset should be relative or not
        if self.relative_reset:
            genn_model[
                "reset_code"
            ] = """
                $(V) -= ($(Vthresh) - $(Vreset));
                """
        else:
            genn_model[
                "reset_code"
            ] = """
                $(V) = $(Vreset);
                """
        # Define integration code based on whether I should be scaled
        if self.scale_i:
            v_update = "$(V) = ($(Alpha) * $(V)) + ((1.0 - $(Alpha)) * $(Isyn));"
        else:
            v_update = "$(V) = ($(Alpha) * $(V)) + $(Isyn);"

        # Define the code for current injections
        spike_isyn = """
            if ($(StartSpike) != $(EndSpike) && $(t) >= $(SpikeTimes)[$(StartSpike)]) {
                $(Isyn)= $(unit_amplitude)*$(SpikePolarity)[$(StartSpike)];
                $(StartSpike)++;
            }
            """
        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["var_name_types"].append(("RefracTime", "scalar"))
            genn_model["param_name_types"].append(("TauRefrac", "scalar"))

            # Build correct sim code depending on whether
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model[
                    "sim_code"
                ] = f"""
                    {spike_isyn}
                    {v_update}
                    if ($(RefracTime) > 0.0) {{
                        $(RefracTime) -= DT;
                    }}
                    """
            else:
                genn_model[
                    "sim_code"
                ] = f"""
                    if ($(RefracTime) > 0.0) {{
                        $(RefracTime) -= DT;
                    }}
                    else {{
                        {spike_isyn}
                        {v_update}
                    }}
                    """

            # Add refractory period initialisation to reset code
            genn_model[
                "reset_code"
            ] += """
                $(RefracTime) = $(TauRefrac);
                """

            # Add refractory check to threshold condition
            genn_model["threshold_condition_code"] += " && $(RefracTime) <= 0.0"
        # Otherwise, build non-refractory sim-code
        else:
            genn_model[
                "sim_code"
            ] = f"""
                {spike_isyn}
                {v_update}
                """

        # Return model
        var_vals = (
            {"StartSpike": 0, "EndSpike": 0}
            if self.tau_refrac is None
            else {"StartSpike": 0, "EndSpike": 0, "RefracTime": 0.0}
        )
        param_vals = {"unit_amplitude": self.unit_amplitude}
        egp_vals = {
            "SpikeTimes": np.empty(self.max_spikes, dtype=np.float32),
            "SpikePolarity": np.empty(self.max_spikes, dtype=np.float32),
        }
        return NeuronModel.from_val_descriptors(
            genn_model,
            "V",
            self,
            dt,
            var_vals=var_vals,
            param_vals=param_vals,
            egp_vals=egp_vals,
        )
