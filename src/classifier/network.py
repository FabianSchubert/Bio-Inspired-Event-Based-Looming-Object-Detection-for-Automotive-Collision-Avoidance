from ml_genn import Connection, Network, Population

from ml_genn.compilers.event_prop_compiler import (
    default_params as evt_prop_def_settings,
)

from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire
from src.classifier.models import LeakyIntegrateFireSpikeInput

from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal

from ml_genn.synapses import Exponential


def generate_full_conn_network(
    n_input: int,
    n_hidden: int,
    n_output: int,
    recurrent: bool = True,
    v_thresh: list[float] = [1.0, 1.0],
    tau_mem: list[float] = [20.0, 20.0, 20.0],
    tau_refrac: list[float | None] = [None, None],
    max_spikes: int = 10000000,
    unit_amplitude: float = 1.2,
    record_spikes: list[bool] = [True, True],
) -> Network:
    network = Network(evt_prop_def_settings)
    with network:
        input_layer = Population(
            LeakyIntegrateFireSpikeInput(
                v_thresh=v_thresh[0],
                tau_mem=tau_mem[0],
                tau_refrac=tau_refrac[0],
                max_spikes=max_spikes,
                unit_amplitude=unit_amplitude,
            ),
            n_input,
            record_spikes=record_spikes[0],
        )

        hidden_layer = Population(
            LeakyIntegrateFire(
                v_thresh=v_thresh[1], tau_mem=tau_mem[1], tau_refrac=tau_refrac[1]
            ),
            n_hidden,
            record_spikes=record_spikes[1],
        )

        output_layer = Population(
            LeakyIntegrate(tau_mem=tau_mem[2], readout="avg_var_exp_weight"), n_output
        )

        Connection(
            input_layer,
            hidden_layer,
            Dense(Normal(mean=0.01, sd=0.01)),
            Exponential(5.0),
        )
        if recurrent:
            Connection(
                hidden_layer,
                hidden_layer,
                Dense(Normal(mean=0.0, sd=0.02)),
                Exponential(5.0),
            )
        Connection(
            hidden_layer,
            output_layer,
            Dense(Normal(mean=0.0, sd=0.03)),
            Exponential(5.0),
        )

    return network
