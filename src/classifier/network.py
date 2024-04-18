from ml_genn import Connection, Network, Population

from ml_genn.compilers.event_prop_compiler import (
    default_params as evt_prop_def_settings,
)

from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire
from src.classifier.models import LeakyIntegrateFireSpikeInput

from ml_genn.connectivity import Dense, Conv2D
from ml_genn.initializers import Normal

from ml_genn.synapses import Exponential

from src.classifier.utils.connectivity import gabor_filters

DEFAULT_KERNEL_SIZE = (10, 10)
DEFAULT_NUM_FILTERS = 3

DEFAULT_KERNEL = gabor_filters(*DEFAULT_KERNEL_SIZE, DEFAULT_NUM_FILTERS)
# DEFAULT_KERNEL = Normal(mean=0.01, sd=0.01)


def generate_cnn_network(
    input_shape: tuple[int, int, int],  # height, width, channels!
    n_hidden: int,
    n_output: int,
    conv_params: dict = {
        "weight_init": DEFAULT_KERNEL,
        "kern_size": DEFAULT_KERNEL_SIZE,  # height, width!
        "stride": (2, 2),  # stride height, stride width
        "padding": "same",  # "valid" or "same"
        "filters": DEFAULT_NUM_FILTERS,  # number of output channels / filters
    },
    recurrent: bool = True,
    v_thresh: list[float] = [1.0, 1.0, 1.0],
    tau_mem: list[float] = [20.0, 20.0, 20.0, 20.0],
    tau_refrac: list[float | None] = [None, None, None],
    max_spikes: int = 10000000,
    unit_amplitude: float = 1.2,
    record_spikes: list[bool] = [True, True, True],
):

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
            input_shape,
            record_spikes=record_spikes[0],
        )

        conv_layer = Population(
            LeakyIntegrateFire(
                v_thresh=v_thresh[1],
                tau_mem=tau_mem[1],
                tau_refrac=tau_refrac[1],
            ),
            None,  # layer size will be dynamically determined from other parameters when connecting
            record_spikes=record_spikes[1],
        )

        hidden_layer = Population(
            LeakyIntegrateFire(
                v_thresh=v_thresh[2], tau_mem=tau_mem[2], tau_refrac=tau_refrac[2]
            ),
            n_hidden,
            record_spikes=record_spikes[2],
        )

        output_layer = Population(
            LeakyIntegrate(tau_mem=tau_mem[3], readout="avg_var_exp_weight"),
            n_output,
        )

        Connection(
            input_layer,
            conv_layer,
            Conv2D(
                conv_params["weight_init"],
                conv_params["filters"],
                conv_params["kern_size"],
                conv_strides=conv_params["stride"],
                flatten=True,
            ),
            Exponential(5.0),
        )

        Connection(
            conv_layer,
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
