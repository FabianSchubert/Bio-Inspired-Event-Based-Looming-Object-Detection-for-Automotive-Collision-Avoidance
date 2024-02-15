import numpy as np

from collections import namedtuple

from typing import Sequence

PreprocessedSpikesPol = namedtuple(
    "PreprocessedSpikesPol", ["end_spikes", "spike_times", "pols"]
)


def preprocess_spikes_pol(
    times: np.ndarray, ids: np.ndarray, pols: np.ndarray, num_neurons: int
) -> PreprocessedSpikesPol:
    # Calculate end spikes
    end_spikes = np.cumsum(np.bincount(ids, minlength=num_neurons))

    # Sort events first by neuron id and then
    # by time and use to order spike times
    times = times[np.lexsort((times, ids))]
    pols = pols[np.lexsort((times, ids))]

    # Return end spike indices and spike times
    return PreprocessedSpikesPol(end_spikes, times, pols)


def preprocess_tonic_spikes_pol(
    events: np.ndarray, ordering: Sequence[str], shape: tuple, time_scale=1.0 / 1000.0
) -> PreprocessedSpikesPol:
    # Check dataset datatype includes time and polarity
    if "t" not in ordering or "p" not in ordering:
        raise RuntimeError(
            "Only tonic datasets with time (t) and "
            "polarity (p) in ordering are supported"
        )

    # If sensor has single polarity
    if shape[2] == 1:
        raise RuntimeError(
            "Only tonic datasets with with polarity == 2 can be used "
            "with polarised spike injection"
        )

    # Calculate cumulative sum of each neuron's spike count
    # ASSUME: the last entry of shape is always the polarity
    num_neurons = np.product(shape[0:2])

    # If sensor is 2D, flatten x and y into event IDs
    if ("x" in ordering) and ("y" in ordering):
        spike_event_ids = events["x"] + (events["y"] * shape[0])
    # Otherwise, if it's 1D, simply use X
    elif "x" in ordering:
        spike_event_ids = events["x"]
    else:
        raise RuntimeError("Only 1D and 2D sensors supported")

    # Preprocess scaled times and flattened IDs
    return preprocess_spikes_pol(
        events["t"] * time_scale, spike_event_ids, events["p"], num_neurons
    )


def batch_spikes_pol(
    spikes: Sequence[PreprocessedSpikesPol], batch_size: int
) -> PreprocessedSpikesPol:
    # Check that there aren't more examples than batch size
    # and that all examples are for same number of neurons
    num_neurons = len(spikes[0].end_spikes)
    if len(spikes) > batch_size:
        raise RuntimeError(
            f"Cannot batch {len(spikes)} PreprocessedSpikes "
            f"when batch size is only {batch_size}"
        )
    if any(len(s.end_spikes) != num_neurons for s in spikes):
        raise RuntimeError(
            "Cannot batch PreprocessedSpikes " "with different numbers of neurons"
        )

    assert all(len(s.end_spikes) == num_neurons for s in spikes)

    # Extract seperate lists of each example's
    # end spike indices and spike times
    end_spikes, spike_times, pols = zip(*spikes)

    # Calculate cumulative sum of spikes counts across batch
    cum_spikes_per_example = np.concatenate(
        ([0], np.cumsum([len(s) for s in spike_times]))
    )

    # Add this cumulative sum onto the end spikes array of each example
    # **NOTE** zip will stop before extra cum_spikes_per_example value
    batch_end_spikes = np.vstack(
        [c + e for e, c in zip(end_spikes, cum_spikes_per_example)]
    )

    # If this isn't a full batch
    if len(spikes) < batch_size:
        # Create spike padding for remainder of batch
        pad_shape = (batch_size - len(spikes), num_neurons)
        spike_padding = np.ones(pad_shape, dtype=int) * cum_spikes_per_example[-1]

        # Stack onto end spikes
        batch_end_spikes = np.vstack((batch_end_spikes, spike_padding))

    # Concatenate together all spike times
    batch_spike_times = np.concatenate(spike_times)
    batch_pols = np.concatenate(pols)

    return PreprocessedSpikesPol(batch_end_spikes, batch_spike_times, batch_pols)
