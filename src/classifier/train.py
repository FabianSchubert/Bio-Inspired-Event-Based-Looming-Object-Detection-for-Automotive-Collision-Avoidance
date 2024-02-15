import numpy as np

from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.serialisers import Numpy
from ml_genn import Network

from ml_genn import Population

from src.classifier.utils import preprocess_tonic_spikes_pol

from time import perf_counter

from typing import Any, Text, TextIO

import os

DT = 1.0


def write_result_line(
    resfile: TextIO, epoch: int, train_res, val_res, spk_stats
) -> None:
    resfile.write(f"{epoch} ")
    resfile.write(f"{train_res} ")
    resfile.write(f"{val_res} ")
    for x in spk_stats:
        resfile.write(f"{x} ")
    resfile.write("\n")
    resfile.flush()


def train_network(
    network: Network,
    data_train: list[tuple],
    data_val: list[tuple],
    sensor_size: tuple[int],
    n_epochs: int,
    shuffle: bool = True,
    event_ordering: tuple[str] = ("t", "x", "y", "p"),
    rec_populations: dict[str, Population] = {},
    resfile_path: str | None = "./",
    network_name: str = "eventprop_net",
    **compiler_args,
):
    save_results = resfile_path is not None
    if save_results:
        if not os.path.exists(resfile_path):
            os.makedirs(resfile_path)

        result_stream = open(
            os.path.join(resfile_path, network_name + "_results.txt"), "a"
        )

    evts_train, labels_train = (
        [dat[0] for dat in data_train],
        [dat[1]["class_id"] for dat in data_train],
    )
    evts_val, labels_val = (
        [dat[0] for dat in data_val],
        [dat[1]["class_id"] for dat in data_val],
    )

    num_output = np.unique(labels_train).max().astype(int) + 1
    max_spikes = 0
    latest_spike_time = 0.0
    if len(data_train) > 0:
        max_spikes = max([len(evt) for evt in evts_train])
        latest_spike_time = max([np.amax(evt["t"]) / 1000.0 for evt in evts_train])

    max_example_timesteps = int(latest_spike_time / DT)

    spikes_val = []
    for events in evts_val:
        spikes_val.append(
            preprocess_tonic_spikes_pol(events, event_ordering, sensor_size)
        )

    compiler = EventPropCompiler(
        example_timesteps=max_example_timesteps,
        **compiler_args,
    )
    compiled_net = compiler.compile(network, network_name)

    # this is not optimal, it assumes that the first population in the list
    # is the input layer, and the last one is the output layer.
    input_pop, output_pop = network.populations[0], network.populations[-1]

    with compiled_net:
        serialiser = Numpy("eventprop_net_checkpoints")
        start_time = perf_counter()
        # callbacks = ["batch_progress_bar", Checkpoint(serialiser),
        #             SpikeRecorder(hidden, record_counts= True)]
        callbacks: list[Any] = [Checkpoint(serialiser, epoch_interval=5)]
        for k, pop in rec_populations.items():
            callbacks.append(SpikeRecorder(pop, record_counts=True, key=f"n_spk_{k}"))

        for ep in range(n_epochs):
            # spikes, labels = [], []
            spikes_train = []
            for events in evts_train:
                spikes_train.append(
                    preprocess_tonic_spikes_pol(events, event_ordering, sensor_size)
                )
            # Train epoch
            metrics, val_metrics, rec_data, val_rec_data = compiled_net.train(
                {input_pop: spikes_train},
                {output_pop: labels_train},
                start_epoch=ep,
                num_epochs=1,
                shuffle=shuffle,
                callbacks=callbacks,
                validation_x={input_pop: spikes_val},
                validation_y={output_pop: labels_val},
            )
            if save_results:
                write_result_line(
                result_stream, ep, metrics[output_pop].result, val_metrics[output_pop].result, ""
            )
            
            end_time = perf_counter()
            print(f"Accuracy = {100 * metrics[output_pop].result}%")
            print(f"Time = {end_time - start_time}s")
