import numpy as np

from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder  # type: ignore
from ml_genn.serialisers import Numpy  # type: ignore
from ml_genn import Network  # type: ignore

from ml_genn import Population

from src.classifier.utils.preprocess_spikes import preprocess_tonic_spikes_pol
from src.classifier.augmentation import AugmentBase

from time import perf_counter, strftime

from typing import Any, Text, TextIO

import os

import pandas as pd

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


class ResultLogger:
    def __init__(self, fold: str, network_name: str):
        self.fold = fold

        if not os.path.exists(fold):
            os.makedirs(fold)

        self.epochs: list[int] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []

        self.filename = f"{network_name}_{strftime('%d_%m_%Y_%Hh-%Mm')}_results.txt"

        self.full_path = os.path.join(self.fold, self.filename)

    def update(self, epoch: int, train_res: float, val_res: float):
        self.epochs.append(epoch)
        self.train_acc.append(train_res)
        self.val_acc.append(val_res)

        df = pd.DataFrame(
            {"epoch": self.epochs, "train acc": self.train_acc, "val acc": self.val_acc}
        )
        df.to_csv(self.full_path)


def train_network(
    network: Network,
    data_train: list,
    data_val: list,
    sensor_size: tuple[int, int, int],
    n_epochs: int,
    shuffle: bool = True,
    augmentation: AugmentBase | None = None,
    event_ordering: tuple[str, str, str, str] = ("t", "x", "y", "p"),
    rec_populations: dict[str, Population] = {},
    resfile_path: str | None = "./",
    network_name: str = "eventprop_net",
    **compiler_args,
):
    save_results = resfile_path is not None
    if save_results:
        resfile_path = str(resfile_path)
        res_logger = ResultLogger(resfile_path, network_name)
    else:
        del resfile_path

    evts_train, labels_train = data_train
    evts_train = list(evts_train)
    labels_train = list(labels_train)
    evts_val, labels_val = data_val
    evts_val = list(evts_val)
    labels_val = list(labels_val)

    max_spikes = 0
    latest_spike_time = 0.0
    if len(data_train) > 0:
        max_spikes = max([len(evt) for evt in evts_train])
        latest_spike_time = max([np.amax(evt["t"]) / 1000.0 for evt in evts_train])

    max_example_timesteps = int(latest_spike_time / DT)

    spikes_val = []
    print("convert validation data...")
    for k in range(len(evts_val)):
        spikes_val.append(
            preprocess_tonic_spikes_pol(evts_val.pop(0), event_ordering, sensor_size)
        )
    print("done")

    print("generating compiler...")
    compiler = EventPropCompiler(
        example_timesteps=max_example_timesteps,
        **compiler_args,
    )
    print("compiling network...")
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

        print("Training...")
        for ep in range(n_epochs):
            # spikes, labels = [], []
            spikes_train = []
            for events in evts_train:
                events_augment = augmentation(events) if augmentation else events
                # events_augment = events
                spikes_train.append(
                    preprocess_tonic_spikes_pol(
                        events_augment, event_ordering, sensor_size
                    )
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
                res_logger.update(
                    ep, metrics[output_pop].result, val_metrics[output_pop].result
                )

            end_time = perf_counter()
            print(f"Train Accuracy = {100 * metrics[output_pop].result}%")
            print(f"Val Accuracy = {100 * val_metrics[output_pop].result}%")
            print(f"Time = {end_time - start_time}s")
