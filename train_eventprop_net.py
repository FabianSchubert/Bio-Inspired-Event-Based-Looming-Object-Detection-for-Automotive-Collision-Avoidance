from src.classifier.network import generate_full_conn_network, generate_cnn_network
from src.classifier.data_io import load_file
from src.classifier.train import train_network
from src.classifier.augmentation import ComposeAugment, FlipHorizontal, Crop
from src.classifier.dataset import EventDataSet

from ml_genn.optimisers import Adam  # type: ignore

from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL

import numpy as np

WIDTH, HEIGHT = 304, 240

N_SUBDIV = 2

dataset_train = EventDataSet([f"./data/box_events/{N_SUBDIV}_tiles/train_a/"], max_samples_per_class=5000)
dataset_val = EventDataSet([f"./data/box_events/{N_SUBDIV}_tiles/val_a/"], max_samples_per_class=1000)

#data_train = load_file("./data/balanced_pruned/train_a_td.npy")
#data_val = load_file("./data/balanced_pruned/val_a_td.npy")

INPUT_WIDTH = int(WIDTH / N_SUBDIV)
INPUT_HEIGHT = int(HEIGHT / N_SUBDIV)

N_IN = INPUT_WIDTH * INPUT_HEIGHT
N_HIDDEN = 512
N_OUT = 3

N_EPOCHS = 100

SENSOR_SIZE = (INPUT_WIDTH, INPUT_HEIGHT, 2)  # preprocess spikes wants this ordering
INPUT_SIZE = (INPUT_HEIGHT, INPUT_WIDTH, 1)  # ml genn wants this ordering
# INPUT_SIZE = (INPUT_WIDTH, INPUT_HEIGHT, 1)

GPU_ID = None

GENN_KWARGS = {"selectGPUByDeviceID": True}
if GPU_ID is not None:
    GENN_KWARGS["deviceSelectMethod"] = DeviceSelect_MANUAL
    GENN_KWARGS["manualDeviceID"] = GPU_ID

COMPILER_ARGS = {
    "losses": "sparse_categorical_crossentropy",
    "reg_lambda_upper": 4e-9,
    "reg_lambda_lower": 4e-9,
    "reg_nu_upper": 5,
    "max_spikes": 1500,
    "optimiser": Adam(0.002),
    "batch_size": 32,
    "kernel_profiling": True,
    **GENN_KWARGS,
}

rng_augment = np.random.default_rng()

augment_pipe = ComposeAugment(
    Crop(SENSOR_SIZE[:2], rng_augment, min_scale=0.75, max_scale=1.0),
    FlipHorizontal(SENSOR_SIZE[:2], rng_augment),
)

# net = generate_full_conn_network(N_IN, N_HIDDEN, N_OUT, recurrent=True)
net = generate_cnn_network(INPUT_SIZE, N_HIDDEN, N_OUT)

train_network(
    net,
    dataset_train,
    dataset_val,
    SENSOR_SIZE,
    N_EPOCHS,
    augmentation=None,  # augment_pipe,
    **COMPILER_ARGS
)

__import__("ipdb").set_trace()
