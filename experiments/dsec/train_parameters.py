from src.pytorch import LoomingDetector
from src.pytorch.data_processing import (
    load_metadata,
    load_raw_data,
    convert_to_tens,
    gen_x_sequ,
)

import torch

import numpy as np

from pathlib import Path

from .settings import base_fold_input_data, base_fold_results
from experiments.carla_sim.random_spawn.settings import (
    base_fold_input_data as base_fold_input_data_carla,
)

##########################

WIDTH, HEIGHT = 640, 480

N_SUBDIV_X = 2
N_SUBDIV_Y = 2
SUBDIV_WIDTH, SUBDIV_HEIGHT = WIDTH // N_SUBDIV_X, HEIGHT // N_SUBDIV_Y
N_DISCARD_TOP = 1
N_DISCARD_BOTTOM = 1
N_DISCARD_LEFT = 1
N_DISCARD_RIGHT = 1

T_STEPS = 200

T_STEPS_T_COLL_MAX = 150  # number of time steps before collision to consider as a positive output if a collision occurs in the sequence

##########################

PATH_SEQUENCES_DSEC = Path(base_fold_input_data)
PATH_SEQUENCES_CARLA = Path(base_fold_input_data_carla)
PATH_RESULTS = Path(base_fold_results)

##########################

LR = 1e-3
N_EPOCHS = 20
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

##########################


def gen_datasets(gen_metadata=False, gen_training=True, gen_test=True):

    if gen_metadata:
        smpls_dsec, lbls_dsec, md_dsec = load_metadata(PATH_SEQUENCES_DSEC, T_STEPS)

        types_dsec, counts_dsec = np.unique(
            [m["coll_type"] for m in md_dsec], return_counts=True
        )
        type_counts_dsec = dict(zip(list(types_dsec), list(counts_dsec)))

        print(type_counts_dsec)

        smpls_carla, lbls_carla, md_carla = load_metadata(PATH_SEQUENCES_CARLA, T_STEPS)

        types_carla, counts_carla = np.unique(
            [m["coll_type"] for m in md_carla], return_counts=True
        )
        type_counts_carla = dict(zip(list(types_carla), list(counts_carla)))

        print(type_counts_carla)

        N_NEG = type_counts_dsec["none_with_traffic"]
        N_POS = type_counts_carla["cars"] + type_counts_carla["pedestrians"]

        if N_NEG > N_POS:
            print("subsampling negative examples")
            N_NEG = N_POS
            # subsample the negative (dsec) examples
            indices = np.where(np.array(lbls_dsec) == 0)[0]
            np.random.shuffle(indices)
            indices = indices[:N_NEG]
            smpls_dsec = [smpls_dsec[i] for i in indices]
            lbls_dsec = [0] * N_NEG
            md_dsec = [md_dsec[i] for i in indices]
        elif N_POS > N_NEG:
            print("subsampling positive examples")
            N_POS = N_NEG
            # subsample the positive (carla) examples
            indices = np.where(np.array(lbls_carla) == 1)[0]
            np.random.shuffle(indices)
            indices = indices[:N_POS]
            smpls_carla = [smpls_carla[i] for i in indices]
            lbls_carla = [1] * N_POS
            md_carla = [md_carla[i] for i in indices]

        N_TOTAL = N_NEG + N_POS
        print(f"Total number of samples: {N_TOTAL}")

        N_TRAIN = int(0.5 * N_TOTAL)
        # N_TEST = int(0.5 * N_TOTAL)

        smpls = smpls_dsec + smpls_carla
        lbls = lbls_dsec + lbls_carla
        md = md_dsec + md_carla

        # pick train and test set
        indices = np.arange(N_TOTAL)
        np.random.shuffle(indices)
        indices_train = indices[:N_TRAIN]
        indices_test = indices[N_TRAIN:]

        smpls_train = [smpls[i] for i in indices_train]
        lbls_train = [lbls[i] for i in indices_train]
        md_train = [md[i] for i in indices_train]

        smpls_test = [smpls[i] for i in indices_test]
        lbls_test = [lbls[i] for i in indices_test]
        md_test = [md[i] for i in indices_test]

        # save train and test (meta)data
        np.savez(
            PATH_RESULTS / "metadata_train.npz",
            smpls=smpls_train,
            lbls=lbls_train,
            md=md_train,
        )
        np.savez(
            PATH_RESULTS / "metadata_test.npz",
            smpls=smpls_test,
            lbls=lbls_test,
            md=md_test,
        )
    else:
        # load train and test (meta)data
        metadata_train = np.load(PATH_RESULTS / "metadata_train.npz", allow_pickle=True)
        smpls_train = metadata_train["smpls"]
        lbls_train = metadata_train["lbls"]
        md_train = metadata_train["md"]

        metadata_test = np.load(PATH_RESULTS / "metadata_test.npz", allow_pickle=True)
        smpls_test = metadata_test["smpls"]
        lbls_test = metadata_test["lbls"]
        md_test = metadata_test["md"]

    if gen_training:

        print("Loading training data...")
        data_train, lbls_train, md_train = load_raw_data(smpls_train)

        print("Converting training data to tensors...")
        X_train_full, y_train_full, md_train = convert_to_tens(
            data_train,
            lbls_train,
            md_train,
            T_STEPS,
            WIDTH,
            HEIGHT,
            SUBDIV_WIDTH,
            SUBDIV_HEIGHT,
            N_DISCARD_TOP,
            N_DISCARD_BOTTOM,
            N_DISCARD_LEFT,
            N_DISCARD_RIGHT,
        )
        assert len(lbls_train) == len(y_train_full)

        print("Saving training data as torch tensors...")
        try:
            torch.save(X_train_full, PATH_RESULTS / "X_train_full.pt")
            torch.save(y_train_full, PATH_RESULTS / "y_train_full.pt")
            torch.save(md_train, PATH_RESULTS / "md_train.pt")
        except Exception as e:
            print(e)

        # free memory of raw data
        del data_train, lbls_train

    if gen_test:

        print("Loading test data...")
        data_test, lbls_test, md_test = load_raw_data(smpls_test)

        print("Converting test data to tensors...")
        X_test_full, y_test_full, md_test = convert_to_tens(
            data_test,
            lbls_test,
            md_test,
            T_STEPS,
            WIDTH,
            HEIGHT,
            SUBDIV_WIDTH,
            SUBDIV_HEIGHT,
            N_DISCARD_TOP,
            N_DISCARD_BOTTOM,
            N_DISCARD_LEFT,
            N_DISCARD_RIGHT,
        )
        assert len(lbls_test) == len(y_test_full)

        print("Saving test data as torch tensors...")
        try:
            torch.save(X_test_full, PATH_RESULTS / "X_test_full.pt")
            torch.save(y_test_full, PATH_RESULTS / "y_test_full.pt")
            torch.save(md_test, PATH_RESULTS / "md_test.pt")
        except Exception as e:
            print(e)

        # free memory of raw data
        del data_test, lbls_test


def gen_data_loaders():

    # load training data
    print("Loading training data...")
    X_train = torch.load(PATH_RESULTS / "X_train_full.pt")
    print("Loading training labels...")
    y_train = torch.load(PATH_RESULTS / "y_train_full.pt")
    print("Loading training metadata...")
    md_train = torch.load(PATH_RESULTS / "md_train.pt")

    # load test data
    print("Loading test data...")
    X_test = torch.load(PATH_RESULTS / "X_test_full.pt")
    print("Loading test labels...")
    y_test = torch.load(PATH_RESULTS / "y_test_full.pt")
    print("Loading test metadata...")
    md_test = torch.load(PATH_RESULTS / "md_test.pt")

    # split test data into training and validation set
    N_VAL = int(0.5 * len(X_test))
    X_val = X_test[:N_VAL].clone()
    y_val = y_test[:N_VAL].clone()
    md_val = md_test[:N_VAL].copy()

    X_test = X_test[N_VAL:]
    y_test = y_test[N_VAL:]
    md_test = md_test[N_VAL:]

    # generate samples (pairs of consecutive frames) from the data
    X_seq_train, y_seq_train = gen_x_sequ(X_train, y_train, T_STEPS_T_COLL_MAX)
    X_seq_val, y_seq_val = gen_x_sequ(X_val, y_val, T_STEPS_T_COLL_MAX)
    X_seq_test, y_seq_test = gen_x_sequ(X_test, y_test, T_STEPS_T_COLL_MAX)

    # create datasets
    print("Creating datasets...")
    train_dataset = torch.utils.data.TensorDataset(X_seq_train, y_seq_train)
    val_dataset = torch.utils.data.TensorDataset(X_seq_val, y_seq_val)
    test_dataset = torch.utils.data.TensorDataset(X_seq_test, y_seq_test)

    # create dataloaders
    print("Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, md_train, md_val, md_test


def train_model(train_loader, val_loader):

    # create model
    print("Creating model...")
    looming_detector = LoomingDetector(width=SUBDIV_WIDTH, height=SUBDIV_HEIGHT)

    looming_detector.to(DEVICE)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(looming_detector.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        looming_detector.train()
        running_loss = 0.0

        looming_detector.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = looming_detector(inputs)

            loss = loss_fn(outputs.flatten(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")

        looming_detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = looming_detector(inputs)
                val_loss += loss_fn(outputs.flatten(), labels.float()).item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Validation loss: {val_loss / len(val_loader)}")

    hyperparams = {
        "sigm_bias": float(looming_detector.sigm.bias.detach().cpu().numpy()[0]),
        "sigm_scale": float(looming_detector.sigm.scale.detach().cpu().numpy()[0]),
        "out_scale": float(looming_detector.out_scale.detach().cpu().numpy()[0]),
        "out_th": float(looming_detector.out_th.detach().cpu().numpy()[0]),
    }
    print(hyperparams)

    print("Saving model...")
    torch.save(looming_detector.state_dict(), PATH_RESULTS / "looming_detector.pt")

    print("Saving hyperparameters...")
    np.savez(PATH_RESULTS / "hyperparams.npz", **hyperparams)

def test_model(test_loader):
    # load the model
    print("Loading model...")
    looming_detector = LoomingDetector(width=SUBDIV_WIDTH, height=SUBDIV_HEIGHT)
    looming_detector.load_state_dict(torch.load(PATH_RESULTS / "looming_detector.pt"))
    looming_detector.to(DEVICE)

    # test the model

    looming_detector.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = looming_detector(inputs)

            predicted = (outputs > 0).int()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the validation set: {100 * correct / total}%")


if __name__ == "__main__":
    gen_datasets(gen_metadata=False, gen_training=False, gen_test=False)
    train_loader, val_loader, test_loader, md_train, md_val, md_test = gen_data_loaders()
    train_model(train_loader, val_loader)
    test_model(test_loader)
