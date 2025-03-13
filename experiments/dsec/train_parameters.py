from src.pytorch import LoomingDetector
from src.pytorch.data_processing import load_metadata, load_raw_data, convert_to_tens

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


def gen_datasets(reload_data=False, gen_training=True, gen_test=True):

    if not reload_data:
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
        N_TEST = int(0.5 * N_TOTAL)

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


if __name__ == "__main__":
    gen_datasets(reload_data=True, gen_training=False, gen_test=True)
