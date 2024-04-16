import numpy as np

from src.looming_sim.lgmd.simulator_LGMD import LGMD_model
from src.utils import make_sdf, compare_boxes_patches, iou
from src.looming_sim.format_spike_data import get_atis_event_array

from sklearn.metrics import f1_score

import os
import glob


def evaluate_response(
    p_optim,
    DATA_FOLD,
    N_SAMPLE,
    SEED_RNG_SAMPLE,
    p_network,
    SIGM_SDF,
    SCALE_SDF,
    metric=f1_score,
):
    p_optim_keys = [
        "SCALE_KERNEL_G",
        "SCALE_KERNEL_D",
        "W_IN_S_E",
        "W_S_LGMD",
        "W_IN_LGMD",
        "TAU_SYN_IN_S_I",
        "TAU_SYN_IN_S_E",
        "TAU_IN_LGMD",
        "THRESH_IN_LGMD",
        "TAU_MEM_P",
        "V_THRESH_P",
        "V_RESET_P",
        "TAU_MEM_S",
        "V_THRESH_S",
        "V_RESET_S",
        "TAU_MEM_LGMD",
        "V_THRESH_LGMD",
        "V_RESET_LGMD",
        "SYN_DELAY_LGMD",
    ]

    p_optim_dict = (
        dict(zip(p_optim_keys, p_optim))
        if (isinstance(p_optim, list) or isinstance(p_optim, np.ndarray))
        else dict(p_optim)
    )

    np.random.seed(SEED_RNG_SAMPLE)

    # select all file ids (the part before td.dat, bbox.npy, etc.) from
    # looming sequence files.
    FILE_IDS = [
        fn.split("bbox_looming_sequences.npy")[0]
        for fn in glob.glob(os.path.join(DATA_FOLD, "*bbox_looming_sequences.npy"))
    ]

    N_FILES = len(FILE_IDS)

    FILES_EVENTS = [fl + "td.dat" for fl in FILE_IDS]
    FILES_SEQUENCES = [fl + "bbox_looming_sequences.npy" for fl in FILE_IDS]

    num_sq_in_files = []

    # determine the number of sequences in each file
    for fl in FILES_SEQUENCES:
        sequences = np.load(fl, allow_pickle=True)
        num_sq_in_files.append(len(sequences))

    # list of pairs of "file idx" and "sequence idx" within that file
    fl_idx_sq_idx = [(i, j) for i in range(N_FILES) for j in range(num_sq_in_files[i])]

    NUM_SEQUENCES = len(fl_idx_sq_idx)

    assert (
        NUM_SEQUENCES >= N_SAMPLE
    ), "The total number of sequences in the data folder is smaller than the number of requested samples"

    # pick N_SAMPLE random samples from fl_idx_sq_idx - i.e. N_SAMPLE random sequences
    glob_sq_idx_subsample = np.arange(NUM_SEQUENCES)
    np.random.shuffle(glob_sq_idx_subsample)
    glob_sq_idx_subsample = np.array(glob_sq_idx_subsample[:N_SAMPLE])

    # after picking a subset of sequences, group them by their respective file,
    # such that we only need to load event data to the device once per file
    # (otherwise we'd be jumping randomly between files).
    sq_idx_dict = {}
    for glob_sq_idx in glob_sq_idx_subsample:
        _file_idx = fl_idx_sq_idx[glob_sq_idx][0]
        _sq_idx = fl_idx_sq_idx[glob_sq_idx][1]
        if _file_idx not in sq_idx_dict.keys():
            sq_idx_dict[_file_idx] = [_sq_idx]
        else:
            sq_idx_dict[_file_idx].append(_sq_idx)

    #### some basic sim settings
    p_network["REC_NEURONS"] = []
    p_network["REC_SPIKES"] = ["LGMD"]
    p_network["SPK_REC_STEPS"] = 100
    p_network["N_SUBDIV_X"] = 4
    p_network["N_SUBDIV_Y"] = 4
    p_network["HALF_STEP_TILES"] = True

    N_TILES_X = (
        (2 * p_network["N_SUBDIV_X"] - 1)
        if p_network["HALF_STEP_TILES"]
        else p_network["N_SUBDIV_X"]
    )
    N_TILES_Y = (
        (2 * p_network["N_SUBDIV_Y"] - 1)
        if p_network["HALF_STEP_TILES"]
        else p_network["N_SUBDIV_Y"]
    )

    # overwrite default settings with parameters that are to be optimised
    for key, val in p_optim_dict.items():
        p_network[key] = val

    network = LGMD_model(p_network)
    # [[tp, fp],
    # [fn, tn]]

    lgmd_predictions = np.empty((0))
    iou_ground_truth = np.empty((0))

    sq_counter = 0

    for fl_idx, sq_indices in sq_idx_dict.items():
        fl_events = FILES_EVENTS[fl_idx]
        fl_sequences = FILES_SEQUENCES[fl_idx]

        sequences = np.load(fl_sequences, allow_pickle=True)

        network.load_input_data_from_file(fl_events)
        network.push_input_data_to_device()

        for sq in sequences[sq_indices]:
            sq_counter += 1
            print(f"PROCESSING SAMPLE {sq_counter}/{N_SAMPLE}")

            # time unit in genn is millisecs, not microsecs
            ts = sq["ts"] / 1e3
            t0 = ts[0]
            t1 = ts[-1]
            t_sim = t1 - t0

            spike_t, spike_ID, rec_vars_n, rec_vars_s = network.run_model(
                t0,
                t_sim,
                event_data=None,
                rec_neurons=p_network["REC_NEURONS"],
                rec_synapses=[],
            )

            lgmd_sdf = {
                k: make_sdf(
                    spike_t[k],
                    SIGM_SDF,
                    p_network["DT_MS"],
                    [t0, t1],
                )[1]
                * SCALE_SDF
                for k in spike_t
                if k.startswith("LGMD")
            }

            lgmd_sdf_grid = [
                [lgmd_sdf[f"LGMD_{i}_{j}"] for j in range(N_TILES_X)]
                for i in range(N_TILES_Y)
            ]

            lgmd_sdf_grid_bin = np.array(lgmd_sdf_grid).sum(axis=2) > 0

            lgmd_predictions = np.append(lgmd_predictions, lgmd_sdf_grid_bin.flatten())

            iou_grid = compare_boxes_patches(
                sq,
                p_network["INPUT_WIDTH"],
                p_network["INPUT_HEIGHT"],
                p_network["N_SUBDIV_X"],
                p_network["N_SUBDIV_Y"],
                half_step=p_network["HALF_STEP_TILES"],
                comp_func=iou,
            )

            iou_bin = np.array(iou_grid).sum(axis=2) > 0.0

            iou_ground_truth = np.append(iou_ground_truth, iou_bin.flatten())

    # free device memory, hopefully ?

    network.model.end()
    network.model.unload()

    return (
        metric(iou_ground_truth, lgmd_predictions),
        iou_ground_truth,
        lgmd_predictions,
    )
