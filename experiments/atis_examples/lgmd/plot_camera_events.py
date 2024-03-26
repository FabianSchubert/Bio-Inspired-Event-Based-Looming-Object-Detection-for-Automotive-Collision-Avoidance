from src.viz import play_event_anim
import numpy as np
import os

base_fold = os.path.join(os.path.dirname(__file__), "../../../data/atis_examples/")

base_fold_results = os.path.join(
    os.path.dirname(__file__),
    "../../../experiments/atis_examples/s_cell_spike_responses/results/lgmd/",
)

for k in range(7):
    results_fold = os.path.join(base_fold_results, f"example_{k}")

    if not os.path.exists(results_fold):
        os.makedirs(results_fold)

    evt_file = os.path.join(base_fold, f"events_test_{k}.npy")
    evts = np.load(evt_file)

    play_event_anim(evts, 0.0, evts["t"].max(), 100., 304, 240, save_frames=results_fold)