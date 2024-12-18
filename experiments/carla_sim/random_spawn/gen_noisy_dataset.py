import numpy as np
import os
import sys

from tqdm import tqdm

from .settings import base_fold_input_data, base_fold_input_data_noisy

from src.config import EVENTS_DTYPE

samples = os.listdir(base_fold_input_data)

NOISE_RATE_PER_SEC_PER_PX = float(sys.argv[1])

WIDTH, HEIGHT = 640, 480

for sample in tqdm(samples):

    events = np.load(os.path.join(base_fold_input_data, sample, "events.npy"))

    t_end_ms = 0 if len(events)==0 else events["t"][-1]

    n_noise_events = int(t_end_ms * NOISE_RATE_PER_SEC_PER_PX * WIDTH * HEIGHT / 1000)

    t_noise = np.random.randint(0, t_end_ms, n_noise_events)
    x_noise = np.random.randint(0, WIDTH, n_noise_events)
    y_noise = np.random.randint(0, HEIGHT, n_noise_events)
    p_noise = np.random.randint(0, 2, n_noise_events)

    noise_events = np.array(list(zip(t_noise, x_noise, y_noise, p_noise)), dtype=EVENTS_DTYPE)

    events_comb = np.append(events, noise_events)

    idx_sort = np.argsort(events_comb["t"])

    events_comb = events_comb[idx_sort]

    noisy_fold = os.path.join(base_fold_input_data_noisy, sample)

    if not os.path.exists(noisy_fold):
        os.makedirs(noisy_fold)

    np.save(os.path.join(noisy_fold, "events.npy"), events_comb)

