#! /usr/bin/env python3
import numpy as np

from src.test_looming_sequences import evaluate_response
from src.default_settings import p as p_default

from sklearn.metrics import f1_score

from scipy.optimize import minimize

from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune

import os

OPT_PARAMS = [
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


def f_optim_neg(x_optim):
    score = -evaluate_response(
        x_optim,
        DATA_FOLD,
        N_SAMPLE,
        SEED_RNG_SAMPLE,
        p_default,
        SIGM_SDF,
        SCALE_SDF,
        metric=f1_score,
    )
    return score


def f_optim(p_optim, *args):
    score = evaluate_response(
        p_optim,
        DATA_FOLD,
        N_SAMPLE,
        SEED_RNG_SAMPLE,
        p_default,
        SIGM_SDF,
        SCALE_SDF,
        metric=f1_score,
    )
    return {"score": score}


def logger_callback(intermediate_result):
    score = -intermediate_result["fun"]
    x = intermediate_result["x"]

    msg_term = f"current_score: {score}"
    print(msg_term, end="\r")

    if not os.path.exists("./optim_log.txt"):
        with open("optim_log.txt", "a") as f:
            header = ",".join(OPT_PARAMS)
            header = "SCORE," + header
            f.write(header + "\n")

    with open("optim_log.txt", "a") as f:
        dat_line = ",".join([str(_x) for _x in x])
        dat_line = str(score) + "," + dat_line
        f.write(dat_line + "\n")


#########################
if __name__ == "__main__":
    DATA_FOLD = "/mnt/data0/prophesee_data/ATIS_Automotive_Detection_Dataset/test_a/"
    N_SAMPLE = 50
    SEED_RNG_SAMPLE = 42
    SIGM_SDF = 100.0
    SCALE_SDF = 1.0

    NUM_ITERATIONS_OPT = 100

    search_bounds = [
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 10.0),
        (0.0, 1.0),
        (-10.0, 0.0),
        (0.0, 1000.0),
        (0.0, 500.0),
        (0.0, 500.0),
        (0.0, 1000.0),
        (0.0, 500.0),
        (0.0, 10.0),
        (-1.0, 1.0),
        (0.0, 250.0),
        (0.0, 10.0),
        (-5.0, 5.0),
        (0.0, 500.0),
        (0.0, 5.0),
        (-5.0, 5.0),
        (0.0, 500.0),
    ]

    SCALE_SIMPLEX_INIT = 0.25

    dim_opt = len(OPT_PARAMS)

    x0 = [p_default[key] for key in OPT_PARAMS]

    init_simplex = np.empty((dim_opt + 1, dim_opt))
    for i in range(dim_opt + 1):
        for j in range(dim_opt):
            low = x0[j] + SCALE_SIMPLEX_INIT * (search_bounds[j][0] - x0[j])
            high = x0[j] + SCALE_SIMPLEX_INIT * (search_bounds[j][1] - x0[j])
            init_simplex[i, j] = np.random.rand() * (high - low) + low

    results = minimize(
        f_optim_neg,
        x0,
        method="Nelder-Mead",
        bounds=search_bounds,
        options={
            "maxiter": NUM_ITERATIONS_OPT,
            "adaptive": True,
            "initial_simplex": init_simplex,
        },
        callback=logger_callback,
    )

    __import__("ipdb").set_trace()

    """
    search_space = {
        "SCALE_KERNEL_G": tune.uniform(0.0, 100.0),
        "SCALE_KERNEL_D": tune.uniform(0.0, 100.0),
        "W_IN_S_E": tune.uniform(0.0, 10.0),
        "W_S_LGMD": tune.uniform(0.0, 1.0),
        "W_IN_LGMD": tune.uniform(-10.0, 0.0),
        "TAU_SYN_IN_S_I": tune.uniform(0.0, 1000.0),
        "TAU_SYN_IN_S_E": tune.uniform(0.0, 500.0),
        "TAU_IN_LGMD": tune.uniform(0.0, 500.0),
        "THRESH_IN_LGMD": tune.uniform(0.0, 1000.0),
        "TAU_MEM_P": tune.uniform(0.0, 500.0),
        "V_THRESH_P": tune.uniform(0.0, 10.0),
        "V_RESET_P": tune.uniform(-1.0, 1.0),
        "TAU_MEM_S": tune.uniform(0.0, 250.0),
        "V_THRESH_S": tune.uniform(0.0, 10.0),
        "V_RESET_S": tune.uniform(-5.0, 5.0),
        "TAU_MEM_LGMD": tune.uniform(0.0, 500.0),
        "V_THRESH_LGMD": tune.uniform(0.0, 5.0),
        "V_RESET_LGMD": tune.uniform(-5.0, 5.0),
        "SYN_DELAY_LGMD": tune.uniform(0.0, 500.0),
    }

    initial_search = {key: p_default[key] for key in search_space.keys()}

    p_default["CUDA_VISIBLE_DEVICES"] = True

    bayesopt = BayesOptSearch(
        metric="score", mode="max", points_to_evaluate=[initial_search]
    )

    trainable_with_gpu = tune.with_resources(f_optim, {"gpu": 1})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            search_alg=bayesopt,
            num_samples=NUM_ITERATIONS_OPT,
        ),
        param_space=search_space,
    )

    tuner.fit()
    """
