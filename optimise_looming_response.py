#! /usr/bin/env python3
import numpy as np

from src.test_looming_sequences import evaluate_response
from src.looming_sim.default_settings import p as p_default

from sklearn.metrics import f1_score

from scipy.optimize import minimize
from scipy.optimize import dual_annealing

from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune

import os

from enum import Enum

from cProfile import Profile
from pstats import SortKey, Stats


class ParamType(Enum):
    POS = 0
    NEG = 1
    POSNEG = 2


OPT_PARAMS = {
    "SCALE_KERNEL_G": ParamType.POS,
    "SCALE_KERNEL_D": ParamType.POS,
    "W_IN_S_E": ParamType.POS,
    "W_S_LGMD": ParamType.POS,
    "W_IN_LGMD": ParamType.NEG,
    "TAU_SYN_IN_S_I": ParamType.POS,
    "TAU_SYN_IN_S_E": ParamType.POS,
    "TAU_IN_LGMD": ParamType.POS,
    "THRESH_IN_LGMD": ParamType.POS,
    "TAU_MEM_P": ParamType.POS,
    "V_THRESH_P": ParamType.POSNEG,
    "V_RESET_P": ParamType.POSNEG,
    "TAU_MEM_S": ParamType.POS,
    "V_THRESH_S": ParamType.POSNEG,
    "V_RESET_S": ParamType.POSNEG,
    "TAU_MEM_LGMD": ParamType.POS,
    "V_THRESH_LGMD": ParamType.POSNEG,
    "V_RESET_LGMD": ParamType.POSNEG,
    "SYN_DELAY_LGMD": ParamType.POS,
}

OPT_PARAMS_LIMITS = []
for k, pt in OPT_PARAMS.items():
    v = p_default[k]
    if pt == ParamType.POS:
        OPT_PARAMS_LIMITS.append((0.0, v * 2.0))
    elif pt == ParamType.NEG:
        OPT_PARAMS_LIMITS.append((v * 2.0, 0.0))
    else:
        OPT_PARAMS_LIMITS.append((-np.abs(2.0 * v), np.abs(2.0 * v)))


def f_optim_neg(x_optim):
    x_optim_transf = [
        lim[0] + x * (lim[1] - lim[0]) for x, lim in zip(x_optim, OPT_PARAMS_LIMITS)
    ]

    # x_optim_transf = [x_optim[k] * search_scale[k] + x0[k] for k in range(len(x_optim))]

    score, *_ = evaluate_response(
        x_optim_transf,
        DATA_FOLD,
        N_SAMPLE,
        SEED_RNG_SAMPLE,
        p_default,
        SIGM_SDF,
        SCALE_SDF,
        metric=f1_score,
    )
    return -score


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
    N_SAMPLE = 3
    SEED_RNG_SAMPLE = 42
    SIGM_SDF = 100.0
    SCALE_SDF = 1.0

    NUM_ITERATIONS_OPT = 1

    # SCALE_SIMPLEX_INIT = 0.25

    dim_opt = len(OPT_PARAMS)

    x0 = [0.5] * dim_opt

    with Profile() as profile:
        minimize(
            f_optim_neg,
            [0.5] * dim_opt,
        )

        dual_annealing(
            f_optim_neg,
            [(0.0, 1.0)] * dim_opt,
            maxiter=NUM_ITERATIONS_OPT,
            no_local_search=True,
        )
        (Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats())

    __import__("ipdb").set_trace()
