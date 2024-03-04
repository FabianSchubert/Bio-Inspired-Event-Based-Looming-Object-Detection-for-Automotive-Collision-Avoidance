import numpy as np
import pandas as pd
from src.test_looming_sequences import evaluate_response

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.default_settings import p as p_default

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# """
DATA_FOLD = "/mnt/data0/prophesee_data/ATIS_Automotive_Detection_Dataset/test_a/"
N_SAMPLE = 10
SEED_RNG_SAMPLE = 52
SIGM_SDF = 100.0
SCALE_SDF = 1.0

optim_results = pd.read_csv("optim_log.txt")

p_optim = dict(optim_results.loc[optim_results.shape[0] - 1])
del p_optim["SCORE"]
p_optim["V_THRESH_LGMD"] = 0.0

score, iou_ground_truth, lgmd_predictions = evaluate_response(
    p_optim, DATA_FOLD, N_SAMPLE, SEED_RNG_SAMPLE, p_default, SIGM_SDF, SCALE_SDF
)

np.savez(
    "test_optim_lgmd_results.npz",
    score=score,
    truth=iou_ground_truth,
    predict=lgmd_predictions,
)
# """

results = np.load("test_optim_lgmd_results.npz")
score = float(results["score"])
truth = results["truth"]
predict = results["predict"]

conf = confusion_matrix(truth, predict)

conf_norm_true = confusion_matrix(truth, predict, normalize="true")
conf_norm_predict = confusion_matrix(truth, predict, normalize="pred")

__import__("pdb").set_trace()

fig, ax = plt.subplots(1, 2)
ax[0].grid(False)
ax[1].grid(False)

sns.heatmap(conf_norm_predict, ax=ax[0])
sns.heatmap(conf_norm_true, ax=ax[1])

plt.show()
