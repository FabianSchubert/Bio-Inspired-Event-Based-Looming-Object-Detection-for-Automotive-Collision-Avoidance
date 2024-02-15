import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.classifier.data_io import load_file
import config
from src.default_settings import p

import os

WIDTH = p["INPUT_WIDTH"] // 4


data = load_file(
    os.path.join(config.PATH_PROCESSED_DATA, "balanced_pruned/train_a_td.npy")
)

while True:
    sample = np.random.randint(len(data))

    test_data = data[sample]

    if test_data[1]["class_id"] == 2:
        break

test_evts = test_data[0]
# test_evts = test_evts[test_evts["t"] <= 100000]
print("last event", test_evts["t"].max())
print("num events", len(test_evts))
test_box = test_data[1]

print(test_box["class_id"])

fig, ax = plt.subplots(1, 1)

ax.scatter(test_evts["x"], test_evts["y"], c=test_evts["p"], s=2, cmap="gray")

box = patches.Rectangle(
    (test_box["x"], test_box["y"]),
    test_box["w"],
    test_box["h"],
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)

ax.add_patch(box)

ax.set_facecolor("gray")

ax.set_aspect("equal")
ax.set_xlim([0, WIDTH])

ax.invert_yaxis()

plt.show()

# __import__("ipdb").set_trace()
