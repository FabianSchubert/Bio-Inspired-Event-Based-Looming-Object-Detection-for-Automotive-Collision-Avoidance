import numpy as np
import matplotlib.pyplot as plt
import sys

from src.default_settings import p

WIDTH, HEIGHT = p["INPUT_WIDTH"], p["INPUT_HEIGHT"]


evt = np.load(sys.argv[1])
plt.figure()
ids = evt[:, 1] * WIDTH + evt[:, 0]
plt.scatter(evt[evt[:, 3] > 0, 2], ids[evt[:, 3] > 0], s=0.2, color="g")
plt.scatter(evt[evt[:, 3] < 0, 2], ids[evt[:, 3] < 0], s=0.2, color="r")
plt.xlabel("event time (s)")
plt.ylabel("neuron ID (unitless)")
plt.title(sys.argv[1])

plt.figure()
plt.plot(evt[:, 2])
plt.xlabel("event index (unitless)")
plt.ylabel("event time (s)")
plt.title(sys.argv[1])
plt.show()
