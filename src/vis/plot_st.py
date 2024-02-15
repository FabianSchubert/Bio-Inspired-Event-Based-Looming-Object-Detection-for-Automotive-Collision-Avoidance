import numpy as np
import matplotlib.pyplot as plt
import sys


t= np.load(sys.argv[1]+"_spike_t.npy")
ID= np.load(sys.argv[1]+"_spike_ID.npy")
print(ID.shape)
print(ID[0,-500:])
plt.figure()
plt.scatter(t,ID,marker=".",s=0.5)
plt.title(sys.argv[1])
plt.show()
