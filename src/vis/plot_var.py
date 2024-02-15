import numpy as np
import matplotlib.pyplot as plt
import sys


var= np.load(sys.argv[1])
plt.figure()
plt.plot(var[:,30000:32000])
plt.title(sys.argv[1])
plt.show()
