import numpy as np
import matplotlib.pyplot as plt
import sys

WIDTH= 304

evt= np.load(sys.argv[1])
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
onevt= evt[evt[:,3]>0]
ax.scatter3D(onevt[:,2],onevt[:,0],onevt[:,1],".",s=0.2, color='g')
offevt= evt[evt[:,3]<0]
ax.scatter3D(offevt[:,2],offevt[:,0],offevt[:,1],".",s=0.2, color='r')
ax.set_xlabel("time (s)")
ax.set_ylabel("x (pixel)")
ax.set_zlabel("y (pixel)")
plt.title(sys.argv[1])

plt.show()
