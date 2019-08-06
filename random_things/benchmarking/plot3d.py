from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import os
import time
import shutil
import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn')


data = np.genfromtxt("./results.csv", delimiter=',')
date_time = datetime.datetime.now()


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos = data[:, -1]
ypos = data[:, -2]
num_elements = len(xpos)
zpos = np.zeros(num_elements)
dx = np.ones(num_elements)*0.5
dy = np.ones(num_elements)*10
dz = data[:, 1]

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
plt.show()
