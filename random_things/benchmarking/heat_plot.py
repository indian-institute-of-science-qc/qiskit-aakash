import subprocess
import numpy as np
import os
import time
import shutil
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


data = np.genfromtxt("./results.csv", delimiter=',')
date_time = datetime.datetime.now()


data1 = data[1:, [0, 1, 2]]
print(data1)

sns.heatmap(data1, linewidth=0.5)
plt.show()
