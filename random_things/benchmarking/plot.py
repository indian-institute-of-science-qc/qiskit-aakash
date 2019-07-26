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

plt.plot(data[:, -2], data[:, :-2])
plt.title(
    f"qubits={int(data[0,-1])} -> {int(data[-1,-1])}, gates={int(data[0,-2])} -> {int(data[-1,-2])}")
plt.xlabel("Number of gates/qubits")
plt.ylabel("Time (sec)")
plt.savefig(f"graph.png")
plt.show()


save_dir = f"./results/qubits={int(data[0,-1])} -> {int(data[-1,-1])},gates={int(data[0,-2])} -> {int(data[-1,-2])},{date_time}/"
os.makedirs(save_dir)

shutil.copy("./graph.png", save_dir +
            f"qubits={int(data[0,-1])} -> {int(data[-1,-1])},gates={int(data[0,-2])} -> {int(data[-1,-2])},{date_time}.png")
shutil.copy("./results.csv", save_dir +
            f"qubits={int(data[0,-1])} -> {int(data[-1,-1])},gates={int(data[0,-2])} -> {int(data[-1,-2])},{date_time}.csv")
