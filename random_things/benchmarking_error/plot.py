import subprocess
import numpy as np
import os
import time
import shutil
import datetime
import matplotlib.pyplot as plt
import sys

title = sys.argv[1]

for i in sys.argv[2:]:
    title += i


plt.style.use('seaborn')


data = np.around(np.genfromtxt("./change.csv", delimiter=',',
                               dtype=np.complex128).real, decimals=2)
other_data = np.around(np.genfromtxt(
    "./results_error.csv", delimiter=','), decimals=2)
date_time = datetime.datetime.now()

# print(data)

plt.xticks(range(data[:, 1].shape[0]), data[:, -1])
plt.plot(data[:, 0]/data[0, 0])

plt.title(title +
          f", qubits={int(other_data[0,-1])}, gates={int(other_data[0,-2])}")
plt.xlabel("Magnitude of error")
plt.ylabel("Deviation from the zero error density matrix")
plt.savefig(f"graph_error.png")
plt.show()


save_dir = f"./results/{title},  qubits={int(other_data[0,-1])} -> {int(other_data[-1,-1])},gates={int(other_data[0,-2])} -> {int(other_data[-1,-2])},{date_time}/"
os.makedirs(save_dir)

shutil.copy("./graph_error.png", save_dir +
            f"{title},  qubits={int(other_data[0,-1])} -> {int(other_data[-1,-1])},gates={int(other_data[0,-2])} -> {int(other_data[-1,-2])},{date_time}.png")
shutil.copy("./results_error.csv", save_dir +
            f"results_error {title},  qubits={int(other_data[0,-1])} -> {int(other_data[-1,-1])},gates={int(other_data[0,-2])} -> {int(other_data[-1,-2])},{date_time}.csv")
shutil.copy("./change.csv", save_dir +
            f"change {title},  qubits={int(other_data[0,-1])} -> {int(other_data[-1,-1])},gates={int(other_data[0,-2])} -> {int(other_data[-1,-2])},{date_time}.csv")
