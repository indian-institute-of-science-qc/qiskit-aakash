import subprocess
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')


def make_circuit(num_qubits, num_instructions):

    outfile = open('circuit.py', 'w')
    p = subprocess.Popen(
        f"python random_circuit_generator.py {num_qubits} {num_instructions}".split(), stdout=outfile)
    outfile.close()
    p.communicate()


with open("results.csv", 'w') as f:
    pass


def record_memory_usage():
    memory_data = glob.glob("*.dat")[0]

    # print(memory_data)

    data = np.genfromtxt(memory_data, skip_header=1)[:, 1:]
    run_time = data[:, 1].max() - data[:, 1].min()
    max_memory = data[:, 0].max() - data[:, 0].min()
    return run_time, max_memory


total_runtime1 = 0.0
total_runtime2 = 0.0
total_memory_usage1 = 0.0
total_memory_usage2 = 0.0

for i in range(3):
    qubits = 3+i
    num_gates = 100
    make_circuit(qubits, num_gates)
    print("Number of qubits,gates", qubits, num_gates)
    print("Running qasm")
    p1 = subprocess.Popen(
        f"mprof clean && mprof run --include-children python circuit.py 0", shell=True)
    p1.communicate()

    total_runtime1, total_memory_usage1 = record_memory_usage()

    print("Running dm")
    p2 = subprocess.Popen(
        f"mprof clean && mprof run --include-children python circuit.py 1", shell=True)
    p2.communicate()
    total_runtime2, total_memory_usage2 = record_memory_usage()

    with open("./results.csv", "a") as f:
        f.write(
            f"{total_memory_usage1},{total_memory_usage2},{total_runtime1},{total_runtime2},{num_gates},{qubits}\n")


subprocess.run("python ./plot.py".split())
