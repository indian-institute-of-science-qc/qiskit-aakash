import subprocess
import numpy as np
import os
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def make_circuit(num_qubits, num_instructions):

    outfile = open('circuit.py', 'w')
    p = subprocess.Popen(
        f"python random_circuit_generator.py {num_qubits} {num_instructions}".split(), stdout=outfile)
    outfile.close()
    p.communicate()


with open("results.csv", 'w') as f:
    pass

for i in range(10):
    qubits = 5
    num_gates = i*100 + 100
    make_circuit(qubits, num_gates)
    print("Number of gates", num_gates)
    # print("Number of qubits", qubits)
    print("Running qasm")
    p1 = subprocess.Popen(f"python circuit.py 0", shell=True)
    p1.communicate()
    print("Running dm")
    p2 = subprocess.Popen(f"python circuit.py 1", shell=True)
    p2.communicate()
    with open("./results.csv", "a") as f:
        f.write(f"{num_gates},{qubits}\n")


subprocess.run("python ./plot.py".split())
