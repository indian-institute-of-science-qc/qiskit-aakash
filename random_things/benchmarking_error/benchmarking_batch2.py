import pickle
import subprocess
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import glob
from scipy import linalg
plt.style.use('seaborn')


def change_options(error, change_per_iter, i):
    options = {

        "chop_threshold": 1e-15,
        "thermal_factor": 0.,
        "decoherence_factor": 1.,
        "depolarization_factor": 1.,
        "bell_depolarization_factor": 1.,
        "decay_factor": 1.,
        "rotation_error": {'rx': [1., 0.], 'ry': [1., 0.], 'rz': [1., 0.]},
        "tsp_model_error": [1., 0.]
    }

    options[error] += change_per_iter*i

    # print(f"Running for {options[error]}")
    with open("./options.pkl", 'wb') as f:
        pickle.dump(options, f)


def make_circuit(num_qubits, num_instructions):

    outfile = open('circuit.py', 'w')
    p = subprocess.Popen(
        f"python random_circuit_generator.py {num_qubits} {num_instructions}".split(), stdout=outfile)
    outfile.close()
    p.communicate()


with open("results_error.csv", 'w') as f:
    pass
with open('./change.csv', 'w') as f:
    pass


def find_change(without_error, with_error):

    return str(np.trace(np.dot(without_error, with_error)))


# def find_change(without_error, with_error):

#     return str(np.trace(linalg.sqrtm(np.dot(np.conjugate(without_error - with_error), (without_error - with_error)))))


def record_memory_usage():
    memory_data = glob.glob("*.dat")[0]

    # print(memory_data)

    data = np.genfromtxt(memory_data, skip_header=1)[:, 1:]
    run_time = data[:, 1].max() - data[:, 1].min()
    max_memory = data[:, 0].max() - data[:, 0].min()
    return run_time, max_memory


total_runtime1 = 0.0
total_memory_usage1 = 0.0


change_options('thermal_factor', 0, 1)

qubits = 5
num_gates = 10
make_circuit(qubits, num_gates)

print("Running without errors")
p2 = subprocess.Popen(
    f"mprof clean && mprof run --include-children python circuit.py 0", shell=True)
p2.communicate()

without_error = np.loadtxt('without_error.csv', dtype=np.complex128)


options = {

    "chop_threshold": 1e-15,
    "thermal_factor": 0.,
    "decoherence_factor": 1.,
    "depolarization_factor": 1.,
    "bell_depolarization_factor": 1.,
    "decay_factor": 1.,
    "rotation_error": {'rx': [1., 0.], 'ry': [1., 0.], 'rz': [1., 0.]},
    "tsp_model_error": [1., 0.]
}
error_vary = 'decay_factor'
change_per_iteration = -0.001
asd = 1
bbb = 'rz'
n = 5

for i in range(n):

    qubits = 5
    num_gates = 10*(i+1)
    make_circuit(qubits, num_gates)

    title = f'{error_vary}= {options[error_vary]+change_per_iteration}'

    change_options('decay_factor', 0, 1)
    print("Running without errors")
    p2 = subprocess.Popen(
        f"mprof clean && mprof run --include-children python circuit.py 0", shell=True)
    p2.communicate()

    without_error = np.loadtxt('without_error.csv', dtype=np.complex128)
    # print(f"Running for {options[error]}")

    change_options('decay_factor', change_per_iteration, 1)
    print("Running with errors")
    p2 = subprocess.Popen(
        f"mprof clean && mprof run --include-children python circuit.py 1", shell=True)
    p2.communicate()
    total_runtime2, total_memory_usage2 = record_memory_usage()

    with open("./results_error.csv", "a") as f:
        f.write(
            f"{total_memory_usage2},{total_runtime2},{num_gates},{qubits}\n")

    with open('./change.csv', 'a') as f:
        f.write(find_change(without_error, np.loadtxt(
            'with_error.csv', dtype=np.complex128)))
        # f.write(f",{options[error_vary]+ change_per_iteration*i}\n")
        f.write(f",{num_gates}\n")

subprocess.run(f"python ./plot2.py {title}".split())
