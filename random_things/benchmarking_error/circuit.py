import numpy as np
import filecmp
import pickle
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

import time
import sys

which_back = int(sys.argv[1])


backend1 = BasicAer.get_backend('dm_simulator')
options = {}
q = QuantumRegister(5)
c = ClassicalRegister(5)
qc = QuantumCircuit(q, c)
qc.ccx(q[0], q[4], q[1])
qc.ccx(q[1], q[3], q[0])
qc.cx(q[0], q[2])
qc.ccx(q[3], q[2], q[4])
qc.u1(2.9327, q[2])
qc.u2(5.81527, 0.88967, q[0])
qc.ccx(q[0], q[2], q[1])
qc.u1(3.06895, q[3])
qc.ccx(q[1], q[0], q[3])
qc.u3(1.94365, 5.83964, 2.17249, q[1])
qc.u3(2.55008, 5.92715, 3.69684, q[1])
qc.ccx(q[1], q[0], q[4])
qc.u1(0.04573, q[0])
qc.u2(2.98289, 0.62354, q[1])
qc.u2(2.65553, 2.31192, q[1])
qc.u3(2.60792, 0.29533, 5.71814, q[3])
qc.ccx(q[1], q[2], q[4])
qc.ccx(q[2], q[1], q[4])
qc.ccx(q[1], q[4], q[3])
qc.u2(6.23308, 0.5164, q[3])
qc.u2(4.78739, 0.29771, q[2])
qc.u2(1.7429, 6.19588, q[4])
qc.u2(2.42789, 0.41046, q[3])
qc.ccx(q[3], q[1], q[4])
qc.u2(1.87799, 5.08101, q[3])
qc.u1(0.93038, q[1])
qc.cx(q[2], q[1])
qc.cx(q[4], q[3])
qc.ccx(q[3], q[0], q[1])
qc.ccx(q[0], q[4], q[1])
qc.cx(q[4], q[0])
qc.u2(2.4685, 2.13931, q[2])
qc.u1(0.86081, q[0])
qc.u1(1.48519, q[2])
qc.u3(2.67487, 5.20018, 4.21994, q[0])
qc.cx(q[2], q[3])
qc.ccx(q[0], q[4], q[2])
qc.ccx(q[2], q[0], q[3])
qc.cx(q[2], q[1])
qc.u3(1.72873, 3.91246, 0.11104, q[2])
qc.u2(5.22097, 3.35408, q[4])
qc.ccx(q[1], q[0], q[4])
qc.ccx(q[3], q[0], q[2])
qc.cx(q[3], q[1])
qc.u2(3.51528, 0.8204, q[4])
qc.cx(q[0], q[1])
qc.ccx(q[4], q[2], q[0])
qc.u1(4.13758, q[1])
qc.ccx(q[4], q[2], q[1])
qc.u1(4.23015, q[3])

with open('./options.pkl', 'rb') as f:
    options = pickle.load(f)


circuits = [qc]
if which_back == 0:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.save("./without_error", result['results'][0]['data']['densitymatrix'])
    # with open("./results_error.csv", 'a') as f:
    #     f.write(f"{run_time},")

if which_back == 1:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.save("./with_error", result['results'][0]['data']['densitymatrix'])
    with open("./results_error.csv", 'a') as f:
        f.write(f"{run_time},")
