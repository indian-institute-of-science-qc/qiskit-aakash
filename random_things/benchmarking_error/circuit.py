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
q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)
qc.cx(q[1],q[2])
qc.u3(0.0559,4.62133,0.457,q[2])
qc.ccx(q[0],q[1],q[2])
qc.cx(q[1],q[0])
qc.ccx(q[0],q[1],q[2])
qc.u3(0.88309,0.14852,5.53667,q[0])
qc.u3(3.02281,3.07426,5.37091,q[0])
qc.u1(1.37729,q[2])
qc.u2(3.72958,1.32292,q[2])
qc.cx(q[2],q[0])

with open('./options.pkl', 'rb') as f:
    options= pickle.load(f)


circuits = [qc]
if which_back == 0:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.savetxt("./without_error.csv", result['results'][0]['data']['densitymatrix'])
    with open("./results_error.csv", 'a') as f:
        f.write(f"{run_time},")

if which_back == 1:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.savetxt("./with_error.csv", result['results'][0]['data']['densitymatrix'])
    with open("./results_error.csv", 'a') as f:
        f.write(f"{run_time},")

