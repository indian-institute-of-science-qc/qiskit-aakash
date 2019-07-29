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
qc.u3(0.00462,5.7863,3.76894,q[4])
qc.cx(q[4],q[3])
qc.u3(0.47063,4.6395,5.61547,q[0])
qc.u2(0.00525,4.68752,q[0])
qc.cx(q[0],q[4])
qc.u1(4.46122,q[0])
qc.cx(q[3],q[0])
qc.cx(q[1],q[3])
qc.ccx(q[0],q[4],q[2])
qc.u3(1.27919,3.58846,2.27442,q[0])
qc.u1(2.43651,q[1])
qc.u3(2.57253,4.1333,4.12562,q[3])
qc.ccx(q[2],q[4],q[0])
qc.u1(2.21122,q[1])
qc.u2(2.82153,3.9878,q[0])
qc.ccx(q[3],q[1],q[4])
qc.u3(1.34068,2.13384,4.28283,q[3])
qc.cx(q[3],q[4])
qc.u3(1.94386,3.6739,0.99485,q[2])
qc.ccx(q[3],q[0],q[1])
qc.u2(6.26086,3.59261,q[0])
qc.cx(q[0],q[3])
qc.u2(5.27153,4.75122,q[0])
qc.u3(2.20888,1.56194,1.41783,q[3])
qc.u3(2.62708,0.90311,3.72108,q[0])

with open('./options.pkl', 'rb') as f:
    options= pickle.load(f)


circuits = [qc]
if which_back == 0:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.savetxt("./without_error.csv", result['results'][0]['data']['densitymatrix'])
    # with open("./results_error.csv", 'a') as f:
    #     f.write(f"{run_time},")

if which_back == 1:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    np.savetxt("./with_error.csv", result['results'][0]['data']['densitymatrix'])
    with open("./results_error.csv", 'a') as f:
        f.write(f"{run_time},")

