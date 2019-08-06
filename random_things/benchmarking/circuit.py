import numpy as np
import filecmp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

import time
import sys

which_back = int(sys.argv[1])


backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator')
options = {}
q = QuantumRegister(10)
c = ClassicalRegister(10)
qc = QuantumCircuit(q, c)
qc.u1(5.34157,q[3])
qc.ccx(q[1],q[3],q[9])
qc.u3(0.01447,1.40313,0.47717,q[1])
qc.u2(6.23842,0.41683,q[2])
qc.u2(2.22555,0.91585,q[6])

circuits = [qc]
if which_back == 0:

    time_start = time.time()
    job = execute(circuits, backend2, **options)
    result = job.result()
    run_time = time.time() - time_start
    with open("./results.csv",'a') as f:
        f.write(f"{run_time},")

if which_back == 1:
    time_start = time.time()
    job = execute(circuits, backend1, **options)
    result = job.result()
    run_time = time.time() - time_start
    with open("./results.csv",'a') as f:
        f.write(f"{run_time},")




