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
qc.cx(q[3],q[0])
qc.ccx(q[4],q[2],q[0])
qc.u2(1.57279,6.174,q[4])
qc.u3(1.92453,4.89742,3.19918,q[3])
qc.u1(1.5909,q[0])
qc.ccx(q[3],q[0],q[2])
qc.u3(2.16565,5.96902,2.20393,q[2])
qc.u3(1.61541,4.03454,0.91897,q[3])
qc.u2(1.19494,6.17827,q[4])
qc.u2(0.31171,3.63562,q[3])
qc.u2(5.3993,3.29372,q[0])
qc.cx(q[3],q[1])
qc.u2(5.03359,0.06738,q[2])
qc.u3(1.47081,5.07824,3.02496,q[3])
qc.u3(2.27287,0.1834,1.72037,q[0])
qc.u2(2.11424,5.94949,q[0])
qc.u1(3.52702,q[4])
qc.cx(q[1],q[3])
qc.u1(5.39909,q[3])
qc.ccx(q[2],q[3],q[0])
qc.u1(4.6977,q[3])
qc.u3(0.26038,6.23279,2.61051,q[2])
qc.u1(4.06848,q[2])
qc.ccx(q[0],q[3],q[4])
qc.ccx(q[4],q[0],q[2])

with open('./options.pkl', 'rb') as f:
    options= pickle.load(f)


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

