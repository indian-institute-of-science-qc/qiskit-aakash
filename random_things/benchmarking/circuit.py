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
q = QuantumRegister(5)
c = ClassicalRegister(5)
qc = QuantumCircuit(q, c)
qc.u1(3.29685,q[0])
qc.u1(4.60542,q[4])
qc.u1(0.20547,q[0])
qc.u1(2.25995,q[4])
qc.u2(1.52241,3.56112,q[3])
qc.ccx(q[3],q[2],q[0])
qc.u1(3.3715,q[2])
qc.u2(0.70507,2.16402,q[3])
qc.u3(0.08407,3.404,0.95869,q[4])
qc.u2(1.10847,2.30582,q[0])
qc.u1(1.2665,q[2])
qc.cx(q[3],q[4])
qc.ccx(q[4],q[3],q[0])
qc.cx(q[4],q[0])
qc.cx(q[4],q[3])
qc.u1(0.62342,q[4])
qc.cx(q[2],q[3])
qc.ccx(q[4],q[3],q[2])
qc.ccx(q[4],q[3],q[0])
qc.u3(2.27064,1.17766,5.63314,q[1])
qc.ccx(q[4],q[1],q[0])
qc.u3(0.60091,1.49441,3.75447,q[3])
qc.u3(2.33646,2.09455,4.27911,q[4])
qc.cx(q[4],q[0])
qc.u3(2.23959,0.41202,1.31574,q[1])
qc.u2(2.89971,1.58028,q[1])
qc.u3(2.83469,1.75224,3.17699,q[4])
qc.u1(2.58588,q[1])
qc.u1(5.41976,q[4])
qc.cx(q[0],q[1])
qc.u2(1.20126,1.29081,q[1])
qc.u2(4.39114,0.32681,q[3])
qc.u2(4.9833,3.06059,q[0])
qc.u3(1.90361,1.99134,3.55517,q[3])
qc.cx(q[1],q[0])
qc.u1(2.45336,q[2])
qc.u3(1.08805,2.16589,0.09071,q[3])
qc.ccx(q[2],q[4],q[0])
qc.cx(q[4],q[3])
qc.ccx(q[0],q[3],q[4])
qc.u3(3.11063,3.54241,6.17759,q[1])
qc.cx(q[0],q[3])
qc.ccx(q[3],q[4],q[0])
qc.u3(3.04612,3.10948,6.25864,q[2])
qc.u1(2.17013,q[3])
qc.ccx(q[2],q[0],q[1])
qc.cx(q[1],q[3])
qc.cx(q[0],q[3])
qc.u3(2.16308,1.07773,6.0339,q[0])
qc.cx(q[4],q[3])
qc.u1(2.02502,q[3])
qc.u1(1.53734,q[0])
qc.u1(5.24093,q[0])
qc.ccx(q[3],q[2],q[4])
qc.ccx(q[1],q[4],q[0])
qc.u2(2.90856,5.46448,q[4])
qc.u3(1.66403,6.22748,0.3194,q[1])
qc.ccx(q[1],q[3],q[4])
qc.cx(q[2],q[4])
qc.u1(3.76752,q[0])
qc.u3(0.29688,1.02963,5.4529,q[0])
qc.u3(1.53022,0.47041,2.52873,q[2])
qc.cx(q[2],q[4])
qc.u3(0.95817,1.77454,4.28308,q[3])
qc.ccx(q[0],q[2],q[4])
qc.cx(q[2],q[0])
qc.ccx(q[1],q[0],q[2])
qc.u2(1.923,0.73922,q[1])
qc.u1(0.86999,q[2])
qc.cx(q[4],q[3])
qc.ccx(q[0],q[2],q[1])
qc.u2(6.16811,3.42893,q[4])
qc.u3(2.83664,1.40614,1.28574,q[3])
qc.u1(0.82993,q[1])
qc.u3(2.31905,2.27534,1.96145,q[4])
qc.u1(5.12073,q[2])
qc.u3(0.25076,6.01647,1.38741,q[3])
qc.u2(4.25612,2.47982,q[1])
qc.u1(5.02056,q[4])
qc.ccx(q[4],q[2],q[3])
qc.cx(q[0],q[3])
qc.cx(q[0],q[1])
qc.cx(q[4],q[1])
qc.u1(2.61999,q[0])
qc.ccx(q[3],q[0],q[2])
qc.u1(4.57313,q[1])
qc.u3(1.71764,5.91688,6.13,q[2])
qc.u3(2.84898,4.90915,0.30935,q[0])
qc.ccx(q[2],q[1],q[4])
qc.u3(1.36766,5.85674,5.37609,q[0])
qc.u1(6.24287,q[1])
qc.u3(1.44903,0.26644,1.67226,q[3])
qc.u2(5.14753,1.33399,q[2])
qc.u3(2.19486,1.04538,4.2062,q[2])
qc.u2(4.15743,2.07575,q[3])
qc.u1(4.56706,q[4])
qc.u3(2.18461,2.03546,2.15608,q[0])
qc.u2(4.761,1.5744,q[4])
qc.u3(2.09066,2.87915,5.50518,q[3])
qc.cx(q[2],q[3])

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




