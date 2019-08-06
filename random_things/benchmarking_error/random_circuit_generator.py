import numpy as np
import random
from random import randint
import sys
import pickle

qubits = int(sys.argv[1])
num_gates = int(sys.argv[2])

print('import numpy as np')
print('import filecmp')
print('import pickle')
print('from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister')
print('from qiskit import BasicAer, execute')
print("""
import time
import sys

which_back = int(sys.argv[1])

""")
print("backend1 = BasicAer.get_backend('dm_simulator')")

print('options = {}')
print(f'q = QuantumRegister({qubits})')
print(f'c = ClassicalRegister({qubits})')
print('qc = QuantumCircuit(q, c)')


noq = qubits
l = ['u1', 'u2', 'u3', 'cx', 'ccx']
for i in range(num_gates):
    x = random.randint(0, 4)
    if x == 0:
        print('qc.{}({},q[{}])'.format((l[x]), round(random.uniform(0, 2*np.pi), 5),
                                       (random.randint(0, noq-1))))
    elif x == 1:
        print('qc.{}({},{},q[{}])'.format(l[x], round(random.uniform(0, 2*np.pi), 5),
                                          round(random.uniform(0, 2*np.pi), 5), random.randint(0, noq-1)))
    elif x == 2:
        print('qc.{}({},{},{},q[{}])'.format(l[x], round(random.uniform(0, np.pi), 5),
                                             round(random.uniform(0, 2*np.pi), 5), round(random.uniform(0, 2*np.pi), 5), random.randint(0, noq-1)))
    elif x == 3:
        p, q = random.sample(range(noq), 2)
        print('qc.{}(q[{}],q[{}])'.format(l[x], p, q))
    elif x == 4:
        a, b, c = random.sample(range(noq), 3)
        print('qc.{}(q[{}],q[{}],q[{}])'.format(l[x], a, b, c))


# print("qc.measure(q[0],c[0],'Z')")
# print("qc.measure(q[1],c[2],'X')")
# print("qc.measure(q[2],c[0],'Z')")
# print("qc.measure(q[1],c[2],'X')")
# # print("qc.measure(q[3],c[0],'Z')")
# print("qc.measure(q[1],c[2],'X')")
# print("qc.measure(q[0],c[0],'Z')")
# print("qc.measure(q[3],c[2],'X')")

print(
    """
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
"""
)
