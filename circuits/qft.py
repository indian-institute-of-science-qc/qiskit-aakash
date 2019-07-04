import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator')
options = {}

def generator(k):
    return (np.pi*2)/(2**k)


num_of_qubits = 5
q = QuantumRegister(num_of_qubits, 'q')
circ = QuantumCircuit(q)

'''circ.h(q[0])
circ.cu1(generator(2),q[1],q[0])

circ.h(q[1])
circ.cu1(generator(3), q[2], q[0])
circ.cu1(generator(2), q[2], q[1])

circ.h(q[2])
circ.cu1(generator(4), q[3], q[0])
circ.cu1(generator(3), q[3], q[1])
circ.cu1(generator(2), q[3], q[2])

circ.h(q[3])
circ.cu1(generator(5), q[4], q[0])
circ.cu1(generator(4), q[4], q[1])
circ.cu1(generator(3), q[4], q[2])
circ.cu1(generator(2), q[4], q[3])

circ.h(q[4])'''

for wire in range (num_of_qubits-1):
    circ.h(q[wire])
    for rotation in range(wire+1):
        circ.cu1(generator(wire+2-rotation), q[wire+1], q[rotation])
circ.h(q[num_of_qubits-1])

circuits = [circ]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)