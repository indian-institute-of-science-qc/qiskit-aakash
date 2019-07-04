import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator')
options = {}

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
circ = QuantumCircuit(q, c)

circ.h(q[1])
circ.cx(q[1], q[2])
circ.h(q[0])
circ.cx(q[0], q[1])
circ.h(q[0])
circ.measure(q[0], c[0])
circ.measure(q[1], c[1])
if c[0] == 1:
    circ.z(q[2])
if c[1] == 1:
    circ.x(q[2])
circ.measure(q[2], c[2])

circuits = [circ]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)
