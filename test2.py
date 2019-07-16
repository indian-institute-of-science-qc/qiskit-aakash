from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
import numpy as np

backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator') # run on local simulator by default
options = {}
q = QuantumRegister(5)
c = ClassicalRegister(5)
circ = QuantumCircuit(q,c)
def generator(k):
    return (np.pi*2)/(2**k)

circ.h(q[0])
circ.h(q[1])
circ.h(q[2])
circ.cx(q[2],q[3])
circ.cx(q[2],q[4])
circ.h(q[1])
circ.cu1(generator(2),q[1],q[0])
circ.h(q[0])
circ.cu1(generator(3), q[1], q[2])
circ.cu1(generator(2), q[0], q[2])
circ.h(q[2])
circ.measure(q[0],c[0])
circ.measure(q[1],c[1])
circ.measure(q[2],c[2])
circuits = [circ]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)