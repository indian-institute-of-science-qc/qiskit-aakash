import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
backend = BasicAer.get_backend('dm_simulator')
options = {}

# generate angle for controlled phase gate R(k)
def generator(k):
    return (np.pi*2)/(2**k)


num_of_qubits = 5
q = QuantumRegister(num_of_qubits, 'q')
c = ClassicalRegister(3,'c')
circ = QuantumCircuit(q,c)
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

circ.draw(output='mpl', line_length=120, scale=0.5)
circuits = [circ]
job = execute(circuits, backend, **options)
result = job.result()
print(result)
