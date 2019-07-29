from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
import matplotlib as mpl
import numpy as np
backend1 = BasicAer.get_backend('dm_simulator')
options = {'plot': True}
q = QuantumRegister(5)
c = ClassicalRegister(5)
circ = QuantumCircuit(q,c)
circ.h(q[0])
circ.x(q[1])
circ.y(q[2])
circ.cx(q[1],q[3])
circ.ccx(q[1],q[3], q[2])
circ.t(q[4])
circ.h(q[0])
circ.measure(q[0],c[0], basis = 'Bell', add_param = '02')
#circ.measure(q, c, basis = 'Z')
circ.measure(q,c,basis = 'XIXXY')
print(circ)
circuits = [circ]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
