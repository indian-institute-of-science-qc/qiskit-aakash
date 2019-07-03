from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
import numpy as np

backend = BasicAer.get_backend('qasm_simulator') # run on local simulator by default
options = {}
q = QuantumRegister(4)
c = ClassicalRegister(4)
qc = QuantumCircuit(q, c)
qc.u3(0.2,1,2.0,q[0])
qc.u3(0.6,2.2,3.1,q[3])
qc.u2(1,2.5,q[3])
qc.u2(0.4,2.5,q[2])
#qc.cx(q[0],q[2])
qc.u1(0.5,q[2])
#qc.cx(q[0],q[3])
qc.x(q[0])
qc.x(q[1])
#qc.ccx(q[0],q[1],q[2])
#qc.ccx(q[1],q[3],q[0])
circuits = [qc]
job = execute(circuits, backend, **options)
result = job.result()
print(result)