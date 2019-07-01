# importing Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

backend = BasicAer.get_backend('dm_simulator') # run on local simulator by default
options = {}
# Creating registers
q = QuantumRegister(3)
c = ClassicalRegister(3)

# quantum circuit to make an entangled bell state
qc = QuantumCircuit(q, c)
qc.x(q[0])
#qc.x(q[0])
#qc.z(q[0])
qc.rx(3,q[0])
qc.rz(3,q[1])
qc.ry(2,q[1])
qc.h(q[0])
qc.z(q[1])
#qc.measure(q[0], c[0])
#qc.y(q[1])
#qc.h(q[1])
qc.x(q[1])
#qc.x(q[1])
#qc.measure(q[0], c[1])
qc.cx(q[1], q[0])
qc.cx(q[0],q[2])
#qc.measure(q,c)
circuits = [qc]
job = execute(circuits, backend, shots=2, **options)
result = job.result()
print(result)

#print(result.get_statevector())
