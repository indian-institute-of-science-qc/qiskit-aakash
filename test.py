# importing Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

backend = BasicAer.get_backend('dm_simulator') # run on local simulator by default

# Creating registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# quantum circuit to make an entangled bell state
qc = QuantumCircuit(q, c)
qc.x(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

circuits = [qc]
job = execute(circuits, backend)
result = job.result()
print(result)
print(result.get_statevector())
