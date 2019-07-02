# importing Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

backend = BasicAer.get_backend('qasm_simulator') # run on local simulator by default
options = {}
#options = {'initial_densitymatrix' :'00101', 'custom_densitymatrix' : 'binary_string'}
# Creating registers
#number_of_qubits = len(options['initial_densitymatrix'])
q = QuantumRegister(2)
c = ClassicalRegister(2)

# quantum circuit to make an entangled bell state
qc = QuantumCircuit(q, c)
#qc.h(q[0])
qc.x(q[0])
qc.z(q[0])
#qc.rx(3,q[0])
#qc.rz(3,q[1])
#qc.ry(2,q[1])
qc.h(q[0])
#qc.z(q[1])
#qc.measure(q[0], c[0])
#qc.y(q[1])
#qc.h(q[1])
#qc.x(q[1])
#qc.x(q[1])
qc.x(q[1])
#qc.measure(q[0], c[1])
#qc.cx(q[1], q[0])
qc.cx(q[0],q[1])
#qc.measure(q,c)
circuits = [qc]
job = execute(circuits, backend, **options)
result = job.result()
print(result)

#print(result.get_statevector())
