# importing Qiskit
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

backend1 = BasicAer.get_backend('dm_simulator') 
#backend2 = BasicAer.get_backend('qasm_simulator')

# run on local simulator by default
#binary_stri = 
#custom = 
#options = {'custom_de':'bin', 'init_de':binar}
options = {}
#options = {'initial_densitymatrix' :'00101', 'custom_densitymatrix' : 'binary_string'}
# Creating registers
#number_of_qubits = len(options['initial_densitymatrix'])
q = QuantumRegister(6)
c = ClassicalRegister(6)
# quantum circuit to make an entangled bell state
qc = QuantumCircuit(q, c)
#qc.h(q[0])
#qc.x(q[0])
#qc.z(q[0])
#qc.rx(3,q[0])
#qc.rz(3,q[1])
#qc.ry(2,q[1])
#qc.h(q[0])
#qc.z(q[1])
#qc.measure(q[0], c[0])
#qc.cx(q[1],q[2])
#qc.x(q[1])
#qc.measure(q, c)
#qc.cx(q[1], q[0])
#qc.cx(q[0],q[2])
#qc.cx(q[1],q[0])
#qc.cx(q[0],q[1])
#qc.x(q[0])
#qc.h(q[1])
#qc.cx(q[1],q[0])
#qc.u3(3,4,5,)
vec = np.array([1,2,3])
qc.measure(q[0],c[0])
#qc.measure(q[1],c[1])
#qc.measure(q[2],c[2])
qc.measure(q[3],c[3], basis='Bell', add_param='34')
qc.measure(q[5],c[5])
#qc.measure(q[2],c[2])
circuits = [qc]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
#job = execute(circuits, backend2, **options)
#result = job.result()
#print(result)
#print(result.get_statevector())