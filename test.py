# importing Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute

<<<<<<< HEAD
backend = BasicAer.get_backend('dm_simulator') # run on local simulator by default
=======
backend1 = BasicAer.get_backend('dm_simulator') 
backend2 = BasicAer.get_backend('qasm_simulator')

# run on local simulator by default
#binary_stri = 
#custom = 
#options = {'custom_de':'bin', 'init_de':binar}
>>>>>>> 6b6a34885b988cf90af5fa71dbadd3037e23af42
options = {}
#options = {'initial_densitymatrix' :'01', 'custom_densitymatrix' : 'binary_string'}
# Creating registers
#number_of_qubits = len(options['initial_densitymatrix'])
q = QuantumRegister(3)
c = ClassicalRegister(3)

# quantum circuit to make an entangled bell state
qc = QuantumCircuit(q, c)
#qc.h(q[0])
#qc.x(q[0])
#qc.z(q[0])
<<<<<<< HEAD
qc.rx(3,q[0])
qc.rz(3,q[0])
qc.ry(2,q[0])
qc.z(q[0])
#qc.h(q[0])
qc.z(q[1])
qc.cx(q[1], q[0])
qc.z(q[1])
qc.cx(q[1], q[0])
qc.z(q[1])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.z(q[1])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
qc.z(q[1])
qc.cx(q[1], q[0])
qc.cx(q[1], q[0])
=======
#qc.rx(3,q[0])
#qc.rz(3,q[1])
#qc.ry(2,q[1])
#qc.h(q[0])
#qc.z(q[1])
>>>>>>> 6b6a34885b988cf90af5fa71dbadd3037e23af42
#qc.measure(q[0], c[0])
qc.u3(0.2, 1, 2.0, q[0])
qc.u3(0.6, 2.2, 3.1, q[1])
#qc.u2(1, 2.5, q[3])
#qc.u1(0.5, q[2])
qc.u2(0.4, 2.5, q[1])
qc.u1(0.5, q[1])
#qc.u2(0.5, 0.1,q[2])
qc.u3(0.2, 1, 2.0, q[1])
qc.u3(0.8, 2, 1, q[0])
qc.cx(q[1], q[0])
#qc.x(q[1])
<<<<<<< HEAD
qc.cx(q[1], q[2])
#qc.x(q[1])
#qc.x(q[1])
#qc.measure(q[0], c[1])
qc.cx(q[1], q[0])
=======
#qc.ccx(q[0],q[1],q[2])
#qc.h(q[1])
#qc.x(q[1])
#qc.measure(q, c)
#qc.cx(q[1], q[0])
>>>>>>> 6b6a34885b988cf90af5fa71dbadd3037e23af42
#qc.cx(q[0],q[1])
#qc.measure(q,c)
circuits = [qc]
job = execute(circuits, backend1, **options)
result = job.result()
print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)
#print(result.get_statevector())
