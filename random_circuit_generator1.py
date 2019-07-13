import numpy as np
import filecmp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator')
#options ={}
options = {'rotation_error':[1,0],'ts_model_error':[1,0],'thermal_factor': 0,'depolarization_factor':1, 'decoherence_factor':[1e-9,1e-4], 'decay_factor':[1e-9,1e-4]}
q = QuantumRegister(6)
c = ClassicalRegister(6)
qc = QuantumCircuit(q, c)
qc.u1(4.44838,q[0])
qc.cx(q[5],q[2])
qc.cx(q[1],q[0])
qc.u3(2.10321,2.07585,3.82738,q[3])
qc.cx(q[3],q[5])
qc.u3(0.29059,5.80797,6.07177,q[1])
qc.u3(0.28921,6.17448,0.03872,q[2])
qc.u2(3.74492,6.05253,q[0])
qc.cx(q[1],q[4])
qc.cx(q[1],q[2])
qc.u2(5.87727,0.64555,q[3])
qc.u1(6.21499,q[4])
qc.u1(4.12189,q[2])
qc.cx(q[4],q[1])
qc.u1(5.12045,q[2])
qc.cx(q[3],q[5])
qc.u2(4.1293,2.5261,q[0])
qc.cx(q[5],q[0])
qc.u1(6.04945,q[1])
qc.u2(0.65597,3.00935,q[4])
qc.ccx(q[3],q[1],q[0])
qc.u2(3.14973,3.71082,q[0])
qc.cx(q[1],q[4])
qc.ccx(q[2],q[0],q[4])
qc.measure(q[2],c[2],'Z')
qc.measure(q[2],c[2],'X')
qc.measure(q[2],c[2],'Y')
qc.measure(q[2],c[2],'N',np.array([1,0,0]))
circuits = [qc]

job = execute(circuits, backend1, **options)
result = job.result()
print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)
a = np.loadtxt('a.txt',dtype=complex)
b = np.loadtxt('a1.txt',dtype=complex)
p = a.real
q = a.imag
c = b.real
d = b.imag
if(np.allclose(p,c) and np.allclose(q,d)):
    print('Your result is right.')
else:
    print('Your result did not match!') 
