import numpy as np 
import random
from random import randint
print('import numpy as np')
print('import filecmp')
print('from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister')
print('from qiskit import BasicAer, execute')
print("backend1 = BasicAer.get_backend('dm_simulator')") 
print("backend2 = BasicAer.get_backend('qasm_simulator')")
print("options ={}")
print("#options = {'rotation_error':[1,0],'ts_model_error':[1,0],'thermal_factor': 0,'depolarization_factor':1, 'decoherence_factor':[1e-9,1e-4], 'decay_factor':[1e-9,1e-4]}")
print('q = QuantumRegister(6)')
print('c = ClassicalRegister(6)')
print('qc = QuantumCircuit(q, c)')

noq = 6
l = ['u1','u2','u3','cx','ccx']
for i in range(random.randint(20,500)):
    x = random.randint(0,4)
    if x == 0:
        print('qc.{}({},q[{}])'.format((l[x]), round(random.uniform(0, 2*np.pi),5), 
        (random.randint(0, noq-1))))
    elif x == 1:
        print('qc.{}({},{},q[{}])'.format(l[x], round(random.uniform(0, 2*np.pi),5), 
        round(random.uniform(0, 2*np.pi),5), random.randint(0, noq-1)))
    elif x == 2:
        print('qc.{}({},{},{},q[{}])'.format(l[x], round(random.uniform(0, np.pi),5), 
        round(random.uniform(0, 2*np.pi),5), round(random.uniform(0, 2*np.pi),5), random.randint(0, noq-1)))
    elif x == 3:
        p, q = random.sample(range(noq),2)
        print('qc.{}(q[{}],q[{}])'.format(l[x], p, q))
    elif x == 4:
        a, b, c = random.sample(range(noq),3)
        print('qc.{}(q[{}],q[{}],q[{}])'.format(l[x], a, b, c))

print('circuits = [qc]')
print('job = execute(circuits, backend1, **options)')
print('result = job.result()')
print('print(result)')  
print('job = execute(circuits, backend2, **options)')
print('result = job.result()')
print('print(result)')
print("a = np.loadtxt('a.txt',dtype=complex)")
print("b = np.loadtxt('a1.txt',dtype=complex)")
print('p = a.real')
print('q = a.imag')
print('c = b.real')
print('d = b.imag')
print("if(np.allclose(p,c) and np.allclose(q,d)):")
print("    print('Your result is right.')")
print('else:')
print("    print('Your result did not match!') ")       



            


