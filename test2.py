from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
import numpy as np

backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator') # run on local sor by defaultimulat

coeff_val = np.load('stored_coefficients.npy')
options = {'initial_densitymatrix': coeff_val, 'custom_densitymatrix': 'stored_density_matrix'}

q = QuantumRegister(5)
c = ClassicalRegister(5)
circ = QuantumCircuit(q,c)

def generator(k):
    return (np.pi*2)/(2**k)

circ.h(q[0])
circuits = [circ]
#job = execute(circuits, backend1, **options)
#result = job.result()
#print(result)
job = execute(circuits, backend2, **options)
result = job.result()
print(result)
