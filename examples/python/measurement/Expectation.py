from qiskit import *
import numpy as np
#%matplotlib inline
qc4 = QuantumCircuit(3,3)
qc4.h(0)
qc4.cx(0,1)
qc4.cx(1,2)
qc4.measure(0,0,basis='Expect',add_param='ZIZ')
backend = BasicAer.get_backend('dm_simulator')
run4 = execute(qc4,backend)
result4 = run4.result()
print(result4['results'][0]['data']['Pauli_string_expectation'])
print('Density Matrix: \n',result4['results'][0]['data']['densitymatrix'])
