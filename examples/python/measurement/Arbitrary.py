from qiskit import *
import numpy as np
#%matplotlib inline
qc1 = QuantumCircuit(1,1)
qc1.s(0)
qc1.measure(0,0, basis='N', add_param=np.array([1,2,3]))         
backend = BasicAer.get_backend('dm_simulator')
run1 = execute(qc1,backend)
result1 = run1.result()
result1['results'][0]['data']['densitymatrix']
print('Density Matrix: \n',result1['results'][0]['data']['densitymatrix'])