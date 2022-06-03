from qiskit import *
import numpy as np
#%matplotlib inline
qc2 = QuantumCircuit(3,2)
qc2.measure(0,0,basis='Bell',add_param='01')
options2 = {
    'plot': True
}
backend = BasicAer.get_backend('dm_simulator')
run2 = execute(qc2,backend,**options2)
result2 = run2.result()
result2['results'][0]['data']['bell_probabilities01']
print('Density Matrix: \n',result2['results'][0]['data']['densitymatrix'])