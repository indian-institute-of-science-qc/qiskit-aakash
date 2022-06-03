from qiskit import *
import numpy as np
#%matplotlib inline
qc3 = QuantumCircuit(3,3)
qc3.h(0)
qc3.cx(0,1)
qc3.cx(1,2)
qc3.measure(0,0,basis='Ensemble',add_param='X')
backend = BasicAer.get_backend('dm_simulator')
options3 = {
    'plot': True
}
run3 = execute(qc3,backend,**options3)
result3 = run3.result()
result3['results'][0]['data']['ensemble_probability']
print('Density Matrix: \n',result3['results'][0]['data']['densitymatrix'])