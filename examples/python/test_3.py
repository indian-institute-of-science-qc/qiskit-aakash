from qiskit import *
backend = BasicAer.get_backend('dm_simulator')
qc2 = QuantumCircuit(2)
options2 = {
    'custom_densitymatrix': 'binary_string',
    'initial_densitymatrix': '01'
}
backend = BasicAer.get_backend('dm_simulator')
run2  = execute(qc2,backend,**options2)
result2 = run2.result()
print('Density Matrix: \n',result2['results'][0]['data']['densitymatrix'])