from qiskit import *

backend = BasicAer.get_backend("dm_simulator")
qc1 = QuantumCircuit(1)
options1 = {"custom_densitymatrix": "max_mixed"}
run1 = execute(qc1, backend, **options1)
result1 = run1.result()
print("Density Matrix: \n", result1.results[0].data.densitymatrix)
