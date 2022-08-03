from qiskit import *
import numpy as np

#%matplotlib inline
qc = QuantumCircuit(1, 1)
qc.s(0)
qc.measure(0, 0, basis="X")
backend = BasicAer.get_backend("dm_simulator_ms")
run = execute(qc, backend)
result = run.result()
result.results[0].data.densitymatrix
print("Density Matrix: \n", result.results[0].data.densitymatrix)
