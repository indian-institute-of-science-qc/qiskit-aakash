from qiskit import QuantumCircuit, BasicAer, execute

qc = QuantumCircuit(2)
# Gates
qc.x(1)
qc.cx(0, 1)
# execution
backend = BasicAer.get_backend("dm_simulator")
run = execute(qc, backend=backend)
result = run.result()
print(result.results[0].data.densitymatrix)
