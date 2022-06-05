from qiskit import *

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)
qc.p(3.6, 0)
qc.cx(0, 1)
qc.p(2.6, 2)
qc.cx(1, 0)
qc.s(2)
qc.y(2)
backend = BasicAer.get_backend("dm_simulator")
options = {"show_partition": True}
run = execute(qc, backend, **options)
