from qiskit import *
q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q,c)
qc.u1(3.6,0)
qc.cx(0,1)
qc.u1(2.6,2)
qc.measure(1,1,basis='X')
qc.measure(0,0,basis='Y')
qc.cx(1,0)
qc.s(2)
qc.y(2)
qc.measure(1,1,basis='Bell',add_param='12')
qc.measure(0,0)
qc.measure(q,c,basis='Ensemble',add_param='X')
backend = BasicAer.get_backend('dm_simulator')
options = {
    'show_partition': True
}
run = execute(qc,backend,**options)