from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit import execute
# from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import BasicAer
backend1 = BasicAer.get_backend('qasm_simulator')
backend2 = BasicAer.get_backend('dm_simulator')

def add(a, b):

    a = [int(x) for x in a]
    b = [int(x) for x in b]
    for x in a:
        assert(x <= 1), ("a is not in binary format")
    for x in b:
        assert(x <= 1), ("b is not in binary format")

    if len(a) > len(b):
        n = len(a)
    else:
        n = len(b)

    aqreg = QuantumRegister(n, 'a')
    bqreg = QuantumRegister(n+1, 'b')
    cqreg = QuantumRegister(n, 'c')
    mqreg = ClassicalRegister(n+1, 'm')

    circuit = QuantumCircuit(aqreg, bqreg, cqreg, mqreg)

    for i in range(len(a)):
        if a[i] == 1:
            circuit.x(aqreg[len(a) - (i+1)])
    for i in range(len(b)):
        if b[i] == 1:
            circuit.x(bqreg[len(b) - (i+1)])
    '''
    for i in range(n-1):
        circuit.ccx(aqreg[i], bqreg[i], cqreg[i+1])
        circuit.cx(aqreg[i], bqreg[i])
        circuit.ccx(cqreg[i], bqreg[i], cqreg[i+1])
    '''
    circuit.ccx(aqreg[n-1], bqreg[n-1], bqreg[n])
    '''
    circuit.cx(aqreg[n-1], bqreg[n-1])
    circuit.ccx(cqreg[n-1], bqreg[n-1], bqreg[n])

    circuit.cx(cqreg[n-1], bqreg[n-1])

    for i in range(n-1):
        circuit.ccx(cqreg[(n-2)-i], bqreg[(n-2)-i], cqreg[(n-1)-i])
        circuit.cx(aqreg[(n-2)-i], bqreg[(n-2)-i])
        circuit.ccx(aqreg[(n-2)-i], bqreg[(n-2)-i], cqreg[(n-1)-i])
        circuit.cx(cqreg[(n-2)-i], bqreg[(n-2)-i])
        circuit.cx(aqreg[(n-2)-i], bqreg[(n-2)-i])
    '''
    #for i in range(n+1):
    #        circuit.measure(bqreg[i], mqreg[i])

    job_sim = execute(circuit, backend1, shots=1)
    result_sim = job_sim.result()
    #print(result_sim)
    #print(result_sim.get_counts())

    job_sim = execute(circuit, backend2)
    result_sim = job_sim.result()
    #print(result_sim)
    #print(result_sim.get_counts())


add("1", "1")
