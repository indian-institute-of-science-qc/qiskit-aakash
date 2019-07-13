from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit import execute
# from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import BasicAer
# backend = BasicAer.get_backend('dm_simulator')
backend = BasicAer.get_backend('qasm_simulator')

n = 5
aqreg = QuantumRegister(n, 'a')
mqreg = ClassicalRegister(n, 'm')
circuit = QuantumCircuit(aqreg, mqreg)
for i in range(n):

    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.x(aqreg[i])
    circuit.cx(aqreg[1], aqreg[0])
    circuit.measure(aqreg[i], mqreg[i])

job_sim = execute(circuit, backend, shots=1024)
