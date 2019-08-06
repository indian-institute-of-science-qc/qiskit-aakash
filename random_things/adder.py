import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit import execute
# from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import BasicAer
backend = BasicAer.get_backend('dm_simulator')


def rev(asd):
    aa = {}
    bb = {}
    for a in asd.keys():
        aa[a[::-1]] = asd[a]

    for a in sorted(aa):
        bb[a] = aa[a]

    return bb


def plot1(prob, name='adder_prob'):

    plt.bar(prob.keys(), prob.values())
    plt.title(
        f"Probability Distribution of states")
    plt.xlabel("States")
    plt.ylabel("Probabilities")
    plt.xticks(rotation=90)
    plt.savefig('./adder_results/' + name + '.png')
    plt.show()


def add(a, b, name='adder'):

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

    for i in range(n-1):
        circuit.ccx(aqreg[i], bqreg[i], cqreg[i+1])
        circuit.cx(aqreg[i], bqreg[i])
        circuit.ccx(cqreg[i], bqreg[i], cqreg[i+1])

    circuit.ccx(aqreg[n-1], bqreg[n-1], bqreg[n])
    circuit.cx(aqreg[n-1], bqreg[n-1])
    circuit.ccx(cqreg[n-1], bqreg[n-1], bqreg[n])

    circuit.cx(cqreg[n-1], bqreg[n-1])

    for i in range(n-1):
        circuit.ccx(cqreg[(n-2)-i], bqreg[(n-2)-i], cqreg[(n-1)-i])
        circuit.cx(aqreg[(n-2)-i], bqreg[(n-2)-i])
        circuit.ccx(aqreg[(n-2)-i], bqreg[(n-2)-i], cqreg[(n-1)-i])
        circuit.cx(cqreg[(n-2)-i], bqreg[(n-2)-i])
        circuit.cx(aqreg[(n-2)-i], bqreg[(n-2)-i])

    for i in range(n+1):
        circuit.measure(bqreg[i], mqreg[i])

    job_sim = execute(circuit, backend, **options)
    result_sim = job_sim.result()
    # result_sim["results"][0]["data"]['densitymatrix']
    # plot1(result_sim["results"][0]["data"]['partial_probability'])
    # plot1(rev(result_sim["results"][0]["data"]['partial_probability']), name)

    # print(result_sim.get_counts())


options = {

    "chop_threshold": 1e-15,
    "thermal_factor": 0.,
    "decoherence_factor": 1.0,
    "depolarization_factor": 1.,
    "bell_depolarization_factor": 1.,
    "decay_factor": 1.,
    "rotation_error": {'rx': [1., 0.], 'ry': [1., 0.0], 'rz': [1., 0.]},
    "tsp_model_error": [1.0, 0.0]
}


add("11", "111")
