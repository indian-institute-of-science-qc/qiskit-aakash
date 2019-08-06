from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
import matplotlib as mpl
import numpy as np
backend1 = BasicAer.get_backend('dm_simulator')
backend2 = BasicAer.get_backend('qasm_simulator')
options = {
    #'custom_densitymatrix': 'max_mixed', 
    'custom_densitymatrix': 'uniform_superpos',
    'plot': True,
    "chop_threshold": 1e-15,
    "thermal_factor": 0.,
    "decoherence_factor": 1.,
    "depolarization_factor": 1.,
    "bell_depolarization_factor": 1.,
    "decay_factor": 1.0,
    "rotation_error": {'rx': [1., 0.], 'ry': [1., 0.], 'rz': [1., 0.]},
    "tsp_model_error": [1., 0.],
    # "store_densitymatrix": True,
    # "compare": True
    
}
q = QuantumRegister(5,'a')
c = ClassicalRegister(5)
circ = QuantumCircuit(q,c)

'''circ.h(q[0])
circ.x(q[1])
# circ.y(q[1])
# circ.z(q[1])
# circ.h(q[1])
# circ.y(q[2])
circ.cx(q[0],q[1])
circ.ccx(q[1],q[3], q[2])
circ.t(q[4])
circ.h(q[0])

circ.measure(q[0], c[0])
#circ.measure(q[0],c[0], basis = 'Bell', add_param = '02')
#circ.measure(q[1], c[1], basis='Bell', add_param='13')
#circ.measure(q, c, basis = 'Z')
circ.measure(q,c,basis = ['ZXIXY'])'''


circ.h(q[0])
circ.x(q[1])
circ.y(q[2])
circ.cx(q[1],q[3])
circ.ccx(q[1],q[3], q[2])
circ.t(q[4])
circ.h(q[0])
circ.measure(q[0],c[0], basis = 'Bell', add_param = '02')
#circ.measure(q, c, basis = 'Z')
circ.measure(q,c,basis = ['XIXXY'])


print("\nINITIAL CIRCUIT\n", circ)
circuits = [circ]
job = execute(circuits, backend1, **options)
result = job.result()
print("\n RESULTS\n",result)
