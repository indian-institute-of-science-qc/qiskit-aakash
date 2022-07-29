# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Session gates."""

from .library.standard_gates.equivalence_library import StandardEquivalenceLibrary
from .equivalence import EquivalenceLibrary

# from qiskit.qasm import pi
# from qiskit.circuit import EquivalenceLibrary, Parameter, QuantumCircuit, QuantumRegister

# from qiskit.circuit.library.generalized_gates.gms import MSGate, MSGate_XX, MSGate_YY
# from qiskit.circuit.library.standard_gates import *


_sel = SessionEquivalenceLibrary = EquivalenceLibrary(base=StandardEquivalenceLibrary)