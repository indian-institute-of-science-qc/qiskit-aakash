# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
SWAP gate.
"""

import numpy

from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cx import CnotGate


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self):
        """Create new SWAP gate."""
        super().__init__("swap", 2, [])

    def _define(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (CnotGate(), [q[0], q[1]], []),
            (CnotGate(), [q[1], q[0]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SwapGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Swap gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=complex)


def swap(self, qubit1, qubit2):
    """Apply SWAP between qubit1 and qubit2."""
   # return self.append(SwapGate(), [qubit1, qubit2], [])
        q_1 = min(qubit_1, qubit_2)
        q_2 = max(qubit_1, qubit_2)

        #update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(self._number_of_qubits-q_2-1), 4, 4**(q_2-q_1-1), 4, 4**q_1))
        for i in range(4**(self._number_of_qubits-q_2-1)):
            for j in range(4**(q_2-q_1-1)):
                for k in range(4**q_1):
                    for l in range(3):
                        for m in range(l+1,4):
                                 temp = self._densitymatrix[i,l,j,m,k].copy()
                                 self._densitymatrix[i,l,j,m,k] = self._densitymatrix[i,m,j,l,k]
                                 self._densitymatrix[i,m,j,l,k] = temp


QuantumCircuit.swap = swap
CompositeGate.swap = swap
