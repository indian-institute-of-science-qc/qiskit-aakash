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

# pylint: disable=invalid-name

"""
Rotation around the y-axis.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u3 import U3Gate

class RYGate(Gate):
    """rotation around the y-axis."""

    def __init__(self, theta):
        """Create new ry single qubit gate."""
        super().__init__("ry", 1, [theta])

    def _define(self):
        """
        gate ry(theta) a { u3(theta, 0, 0) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(self.params[0], 0, 0), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        return RYGate(-self.params[0])


def ry(self, theta, q):
    return self.append(RYGate(theta), [q], [])
    """
        Apply RY to qubit q in density matrix register self.
        Density matrix remains in the same register.
        Args:
            q (int): q is the qubit where the gate RY is applied.
            theta: Rotation angle is theta.

    # update density matrix
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    self._densitymatrix = np.reshape(self._densitymatrix,(4**(q),4,4**(self._number_of_qubits-q-1)))
    for j in range(4**(self._number_of_qubits-q-1)):
        for i in range(4**(q)):
            temp1 = self._densitymatrix[i,1,j].copy()
            temp2 = self._densitymatrix[i,3,j].copy()
            self._densitymatrix[i,1,j] = c*temp1 + s*temp2
            self._densitymatrix[i,3,j] = c*temp2 - s*temp1
    """

QuantumCircuit.ry = ry
CompositeGate.ry = ry
