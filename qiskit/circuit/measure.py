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
Quantum measurement in the computational basis.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self, basis, add_param):
        """Create new measurement instruction."""
        avail_basis = ['X', 'Y', 'Z', 'Bell', 'N']
        
        if basis is not None and add_param is None:
            super().__init__("measure", 1, 1, [basis])
        elif basis == 'N' and add_param is not None:
            super().__init__("measure", 1, 1, [basis, add_param])
        elif basis == 'Bell' and add_param is not None:
            super().__init__("measure", 1, 1, [basis, add_param])
        elif basis != 'N' and add_param is not None:
            raise QiskitError('Vector cannot be provided with this measurement basis.')
        elif basis == 'N' and add_param is None:
            raise QiskitError('Vector should be provided with this measurement basis.')
        elif basis == 'Bell' and add_param is None:
            raise QiskitError('Bell string should be provided with this measurement basis.')            
        elif basis is not None and basis not in avail_basis:
            raise QiskitError('Invalid basis provided.')
        else:
            super().__init__("measure", 1, 1, [])

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise QiskitError('register size error')


def measure(self, qubit, cbit, basis=None, add_param=None):
    """Measure quantum bit into classical bit (tuples).

    Args:
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.Instruction: the attached measure instruction.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    return self.append(Measure(basis, add_param), [qubit], [cbit])


QuantumCircuit.measure = measure
