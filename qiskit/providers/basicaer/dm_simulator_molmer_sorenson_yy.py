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

# pylint: disable=arguments-differ

"""Contains a (slow) python simulator using the density matrix backend.
It simulates a qasm quantum circuit (an experiment) that has been processed
by the transpiler. Its complexity is exponential in the number of qubits.

The density matrix formalism allows inclusion of environmental noise in the simulation.
The noise is specified in terms of several parameters, which the user has to provide.
The density matrix is evolved according to the Kraus superoperator representation.

The simulator is run using

.. code-block:: python

    DmSimulatorPy().run(qobj)

Here the input is a Qobj object and the output is a BasicAerJob object,
which can later be queried for the Result object.

This is a derivative work of the Qiskit project. If you use it, please acknowledge
H. Chaudhary, B. Mahato, L. Priyadarshi, N. Roshan, Utkarsh and A. Patel, arXiv:1908.?????
Upgraded to Terra Stable v0.20 by Samarth Hawaldar, Nikhil Nair, Purnendu Sen, Shivalee Shah, Debabrata Bhattacharya, Apoorva Patel
"""

import uuid
import time
import logging
import warnings

from math import log2
from collections import Counter
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.providers.backend import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from .exceptions import BasicAerError
from .basicaertools import *

from .dm_simulator_base import DmSimulatorPy_Base

logger = logging.getLogger(__name__)


class DmSimulatorMSYYPy(DmSimulatorPy_Base):
    """Python implementation of a Density Matrix simulator.
    The density matrix is expressed in the orthogonal Pauli basis as
    rho = sum_{ij...} a_{ij...} sigma_i x sigma_j x ...
    The array "densitymatrix" contains the 4**n real coefficients a_{ij...},
    with each subscript taking values 0,1,2,3 (equivalently I,X,Y,Z).
    The default shape for the array a_{ij...} is n*[4].
    """

    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()["memory"] * (1024**3) / 16))

    DEFAULT_CONFIGURATION = {
        "backend_name": "dm_simulator_ms_yy",
        "backend_version": "2.0.0",
        "n_qubits": MAX_QUBITS_MEMORY,
        "url": "https://github.com/Qiskit/qiskit-terra",
        "simulator": True,
        "local": True,
        "conditional": True,
        "open_pulse": False,
        "memory": True,
        "max_shots": 1,
        "coupling_map": None,
        "description": "A python simulator for qasm experiments",
        "basis_gates": ["u1", "u2", "u3", "ms_yy", "id", "unitary"],
        "gates": [
            {
                "name": "u1",
                "parameters": ["lambda"],
                "qasm_def": "gate u1(lambda) q { U(0,0,lambda) q; }",
            },
            {
                "name": "u2",
                "parameters": ["phi", "lambda"],
                "qasm_def": "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }",
            },
            {
                "name": "u3",
                "parameters": ["theta", "phi", "lambda"],
                "qasm_def": "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }",
            },
            {"name": "ms_yy", "parameters": ["c", "t"], "qasm_def": "gate ms_yy c,t { ryy(pi/2) c,t; }"},
            {"name": "id", "parameters": ["a"], "qasm_def": "gate id a { U(0,0,0) a; }"},
            {"name": "unitary", "parameters": ["matrix"], "qasm_def": "unitary(matrix) q1, q2,..."},
        ],
    }

    # Class level variable to return the final state at the end of simulation
    # This should be set to True for the densitymatrix simulator
    SHOW_FINAL_STATE = True
    PLOTTING = False
    SHOW_PARTITION = False
    DEBUG = True
    STORE_LOCAL = False
    COMPARE = False
    FILE_EXIST = False
    MERGE = True

    def __init__(self, configuration=None, provider=None):

        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)
            ),
            provider=provider,
        )

        # Define two-qubit gate type
        self._two_qubit_gates = ('ms_yy', 'MS_YY')
        self._two_qubit_gate_qasm = 'ms_yy'
        self._two_qubit_gate_descriptor = 'MS_YY' # This is the string that would be printed if  SHOW_PARTITION is True
        self._two_qubit_gate_partition_name = 'MS_YY'

        # Define which errors to be applied
        self._decoherence_and_amp_decay_applied = True
        self._depolarization_applied = False


    def _add_unitary_two(self, qubit0, qubit1):
        """Apply a two-qubit unitary transformation (only ms gate is included).

        Args:
            qubit0 (int): control qubit
            qubit1 (int): target qubit
        """

        self._densitymatrix = ms_gate_yy_dm_matrix(
            self._densitymatrix,
            qubit0,
            qubit1,
            self._error_params["two_qubit_gates"],
            self._number_of_qubits,
        )