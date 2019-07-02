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

"""Contains a (slow) python simulator.

It simulates a qasm quantum circuit (an experiment) that has been compiled
to run on the simulator. It is exponential in the number of qubits.

The simulator is run using

.. code-block:: python

    DmSimulatorPy().run(qobj)

Where the input is a Qobj object and the output is a BasicAerJob object, which can
later be queried for the Result object. The result will contain a 'memory' data
field, which is a result of measurements for each shot.
"""

import uuid
import time
import logging

from math import log2
from collections import Counter
import numpy as np
import itertools

from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.providers import BaseBackend
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from .exceptions import BasicAerError
from .basicaertools import *

logger = logging.getLogger(__name__)


class DmSimulatorPy(BaseBackend):
    """Python implementation of a Density Matrix simulator."""

    MAX_QUBITS_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'dm_simulator',
        'backend_version': '2.0.0',
        'n_qubits': min(24, MAX_QUBITS_MEMORY),
        'url': 'https://github.com/Qiskit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 65536,
        'coupling_map': None,
        'description': 'A python simulator for qasm experiments',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id', 'unitary'],
        'gates': [
            {
                'name': 'u1',
                'parameters': ['lambda'],
                'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'
            },
            {
                'name': 'u2',
                'parameters': ['phi', 'lambda'],
                'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'
            },
            {
                'name': 'u3',
                'parameters': ['theta', 'phi', 'lambda'],
                'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'
            },
            {
                'name': 'cx',
                'parameters': ['c', 't'],
                'qasm_def': 'gate cx c,t { CX c,t; }'
            },
            {
                'name': 'id',
                'parameters': ['a'],
                'qasm_def': 'gate id a { U(0,0,0) a; }'
            },
            {
                'name': 'unitary',
                'parameters': ['matrix'],
                'qasm_def': 'unitary(matrix) q1, q2,...'
            }
        ]
    }

    DEFAULT_OPTIONS = {
        "initial_densitymatrix": None,
        "chop_threshold": 1e-15
    }

    # Class level variable to return the final state at the end of simulation
    # This should be set to True for the densitymatrix simulator
    SHOW_FINAL_STATE = True

    def __init__(self, configuration=None, provider=None):

        super().__init__(configuration=(
            configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)),
                         provider=provider)

        # Define attributes in __init__.
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._densitymatrix = 0
        self._probability_of_zero = 0.0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._error_params = None
        self._memory = False
        self._rotation_err_mean = 0     # Rotation Gates
        self._rotation_err_fluc = 1     # Rotation Gates
        self._thermal_factor = 0        # p
        self._decoherence_factor = 1    # f
        self._decay_factor = 1          # g
        # (Bit flip and Depolarization have the same effect)
        self._depolarization_factor = 1 # During Measurement 
        self._custom_densitymatrix = None
        self._initial_densitymatrix = self.DEFAULT_OPTIONS["initial_densitymatrix"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        self._qobj_config = None
        # TEMP
        self._sample_measure = False

    def _add_unitary_single(self, gate, qubit):
        """Apply an arbitrary 1-qubit unitary matrix.

        Args:
            params (list): list of parameters for U1,U2 and U3 gate.
            qubit (int): the qubit to apply gate to
        """
        
        # changing density matrix
        lt, mt, rt = 4 ** qubit, 4, 4 ** (self._number_of_qubits-qubit-1)
        self._densitymatrix = np.reshape(self._densitymatrix, (lt, mt, rt)) 
        
        #self._densitymatrix = np.reshape(
        #    self._densitymatrix, (4**qubit, 4, 4**(self._number_of_qubits-qubit-1)))
        
        # After doing a ZY decomposition of unitary gate, we iteratively apply the rotation gates
        #print(gate)

        self._densitymatrix = rt_gate_dm_matrix_1(gate = gate, 
                    err_param = self._error_params['single_gate'], 
                    state = self._densitymatrix, q = (lt, mt, rt), 
                    num_qubits = self._number_of_qubits)
        '''
        for idx in gate: # For Rotations in the Decomposed Gate list
            #print(idx, self._densitymatrix)
    
            self._densitymatrix = rt_gate_dm_matrix(
                idx[0], idx[1], self._error_params['single_gate'], self._densitymatrix, qubit, self._number_of_qubits)
            
            #print(self._densitymatrix, idx, self._error_params['single_gate'])
        '''

        self._densitymatrix = np.reshape(self._densitymatrix,
                                    self._number_of_qubits * [4])

    def _add_unitary_two(self, qubit0, qubit1):
        """Apply a two-qubit unitary matrix.

        Args:
            gate (matrix_like): a the two-qubit gate matrix
            qubit0 (int): gate qubit-0
            qubit1 (int): gate qubit-1
        """ 
        
        self._densitymatrix = cx_gate_dm_matrix(self._densitymatrix,
                                                qubit0, qubit1, self._number_of_qubits)
        
        self._densitymatrix = np.reshape(self._densitymatrix,
                                        self._number_of_qubits * [4])
   
    #TODO Combine the decoherence and decay (1. Off-Diagonal elements multiplied by sqrt(g)*f)
    def _add_decoherence_and_amp_decay(self, level, f, p, g):
        """ Apply decoherence transofrmation and amplitude decay transformation independently 
            to all the qubits. Off-diagonal elements of the density get contracted by a factor
            'f' due to decoherence and 'sqrt(g)' due to amplitude decay. Diagonal elements decay
            towards the thermal state.
        Args:
            level (int):    Clock cycle number
            f     (float):  Contraction of diagonal elements due to T_2 (coherence time) 
            p     (float):  Thermal factor corresponding to the asymptotic state
            g     (float):  Decay rate for the excited state component
        """ 
        sg = np.sqrt(g)
        off_diag_contract = np.sqrt(g) * f
        diag_decay = (1-g)*(1-2*p)

        for qb in range(self._number_of_qubits):

            lt, mt, rt = 4 ** qb, 4, 4 ** (self._number_of_qubits - qb - 1)
            self._densitymatrix = np.reshape(self._densitymatrix, (lt, mt, rt))
            temp = self._densitymatrix.copy()  # qc.measure(q[0], c[0])
            for j in range(rt):
                for i in range(lt):
                    self._densitymatrix[i, 1, j] = off_diag_contract * \
                                                    temp[i, 1, j]
                    self._densitymatrix[i, 2, j] = off_diag_contract * \
                                                    temp[i, 2, j]
                    self._densitymatrix[i, 3, j] =  g * temp[i, 3, j] + \
                                                    diag_decay * temp[i, 0, j]
        
        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])
    '''
    def _add_decoherence(self, level, f):
        """ Apply decoherence transformation independently to all the qubits.
            Off-diagonal elements of the density matrix get contracted by a factor 'f'.
        
        Args:
            level (int):    Clock cycle number
            f     (float):  Contraction of diagonal elements due to T_2 (coherence time)
        """

        for qb in range(self._number_of_qubits):
            # changing density matrix
            self._densitymatrix = np.reshape(
                self._densitymatrix, (4**qb, 4, 4**(self._number_of_qubits-qb-1)))
            
            temp = self._densitymatrix.copy()

            for j in range(4**(self._number_of_qubits-qb-1)):
                for i in range(4**(qb)):
                    self._densitymatrix[i, 1, j] = f*temp[i,1,j]
                    self._densitymatrix[i, 2, j] = f*temp[i,2,j]

        self._densitymatrix = np.reshape(self._densitymatrix,
                                            self._number_of_qubits * [4])

    def _add_amplitude_decay(self, level, p, g):
        """ Apply amplitude decay transformation independently to all the qubits.
            Off-diagonal elements of the density matrix get contracted by a factor ''.
            Diagonal elements decay towards the thermal state

        Args:
            level (int):    Clock cycle number
            p     (float):  Thermal factor corresponding to the asymptotic state
            g     (float):  Decay rate for the excited state component
        """
        sg = np.sqrt(g) #Square root of g
        dc = (1-g)*(1-2*p) #Decay Rate parameter

        for qb in range(self._number_of_qubits):
            # changing density matrix
            self._densitymatrix = np.reshape(
                self._densitymatrix, (4**qb, 4, 4**(self._number_of_qubits-qb-1)))
            
            temp = self._densitymatrix.copy()

            for j in range(4**(self._number_of_qubits-qb-1)):
                for i in range(4**(qb)):
                    self._densitymatrix[i, 1, j] = sg*temp[i, 1, j]
                    self._densitymatrix[i, 2, j] = sg*temp[i, 2, j]
                    self._densitymatrix[i, 3, j] = g*temp[i, 3, j] + dc*temp[i, 0, j]

        self._densitymatrix = np.reshape(self._densitymatrix, 
                                            self._number_of_qubits * [4])
    '''
    def _get_measure_outcome(self, qubit):
        """Simulate the outcome of measurement of a qubit.

        Args:
            qubit (int): the qubit to measure

        Return:
            tuple: pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._densitymatrix) ** 2, axis=tuple(axis))

        measure_ind = [x for x in itertools.product(
            [0, 3], repeat=self._number_of_qubits)]
        operator_ind = [self._densitymatrix[x] for x in measure_ind]
        operator_mes = np.array([[1, 1], [1, -1]])
        for i in range(self._number_of_qubits-1):
            operator_mes = np.kron(np.array([[1, 1], [1, -1]]), operator_mes)

        probabilities = np.reshape((1/2**self._number_of_qubits) * np.array([np.sum(
            np.multiply(operator_ind, x)) for x in operator_mes]),  self._number_of_qubits * [2])
        #print('Probability Before: ', probabilities)

        probabilities = np.reshape(
            np.sum(probabilities, axis=tuple(axis)), 2)
        #print('Probability After: ', probabilities)

        # Compute einsum index string for 1-qubit matrix multiplication
        random_number = self._local_random.rand()
        if random_number < probabilities[0]:
            return '0', probabilities[0]
        # Else outcome was '1'
        return '1', probabilities[1]

    def _add_ensemble_measure(self, err_param):
        """Perform complete computational basis measurement for current densitymatrix.

        Args:
            err_param   (float): Reduction in polarization during measurement
        Returns:
            list: Complete list of probabilities. 
        """
        # We get indices used for Probability Measurement via this.
        measure_ind = [x for x in itertools.product([0,3], repeat=self._number_of_qubits)]
        # We get coefficient values stored at those indices via this. 
        operator_ind = [self._densitymatrix[x] for x in measure_ind]
        # We get permutations of signs for summing those coefficient values.
        operator_mes = np.array([[1, err_param], [1, -err_param]], dtype=float)
        for i in range(self._number_of_qubits-1):
            operator_mes = np.kron(np.array([[1, err_param], [1, -err_param]]), operator_mes)

        # We get 2**n probabilities via this.
        probabilities = np.reshape((0.5**self._number_of_qubits)*np.array([np.sum(np.multiply(operator_ind, x)) for x in operator_mes]), 2**self._number_of_qubits)
        key = [x for x in itertools.product([0,1],repeat = self._number_of_qubits)]
        prob_key = [''.join(str(y) for y in x) for x in key]
        prob = {}
        for i in range(2**self._number_of_qubits):
            prob.update({prob_key[i]: probabilities[i]})
        print(prob)    
        return probabilities

    def _add_bell_basis_measure(self, qubit_1, qubit_2):
        """
        Apply a Bell basisi measure instruction to two qubits.
        Post measurement density matrix is returned in the same array.

        Args:
            qubit_1 (int): first qubit of Bell pair.
            qubit_2 (int): second qubit of Bell pair.
        
        Returns:
            Four probabilities in the (|00>+|11>,|00>-|11>,|01>+|10>,|01>-|10>) basis.
        """
        q_1 = min(qubit_1, qubit_2)
        q_2 = max(qubit_1, qubit_2)

        #update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(self._number_of_qubits-q_2-1), 4, 4**(q_2-q_1-1), 4, 4**q_1))
        bell_probabilities = [0,0,0,0]
        for i in range(4**(self._number_of_qubits-q_2-1)):
            for j in range(4**(q_2-q_1-1)):
                for k in range(4**q_1):
                    for l in range(4):
                        for m in range(4):
                            if l != m:
                                self._densitymatrix[i,l,j,m,k] = 0
                    bell_probabilities[0] += 0.25*(self._densitymatrix[i,0,j,0,k] + self._densitymatrix[i,1,j,1,k] - self._densitymatrix[i,2,j,2,k] + self._densitymatrix[i,3,j,3,k])
                    bell_probabilities[1] += 0.25*(self._densitymatrix[i,0,j,0,k] - self._densitymatrix[i,1,j,1,k] + self._densitymatrix[i,2,j,2,k] + self._densitymatrix[i,3,j,3,k])
                    bell_probabilities[2] += 0.25*(self._densitymatrix[i,0,j,0,k] + self._densitymatrix[i,1,j,1,k] + self._densitymatrix[i,2,j,2,k] - self._densitymatrix[i,3,j,3,k])
                    bell_probabilities[3] += 0.25*(self._densitymatrix[i,0,j,0,k] - self._densitymatrix[i,1,j,1,k] - self._densitymatrix[i,2,j,2,k] - self._densitymatrix[i,3,j,3,k])
        return bell_probabilities

    def _add_qasm_measure_Z(self, qubit, err_param):
        """Apply a Z basis measure instruction to a qubit. 
        Post-measurement density matrix is returned in the same array.

        Args:
            qubit (int): qubit is the qubit measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))
        p_3 = 0.0
        for j in range(4**(self._number_of_qubits-qubit-1)):
            for i in range(4**(qubit)):
                self._densitymatrix[i,1,j] = 0
                self._densitymatrix[i,2,j] = 0
                self._densitymatrix[1,3,j] *= err_param
                p_3 += self._densitymatrix[i,3,j]
        
        probability_of_zero = 0.5 * (1 + p_3)
        probability_of_one = 1 - probability_of_zero
        if probability_of_zero > probability_of_one:
            return 0, probability_of_zero
        else:
            return 1, probability_of_one
        #return probability_of_zero
    
    def _add_qasm_measure_X(self, qubit, err_param):
        """Apply a X basis measure instruction to a qubit. 
        Post-measurement density matrix is returned in the same array.

        Args:
            qubit (int): qubit is the qubit measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))
        p_1 = 0.0
        for j in range(4**(self._number_of_qubits-qubit-1)):
            for i in range(4**(qubit)):
                self._densitymatrix[i,2,j] = 0
                self._densitymatrix[i,3,j] = 0
                self._densitymatrix[i,1,j] *= err_param
                p_1 += self._densitymatrix[i,1,j]
        
        probability_of_zero = 0.5 * (1 + p_1)
        probability_of_one = 1 - probability_of_zero
        return probability_of_zero

    def _add_qasm_measure_Y(self, qubit, err_param):
        """Apply a Y basis measure instruction to a qubit. 
        Post-measurement density matrix is returned in the same array.

        Args:
            qubit (int): qubit is the qubit measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))
        p_2 = 0.0
        for j in range(4**(self._number_of_qubits-qubit-1)):
            for i in range(4**(qubit)):
                self._densitymatrix[i,1,j] = 0
                self._densitymatrix[i,3,j] = 0
                self._densitymatrix[i,2,j] *= err_param
                p_2 += self._densitymatrix[i,2,j]
        
        probability_of_zero = 0.5 * (1 + p_2)
        probability_of_one = 1 - probability_of_zero
        return probability_of_zero

    def _add_qasm_measure_N(self, qubit, n, err_param):
        """Apply a general n-axis measure instruction to a qubit. 
        Post-measurement density matrix is returned in the same array.

        Args:
            qubit       (int): Qubit is the qubit measured.
            n           (vec): Axis of measurement.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))

        p_n = 0.0

        for j in range(4**(self._number_of_qubits-qubit-1)):
            for i in range(4**(qubit)):
                temp = n[0]*self._densitymatrix[i,1,j] + n[1]*self._densitymatrix[i,2,j] + \
                       n[2]*self._densitymatrix[i,3,j]
                
                temp *= err_param
                
                self._densitymatrix[i,1,j] = temp*n[0] 
                self._densitymatrix[i,2,j] = temp*n[1]
                self._densitymatrix[1,3,j] = temp*n[2]

                p_n +=  temp
        
        probability_of_zero = 0.5 * (1 + p_n)
        probability_of_one = 1 - probability_of_zero
        return probability_of_zero
    
    def _add_qasm_measure(self, qubit, cmembit, cregbit=None):
        """Apply a measure instruction to a qubit.
        Args:
            qubit (int): qubit is the qubit measured.
            cmembit (int): is the classical memory bit to store outcome in.
            cregbit (int, optional): is the classical register bit to store outcome in.
        """
        # get measure outcome
        outcome, probability = self._add_qasm_measure_Z(qubit, self._depolarization_factor)
        # update classical state
        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (
            ~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (
                    int(outcome) << cregbit)
    
    def _add_qasm_reset(self, qubit):
        """Apply a reset instruction to a qubit.

        Args:
            qubit (int): the qubit being rest

        This is done by doing a simulating a measurement
        outcome and projecting onto the outcome state while
        renormalizing.
        """
        # get measure outcome
        outcome, probability = self._get_measure_outcome(qubit)
        # update quantum state

        if outcome == '0':
            update = 1/np.sqrt(probability)*np.array(
                [[1,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]], dtype=float)
            self._add_unitary_single(update, qubit)
        else:
            update = 1/np.sqrt(probability)*np.array(
                [[1,0,0,-1], [0,1,0,0], [0,0,1,0], [1,0,0,-1]], dtype=float)
        # update classical state
            self._add_unitary_single(update, qubit)

    def _validate_initial_densitymatrix(self):
        """Validate an initial densitymatrix"""
        # If initial densitymatrix isn't set we don't need to validate
        if self._initial_densitymatrix is None:
            return
        # Check densitymatrix is correct length for number of qubits
        length = np.size(self._initial_densitymatrix)
        #print(length, self._number_of_qubits)
        required_dim = 4 ** self._number_of_qubits
        #if length != required_dim:    #TODO
            #raise BasicAerError('initial densitymatrix is incorrect length: ' + '{} != {}'.format(length, required_dim))
        # Check if Trace is 0
        #if self._densitymatrix[0] != 1:   #TODO
            #raise BasicAerError('Trace of initial densitymatrix is not one: ' + '{} != {}'.format(self._densitymatrix[0], 1))

    def _set_options(self, qobj_config=None, backend_options=None):
        """Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_densitymatrix = self.DEFAULT_OPTIONS["initial_densitymatrix"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        if 'error_params' in backend_options:
            self._error_params = backend_options['error_params']
        else:
            # Default Value - Has no effect
            self._error_params = { 'single_gate' : [1, 0] } 
        if backend_options is None:
            backend_options = {}
        # Check for custom initial densitymatrix in backend_options first,
        # then config second
        if 'initial_densitymatrix' in backend_options:
            self._initial_densitymatrix = np.array(backend_options['initial_densitymatrix'], dtype=float)
        elif hasattr(qobj_config, 'initial_densitymatrix'):
            self._initial_densitymatrix = np.array(qobj_config.initial_densitymatrix,
                                                 dtype=float)

        #TODO Create coefficient matrix for a custom density matrix
        print(backend_options)
        if 'custom_densitymatrix' in backend_options:
            self._custom_densitymatrix = backend_options['custom_densitymatrix']
            if self._custom_densitymatrix == 'binary_string':
                self._initial_densitymatrix = backend_options['initial_densitymatrix']


        if 'thermal_factor' in backend_options:
            self._thermal_factor = backend_options['thermal_factor']

        if 'decoherence_factor' in backend_options:
            del_T = backend_options['decoherence_factor'][0]
            T_2 = backend_options['decoherence_factor'][1]
            self._decoherence_factor = np.exp(-del_T/T_2)

        if 'decay_factor' in backend_options:
            del_T = backend_options['decay_factor'][0]
            T_1 = backend_options['decay_factor'][1]
            self._decay_factor = np.exp(-del_T/T_1)

        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

    def _initialize_densitymatrix(self):
        """
            Set the initial densitymatrix for simulation
            Default: All Zero [((I+sigma(3))/2)**num_qubits]
            Custom: max_mixed - Maximally Mixed [(I/2)**num_qubits]
                    uniform_superpos - Uniform Superposition [((I+sigma(1))/2)**num_qubits]
                    thermal_state - Thermalized State 
                    [([[1-p, 0],[0, p]])**num_qubits]
            ** -> Tensor product.
       """


        if self._initial_densitymatrix is None and self._custom_densitymatrix is None:
            self._densitymatrix = np.array([1,0,0,1], dtype=float)
            for i in range(self._number_of_qubits-1):
                self._densitymatrix = np.kron([1,0,0,1],self._densitymatrix)
        elif self._initial_densitymatrix is None and self._custom_densitymatrix == 'max_mixed':
            self._densitymatrix = np.array([1,0,0,0], dtype=float)
            for i in range(self._number_of_qubits-1):
                self._densitymatrix = np.kron([1,0,0,0], self._densitymatrix)
        elif self._initial_densitymatrix is None and self._custom_densitymatrix == 'uniform_superpos':
            self._densitymatrix = np.array([1,1,0,0], dtype=float)
            for i in range(self._number_of_qubits-1):
                self._densitymatrix = np.kron([1,1,0,0], self._densitymatrix)
        elif self._initial_densitymatrix is None and self._custom_densitymatrix == 'thermal_state':
            tf = 1-2*self._thermal_factor
            self._densitymatrix = np.array([1,0,0,tf], 
                                                dtype=float)
            for i in range(self._number_of_qubits-1):
                self._densitymatrix = np.kron([1,0,0,tf],
                                                    self._densitymatrix)
        elif self._initial_densitymatrix is not None and self._custom_densitymatrix == 'binary_string':
            print(self._initial_densitymatrix)
            print(self._custom_densitymatrix)
            if len(self._initial_densitymatrix) != self._number_of_qubits:
                raise BasicAerError('Wrong input binary string length')
            if self._initial_densitymatrix[-1] == '0':
                self._densitymatrix = np.array([1,0,0,1], dtype=float)
            else:
                self._densitymatrix = np.array([1,0,0,-1], dtype=float) 
            for idx in reversed(self._initial_densitymatrix[:-1]):
                if idx == '0':
                    self._densitymatrix = np.kron([1,0,0,1],self._densitymatrix)
                else:
                    self._densitymatrix = np.kron([1,0,0,-1],self._densitymatrix)
            self._initialize_densitymatrix = None      #For Normalization        
                                                          
        else:
            self._densitymatrix = self._initial_densitymatrix.copy()
        
        # Normalize
        if self._initial_densitymatrix is None:
            self._densitymatrix *= 0.5**(self._number_of_qubits)
        
        # Reshape to rank-N tensor
        self._densitymatrix = np.reshape(self._densitymatrix,
                                       self._number_of_qubits * [4])


    def _get_densitymatrix(self):
        """Return the current densitymatrix in JSON Result spec format"""
        # Coefficients
        vec = np.reshape(self._densitymatrix.real, 4 ** self._number_of_qubits)
        
        p_0 = np.array([[1, 0], [0, 1]], dtype = complex)
        p_1 = np.array([[0, 1], [1, 0]], dtype = complex)
        p_2 = np.array([[0, -1j], [1j, 0]], dtype = complex)
        p_3 = np.array([[1, 0], [0, -1]], dtype = complex)
        pauli_basis = [p_0, p_1, p_2, p_3]
        den_creat = [x for x in itertools.product([0,1,2,3], repeat=self._number_of_qubits)]
        den = []
        
        for creat in den_creat:
            op = pauli_basis[creat[0]]
            for idx in range(1, len(creat)):
                op = np.kron(op, pauli_basis[creat[idx]])
            den.append(op)
            op = None

        densitymatrix = vec[0]*den[0]

        for i in range(1, 4**self._number_of_qubits):
            densitymatrix += vec[i]*den[i]
        
        # Expand float numbers
        # Truncate small values
        vec[abs(vec) < self._chop_threshold] = 0.0
        return vec, densitymatrix

    def _validate_measure_sampling(self, experiment):
        """Determine if measure sampling is allowed for an experiment

        Args:
            experiment (QobjExperiment): a qobj experiment.
        """
        # If shots=1 we should disable measure sampling.
        # This is also required for densitymatrix simulator to return the
        # correct final densitymatrix without silently dropping final measurements.
        if self._shots <= 1:
            self._sample_measure = False
            return

        # Check for config flag
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        # If flag isn't found do a simple test to see if a circuit contains
        # no reset instructions, and no gates instructions after
        # the first measure.
        else:
            measure_flag = False
            for instruction in experiment.instructions:
                # If circuit contains reset operations we cannot sample
                if instruction.name == "reset":
                    self._sample_measure = False
                    return
                # If circuit contains a measure option then we can
                # sample only if all following operations are measures
                if measure_flag:
                    # If we find a non-measure instruction
                    # we cannot do measure sampling
                    if instruction.name not in ["measure", "barrier", "id", "u0"]:
                        self._sample_measure = False
                        return
                elif instruction.name == "measure":
                    measure_flag = True
            # If we made it to the end of the circuit without returning
            # measure sampling is allowed
            self._sample_measure = True

    def run(self, qobj, backend_options=None):
        """Run qobj asynchronously.

        Args:
            qobj (Qobj): payload of the experiment
            backend_options (dict): backend options

        Returns:
            BasicAerJob: derived from BaseJob

        Additional Information:
            backend_options: Is a dict of options for the backend. It may contain
                * "initial_densitymatrix": vector_like

            The "initial_densitymatrix" option specifies a custom initial
            initial densitymatrix for the simulator to be used instead of the all
            zero state. This size of this vector must be correct for the number
            of qubits in all experiments in the qobj.

            Example::

                backend_options = {
                    "initial_densitymatrix": np.array([1, 0, 0, 1j]) / np.sqrt(2),
                }
        """
        self._set_options(qobj_config=qobj.config,
                          backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = BasicAerJob(self, job_id, self._run_job, qobj)
        job.submit()
        return job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj

        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description

        Returns:
            Result: Result object
        """
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, 'memory', False)
        self._qobj_config = qobj.config
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {'backend_name': self.name(),
                  'backend_version': self._configuration.backend_version,
                  'qobj_id': qobj.qobj_id,
                  'job_id': job_id,
                  'results': result_list,
                  'status': 'COMPLETED',
                  'success': True,
                  'time_taken': end-start,
                  'header': qobj.header.as_dict()}

        return Result.from_dict(result)

    def run_experiment(self, experiment):
        """Run an experiment (circuit) and return a single experiment result.

        Args:
            experiment (QobjExperiment): experiment from qobj experiments list

        Returns:
             dict: A result dictionary which looks something like::

                {
                "name": name of this experiment (obtained from qobj.experiment header)
                "seed": random seed used for simulation
                "shots": number of shots used in the simulation
                "data":
                    {
                    "counts": {'0x9: 5, ...},
                    "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                    },
                "status": status string for the simulation
                "success": boolean
                "time_taken": simulation time of this single experiment
                }
        Raises:
            BasicAerError: if an error occurred.
        """
        start_processing = time.time()
        self._number_of_qubits = experiment.config.n_qubits
        self._number_of_cmembits = experiment.config.memory_slots
        self._densitymatrix = 0
        self._classical_memory = 0
        self._classical_register = 0
        #self._sample_measure = False
        # Validate the dimension of initial densitymatrix if set
        self._validate_initial_densitymatrix()
        # Get the seed looking in circuit, qobj, and then random.
        if hasattr(experiment.config, 'seed_simulator'):
            seed_simulator = experiment.config.seed_simulator
        elif hasattr(self._qobj_config, 'seed_simulator'):
            seed_simulator = self._qobj_config.seed_simulator
        else:
            # For compatibility on Windows force dyte to be int32
            ## TODO
            # and set the maximum value to be (4 ** 31) - 1
            seed_simulator = np.random.randint(2147483647, dtype='int32')

        self._local_random.seed(seed=seed_simulator)
        # Check if measure sampling is supported for current circuit
        #self._validate_measure_sampling(experiment)

        # List of final counts for all shots
        memory = []
        # Check if we can sample measurements, if so we only perform 1 shot
        # and sample all outcomes from the final state vector
        if self._sample_measure:
            measure_sample_ops = []
            # Store (qubit, cmembit) pairs for all measure ops in circuit to
            # be sampled
        else:
            shots = self._shots

        self._initialize_densitymatrix()
        # Initialize classical memory to all 0
        self._classical_memory = 0
        self._classical_register = 0
        
        print('Initial: ', experiment.instructions)
        
        experiment.instructions = single_gate_merge(experiment.instructions,
                                                    self._number_of_qubits)
        print('Merged: ', experiment.instructions)

        partitioned_instructions, levels = partition(experiment.instructions, 
                                                self._number_of_qubits)
        
        print('Partitioned: ', partitioned_instructions)
        
        end_processing = time.time()
        start_runtime = time.time()
        

        self._add_ensemble_measure(0.99)
        for clock in range(levels):

            print('Level: ', clock, partitioned_instructions[clock])
            for operation in partitioned_instructions[clock]:
                conditional = getattr(operation, 'conditional', None)
                if isinstance(conditional, int):
                    conditional_bit_set = (self._classical_register >> conditional) & 1
                    if not conditional_bit_set:
                        continue
                elif conditional is not None:
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_memory & mask
                        while (mask & 0x1) == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue
    
                if operation.name in ('U', 'u1', 'u2', 'u3'):
                    params = getattr(operation, 'params', None)
                    gate = single_gate_dm_matrix(operation.name, params)
                    qubit = operation.qubits[0]
                    self._add_unitary_single(gate, qubit)
                elif operation.name in ('id', 'u0'):
                    pass
                # Check if CX gate
                elif operation.name in ('CX', 'cx'):
                    qubit0 = operation.qubits[0]
                    qubit1 = operation.qubits[1]
                    self._add_unitary_two(qubit0, qubit1)
                # Check if reset
                elif operation.name == 'reset':
                    qubit = operation.qubits[0]
                    self._add_qasm_reset(qubit)
                # Check if barrier
                elif operation.name == 'barrier':
                    pass
                # Check if measure
                #TODO FIX MEASURE
                elif operation.name == 'measure':
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(operation, 'register') else None
                    
                    if len(partitioned_instructions[clock]) == 1:
                        self._add_qasm_measure_Z(qubit, self._probability_of_zero)
                    else:
                        self._add_ensemble_measure(1)
                    
                    #if self._sample_measure:
                        # If sampling measurements record the qubit and cmembit
                        # for this measurement for later sampling
                        # measure_sample_ops.append((qubit, cmembit))
                    #else:
                        # If not sampling perform measurement as normal
                        #self._add_qasm_measure(qubit, cmembit, cregbit)
                elif operation.name == 'bfunc':
                    mask = int(operation.mask, 16)
                    relation = operation.relation
                    val = int(operation.val, 16)
                    cregbit = operation.register
                    cmembit = operation.memory if hasattr(operation, 'memory') else None
                    compared = (self._classical_register & mask) - val
                    if relation == '==':
                        outcome = (compared == 0)
                    elif relation == '!=':
                        outcome = (compared != 0)
                    elif relation == '<':
                        outcome = (compared < 0)
                    elif relation == '<=':
                        outcome = (compared <= 0)
                    elif relation == '>':
                        outcome = (compared > 0)
                    elif relation == '>=':
                        outcome = (compared >= 0)
                    else:
                        raise BasicAerError('Invalid boolean function relation.')
                    # Store outcome in register and optionally memory slot
                    regbit = 1 << cregbit
                    self._classical_register = \
                        (self._classical_register & (~regbit)) | (int(outcome) << cregbit)
                    if cmembit is not None:
                        membit = 1 << cmembit
                        self._classical_memory = \
                            (self._classical_memory & (~membit)) | (int(outcome) << cmembit)
                else:
                    backend = self.name()
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise BasicAerError(err_msg.format(backend, operation.name))

            # Add Memory errors at the end of each clock cycle
            for qb in range(self._number_of_qubits):
                self._add_decoherence_and_amp_decay(clock, f = self._decoherence_factor, 
                                        p = self._thermal_factor, g = self._decay_factor)

        # Add final creg data to memory list
        if self._number_of_cmembits > 0:
            if self._sample_measure:
                # If sampling we generate all shot samples from the final densitymatrix
                memory = self._add_sample_measure(measure_sample_ops, self._shots)
            else:
                # Turn classical_memory (int) into bit string and pad zero for unused cmembits
                outcome = bin(self._classical_memory)[2:]
                memory.append(hex(int(outcome, 2)))

        # Add data
        data = {'counts': dict(Counter(memory))}
        # Optionally add memory list
        if self._memory:
            data['memory'] = memory
        # Optionally add final densitymatrix
        if self.SHOW_FINAL_STATE:
            data['coeffmatrix'], data['densitymatrix'] = self._get_densitymatrix()
            # Remove empty counts and memory for densitymatrix simulator
            if not data['counts']:
                data.pop('counts')
            if 'memory' in data and not data['memory']:
                data.pop('memory')
        end_runtime = time.time()
        return {'name': experiment.header.name,
                'seed_simulator': seed_simulator,
                'shots': self._shots,
                'data': data,
                'status': 'DONE',
                'success': True,
                'processing_time_taken': -start_processing+end_processing,
                'running_time_taken': -start_runtime+end_runtime,
                'header': experiment.header.as_dict()}

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas."""
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise BasicAerError('Number of qubits {} '.format(n_qubits) +
                                'is greater than maximum ({}) '.format(max_qubits) +
                                'for "{}".'.format(self.name()))
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s", '
                               'counts will be empty.', name)
            elif 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s", '
                               'classical register will remain all zeros.', name)
