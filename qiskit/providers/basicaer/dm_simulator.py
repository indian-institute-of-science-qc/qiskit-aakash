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
        'n_qubits': MAX_QUBITS_MEMORY,
        'url': 'https://github.com/Qiskit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 1,
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
        "chop_threshold": 1e-15,
        "thermal_factor": 0.,
        "decoherence_factor": 1.,
        "depolarization_factor": 1.,
        "bell_depolarization_factor": 1.,
        "decay_factor": 1.,
        "rotation_error": {'rx':[1., 0.], 'ry':[1., 0.], 'rz': [1., 0.]},
        "tsp_model_error": [1., 0.]
    }

    # Class level variable to return the final state at the end of simulation
    # This should be set to True for the densitymatrix simulator
    SHOW_FINAL_STATE = False
    DEBUG = True

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
        self._memory = False
        self._custom_densitymatrix = None
        self._initial_densitymatrix = self.DEFAULT_OPTIONS["initial_densitymatrix"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        self._qobj_config = None
        # Errors
        self._error_params = {}
        self._rotation_error = None         # [<cos(fluctuation)>, mean] , Single Rotation gates errors
        self._tsp_model_error = None        # [<cos(fluctuation)>, mean]  , Transition selective pulse error 
        self._thermal_factor = None         # p
        self._decoherence_factor = None     # f
        self._decay_factor = None           # g
        self._depolarization_factor = None  # During Measurement (Bit flip and Depolarization have the same 
        effect)
        self._bell_depolarization_factor = None
        # TEMP
        self._sample_measure = False
        self._get_den_mat = False#True
        self._error_included = False

    def _add_unitary_single(self, gate, qubit):
        """Apply an arbitrary 1-qubit unitary matrix.

        Args:
            params (list): list of parameters for U1,U2 and U3 gate.
            qubit (int): the qubit to apply gate to
        """
        
        # changing density matrix
        lt, mt, rt = 4 ** qubit, 4, 4 ** (self._number_of_qubits-qubit-1)
        self._densitymatrix = np.reshape(self._densitymatrix, (lt, mt, rt))

        for idx in gate: # For Rotations in the Decomposed Gate list
            self._densitymatrix = rt_gate_dm_matrix(
                idx[0], idx[1], self._error_params['one_qubit_gates'][idx[0]], self._densitymatrix, qubit, self._number_of_qubits)

        self._densitymatrix = np.reshape(self._densitymatrix,
                                    self._number_of_qubits * [4])

    def _add_unitary_two(self, qubit0, qubit1):
        """Apply a two-qubit unitary matrix.

        Args:
            gate (matrix_like): a the two-qubit gate matrix
            qubit0 (int): control qubit 
            qubit1 (int): target qubit
        """ 
        
        self._densitymatrix = cx_gate_dm_matrix(self._densitymatrix,
                                                qubit0, qubit1, self._error_params['two_qubit_gates'],self._number_of_qubits)
        
        self._densitymatrix = np.reshape(self._densitymatrix,
                                        self._number_of_qubits * [4])

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
       
            self._densitymatrix[:, 1, :] = off_diag_contract * \
                                            temp[:, 1, :]
            self._densitymatrix[:, 2, :] = off_diag_contract * \
                                            temp[:, 2, :]
            self._densitymatrix[:, 3, :] =  g * temp[:, 3, :] + \
                                                    diag_decay * temp[:, 0, :]
        
        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

    def _add_ensemble_measure(self, basis, err_param):
        """Perform complete computational basis measurement for current densitymatrix.

        Args:
            err_param   (float): Reduction in polarization during measurement
        Returns:
            list: Complete list of probabilities. 
        """
        # TODO Generalize it
        supplement_data = {'X': [0, 1], 'Y': [0, 2], 'Z': [0, 3]}

        if basis != 'N':
            # We get indices used for Probability Measurement via this.
            measure_ind = [x for x in itertools.product(supplement_data[basis], repeat=self._number_of_qubits)]
            # We get coefficient values stored at those indices via this. 
            operator_ind = [self._densitymatrix[x] for x in measure_ind]
            # We get permutations of signs for summing those coefficient values.
            operator_mes = np.array([[1, err_param], [1, -err_param]], dtype=float)
            for i in range(self._number_of_qubits-1):
                operator_mes = np.kron(np.array([[1, err_param], [1, -err_param]]), operator_mes)

            # We get 2**n probabilities via this.
            probabilities = np.reshape(
                                np.array([np.sum(np.multiply(operator_ind, x)) for x in operator_mes]), 
                                2**self._number_of_qubits)

            key = [x for x in itertools.product([0,1],repeat = self._number_of_qubits)]
            prob_key = [''.join(str(y) for y in x) for x in key]
            prob = {}

            for i in range(2**self._number_of_qubits):
                prob.update({prob_key[i]: probabilities[i]})


    def _add_partial_measure(self, qubits, cmembits , cregbits , err_param, basis, add_param = None):
        """Perform complete computational basis measurement for current densitymatrix.

        Args:
            err_param   (float): Reduction in polarization during measurement
        Returns:
            list: Complete list of probabilities. 
        """
        supplement_data = { 'X':[self._add_qasm_measure_X, [0, 1]], 
                            'Y':[self._add_qasm_measure_Y, [0, 2]], 
                            'Z':[self._add_qasm_measure_Z, [0, 3]] 
                        }

        if basis != 'N':
            measured_qubits = qubits #list({qubit for qubit, cmembit in measure_params})
            num_measured = len(measured_qubits)

            axis = list(range(self._number_of_qubits))
            for qubit in reversed(measured_qubits):
                axis.remove(qubit)

            # We get indices used for Probability Measurement via this.
            measure_ind = [x for x in itertools.product(
                supplement_data[basis][1], repeat=self._number_of_qubits)]
            # We get coefficient values stored at those indices via this.
            operator_ind = [self._densitymatrix[x] for x in measure_ind]
            # We get permutations of signs for summing those coefficient values.
            operator_mes = np.array([[1, err_param], [1, -err_param]], dtype=float)
            for i in range(self._number_of_qubits-1):
                operator_mes = np.kron(
                    np.array([[1, err_param], [1, -err_param]]), operator_mes)

            probabilities = np.reshape(np.sum(np.reshape(np.array([np.sum(np.multiply(
                operator_ind, x)) for x in operator_mes]), self._number_of_qubits * [2]),  
                axis=tuple(axis)), 2**num_measured)
        
            key = [x for x in itertools.product(
                [0, 1], repeat=num_measured)]
            prob_key = [''.join(str(y) for y in x) for x in key]
            prob = {}

            for i in range(2**num_measured):
                prob.update({prob_key[i]: probabilities[i]})
        
        for mqb,mcb,mcregb in list(zip(measured_qubits,cmembits,cregbits)):
            if basis == 'N' and add_param is not None:
                self._add_qasm_measure_N(
                    mqb, mcb, mcregb, add_param, self._error_params['measurement'])
            else:
                supplement_data[basis][0](mqb, mcb, mcregb, 
                                self._error_params['measurement'])
        
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
        bell_probabilities = [0.0,0.0,0.0,0.0]
        for i in range(4):
            for j in range(4):
                if i != j:
                    self._densitymatrix[:,i,:,j,:] = 0
        
        k_0 = self._densitymatrix[:,0,:,0,:].sum()
        k_1 = self._densitymatrix[:,1,:,1,:].sum()
        k_2 = self._densitymatrix[:,2,:,2,:].sum()
        k_3 = self._densitymatrix[:,3,:,3,:].sum()
        bell_probabilities[0] = 0.25*(k_0 + k_1 - k_2 + k_3)
        bell_probabilities[1] = 0.25*(k_0 - k_1 + k_2 + k_3)
        bell_probabilities[2] = 0.25*(k_0 + k_1 + k_2 - k_3)
        bell_probabilities[3] = 0.25*(k_0 - k_1 - k_2 - k_3)

        return bell_probabilities
    
    def _add_qasm_measure_X(self, qubit, cmembit,cregbit=None, err_param=1.0):
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

        self._densitymatrix[:,2,:] = 0
        self._densitymatrix[:,3,:] = 0
        self._densitymatrix[:,1,:] *= err_param
        p_1 = self._densitymatrix[:, 1, :].sum()
               
        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        probability_of_zero = 0.5 * (1 + p_1)
        probability_of_one = 1 - probability_of_zero


        if probability_of_zero > probability_of_one:
            outcome, probability = 0,probability_of_zero
        else:
            outcome, probability = 1, probability_of_one

        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (
            ~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (
                    int(outcome) << cregbit)

        return outcome,probability

    def _add_qasm_measure_Y(self, qubit, cmembit, cregbit=None, err_param=1.0):
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

        self._densitymatrix[:,1,:] = 0
        self._densitymatrix[:,3,:] = 0
        self._densitymatrix[:,2,:] *= err_param
        p_2 = self._densitymatrix[:,2,:].sum()

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        probability_of_zero = 0.5 * (1 + p_2)
        probability_of_one = 1 - probability_of_zero


        if probability_of_zero > probability_of_one:
            outcome, probability = 0, probability_of_zero
        else:
            outcome, probability = 1, probability_of_one

        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (
            ~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (
                    int(outcome) << cregbit)

        return outcome, probability

    def _add_qasm_measure_Z(self, qubit, cmembit, cregbit=None, err_param=1.0):
        """Apply a Z basis measure instruction to a qubit. 
        Post-measurement density matrix is returned in the same array.

        Args:
            qubit (int): qubit is the qubit measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        #print(err_param)
        self._densitymatrix = np.reshape(
            self._densitymatrix, (4**(qubit), 4, 4**(self._number_of_qubits-qubit-1)))
        p_3 = 0.0

        self._densitymatrix[:, 1, :] = 0
        self._densitymatrix[:, 2, :] = 0
        self._densitymatrix[:, 3, :] *= err_param
        p_3 = self._densitymatrix[:, 3, :].sum()

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        probability_of_zero = 0.5 * (1 + p_3)
        probability_of_one = 1 - probability_of_zero

        if probability_of_zero > probability_of_one:
            outcome, probability = 0, probability_of_zero
        else:
            outcome, probability = 1, probability_of_one

        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (
            ~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (
                    int(outcome) << cregbit)

        return outcome, probability

    def _add_qasm_measure_N(self, qubit , cmembit , cregbit = None, n = np.array([0.0,0.0,1.0]), err_param = 1.0):
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

        temp = n[0]*self._densitymatrix[:,1,:] + n[1]*self._densitymatrix[:,2,:] + \
                       n[2]*self._densitymatrix[:,3,:]
        temp *= err_param
                
        self._densitymatrix[:,1,:] = temp*n[0] 
        self._densitymatrix[:,2,:] = temp*n[1]
        self._densitymatrix[:,3,:] = temp*n[2]

        p_n =  temp.sum()

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        probability_of_zero = 0.5 * (1 + p_n)
        probability_of_one = 1 - probability_of_zero


        if probability_of_zero > probability_of_one:
            outcome, probability = 0, probability_of_zero
        else:
            outcome, probability = 1, probability_of_one

        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (
            ~membit)) | (int(outcome) << cmembit)

        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = \
                (self._classical_register & (~regbit)) | (
                    int(outcome) << cregbit)

        return outcome, probability

    def _add_qasm_reset(self, qubit):
        """Apply a reset instruction to a qubit.

        Args:
            qubit (int): the qubit being reset

        This is done by setting the measured qubit to the zero state.
        It is equivalent to performing P0*rho*P0+X*P1*rho*P1*X.
        """

        # update density matrix
        self._densitymatrix =  np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))

        self._densitymatrix[:,1,:] = 0
        self._densitymatrix[:,2,:] = 0
        self._densitymatrix[:,3,:] = self._densitymatrix[:,0,:].copy()


    def _validate_initial_densitymatrix(self):
        """Validate an initial densitymatrix"""
        # If initial densitymatrix isn't set we don't need to validate
        if self._initial_densitymatrix is None:
            return
        if self._custom_densitymatrix == 'binary_string':
            return 
        # Check densitymatrix is correct length for number of qubits
        length = np.size(self._initial_densitymatrix)
        ##print(length, self._number_of_qubits)
        required_dim = 4 ** self._number_of_qubits
        
        if length != required_dim:
            raise BasicAerError('initial densitymatrix is incorrect length: ' + '{} != {}'.format(length, required_dim))

        if self._densitymatrix[0] != 1:
            raise BasicAerError('Trace of initial densitymatrix is not one: ' + '{} != {}'.format(self._densitymatrix[0], 1))

    def _set_options(self, qobj_config=None, backend_options=None):
        """Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_densitymatrix = self.DEFAULT_OPTIONS["initial_densitymatrix"]
        self._chop_threshold = self.DEFAULT_OPTIONS["chop_threshold"]
        self._rotation_error = self.DEFAULT_OPTIONS["rotation_error"]
        self._tsp_model_error = self.DEFAULT_OPTIONS["tsp_model_error"]
        self._thermal_factor = self.DEFAULT_OPTIONS["thermal_factor"]
        self._decoherence_factor = self.DEFAULT_OPTIONS["decoherence_factor"]
        self._decay_factor = self.DEFAULT_OPTIONS["decay_factor"]
        self._depolarization_factor = self.DEFAULT_OPTIONS["depolarization_factor"]
        self._bell_depolarization_factor = self.DEFAULT_OPTIONS["bell_depolarization_factor"]

        if backend_options is None:
            backend_options = {}
        
        # Check for custom initial densitymatrix in backend_options first,
        # then config second
        if 'initial_densitymatrix' in backend_options:
            self._initial_densitymatrix = np.array(backend_options['initial_densitymatrix'], dtype=float)
        elif hasattr(qobj_config, 'initial_densitymatrix'):
            self._initial_densitymatrix = np.array(qobj_config.initial_densitymatrix,
                                                 dtype=float)

        if 'custom_densitymatrix' in backend_options:
            self._custom_densitymatrix = backend_options['custom_densitymatrix']
            if self._custom_densitymatrix == 'binary_string':
                self._initial_densitymatrix = backend_options['initial_densitymatrix']

        # Error for Rotation Gates
        if 'rotation_error' in backend_options:
            if type(backend_options['rotation_error']) != dict or not all(x in ['rx', 'ry', 'rz'] for x in backend_options['rotation_error']) :
                raise BasicAerError('Error! Incorrect Rotation Error parameters, Expected argument : A dict with rotation gate as key and a list of 2 reals ranging between 0 and 1 both inclusive as their values.')
            else:
                for gt, vl in backend_options['rotation_error'].items():
                    self._rotation_error.update({gt:vl})

        # Error in CX based on Transition Selective model
        if 'tsp_model_error' in backend_options:
            if type(backend_options['tsp_model_error']) != list or len(backend_options['tsp_model_error']) !=2 or backend_options['tsp_model_error'][0] > 1 or backend_options['tsp_model_error'][1] > 1:
                raise BasicAerError('Error! Incorrect transition model error parameter, Expected argument : A list of 2 reals ranging between 0 and 1 both inclusive.')
            else:
                self._ts_model_error = backend_options['tsp_model_error']

        # Error by Thermalization
        if 'thermal_factor' in backend_options:
            self._thermal_factor = backend_options['thermal_factor']

        # Error by Decoherence
        if 'decoherence_factor' in backend_options:
            del_T = backend_options['decoherence_factor'][0]
            T_2 = backend_options['decoherence_factor'][1]
            self._decoherence_factor = np.exp(-del_T/T_2)

        # Error by state decay
        if 'decay_factor' in backend_options:
            del_T = backend_options['decay_factor'][0]
            T_1 = backend_options['decay_factor'][1]
            self._decay_factor = np.exp(-del_T/T_1)

        if 'depolarization_factor' in backend_options:
            self._depolarization_factor = backend_options['depolarization_factor']

        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

        if 'compute_densitymatrix' in backend_options:
            self._get_den_mat = backend_options['compute_densitymatrix']
        
        if 'debug' in backend_options:
            DEBUG = backend_options['debug']


    def _initialize_errors(self):

        self._error_params.update({'one_qubit_gates':self._rotation_error})
        self._error_params.update({'two_qubit_gates':self._tsp_model_error})
        self._error_params.update({'memory':{'thermalization':self._thermal_factor,
                                             'decoherence':self._decoherence_factor, 
                                             'amplitude_decay':self._decay_factor}
                                            })
        self._error_params.update({'measurement':self._depolarization_factor})

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
            if len(self._initial_densitymatrix) != self._number_of_qubits:
                raise BasicAerError('Wrong input binary string length')
            if self._initial_densitymatrix[0] == '0':
                self._densitymatrix = np.array([1,0,0,1], dtype=float)
            else:
                self._densitymatrix = np.array([1,0,0,-1], dtype=float) 
            for idx in self._initial_densitymatrix[1:]:
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

    def _compute_densitymatrix(self, vec):
        '''
            Generates density matrix from a given coefficient matrix
        '''

        p_0 = np.array([[1, 0], [0, 1]], dtype=complex)
        p_1 = np.array([[0, 1], [1, 0]], dtype=complex)
        p_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        p_3 = np.array([[1, 0], [0, -1]], dtype=complex)
        pauli_basis = [p_0, p_1, p_2, p_3]
        den_creat = [x for x in itertools.product(
            [0, 1, 2, 3], repeat=self._number_of_qubits)]
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
        
        if not self._error_included:
            np.savetxt("a.txt", np.asarray(
                np.round(densitymatrix, 4)), fmt='%1.3f', newline="\n")
        else:
            np.savetxt("a1.txt", np.asarray(
                np.round(densitymatrix, 4)), fmt='%1.3f', newline="\n")

        return densitymatrix

    def _get_densitymatrix(self):
        """Return the current densitymatrix in JSON Result spec format"""
        # Coefficients
        vec = np.reshape(self._densitymatrix.real, 4 ** self._number_of_qubits)
        vec[abs(vec) < self._chop_threshold] = 0.0
        #pprint.pprint(vec)
        if self._get_den_mat:
            densitymatrix = self._compute_densitymatrix(vec)
            return vec, densitymatrix
        else:
            densitymatrix = None
            return vec

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

        return result_list

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
        self._initialize_errors()
        # Initialize classical memory to all 0
        self._classical_memory = 0
        self._classical_register = 0
        
        experiment.instructions = single_gate_merge(experiment.instructions,
                                                    self._number_of_qubits)
            
        partitioned_instructions, levels = partition(experiment.instructions, 
                                                self._number_of_qubits)

        end_processing = time.time()
        start_runtime = time.time()

        for clock in range(levels):

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
                    #a, b = self._get_densitymatrix()
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
                elif operation.name == 'measure':
                    params = getattr(operation, 'params', None)
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(
                        operation, 'register') else None

                    sngl_measure = True
                    part_measure = True
                    ensm_measure = True

                    len_pi = len(partitioned_instructions[clock])

                    for mt in partitioned_instructions[clock]:
  
                        para = getattr(mt, 'params', None)

                        if para is not None and params is not None and para != params[0]:
                            ensm_measure = False                            
                            part_measure = False
                            break

                    if params is not None:
                        params[0] = str(params[0])
                    else:
                        params = ['Z']

                    if params[0] == 'Bell':
                        part_measure = False
                        ensm_measure = False
                    
                    if part_measure or ensm_measure:
                        sngl_measure = False

                    if self._sample_measure:
                        sngl_measure = False
                        ensm_measure = True

                    cregbit = operation.register[0] if hasattr(operation, 'register') else None

                    if len_pi == 1 or sngl_measure:
                        if params[0] == 'X':
                            self._add_qasm_measure_X(
                                qubit, cmembit, cregbit, self._error_params['measurement'])
                        elif params[0] == 'Y':
                            self._add_qasm_measure_Y(
                                qubit, cmembit, cregbit, self._error_params['measurement'])
                        elif params[0] == 'N':
                            self._add_qasm_measure_N(
                                qubit, cmembit, cregbit, params[1], self._error_params['measurement'])
                        elif params[0] == 'Bell':
                            self._add_bell_basis_measure(int(params[1][0], int(params[1][1])))
                        else:
                            self._add_qasm_measure_Z(
                                qubit,cmembit,cregbit,self._error_params['measurement'])
                        # Check for the next groupings
                        partitioned_instructions[clock].remove(operation)       

                    elif part_measure and len_pi > 1 and len_pi < self._number_of_qubits:
                        qu_mes_list = [x.qubits[0] for x in partitioned_instructions[clock]]
                        cmem_mes_list = [x.memory[0] for x in partitioned_instructions[clock]]
                        creg_mes_list = []

                        for x in partitioned_instructions[clock]:
                            cregbit = x.register[0] if hasattr(x, 'register') else None
                            creg_mes_list.append(cregbit)

                        if params[0] != 'N':
                            self._add_partial_measure(
                                qu_mes_list, cmem_mes_list, creg_mes_list, self._error_params['measurement'], params[0])
                        else:
                            self._add_partial_measure(
                                qu_mes_list, cmem_mes_list, creg_mes_list, self._error_params['measurement'], params[0], params[1])
                        break
                    
                    else:
                        self._add_ensemble_measure(params[0], self._error_params['measurement'])
                        break

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
                    self._add_decoherence_and_amp_decay(clock, 
                                f = self._error_params['memory']['decoherence'], 
                                p = self._error_params['memory']['thermalization'], 
                                g = self._error_params['memory']['amplitude_decay']
                            )

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
            if self._get_den_mat:
                data['coeffmatrix'], data['densitymatrix'] = self._get_densitymatrix()
            else:
                data['coeffmatrix'] = self._get_densitymatrix()

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
