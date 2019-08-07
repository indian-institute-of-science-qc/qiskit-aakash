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
"""

import uuid
import time
import logging

from math import log2
from collections import Counter
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.providers import BaseBackend
from qiskit.providers.basicaer.basicaerjob import BasicAerJob
from .exceptions import BasicAerError
from .basicaertools import *

logger = logging.getLogger(__name__)


class DmSimulatorPy(BaseBackend):
    """Python implementation of a Density Matrix simulator.
    The density matrix is expressed in the orthogonal Pauli basis as
    rho = sum_{ij...} a_{ij...} sigma_i x sigma_j x ...
    The array "densitymatrix" contains the 4**n real coefficients a_{ij...},
    with each subscript taking values 0,1,2,3 (equivalently I,X,Y,Z).
    The default shape for the array a_{ij...} is n*[4].
    """

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
    SHOW_FINAL_STATE = True
    PLOTTING = False
    SHOW_PARTITION = False
    DEBUG = True
    STORE_LOCAL = False
    COMPARE = False
    FILE_EXIST = False

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
        self._depolarization_factor = None  # During Measurement (Bit flip and Depolarization have the same effect)
        self._bell_depolarization_factor = None
        # TEMP
        self._sample_measure = False
        self._get_den_mat = True
        self._error_included = False
        # self.result_dict = None
        self.fidelity = None 
        self.density_matrix_0 = None 

    def _add_unitary_single(self, gate, qubit):
        """Apply an arbitrary 1-qubit unitary transformation.

        Args:
            gate (list): the type of gate (u1, u2 or u3) together with its parameters.
            qubit (int): the qubit to apply the gate to.
        """
        
        # changing density matrix
        lt, mt, rt = 4 ** qubit, 4, 4 ** (self._number_of_qubits-qubit-1)
        self._densitymatrix = np.reshape(self._densitymatrix, (lt, mt, rt))

        for idx in gate: # For Rotations in the Decomposed Gate list
            self._densitymatrix = rot_gate_dm_matrix(
                idx[0], idx[1], self._error_params['one_qubit_gates'][idx[0]], self._densitymatrix, qubit, self._number_of_qubits)

        self._densitymatrix = np.reshape(self._densitymatrix,
                                    self._number_of_qubits * [4])

    def _add_unitary_two(self, qubit0, qubit1):
        """Apply a two-qubit unitary transformation (only cx gate is included).

        Args:
            qubit0 (int): control qubit 
            qubit1 (int): target qubit
        """ 
        
        self._densitymatrix = cx_gate_dm_matrix(self._densitymatrix,
                                                qubit0, qubit1, self._error_params['two_qubit_gates'],self._number_of_qubits)
        
    def _add_decoherence_and_amp_decay(self, level, f, p, g):
        """ Apply decoherence transofrmation and amplitude decay transformation independently 
            to all the qubits. Off-diagonal elements of the density get contracted by a factor
            'f' due to decoherence and 'sqrt(g)' due to amplitude decay. Diagonal elements decay
            with rate 'g' towards the thermal state specified by 'p'.
        Args:
            level (int):    Clock cycle number (not used)
            f     (float):  Contraction of diagonal elements due to T_2 (decoherence time) 
            p     (float):  Thermal factor corresponding to the asymptotic state
            g     (float):  Decay of the excited state component due to T_1 (relaxation time)
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

    def _add_ensemble_measure(self, basis, add_param, err_param):
        """ Perform complete computational basis measurement for current density matrix.
            The density matrix is not updated.
        Args:
            basis       (string): Direction of measurement (same for all qubits)- 'X', 'Y', 'Z' or 'N'.
            err_param   (float): Reduction in polarization during measurement
            add_param : parameters specifying components of N
        Returns:
            prob (dictionray): 2**n possible results with their probabilities
            max_str (string): location of the result with maximum probability
            max_prob (float): value of the maximum probability
        """
        supplement_data = {'X': [0, 1], 'Y': [
            0, 2], 'Z': [0, 3], 'N': [0, 1, 2, 3]}

        # We get indices used for Probability Measurement via this.
        
        measure_ind = [x for x in itertools.product(
            supplement_data[basis], repeat=self._number_of_qubits)]
        # We get coefficient values stored at those indices via this.
        operator_ind = [self._densitymatrix[x] for x in measure_ind]
        # We get permutations of signs for summing those coefficient values.

        if basis != 'N':
            operator_mes = np.array(
                [[1, err_param], [1, -err_param]], dtype=float)
            for i in range(self._number_of_qubits-1):
                operator_mes = np.kron(
                    np.array([[1, err_param], [1, -err_param]]), operator_mes)
        else:
            n = add_param*err_param
            operator_mes = np.array(
                [[1, n[0], n[1], n[2]], [1, -n[0], -n[1], -n[2]]])
            for i in range(self._number_of_qubits-1):
                operator_mes = np.kron(np.array([[1, n[0], n[1], n[2]], [1, -n[0], -n[1], -n[2]]]),
                                       operator_mes)

        # We get 2**n probabilities via this.
        probabilities = np.reshape(
            np.array([np.sum(np.multiply(operator_ind, x))
                      for x in operator_mes]),
            2**self._number_of_qubits)

        prob_key = ["".join(s) for s in itertools.product("01", repeat=self._number_of_qubits)]
        prob = dict(zip(prob_key, probabilities))
        max_str = max(prob, key=prob.get)
        max_prob = prob[max_str]
        
        # Store the density matrix in a local file
        if self.STORE_LOCAL:
            self.store_density_matrix()
        # Compare the current density matrix with stored density matrix
        if self.COMPARE and self.FILE_EXIST:
            self.fidelity = self.state_overlap(self.density_matrix_0, np.reshape(self._densitymatrix,4**self._number_of_qubits))
        return prob, max_str, max_prob
    
    def _plot_ensemble_measure(self,prob,basis):
        ''' Plots the probability distribution of 2**n possible results if the 'plot' flag is on.
        '''
        if self.PLOTTING:
            plt.bar(prob.keys(),prob.values())
            plt.title(f"Probability Distribution for ensemble measurement in {basis} basis")
            plt.xticks(rotation='vertical')
            plt.show()

    def _add_partial_measure(self, measured_qubits, cmembits, cregbits, err_param, basis, add_param=None):
        """ Perform partial measurement for current density matrix on the specified qubits along the given common basis direction.
            Post measurement density matrix is updated in the same array.

        Args:
            measured_qubits (int) : list of measured qubits
            cmembits: classical memory bits
            cregbits: classical register bits
            basis       (string): Direction of measurement (same for all qubits) 'X', 'Y', 'Z' or 'N'.
            err_param   (float): Reduction of  polarization during measurement
            add_param : parameters specifying components of unit vector N
        Returns:
            partial_prob (dictionray): possible results for measured qubits with their probabilities
            max_str (string): location of the result with maximum probability
            max_prob (float): value of the maximum probability   
        """

        supplement_data = { 'X':[self._add_qasm_measure_X, [0, 1]], 
                            'Y':[self._add_qasm_measure_Y, [0, 2]], 
                            'Z':[self._add_qasm_measure_Z, [0, 3]],
                            'N':[self._add_qasm_measure_N, [0, 1, 2, 3]] 
                        }
        num_measured = len(measured_qubits)
        axis = list(set(range(self._number_of_qubits)) - set(measured_qubits))

        # Calculate the probabilities
        prob_dict, max_str, max_prob = self._add_ensemble_measure(basis, add_param, err_param)
        prob_ensemble = np.array(list(prob_dict.values()))
        probabilities = np.reshape(np.sum(np.reshape(prob_ensemble, self._number_of_qubits * [2]),  
            axis=tuple(axis)), 2**num_measured)
        
        prob_key = ["".join(s) for s in itertools.product("01", repeat=num_measured)]
        partial_prob = dict(zip(prob_key, probabilities))
        
        max_str = max(partial_prob, key=partial_prob.get)
        max_prob = partial_prob[max_str]

        # Update the density matrix
        for mqb,mcb,mcregb in list(zip(measured_qubits,cmembits,cregbits)):
            if basis == 'N' and add_param is not None:
                self._add_qasm_measure_N(
                    mqb, mcb, mcregb, add_param, self._error_params['measurement'])
            else:
                supplement_data[basis][0](mqb, mcb, mcregb, 
                                self._error_params['measurement'])
        return partial_prob, max_str, max_prob
        
    def _pauli_string_expectation(self, basis, err_param, add_param = None):
        """
        Returns expectation value for a given string of pauli matrices.
        Post measurement density matrix is updated in the same array.

        Args:
            basis (list): pauli string (alphabet from {'I','X','Y','Z'}) corresponding to the measured operator.
            err_param (float): Reduction in polarization during measurement.
        Returns:
            expectation (float): expectation value of the pauli string operator.

        """
        bas_ind = {'I':0, 'X':1, 'Y':2, 'Z':3}
        for i in range(self._number_of_qubits):
            self._densitymatrix = np.reshape(self._densitymatrix,(4**(i),4,4**(self._number_of_qubits-i-1)))
            if basis[i] == 'X':
                self._densitymatrix[:,1,:] *= err_param 
                self._densitymatrix[:,2,:] = 0
                self._densitymatrix[:,3,:] = 0
            elif basis[i] == 'Y':
                self._densitymatrix[:,1,:] = 0
                self._densitymatrix[:,2,:] *= err_param
                self._densitymatrix[:,3,:] = 0
            elif basis[i] == 'Z':
                self._densitymatrix[:,1,:] = 0
                self._densitymatrix[:,2,:] = 0
                self._densitymatrix[:,3,:] *= err_param
        
        self._densitymatrix = np.reshape(self._densitymatrix, self._number_of_qubits * [4])
        index = tuple([bas_ind[x] for x in basis])
        expectation = self._densitymatrix[index] * 2**self._number_of_qubits
        
        return expectation

    def _add_bell_basis_measure(self, qubit_1, qubit_2, err_param):
        """
        Apply a Bell basis measure instruction for two qubits.
        Post measurement density matrix is updated in the same array.
        Four Bell probabilities are calculated in the (|00>+|11>,|00>-|11>,|01>+|10>,|01>-|10>) basis. 
        Two qubit reduced density matrix is plotted if 'plot' flag is on.

        Args:
            qubit_1 (int): first qubit of the Bell pair.
            qubit_2 (int): second qubit of the Bell pair.
            err_param (float): Reduction in polarization during measurement.
        Returns (as global variables):
            reduced_bell_densitymatrix for the two qubits (prior to measurement)
            bell_probabilities for the four orthogonal Bell states (after measurement)
        """
        q_1 = min(qubit_1, qubit_2)
        q_2 = max(qubit_1, qubit_2)

        #update density matrix
        
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(self._number_of_qubits-q_2-1), 4, 4**(q_2-q_1-1), 4, 4**q_1))
        
        # Reduced density matrix 
        reduced_bell_densitymatrix = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                reduced_bell_densitymatrix[i,j] = self._densitymatrix[0,i,0,j,0] * 2**(self._number_of_qubits - 2)

        for i in range(4):
            for j in range(4):
                if i != j:
                    self._densitymatrix[:,i,:,j,:] = 0

        self._densitymatrix[:,1,:,1,:] *= err_param
        self._densitymatrix[:,2,:,2,:] *= err_param
        self._densitymatrix[:,3,:,3,:] *= err_param

        k = [0.0,0.0,0.0,0.0]
        for i in range(4):
            k[i] = self._densitymatrix[0,i,0,i,0] * 2**self._number_of_qubits

        bell_probabilities = [0., 0., 0., 0.]
        bell_probabilities[0] = 0.25*(k[0] + k[1] - k[2] + k[3])
        bell_probabilities[1] = 0.25*(k[0] - k[1] + k[2] + k[3])
        bell_probabilities[2] = 0.25*(k[0] + k[1] + k[2] - k[3])
        bell_probabilities[3] = 0.25*(k[0] - k[1] - k[2] - k[3])

        # bell_states = [r'$\frac{|00\rangle + |11\rangle}{\sqrt(2)}$', r'$\frac{|00\rangle - |11\rangle}{\sqrt(2)}$', r'$\frac{|01\rangle + |10\rangle}{\sqrt(2)}$', r'$\frac{|01\rangle - |10\rangle}{\sqrt(2)}$']
        bell_states = ['Bell_1','Bell_2','Bell_3','Bell_4']
        bell_probabilities = dict(zip(bell_states,bell_probabilities))

        self._densitymatrix = np.reshape(self._densitymatrix,self._number_of_qubits*[4])

        # plot the resultant reduced density matrix
        if self.PLOTTING:
            self._plot_reduced_bell_basis(reduced_bell_densitymatrix)
        return bell_probabilities, reduced_bell_densitymatrix
    
    def _plot_reduced_bell_basis(self,reduced_bell_densitymatrix):
        '''
        Plots the two qubit reduced density matrix.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        _x = range(4)
        _y = range(4)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        top = reduced_bell_densitymatrix[x,y]
        bottom = np.zeros_like(top)
        width = 0.5
        depth = 0.5
        values = np.linspace(0.2, 1., x.ravel().shape[0])
        colors = cm.rainbow(values)
        z_up_lim = np.amax(reduced_bell_densitymatrix)
        z_low_lim = np.amin(reduced_bell_densitymatrix)

        ax.set_zlim3d(z_low_lim,z_up_lim)
        ax.w_xaxis.set_ticks(x)
        ax.w_yaxis.set_ticks(y)
        ax.set_title("Reduced Density Matrix in Pauli Basis")
        ax.set_xlabel("First Qubit")
        ax.set_ylabel("Second qubit")
        ax.set_zlabel("Coefficient value")

        ax.bar3d(x-0.25, y-0.25, bottom, width, depth, top, color=colors, alpha=0.8, shade=True)

        plt.show()
    
    def _add_qasm_measure_X(self, qubit, cmembit,cregbit=None, err_param=1.0):
        """Apply a X basis measure instruction to a single qubit. 
        Post measurement density matrix is updated in the same array.

        Args:
            qubit (int): the qubit being measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))
   
        self._densitymatrix[:,1,:] *= err_param
        self._densitymatrix[:,2,:] = 0
        self._densitymatrix[:,3,:] = 0
        
        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        index = [0 for x in range(self._number_of_qubits)]
        index[qubit] = 1
        p_1 = self._densitymatrix[tuple(index)] * 2**self._number_of_qubits

        probability_of_zero = 0.5 * (1 + p_1)
        probability_of_one = 1 - probability_of_zero

        return probability_of_zero

    def _add_qasm_measure_Y(self, qubit, cmembit, cregbit=None, err_param=1.0):
        """Apply a Y basis measure instruction to a single qubit. 
        Post measurement density matrix is updated in the same array.

        Args:
            qubit (int): the qubit being measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))

        self._densitymatrix[:,1,:] = 0
        self._densitymatrix[:,3,:] = 0
        self._densitymatrix[:,2,:] *= err_param

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        index = [0 for x in range(self._number_of_qubits)]
        index[qubit] = 2
        p_2 = self._densitymatrix[tuple(index)] * 2**self._number_of_qubits

        probability_of_zero = 0.5 * (1 + p_2)
        probability_of_one = 1 - probability_of_zero

        return probability_of_zero

    def _add_qasm_measure_Z(self, qubit, cmembit, cregbit=None, err_param=1.0):
        """Apply a Z basis measure instruction to a single qubit. 
        Post measurement density matrix is updated in the same array.

        Args:
            qubit (int): the qubit being measured.
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """

        # update density matrix
        #print(err_param)
        self._densitymatrix = np.reshape(
            self._densitymatrix, (4**(qubit), 4, 4**(self._number_of_qubits-qubit-1)))
        
        self._densitymatrix[:, 1, :] = 0
        self._densitymatrix[:, 2, :] = 0
        self._densitymatrix[:, 3, :] *= err_param

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])
        
        index = [0 for x in range(self._number_of_qubits)]
        index[qubit] = 3
        p_3 = self._densitymatrix[tuple(index)] * 2**self._number_of_qubits

        probability_of_zero = 0.5 * (1 + p_3)
        probability_of_one = 1 - probability_of_zero

        return probability_of_zero

    def _add_qasm_measure_N(self, qubit , cmembit , cregbit = None, n = np.array([0.0,0.0,1.0]), err_param = 1.0):
        """Apply a general n-axis measure instruction to a single qubit. 
        Post measurement density matrix is updated in the same array.

        Args:
            qubit (int): the qubit being measured.
            n           (vec): Axis of measurement (unit vector).
            err_param   (float): Reduction in polarization during measurement
        Return
            probability_of_zero (float): is the probability of getting zero state as outcome.   
        """
            
        # update density matrix
        self._densitymatrix = np.reshape(self._densitymatrix,(4**(qubit),4,4**(self._number_of_qubits-qubit-1)))

        temp = n[0]*self._densitymatrix[:,1,:] + n[1]*self._densitymatrix[:,2,:] + \
                       n[2]*self._densitymatrix[:,3,:]
        temp *= err_param
                
        self._densitymatrix[:,1,:] = temp*n[0] 
        self._densitymatrix[:,2,:] = temp*n[1]
        self._densitymatrix[:,3,:] = temp*n[2]

        self._densitymatrix = np.reshape(self._densitymatrix,
                                         self._number_of_qubits * [4])

        index = [0 for x in range(self._number_of_qubits)]
        
        p_n = 0.0
        for i in range(3):
            index[qubit] = i+1
            p_n += self._densitymatrix[tuple(index)]
        
        p_n *= 2**self._number_of_qubits
        
        probability_of_zero = 0.5 * (1 + p_n)
        probability_of_one = 1 - probability_of_zero

        return probability_of_zero
        
    def _add_qasm_reset(self, qubit):
        """ Reset the qubit to the zero state.
            It is equivalent to performing P0*rho*P0+X*P1*rho*P1*X.

        Args:
            qubit (int): the qubit being reset

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
        if self._custom_densitymatrix == 'stored_density_matrix':
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
        
        # Check for custom initial density matrix in backend_options first,
        # otherwise take it from config.
        if 'initial_densitymatrix' in backend_options:
            self._initial_densitymatrix = np.array(backend_options['initial_densitymatrix'], dtype=float)
        elif hasattr(qobj_config, 'initial_densitymatrix'):
            self._initial_densitymatrix = np.array(qobj_config.initial_densitymatrix, dtype=float)

        if 'custom_densitymatrix' in backend_options:
            self._custom_densitymatrix = backend_options['custom_densitymatrix']
            if self._custom_densitymatrix == 'binary_string':
                self._initial_densitymatrix = backend_options['initial_densitymatrix']
            elif self._custom_densitymatrix == 'stored_density_matrix':
                self._initial_densitymatrix = backend_options['initial_densitymatrix']

        # Error for Single Qubit Rotation Gates
        if 'rotation_error' in backend_options:
            if type(backend_options['rotation_error']) != dict or not all(x in ['rx', 'ry', 'rz'] for x in backend_options['rotation_error']) :
                raise BasicAerError('Error! Incorrect Rotation Error parameters, Expected argument : A dict with rotation gate as key and a list of 2 reals ranging between 0 and 1 both inclusive as their values.')
            else:
                for gt, vl in backend_options['rotation_error'].items():
                    self._rotation_error.update({gt:vl})

        # Error for C-NOT based on Transition Selective Pulse model
        if 'tsp_model_error' in backend_options:
            if type(backend_options['tsp_model_error']) != list or len(backend_options['tsp_model_error']) !=2 or backend_options['tsp_model_error'][0] > 1 or backend_options['tsp_model_error'][1] > 1:
                raise BasicAerError('Error! Incorrect transition model error parameter, Expected argument : A list of 2 reals ranging between 0 and 1 both inclusive.')
            else:
                self._tsp_model_error = backend_options['tsp_model_error']

                
        # Error due to Thermalization
        if 'thermal_factor' in backend_options:
            self._thermal_factor = backend_options['thermal_factor']

        # Error due to Decoherence: decoherence factor = exp(-del_T/T_2)
        if 'decoherence_factor' in backend_options:
            self._decoherence_factor = backend_options['decoherence_factor']

        # Error due to State Decay (1 -> 0): decay_factor = exp(-del_T/T_1)
        if 'decay_factor' in backend_options:
            self._decay_factor = backend_options['decay_factor']

         # Error due to Depolarization (or bit-flip) during measurement
        if 'depolarization_factor' in backend_options:
            self._depolarization_factor = backend_options['depolarization_factor']
        
        # Error due to Depolarization during measurement
        if 'bell_depolarization_factor' in backend_options:
            self.bell_depolarization_factor = backend_options['bell_depolarization_factor']

        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

        if 'compute_densitymatrix' in backend_options:
            self._get_den_mat = backend_options['compute_densitymatrix']
        
        if 'debug' in backend_options:
            DEBUG = backend_options['debug']
        
        if 'plot' in backend_options:
            self.PLOTTING = backend_options['plot'] 
        
        if 'show_partition' in backend_options:
            self.SHOW_PARTITION = backend_options['show_partition']
        
        if 'store_densitymatrix' in backend_options:
            self.STORE_LOCAL = backend_options['store_densitymatrix']

        if 'compare' in backend_options:
            self.COMPARE = backend_options['compare']
            try:
                self.density_matrix_0 = np.load('stored_coefficients.npy')
                self.FILE_EXIST = True
            except FileNotFoundError:
                print('Stored Coefficient File does not exist')

    def _initialize_errors(self):

        self._error_params.update({'one_qubit_gates':self._rotation_error})
        self._error_params.update({'two_qubit_gates':self._tsp_model_error})
        self._error_params.update({'memory':{'thermalization':self._thermal_factor,
                                             'decoherence':self._decoherence_factor, 
                                             'amplitude_decay':self._decay_factor}
                                            })
        self._error_params.update({'measurement': self._depolarization_factor,
                                   'measurement_bell': self._bell_depolarization_factor})

    def _initialize_densitymatrix(self):
        """
            Initialize the density matrix for simulation.
            Default: All Zero State [((I+sigma(3))/2)**num_qubits]
            Custom:  max_mixed - Maximally Mixed State [(I/2)**num_qubits]
                     uniform_superpos - Uniform Superposition State [((I+sigma(1))/2)**num_qubits]
                     thermal_state - Thermalized State [([[1-p, 0],[0, p]])**num_qubits]
                     binary string - Specified sequence of Zero and One qubit states
                     stored density matrix - Initialize to a specified density matrix
            ** -> Tensor product.
       """

        if self._initial_densitymatrix is None:
            if self._custom_densitymatrix is None:
                self._densitymatrix = np.array([1,0,0,1], dtype=float)
                for i in range(self._number_of_qubits-1):
                    self._densitymatrix = np.kron([1,0,0,1],self._densitymatrix)
            elif self._custom_densitymatrix == 'max_mixed':
                self._densitymatrix = np.array([1,0,0,0], dtype=float)
                for i in range(self._number_of_qubits-1):
                    self._densitymatrix = np.kron([1,0,0,0], self._densitymatrix)
            elif self._custom_densitymatrix == 'uniform_superpos':
                self._densitymatrix = np.array([1,1,0,0], dtype=float)
                for i in range(self._number_of_qubits-1):
                    self._densitymatrix = np.kron([1,1,0,0], self._densitymatrix)
            elif self._custom_densitymatrix == 'thermal_state':
                tf = 1-2*self._thermal_factor
                self._densitymatrix = np.array([1,0,0,tf], dtype=float)
                for i in range(self._number_of_qubits-1):
                    self._densitymatrix = np.kron([1,0,0,tf], self._densitymatrix)
            else:
                raise BasicAerError('_custom_densitymatrix value is invalid')
            # Normalize the density matrix
            self._densitymatrix *= 0.5**(self._number_of_qubits)


        else:
            # Binary string is encoded in the self._initial_densitymatrix
            if self._custom_densitymatrix == 'binary_string':
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
                # Normalize the density matrix
                self._densitymatrix *= 0.5**(self._number_of_qubits)

            # Stored density matrix is encoded in file 'stored_density_matrix.npy' 
            elif self._custom_densitymatrix == 'stored_density_matrix':
                try:
                    self._densitymatrix = np.load('stored_density_matrix.npy')
                    if len(self._densitymatrix) != 4**self._number_of_qubits:
                        raise BasicAerError('Wrong input stored density matrix')
                except FileNotFoundError:
                    print('Stored Coefficient File does not exist')
            else:
                raise BasicAerError('_custom_densitymatrix value is invalid')
        
       # Reshape to rank-N tensor
        self._densitymatrix = np.reshape(self._densitymatrix,
                                       self._number_of_qubits * [4])

    # def _compute_densitymatrix1(self, vec):
    #     '''
    #         Generates density matrix from a given coefficient matrix
    #     '''

    #     p_0 = np.array([[1, 0], [0, 1]], dtype=complex)
    #     p_1 = np.array([[0, 1], [1, 0]], dtype=complex)
    #     p_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    #     p_3 = np.array([[1, 0], [0, -1]], dtype=complex)
    #     pauli_basis = [p_0, p_1, p_2, p_3]
    #     den_creat = [x for x in itertools.product(
    #         [0, 1, 2, 3], repeat=self._number_of_qubits)]
    #     densitymatrix = np.zeros((2**self._number_of_qubits, 2**self._number_of_qubits), dtype=complex)
     
    #     for i in range(len(den_creat)):
    #         creat = den_creat[i]
    #         op = pauli_basis[creat[0]]
    #         for idx in range(1, len(creat)):
    #             op = np.kron(op, pauli_basis[creat[idx]])
    #         densitymatrix += op*vec[i]
    #         op = None

    #     return densitymatrix

    def _compute_densitymatrix(self, dmpauli):
        '''
            Converts the density matrix from the Pauli basis to the standard matrix basis.
            rho = sum_{i,j,...=0,1,2,3} a_{i,j...} sigma_i x sigma_j x ...    [Pauli basis]
                = sum_{mu,nu,...=00,01,10,11} b_{mu,nu,...} e_mu x e_nu x ... [matrix basis]
            a_{i,j,...}   : 4**n real components
            b_{mu,nu,...} : 4**n complex components   
            Both the bases are orthogonal, and nonzero overlaps of the basis vectors are:
                i={0,3} <-> mu={00,11} and i={1,2} <-> mu={01,10}
            Matching components using orthogonal projections gives:
            b_{mu,nu,...} = sum_{i,j,...} a_{i,j,...} <e_mu,sigma_i> x <e_nu,sigma_j> x ...
            Only 2**n terms on the r.h.s. contribute with nonzero overlaps.
            The sum on the r.h.s. is evaluated in n steps, converting one basis index at a time,
            i.e. first i is converted to mu, then j is converted to nu, etc.

            Arg:
            dmpauli (float) : 4**n real components in the Pauli basis for n qubits

            Return:
            densitymatrix (complex) : 2**n x 2**n components in the matrix basis
        '''

        densitymatrix = np.zeros((2**self._number_of_qubits, 2**self._number_of_qubits), dtype=complex)

        dmcomplex = dmpauli.astype(complex)
        dot_prod = np.array([[1, 0, 0, 1],
                             [0, 1, complex(0, -1), 0],
                             [0, 1, complex(0, 1), 0],
                             [1, 0, 0, -1]]
                            )

        nonzero_overlaps = [(0, 3), (1, 2), (1, 2), (0, 3)]
        binary_index_value = [2**(self._number_of_qubits-i-1) for i in range(self._number_of_qubits)]
        ind_mat = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        for qubit in range(self._number_of_qubits):
            densitymatrix_b = np.zeros((4**(qubit), 4, 4**(self._number_of_qubits-qubit-1)), dtype=complex)
            dmcomplex = np.reshape(dmcomplex, (4**(qubit), 4, 4**(self._number_of_qubits-qubit-1)))

            for idxb in range(4):

                ind_i = nonzero_overlaps[idxb]

                for idxa in ind_i:
                    total_overlap = dot_prod[tuple((idxb,idxa))]
                    densitymatrix_b[:,idxb,:] += total_overlap*dmcomplex[:,idxa,:]

            dmcomplex = densitymatrix_b

        densitymatrix_b = np.reshape(densitymatrix_b, [4]*self._number_of_qubits)
        ind_mu = [x for x in itertools.product([0, 1, 2, 3], repeat=self._number_of_qubits)]

        for idxb in ind_mu:
            index_list = [ind_mat[i] for i in idxb]
            final_index = tuple(sum([binary_index_value[i]*index_list[i] for i in range(self._number_of_qubits)]))
            densitymatrix[final_index] = densitymatrix_b[idxb]



        np.savetxt("a.txt", np.asarray(
            np.round(densitymatrix, 4)), fmt='%1.3f', newline="\n")


        return densitymatrix

    def _get_densitymatrix(self):
        """Return the current densitymatrix in JSON Result spec format"""

        if self._get_den_mat:
            densitymatrix = self._compute_densitymatrix(self._densitymatrix)
            vec = np.reshape(self._densitymatrix, 4 ** self._number_of_qubits)
            vec[abs(vec) < self._chop_threshold] = 0.0
            return vec, densitymatrix
        else:
            densitymatrix = None
            vec = np.reshape(self._densitymatrix, 4 ** self._number_of_qubits)
            vec[abs(vec) < self._chop_threshold] = 0.0
            return vec

    def _unit_vector_normalisation(self, n):
        """ Checks if the given direction vector for measurement in N basis is valid or not.
         If not, it is normalised to be a unit vector.

        Arg:
            n (list): measurement direction
        """
        norm = np.linalg.norm(n)
        if norm == 1:
            pass
        else:
            n = n/norm
            logger.warning('Given direction for the measurement was not normalised. It has been normalised to be unit vector!!')
        return n

    def _validate_measure(self, insts):
        """ Determines whether ensemble measurement is needed to be done for the experiment.
            The instruction sequence is repartitioned in case of Bell basis measurement. 

        Args:
            experiment (QobjExperiment): a qobj experiment.
        """
        validated_inst = []
        measure_flag = False
        self._sample_measure = True
        set_flag = False

        for part in insts:
            if not part:
                continue
            if part[0].name != 'measure':
                if part[0].name != 'barrier':
                    self._sample_measure = False
                validated_inst.append(part)
                continue
            else:
                measure_flag = True
                bf_id = 0
                temppart = []
                for idx in range(len(part)):
                        
                    para = getattr(part[idx], 'params', None)

                    if para is None:
                        set_flag = True
                        setattr(part[idx], 'params', ['Z'])
                
                    part[idx].params[0] = str(para[0])
                    if str(para[0]) == 'Bell':
                        part[idx].params[1] = str(para[1])
                        if part[bf_id:idx]:
                            validated_inst.append(part[bf_id:idx])
                        validated_inst.append([part[idx]])
                        bf_id = idx+1
                    else:
                        temppart.append(part[idx])

                if temppart:
                    validated_inst.append(temppart)

        if set_flag:
            logger.warning('No basis choice provided for measurement. Default value set to Pauli Z [Computational Basis]')
        
        return validated_inst, len(validated_inst)


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

        return result

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

        # Validate the dimension of initial densitymatrix if set
        # self._validate_initial_densitymatrix()

        # Add data
        data = {}

        self._initialize_densitymatrix()
        self._initialize_errors()
        # Initialize classical memory to all 0
        self._classical_memory = 0
        self._classical_register = 0
        #print("MERGING U1 and U3 GATES\n")
        experiment.instructions = single_gate_merge(experiment.instructions,
                                                    self._number_of_qubits)
        partitioned_instructions, levels = partition(experiment.instructions, 
                                                self._number_of_qubits)
        # if self.SHOW_PARTITION:
        #     print("\nINITIAL PARTITION")
        #     self.describe_partition(partitioned_instructions)
        #partitioned_instructions, levels =  self._validate_measure(partitioned_instructions)
        if self.SHOW_PARTITION:
            print("\nPARTITIONED CIRCUIT")
            self.describe_partition(partitioned_instructions)
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
 
                if operation.name in ('u1','u3'):
                    params = getattr(operation, 'params', None)
                    gate = single_gate_dm_matrix(operation.name, params)
                    qubit = operation.qubits[0]
                    self._add_unitary_single(gate, qubit)
                # Check if C-NOT gate
                elif operation.name == 'cx':
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
                    params = operation.params
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(
                        operation, 'register') else None

                    sngl_measure = False
                    part_measure = False
                    exp_measure = False
                    ensm_measure = False

                    len_pi = len(partitioned_instructions[clock])

                    
                    if str(params[0]) == 'Ensemble':
                        ensm_measure = True
                    elif str(params[0]) == 'Expect':
                        exp_measure = True
                    elif len_pi == 1:
                        sngl_measure = True
                    else:
                        part_measure = True

                    if len(params) == 1:
                        params.append(None)

                    cregbit = operation.register[0] if hasattr(operation, 'register') else None 
                    
                    if sngl_measure:
                        if str(params[0]) == 'X':
                            self._add_qasm_measure_X(
                                qubit, cmembit, cregbit, self._error_params['measurement'])
                        elif str(params[0]) == 'Y':
                            self._add_qasm_measure_Y(
                                qubit, cmembit, cregbit, self._error_params['measurement'])
                        elif str(params[0]) == 'N':
                            params[1] = self._unit_vector_normalisation(params[1])
                            self._add_qasm_measure_N(
                                qubit, cmembit, cregbit, params[1], self._error_params['measurement'])
                        elif str(params[0]) == 'Bell':
                            bell_probabilities, reduced_bell_densitymatrix = self._add_bell_basis_measure(int(str(params[1])[0]), int(str(params[1])[1]), err_param = self._error_params['measurement_bell'])
                            data[f'bell_probabilities{str(params[1])[0]}{str(params[1])[1]}'] = bell_probabilities
                            data[f'reduced_bell_densitymatrix{str(params[1])[0]}{str(params[1])[1]}'] = reduced_bell_densitymatrix
                        else:
                            self._add_qasm_measure_Z(
                                qubit,cmembit,cregbit,self._error_params['measurement'])       
                        partitioned_instructions[clock].remove(operation)

                    elif part_measure:
                        qubit_mes_list = [x.qubits[0] for x in partitioned_instructions[clock]]
                        cmem_mes_list = [x.memory[0] for x in partitioned_instructions[clock]]
                        creg_mes_list = []

                        for x in partitioned_instructions[clock]:
                            cregbit = x.register[0] if hasattr(
                                x, 'register') else None
                            creg_mes_list.append(cregbit)
                        if str(params[1]) == 'N':
                            add_param = params[1][1]
                            basis = str(params[1][0])
                        else:
                            add_param = None
                            basis = str(params[1])

                        data['partial_probability'], max_str, max_prob = self._add_partial_measure(
                            qubit_mes_list, cmem_mes_list, creg_mes_list,
                            self._error_params['measurement'], basis, add_param)
                        break

                    elif exp_measure:
                        data['Pauli_string_expectation'] = self._pauli_string_expectation(str(params[1]), self._error_params['measurement'])
                        break

                    elif ensm_measure:
                        if len(str(params[1])) == 1:
                            add_param = None
                            basis = str(params[1])
                        elif params[1][0]== 'N':
                            add_param = params[1][1]
                            basis = str(params[1][0])
                        prob, max_str, max_prob = self._add_ensemble_measure(basis, add_param, self._error_params['measurement'])
                        self._plot_ensemble_measure(prob,params[0]) 
                        data['ensemble_probability'] = prob       
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
        
        if self.SHOW_FINAL_STATE:
            if self._get_den_mat:
                data['coeffmatrix'], data['densitymatrix'] = self._get_densitymatrix()
            else:
                data['coeffmatrix'] = self._get_densitymatrix()
            if self.fidelity != None:
                data['fidelity'] = self.fidelity

        end_runtime = time.time()
        # self.result_dict = {'name': experiment.header.name,
        #         'number_of_clock_cycles': levels,
        #         'data': data,
        #         'status': 'DONE',
        #         'success': True,
        #         'processing_time_taken': -start_processing+end_processing,
        #         'running_time_taken': -start_runtime+end_runtime,
        #         'header': experiment.header.as_dict()}

        return {'name': experiment.header.name,
                'number_of_clock_cycles': levels,
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
            if 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s", '
                               'classical register will remain all zeros.', name)

    def store_density_matrix(self):
        """ Store the density matrix in the specified file.
        """
        densitymatrix = np.reshape(self._densitymatrix, 4**self._number_of_qubits)
        np.save("stored_coefficients", densitymatrix)

    def describe_partition(self, partition):
        """ Partitioned instructions are printed as a table (with all parameters) for visualization.
        """
        
        for current in range(len(partition)):
            print("\nPartition ", current)
            current_partition = partition[current]
            for instruction in range(len(current_partition)):
                inst = current_partition[instruction]
                name, qubit = inst.name, inst.qubits
                if name == 'u3':
                    param = [round(x, 6) for x in inst.params]
                    print("U3", "   qubit", qubit, "    ", param)
                if name == 'u1':
                    param = [round(x, 6) for x in inst.params]
                    print("U1", "   qubit", qubit, "    ", param)
                elif name == 'cx':
                    print("C-NOT", "   qubit", qubit)
                if name == 'measure':
                    parameter = getattr(inst,'params',None)
                    if parameter is None:
                        setattr(inst,'params',['Z'])
                    param = inst.params
                    if param[0] == 'Bell':
                        name = 'Bell Measure'
                        qubit = [int(x) for x in param[1]]
                        param = param[0:0]
                        print(name, "   qubit", qubit)
                    else:
                        print(name, "   qubit", qubit, "    ", param)


    def state_overlap(self, density_matrix_1, density_matrix_2):
        """ Calculate the state overlap:  Tr(density_matrix_1,density_matrix_2)
        Args   : density_matrix_1 (4**n) and density_matrix_2 (4**n) in Pauli basis
        Return : Value of overlap between density_matrix_1 and density_matrix_2
        """

        return np.dot(density_matrix_1, density_matrix_2) * 2**self._number_of_qubits
