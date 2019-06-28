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

"""Contains functions used by the basic aer simulators.

"""

from string import ascii_uppercase, ascii_lowercase
import numpy as np
from copy import deepcopy
from qiskit.exceptions import QiskitError


def single_gate_params(gate, params=None):
    """Apply a single qubit gate to the qubit.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        tuple: a tuple of U gate parameters (theta, phi, lam)
    Raises:
        QiskitError: if the gate name is not valid
    """
    if gate in ('U', 'u3'):
        return params[0], params[1], params[2]
    elif gate == 'u2':
        return np.pi / 2, params[0], params[1]
    elif gate == 'u1':
        return 0, 0, params[0]
    elif gate == 'id':
        return 0, 0, 0
    else:
        raise QiskitError('Gate is not among the valid types: %s' % gate)


def single_gate_matrix(gate, params=None):
    """Get the matrix for a single qubit.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        array: A numpy array representing the matrix
    """

    # Converting sym to floats improves the performance of the simulator 10x.
    # This a is a probable a FIXME since it might show bugs in the simulator.
    (theta, phi, lam) = map(float, single_gate_params(gate, params)) 

    return np.array([[np.cos(theta / 2),
                      -np.exp(1j * lam) * np.sin(theta / 2)],
                     [np.exp(1j * phi) * np.sin(theta / 2),
                      np.exp(1j * phi + 1j * lam) * np.cos(theta / 2)]])


def single_gate_dm_matrix(gate, params=None):
    """Get the matrix for a single qubit in density matrix formalism.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        array: A numpy array representing the matrix
    """

    # Converting sym to floats improves the performance of the simulator 10x.
    # This a is a probable a FIXME since it might show bugs in the simulator.
    
    decomp_gate = []
    param = list(map(float, single_gate_params(gate, params)))

    # Decomposition is Rz(Phi)Ry(Theta)Rz(Lamb) 
    # Order of application goes Right to Left

    if param[2]:    # Lamb
        decomp_gate.append(['rz', param[2]])
    if param[0]:    # Theta
        decomp_gate.append(['ry', param[0]])
    if param[1]:    # Phi
        decomp_gate.append(['rz', param[1]])

    return decomp_gate
    
    '''
    return np.array([[1,0,0,0],
                    [0,np.sin(lam)*np.sin(phi)+ np.cos(theta)*np.cos(phi)*np.cos(lam),np.cos(theta)*np.cos(phi)*np.sin(lam)- np.cos(lam)*np.sin(phi),np.sin(theta)*np.cos(phi)],
                    [0,np.cos(theta)*np.sin(phi)*np.cos(lam)- np.sin(lam)*np.cos(phi),np.cos(phi)*np.cos(lam) + np.cos(theta)*np.sin(phi)*np.sin(lam), np.sin(theta)*np.sin(phi)],
                    [0,-np.cos(lam)*np.sin(theta), np.sin(theta)*np.sin(lam), np.cos(theta)]
                    ])
    '''
def rt_gate_dm_matrix(gate, param, err_param, state, q, num_qubits):

    """   
    The error model adds a fluctuation to the angle param, with mean err_param[1] and variance parametrized in terms of err_param[0].
    Args:
        err_param[1] is the mean error in the angle param.
        err_param[0] is the reduction in the radius after averaging over fluctuations in the angle param.
    """
    c = err_param[0] * np.cos(param + err_param[1])
    s = err_param[0] * np.sin(param + err_param[1])

    if gate == 'rz':
        k = [1, 2]
    elif gate  == 'ry':
        k = [3, 1]
    elif gate == 'rx':
        k = [2, 3]
    else:
        raise QiskitError('Gate is not among the valid decomposition types: %s' % gate)   
    
    #print(gate, state, 4**(num_qubits-q-1), 4**q)

    for j in range(4**(num_qubits-q-1)):
        for i in range(4**(q)):

            temp1 = state[i, k[0], j] 
            temp2 = state[i, k[1], j]
            
            state[i, k[0], j] = c*temp1 - s*temp2
            state[i, k[1], j] = c*temp2 + s*temp1

    #print(state[0,0,0], state[0,1,0], state[0,2,0], state[0,3,0])
    return state

def U3_merge(theta, phi, lamb, tol):
    """Performs merge operation when both the gates are U3 by transforming the Y-Z decomposition of the gates to the Z-Y decomposition.
        Args:
            theta   (float) :  Ry(theta1) 
            phi     (float) :  Rz(theta2)
            lamb    (float) :  Rz(theta3)
            tol     (float) :  Tolerance limit
        Return
            [β, α, γ] (list, type:float ):  {Rz(α) , Ry(β) , Rz(γ)}
    
    Matrix Form
    {
        E^(-((I xi)/2))*cos[theta1/2]*cos[theta2/2] - 
        E^((I xi)/2)*sin[theta1/2]*sim[theta2/2]	(1,1)
       
       -E^(((I xi)/2))*cos[theta2/2]*sin[theta1/2] - 
        E^(-((I xi)/2))*cos[theta1/2]*sin[theta2/2]  (1,2)
       
        E^(-((I xi)/2))*cos[theta2/2]*sin[theta1/2] + 
        E^((I xi)/2)*cos[theta1/2]*sin[theta2/2]	(2,1)
       
        E^((I xi)/2)*cos[theta1/2]*cos[theta2/2] - 
        E^(-((I xi)/2))*sin[theta1/2]*sin[theta2/2]  (2,2)
    }
    
    """
    xi = theta
    theta1 = phi
    theta2 = lamb
    atol = 1e-8
    # for storing all the solutions
    solutions = []

    if np.abs(np.cos(xi)) < tol:
        return [theta2-theta1, xi, 0]
    elif np.abs(np.sin(theta1+theta2)) < tol:
        phi_minus_lambda = [np.pi/2, 3*np.pi/2, np.pi/2, 3*np.pi/2]
        stheta_1 =  np.arcsin(np.sin(xi) * np.sin(-theta1 + theta2))
        stheta_2 = -stheta_1
        stheta_3 =  np.pi - stheta_1
        stheta_4 =  np.pi - stheta_2
        stheta = [stheta_1, stheta_2, stheta_3, stheta_4]
        phi_plus_lambda = list(map(lambda x:
                                   np.arccos(np.cos(theta1 + theta2) *
                                            np.cos(xi) / np.cos(x)),
                                            stheta))
        sphi = [(term[0] + term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        slam = [(term[0] - term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        solutions = list(zip(stheta, sphi, slam))
    elif np.abs(np.cos(theta1+theta2)) < tol:
        phi_plus_lambda = [np.pi/2, 3*np.pi/2, np.pi/2, 3*np.pi/2]
        stheta_1 =  np.arccos(np.sin(xi) * np.cos(theta1 - theta2))
        stheta_2 =  stheta_1
        stheta_3 = -stheta_1
        stheta_4 = -stheta_2
        stheta = [stheta_1, stheta_2, stheta_3, stheta_4]
        phi_minus_lambda = list(map(lambda x:
                                    np.arccos(np.sin(theta1 + theta2) *
                                             np.cos(xi) / np.sin(x)),
                                             stheta))
        sphi = [(term[0] + term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        slam = [(term[0] - term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        solutions = list(zip(stheta, sphi, slam))
    else:
        sinxi = np.sin(xi)
        cosxi = np.cos(xi)
        costheta12 = np.cos(theta1 + theta2)
        phi_plus_lambda = np.arctan(sinxi * np.cos(theta1 - theta2) /
                                     (cosxi * costheta12))
        phi_minus_lambda = np.arctan(sinxi * np.sin(-theta1 +
                                                        theta2) /
                                      (cosxi * np.sin(theta1 +
                                                         theta2)))
        sphi = (phi_plus_lambda + phi_minus_lambda) / 2
        slam = (phi_plus_lambda - phi_minus_lambda) / 2
        cossphislam = np.cos(sphi + slam)
        arccos = np.arccos(cosxi * costheta12 / cossphislam)
        solutions.append((arccos, sphi, slam))
        solutions.append((arccos, sphi + np.pi / 2, slam + np.pi / 2))
        solutions.append((arccos, sphi + np.pi / 2, slam - np.pi / 2))
        solutions.append((arccos, sphi + np.pi, slam))
    
    # Choose the first solution with desired accuracy
    for ans in solutions:

        sinxi = np.sin(xi)
        cosxi = np.cos(xi)
        sintheta = np.sin(ans[0])
        costheta = np.cos(ans[0])
        cost1 = ans[1] + ans[2]
        cost2 = ans[1] - ans[2]
        sint1 = theta1 + theta2 
        sint2 = theta1 - theta2

        delta1 = np.abs(np.cos(cost1) * costheta - cosxi * np.cos(sint1))
        if delta1 > tol:
            continue
        
        delta2 = np.abs(np.sin(cost1) * costheta - sinxi * np.cos(sint2))
        if delta2 > tol:
            continueX

        delta3 = np.abs(np.cos(cost2) * sintheta - cosxi * np.sin(sint1))
        if delta3 > tol:
            continue

        delta4 = np.abs(np.sin(cost2) * sintheta - sinxi * np.sin(-sint2))
        if delta4 > tol:
            continue

        return ans

def mergeU(gate1, gate2):
    """
    Merges Unitary Gates acting consecutively on a same qubit within in partions
    Args:
        Gate1   ([Inst, index])
        Gate2   ([Inst, index])
    Return:
        Gate    ([Inst, index])
    """

    temp = None
    # To preserve the sequencing we choose the smaller index while merging.
    
    if gate1[1] < gate2[1]:
        temp = deepcopy(gate1)
    else:
        temp = deepcopy(gate2)

    if gate1[0].name == 'u1' and gate2[0].name == 'u1':
        temp[0].params[0] = gate1[0].params[0] + gate2[0].params[0] 
    elif gate1[0].name == 'u1' or gate2[0].name == 'u1':   
        # If first gate is U1
        if temp[0].name == 'u1':
            temp[0].name = 'u3'
            for i in range(2):
                temp[0].params.append(0)

        if gate1[0].name == 'u1' and gate2[0].name == 'u3':
            temp[0].params[0] = gate2[0].params[0]
            temp[0].params[1] = gate2[0].params[1] 
            temp[0].params[2] = gate2[0].params[2] + gate1[0].params[0]
        elif gate1[0].name == 'u3' and gate2[0].name == 'u1':
            temp[0].params[0] = gate1[0].params[0]
            temp[0].params[1] = gate1[0].params[1] + gate2[0].params[0]
            temp[0].params[2] = gate1[0].params[2]
    elif gate1[0].name == 'u3' and gate2[0].name == 'u3':
        atol = 1e-8
        #phi = float(gate1[0].params[0]) * 0.5
        #theta = float(gate1[0].params[2] + gate2[0].params[1]) * 0.5
        #lamb = float(gate2[0].params[0]) * 0.5
        theta = float(gate2[0].params[2] + gate1[0].params[1]) * 0.5
        phi = float(gate2[0].params[0]) * 0.5
        lamb = float(gate1[0].params[0]) * 0.5

        res = U3_merge(theta, phi, lamb, atol)        
        
        temp[0].params[0] = 2*res[0]
        temp[0].params[1] = gate2[0].params[1] + 2*res[1]
        temp[0].params[2] = gate1[0].params[2] + 2*res[2]
    else:
        raise QiskitError('Encountered unrecognized instructions: %s, %s' % gate1[0].name, gate2[0].name)
    return temp

def merge_gates(inst):
    """
    To merge unitary gate calls the helper function iteratively on the pair of consecutive qubits.
    Args:
        Inst [[inst, index]]:   Instructions to be merged
    Return
        Inst [Qasm Inst]:       Merged List
    """

    if len(inst) < 2:
        return inst[0][0]
    else:
        temp = mergeU(inst[0], inst[1])
        for idx in range(2, len(inst)):
            param = []
            temp = mergeU(temp, inst[idx])
        return temp[0]


def single_gate_merge(inst, num_qubits):
    """
        Merges the single gates applied consecutively on a circuit
        Args:
            inst [QASM Inst]:   List of instructions (original)
        Return
            inst [QASM Inst]:   List of instructions with merging
    """

    single_gt = [[] for x in range(num_qubits)]
    inst_merged = []

    for ind, op in enumerate(inst):
        # To preserve the sequencing of the instruments
        opx = [op, ind]
        # Check if non-unitary gate marks a partition
        if opx[0].name in ('CX', 'cx', 'measure', 'bfunc', 'reset'):
            for idx, sg in enumerate(single_gt):
                if sg:
                    inst_merged.append(merge_gates(sg))
                    single_gt[idx] = []
            inst_merged.append(opx[0])
        # Unitary gates are appended to their respective qubits
        elif opx[0].name in ('U', 'u1', 'u2', 'u3'):
            if opx[0].name == 'u2':
                opx[0].name = 'u3'
                opx[0].params.insert(0, np.pi/2)
            single_gt[op.qubits[0]].append(opx)
        else:
            raise QiskitError('Encountered unrecognized instruction: %s' % op)

    # To merge the final left out gates 
    for gts in single_gt:
        if gts:
            inst_merged.append(merge_gates(gts))

    return inst_merged


def cx_gate_dm_matrix(state, q_1, q_2, num_qubits):
    """Apply C-NOT gate in density matrix formalism.

        Args:
        state - density matrix
        q_1 (int): Control qubit 
        q_2 (int): Target qubit"""

    # Remark - ordering of qubits (MSB right, LSB left)
    print(q_1, q_2)

    if (q_1 == q_2) or (q_1 >= num_qubits) or (q_2 >= num_qubits):
        raise QiskitError('Qubit Labels out of bound in CX Gate')
    elif q_2 > q_1:
        # Reshape Density Matrix
        state = np.reshape(state, (4**(num_qubits-q_2-1),
                                   4, 4**(q_2-q_1-1), 4, 4**q_1))
        temp_dm = state.copy()
        #print(state)
        # Update Density Matrix
        for i in range(4**(num_qubits-q_2-1)):
            for j in range(4**(q_2-q_1-1)):
                for k in range(4**q_1):
                    #print(i,j,k)
                    state[i, 0, j, 2, k] = temp_dm[i, 3, j, 2, k]
                    state[i, 0, j, 3, k] = temp_dm[i, 3, j, 3, k]
                    state[i, 1, j, 0, k] = temp_dm[i, 1, j, 1, k]
                    state[i, 1, j, 1, k] = temp_dm[i, 1, j, 0, k]
                    state[i, 1, j, 2, k] = temp_dm[i, 2, j, 3, k]
                    state[i, 1, j, 3, k] = -temp_dm[i, 2, j, 2, k]
                    state[i, 2, j, 0, k] = temp_dm[i, 2, j, 1, k]
                    state[i, 2, j, 1, k] = temp_dm[i, 2, j, 0, k]
                    state[i, 2, j, 2, k] = -temp_dm[i, 1, j, 3, k]
                    state[i, 2, j, 3, k] = temp_dm[i, 1, j, 2, k]
                    state[i, 3, j, 2, k] = temp_dm[i, 0, j, 2, k]
                    state[i, 3, j, 3, k] = temp_dm[i, 0, j, 3, k]
    else:
        # Reshape Density Matrix
        state = np.reshape(state, (4**(num_qubits-q_1-1),
                                   4, 4**(q_1-q_2-1), 4, 4**q_2))
        temp_dm = state.copy()
        #print(state)
        # Update Density Matrix
        for i in range(4**(num_qubits-q_1-1)):
            for j in range(4**(q_1-q_2-1)):
                for k in range(4**q_2):
                    #print(i, j, k)
                    #print(temp_dm[i, 2, j, 2, k], temp_dm[i, 3, j, 1, k])
                    state[i, 2, j, 0, k] = temp_dm[i, 2, j, 3, k]
                    state[i, 3, j, 0, k] = temp_dm[i, 3, j, 3, k]
                    state[i, 0, j, 1, k] = temp_dm[i, 1, j, 1, k]
                    state[i, 1, j, 1, k] = temp_dm[i, 0, j, 1, k]
                    state[i, 2, j, 1, k] = temp_dm[i, 3, j, 2, k]
                    state[i, 3, j, 1, k] = -temp_dm[i, 2, j, 2, k]
                    state[i, 0, j, 2, k] = temp_dm[i, 1, j, 2, k]
                    state[i, 1, j, 2, k] = temp_dm[i, 0, j, 2, k]
                    state[i, 2, j, 2, k] = -temp_dm[i, 3, j, 1, k]
                    state[i, 3, j, 2, k] = temp_dm[i, 2, j, 1, k]
                    state[i, 2, j, 3, k] = temp_dm[i, 2, j, 0, k]
                    state[i, 3, j, 3, k] = temp_dm[i, 3, j, 0, k]
        print(state)
    return state

def cx_gate_matrix():
    """Get the matrix for a controlled-NOT gate."""
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]], dtype=complex)


def einsum_matmul_index(gate_indices, number_of_qubits):
    """Return the index string for Numpy.eignsum matrix-matrix multiplication.

    The returned indices are to perform a matrix multiplication A.B where
    the matrix A is an M-qubit matrix, matrix B is an N-qubit matrix, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on B.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.
    """

    mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices,
                                                                    number_of_qubits)

    # Right indices for the N-qubit input and output tensor
    tens_r = ascii_uppercase[:number_of_qubits]

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return "{mat_l}{mat_r}, ".format(mat_l=mat_l, mat_r=mat_r) + \
           "{tens_lin}{tens_r}->{tens_lout}{tens_r}".format(tens_lin=tens_lin,
                                                            tens_lout=tens_lout,
                                                            tens_r=tens_r)


def einsum_vecmul_index(gate_indices, number_of_qubits):
    """Return the index string for Numpy.eignsum matrix-vector multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, vector v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                  to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.
    """

    mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices,
                                                                    number_of_qubits)

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return "{mat_l}{mat_r}, ".format(mat_l=mat_l, mat_r=mat_r) + \
           "{tens_lin}->{tens_lout}".format(tens_lin=tens_lin,
                                            tens_lout=tens_lout)


def _einsum_matmul_index_helper(gate_indices, number_of_qubits):
    """Return the index string for Numpy.eignsum matrix multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, matrix v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        tuple: (mat_left, mat_right, tens_in, tens_out) of index strings for
        that may be combined into a Numpy.einsum function string.

    Raises:
        QiskitError: if the total number of qubits plus the number of
        contracted indices is greater than 26.
    """

    # Since we use ASCII alphabet for einsum index labels we are limited
    # to 26 total free left (lowercase) and 26 right (uppercase) indexes.
    # The rank of the contracted tensor reduces this as we need to use that
    # many characters for the contracted indices
    if len(gate_indices) + number_of_qubits > 26:
        raise QiskitError("Total number of free indexes limited to 26")

    # Indices for N-qubit input tensor
    tens_in = ascii_lowercase[:number_of_qubits]

    # Indices for the N-qubit output tensor
    tens_out = list(tens_in)

    # Left and right indices for the M-qubit multiplying tensor
    mat_left = ""
    mat_right = ""

    # Update left indices for mat and output
    for pos, idx in enumerate(reversed(gate_indices)):
        mat_left += ascii_lowercase[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = ascii_lowercase[-1 - pos]
    tens_out = "".join(tens_out)

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return mat_left, mat_right, tens_in, tens_out

def is_single(gate):
    # Checks if gate is single
    return True if gate.name in ['u3','u1'] else False

def is_cx(gate):
    # Checks if gate is CX
    return True if gate.name in ['CX','cx'] else False

def is_measure(gate): #TODO Seperate Them
    # Checks if gate is measure
    return True if gate.name == 'measure' else False

def is_reset(gate):
    # Checks if gate is reset
    return True if gate.name == 'reset' else False

def is_measure_dummy(gate):
    # Checks if gate is dummy measure
    return True if gate.name == 'dummy_measure' else False

def is_reset_dummy(gate):
    # Checks if gate is dummy reset
    return True if gate.name == 'dummy_reset' else False

def qubit_stack(i_set, num_qubits):
    instruction_set = [[] for _ in range(num_qubits)]
    for instruction in i_set:
        #print('Inst: ', instruction)
        if not is_measure(instruction) and not is_reset(instruction):
            for qubit in instruction.qubits:
                instruction_set[qubit].append(instruction)
        elif is_measure(instruction):
            if not is_measure_dummy(instruction_set[instruction.qubits[0]][-1]):
                instruction_set[instruction.qubits[0]].append(instruction)
                dummy = deepcopy(instruction)
                dummy.name = 'dummy_measure'
                dummy.qubits[0] = -1
                for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                    instruction_set[qubit].append(dummy)
            else:
                if instruction_set[instruction.qubits[0]]: # Checks if the first instruction is measure
                    instruction_set[instruction.qubits[0]][-1] = instruction
                else:
                    instruction_set[instruction.qubits[0]].append(instruction)
        elif is_reset(instruction):
            if not is_reset_dummy(instruction_set[instruction.qubits[0]][-1]):
                instruction_set[instruction.qubits[0]].append(instruction)
                dummy = deepcopy(instruction)
                dummy.name = 'dummy_reset'
                dummy.qubits[0] = -1
                for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                    instruction_set[qubit].append(dummy)
            else:
                if instruction_set[instruction.qubits[0]]:
                    instruction_set[instruction.qubits[0]][-1] = instruction
                else:
                    instruction_set[instruction.qubits[0]].append(instruction)

    stack_depth = max([len(stack) for stack in instruction_set])
    return instruction_set, stack_depth

def partition(i_set,num_qubits):
    i_stack, depth = qubit_stack(i_set, num_qubits)
    level, sequence = 0, [[] for _ in range(depth)]
    #print('Stacked: ', i_stack)
    #for idx, st in enumerate(i_stack):
    #    print('Stack for Qubit: ', idx, st)
    while i_set:
        # Qubits included in the partition
        qubit_included = [] 
        for qubit in range(num_qubits):
            gate = i_stack[qubit][0]

            # Check for dummy gate
            if is_measure_dummy(gate) or is_reset_dummy(gate):
                continue
            # Check for single gate
            elif is_single(gate):
                sequence[level].append(gate)
                qubit_included.append(qubit)
                i_set.remove(gate)      # Remove from Set
                i_stack[qubit].pop(0)   # Remove from Stack
            # Check for CX gate
            elif is_cx(gate):
                #print(set(gate.qubits))
                #print(set(gate.qubits).difference(set([qubit])))
                second_qubit = list(set(gate.qubits).difference(set([qubit])))[0]
                buffer_gate = i_stack[second_qubit][0] 

                # Checks if gate already included in the partition
                if qubit in qubit_included or second_qubit in qubit_included:
                    continue
                
                # Check if CX is top in stacks of both of its indexes. 
                if gate == buffer_gate:
                    qubit_included.append(qubit)
                    qubit_included.append(second_qubit)
                    sequence[level].append(gate)
                    i_set.remove(gate)
                    i_stack[qubit].pop(0)
                    i_stack[second_qubit].pop(0)
                # If not then don't add it.
                else:
                    continue
            elif is_measure(gate):
                all_dummy = True
                for x in range(num_qubits):
                    # Intersection of both should be used 
                    if not is_measure(i_stack[x][0]) and not is_measure_dummy(i_stack[x][0]):
                        all_dummy = False
                        break 

                if all_dummy:
                    # Check if current level already has gates
                    if sequence[level]:
                        qubit_included = []
                        level += 1 # Increment the level

                    for x in range(num_qubits):
                        # Check if measure
                        if is_measure(i_stack[x][0]):
                            qubit_included.append(x)
                            sequence[level].append(i_stack[x][0])
                            i_set.remove(i_stack[x][0]) # Remove from Instruction list
                        i_stack[x].pop(0)
                    break # To restart the Qubit loop from 0
            elif is_reset(gate):
                all_dummy=True
                for x in range(num_qubits):
                    # Intersection of both should be used
                    if not is_reset(i_stack[x][0]) and not is_reset_dummy(i_stack[x][0]):
                        all_dummy=False
                        break

                if all_dummy:
                    # Check if current level already has gates
                    if sequence[level]:
                        qubit_included = []
                        level += 1  # Increment the level

                    for x in range(num_qubits):
                        # Check if measure
                        if is_reset(i_stack[x][0]):
                            qubit_included.append(x)
                            sequence[level].append(i_stack[x][0])
                            # Remove from Instruction list
                            i_set.remove(i_stack[x][0])
                        i_stack[x].pop(0)
                    break  # To restart the Qubit loop from 0
        
            # Check if the instruction list is empty
            if not i_set:
                break

        level += 1
    return sequence, level
