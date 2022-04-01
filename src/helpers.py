import numpy as np

def binary_string_to_spin_list(obj, reverse = False):
    # trafo '1001' <=> [-1,1,1,-1]
    # s_i = 1-2x_i

    new_obj = obj
    
    if not reverse and isinstance(obj, str):
        string_list = list(obj)
        new_obj = []
        for elem in string_list:
            assert elem in ('0', '1'), 'No valid bitstring'
            if elem == '1':
                new_obj.append(-1)
            else:
                new_obj.append(1)
    if reverse and (isinstance(obj, list) or isinstance(obj, np.ndarray)):
        new_obj = ''
        for elem in obj:
            assert elem in (1, -1), 'No valid spin list'
            if elem == 1:
                new_obj += '0'
            else:
                new_obj += '1'    
    return new_obj

def kronecker_product(op_string):
    """
    Given an op_string in Pauli decomposition, this function calculates the tensor product of 
    the matrices to give the whole matrix (if n=len(op_string) the matrix has shape (2**n, 2**n)).
    """
    paulix = np.array([[0,1],[1,0]])
    pauliy = np.array([[0, complex(0,-1)],[complex(0,1),0]])
    pauliz = np.array([[1,0],[0,-1]])
    identity = np.array([[1,0],[0,1]])

    op = op_string[0]
    if op == 'X':
        matrix = paulix
    if op == 'Y':
        matrix = pauliy
    if op == 'Z':
        matrix = pauliz
    if op == 'I':
        matrix = identity

    if len(op_string) == 1:
        return matrix
    else:
        return np.kron(matrix, kronecker_product(op_string[1:]))
    
def analytic_groundstate(hamilton):
    """
    Given a Hamiltonian in the form sum(coeff*op_string), calculate the groundstate energy, as 
    well as the corresponding eigenstates. 

    Args
        hamilton: Dict(String: Float)
            Dictionary containing the information about the Hamiltonian written in terms of 
            Pauli strings.
    Returns
        sol_dict: Dict(string: Any)

    """
    dimension = len(list(hamilton.keys())[0])
    matrix = np.zeros((2**dimension, 2**dimension), dtype = complex)
    for key, value in hamilton.items():
        matrix += value*kronecker_product(key)
    
    solutions = np.linalg.eigh(matrix)
    eigv_idx = np.where(solutions[1][:,0] != 0)[0]
    
    bin_list = []
    ampl_list = []
    for idx in eigv_idx:
        ampl = solutions[1][idx, 0]
        if np.abs(ampl) < 1e-10:
            continue
        ampl_list.append(solutions[1][idx,0])
        bin_state = bin(idx)[2:]
        if len(bin_state) != dimension:
            delta = dimension - len(bin_state)
            bin_state = delta*'0' + bin_state
        bin_list.append(bin_state)
    sol_dict = {}
    sol_dict['energy'] = solutions[0][0]
    sol_dict['states'] = bin_list
    sol_dict['amplitudes'] = ampl_list
    return sol_dict

def matrix_exp_val(state_amplitudes, op_string):
    """
    Given a state_amplitudes object and an op_string, the function calculates
    the matrix product <\psi|op_string|\psi> 
    """
    state_list = list(state_amplitudes.keys())
    assert len(state_list[0])==len(op_string), "states and operator have mismatching dimensions"
    state_list.sort()
    statevec = np.array([state_amplitudes[key] for key in state_list])
    statevec_dg = statevec.conjugate()
    
    op_matrix = kronecker_product(op_string)
    
    return statevec_dg.dot(op_matrix.dot(statevec))

