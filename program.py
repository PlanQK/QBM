from src import QBMBuilder, QBMData, ExpValClass

import numpy as np

def optimize_qbm_from_hamiltonian(hamiltonian,
                                  num_hidden_qubits=None,
                                  backend=None,
                                  print_results=False,
                                  **kwargs):
    """
    Function for calculating the groundstate energy and wave function for a given qubit Hamiltonian.
    Args:
        hamiltonian: Dict(String: Float)
            Dictionary containing Pauli-Strings as keys and their respective coefficients as
            values, e.g.: {'XIXZ': 0.32, ...}
        num_hidden_qubits: Int
            # qubits for amplitude manipulation. Normally, it should be O(n_visible_qubits), which 
            is determined by the Input Hamiltonian. If None, will be set to num_visible_qubits
        backend: qiskit Backend
            Backend used for QBM sampling. Right now, for the execution, only qasm_simulator should 
            was tested. Default is None, which results in using the local qasm_simulator.
        print_results: Bool
            If set to True, prints hamiltonian expectation values during the optimization. Default
            is False.
            
    Returns:
        result: Dict
            Dictionary with keys
                state_amplitudes: describes the ground state wave function
                hamiltonian_groundstate: minimum eigenvalue of qubit hamiltonian
                qbm_params: parameters for recreating the resulting state_amplitudes
                
    kwargs include: 
        eta_0, eta_f: default=1, for adaptive learning rates in qbm optimization
        max_iter: default=15, iterations for qbm optimization
        shots: default=5000, repetitions of the same qbm circuit for wave function sampling
        print_iter: default=False, print hamiltonian exp. values during optimization

    """

    pauli_string_len = set(len(key) for key in hamiltonian.keys())
    assert(len(pauli_string_len) == 1), "No uniform length of resulting paul-strings"
    num_vis = next(iter(pauli_string_len))
    num_hid = num_hidden_qubits if num_hidden_qubits is not None else num_vis

    qbm_data = QBMData(num_vis, num_hid)
    qbm_builder = QBMBuilder(qbm_data, backend)

    eta_0 = kwargs.get('eta_0', 1)
    eta_f = kwargs.get('eta_f', 1)
    max_iter = kwargs.get('max_iter', 20)
    decay_rate = np.log(eta_f/eta_0)/max_iter

    shots = kwargs.get('shots', 5000)

    print_iter = kwargs.get('print_iter', 5)

    optimal_cost = 1e5

    # initalization as in paper of xia & kais
    angles = 0.02*(2*np.random.rand(qbm_data.num_params) - 1)

    for i in range(max_iter + 1):
        
        # generate probability distribution based on list of parameters
        state_amplitudes = qbm_builder.get_state_amplitudes(params_list=angles,
                                                            joined=True,
                                                            shots=shots)
        # initialize post-process with Hamiltonian of interest
        post_process = ExpValClass(state_amplitudes, qbm_data, hamiltonian)
        
        
        # evaluate expectation value of Hamiltonian
        hamilton_exp_val, cached_op_string_exp_vals = \
            post_process.get_hamilton_exp_value()
            
        # save result if it better than before
        if hamilton_exp_val < optimal_cost:
            optimal_cost = hamilton_exp_val
            optimal_angles = angles
            optimal_amplitudes = state_amplitudes

        if i % print_iter == 0 and print_results:
            print(f'cost after {i} iters: {hamilton_exp_val}')

        # gradien descent with decreating learning rate for every iteration if eta_f < eta_0 
        angles -= np.exp(decay_rate * i) * eta_0 * post_process.get_grad_h(angles).astype(float)
        
    result = {}
    result['state_amplitudes'] = optimal_amplitudes
    result['hamiltonian_ground_state_energy'] = optimal_cost
    result['qbm_params'] = optimal_angles

    return result


if __name__ == '__main__':
    import json

    # define name of input hamiltonian within input dir
    hamilton_json_str = '2_qubit_z_hamiltonian'

    with open(f"input/{hamilton_json_str}.json") as file:
        hamilton = json.load(file)

    # test params
    backend = None
    num_hidden_qubits = 2
    eta_0 = 1
    eta_f = 1
    max_iter = 20
    print_results = False

    result = optimize_qbm_from_hamiltonian(hamiltonian=hamilton,
                                           num_hidden_qubits=num_hidden_qubits)
    print(result)