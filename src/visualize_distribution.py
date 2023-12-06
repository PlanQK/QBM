import numpy as np
from itertools import product
from .helpers import binary_string_to_spin_list
from .QBMBuilder import QBMBuilder


def possible_binary_states(qbm_data):
    num_qbm_qubits = qbm_data.num_qbm_qubits
    combi_list = []
    for elem in product([0, 1], repeat=num_qbm_qubits):
        string = ''.join(list(map(str, elem)))
        combi_list.append(string)

    return combi_list


def boltzmann_numerator(input_string, qbm_data, params_list):
    params_dict = qbm_data.get_params_dict(params_list)
    visible_biases = params_dict['visible_biases']
    hidden_biases = params_dict['hidden_biases']
    weights = params_dict['weights']
    regulator = params_dict['regulator']

    spin_states = np.array(binary_string_to_spin_list(input_string))

    vis_spin_states = spin_states[:qbm_data.num_visible_qubits]
    hid_spin_states = spin_states[qbm_data.num_visible_qubits:]

    energy = visible_biases.dot(vis_spin_states) + hidden_biases.dot(hid_spin_states) \
        + vis_spin_states.dot(weights.dot(hid_spin_states))
    return np.exp(energy/regulator)


def boltzmann_denominator(qbm_data, params_list):
    all_terms = np.array([])

    for state in product([0, 1], repeat=qbm_data.num_qbm_qubits):
        string = ''.join(list(map(str, state)))
        exp_term = boltzmann_numerator(string, qbm_data, params_list)
        all_terms = np.append(all_terms, exp_term)

    partition_function = np.sum(all_terms)
    return partition_function, all_terms


def ideal_distribution(qbm_data, params_list):
    combi_list = possible_binary_states(qbm_data)
    result = {key: boltzmann_numerator(key, qbm_data, params_list)
              / boltzmann_denominator(qbm_data, params_list)[0] for key in combi_list}
    return result


def measured_distribution(qbm_data, params_list, shots: int = 100_000, ordered=False):
    qbm_builder = QBMBuilder(qbm_data)

    results = qbm_builder.get_qbm_results(params_list, shots=shots)
    valid_results_ordered = results['valid_results_ordered']

    measured_distribution_var = {key: valid_results_ordered[key]/sum(valid_results_ordered.values())
                                 for key in valid_results_ordered.keys()}

    return measured_distribution_var
