import numpy as np
import qiskit as q

# While being in QBM repo: python -m circuit

num_visible_qubits = 2
num_hidden_qubits = 2
reuse_ancilla = True


def get_circuit(**kwargs):
    n_vis = kwargs.get("num_visible_qubits")
    n_hid = kwargs.get("num_hidden_qubits")
    reuse_ancilla = kwargs.get("reuse_ancilla", True)
    qbm_data = QBMData(n_vis, n_hid, reuse_ancilla=reuse_ancilla)
    qbm_builder = QBMBuilder(qbm_data)
    circuit = qbm_builder.prepare_qbm(params_list=np.random.rand(qbm_data.num_params))

    return circuit


def get_circuit_with_parameters(**kwargs):
    n_vis = kwargs.get("num_visible_qubits")
    n_hid = kwargs.get("num_hidden_qubits")
    reuse_ancilla = kwargs.get("reuse_ancilla", True)
    qbm_data = QBMData(n_vis, n_hid, reuse_ancilla=reuse_ancilla)
    qbm_builder = QBMBuilder(qbm_data)

    params = kwargs.get("params")
    if params is None:
        params = np.random.rand(qbm_data.num_params)
    circuit = qbm_builder.prepare_qbm(params)
    return circuit

# source code:


class QBMData():
    '''
    Class containing all important information for the execution of the QBM. Expects #visible
    qubits (which is determined by length of the Pauli-strings within the Hamiltonian of 
    interest) as well as #hidden qubits (which can be arbitrary, though it is recommended
    that #hidden ~ O(#visible)). 
    '''

    def __init__(self, num_visible_qubits, num_hidden_qubits, **kwargs):
        self.num_visible_qubits = num_visible_qubits
        self.num_hidden_qubits = num_hidden_qubits
        self.reuse_ancilla = kwargs.get('reuse_ancilla', True)
        self.node_type = 'sign'

        self.num_qbm_qubits = self.get_num_qbm_qubits()
        self.num_layer_connections = self.get_num_layer_connections()
        self.num_all_qubits = self.get_num_all_qubits()
        self.num_ancilla_qubits = self.get_num_ancilla_qubits()
        self.num_params = self.get_num_params()
        self.num_c_params = self.get_num_c_params()
        self.num_q_params = self.get_num_q_params()

    def __str__(self):
        '''
        Print out all attribute dictionary.
        '''
        return str(vars(self))

    def get_num_qbm_qubits(self):
        return self.num_visible_qubits + self.num_hidden_qubits

    def get_num_layer_connections(self):
        return self.num_hidden_qubits*self.num_visible_qubits

    def get_num_ancilla_qubits(self):
        if self.reuse_ancilla:
            return 1
        return self.get_num_layer_connections()

    def get_num_all_qubits(self):
        return self.get_num_qbm_qubits() + self.get_num_ancilla_qubits()

    def get_num_params(self):
        num_params = 2*self.num_visible_qubits + self.num_hidden_qubits\
            + self.num_visible_qubits*self.num_hidden_qubits + 1
        if self.node_type == 'phase':
            num_params += self.num_visible_qubits + 1
        return num_params

    def get_num_c_params(self):
        '''
        Number of classical parameters to be optimized (namely the arguments within the classical
        node).
        '''
        num_params = self.num_visible_qubits + 1
        if self.node_type == 'phase':
            num_params *= 2
        return num_params

    def get_num_q_params(self):
        '''
        Number of quantum parameters to be optimized (namely the arguments within the QBM).
        '''
        return self.get_num_params() - self.get_num_c_params()

    def get_params_dict(self, params_list):
        '''
        Main object for getting access to the relevant parameters. For optimization all parameters
        must be in a (n,) array, which makes the distinction difficult. When one is interested
        in the weight-matrix for a given list of parameters, the params_dict contains that
        information.
        '''
        params_dict = {}

        # list of all parameters concatenated, important for optimization
        circuit_params = params_list[:self.num_q_params]
        visible_biases = circuit_params[:self.num_visible_qubits]
        hidden_biases = circuit_params[self.num_visible_qubits:self.num_qbm_qubits]
        weights = np.reshape(
            circuit_params[self.num_qbm_qubits:self.num_qbm_qubits + self.num_layer_connections],
            newshape=(self.num_visible_qubits, self.num_hidden_qubits))

        sign_biases = params_list[self.num_q_params:self.num_q_params+self.num_visible_qubits]
        sign_shift = np.array(params_list[self.num_q_params+self.num_visible_qubits])

        params_dict['params_list'] = params_list
        params_dict['circuit_params'] = circuit_params
        params_dict['visible_biases'] = visible_biases
        params_dict['hidden_biases'] = hidden_biases
        params_dict['weights'] = weights
        params_dict['sign_biases'] = sign_biases
        params_dict['sign_shift'] = sign_shift

        if self.node_type == 'phase':
            phase_biases = params_list[self.num_q_params + self.num_visible_qubits + 1:
                                       self.num_q_params + 2*self.num_visible_qubits + 1]
            phase_shift = np.array(params_list[-1])

            params_dict['phase_biases'] = phase_biases
            params_dict['phase_shift'] = phase_shift

        abs_weight_sum = float(np.sum(np.abs(weights)))
        regulator = max(abs_weight_sum/2, 1)
        params_dict['regulator'] = regulator

        return params_dict

    def get_node_value(self, spin_state, params_dict):
        '''
        Calculates the value of the sign-/phase-node for a given spin-state and parameters
        provided in params_dict.
        '''
        sign_biases = params_dict['sign_biases']
        sign_shift = params_dict['sign_shift']
        if self.node_type == 'phase':
            phase_biases = params_dict['phase_biases']
            phase_shift = params_dict['phase_shift']

        if self.node_type == 'sign':
            node_arg = spin_state.dot(sign_biases) + sign_shift
        elif self.node_type == 'phase':
            node_arg_real = spin_state.dot(sign_biases) + sign_shift
            node_arg_imag = spin_state.dot(phase_biases) + phase_shift

            node_arg = complex(node_arg_real, node_arg_imag)

        return np.tanh(node_arg)


class QBMBuilder(QBMData):
    '''
    Class for building the QBM and extract wave-function amplitudes based on the measurement
    results.
    '''

    def __init__(self, qbm_data, backend=None):
        self.num_visible_qubits = qbm_data.num_visible_qubits
        self.num_hidden_qubits = qbm_data.num_hidden_qubits
        self.reuse_ancilla = qbm_data.reuse_ancilla
        self.node_type = 'sign'
        self.backend = q.Aer.get_backend('qasm_simulator') if backend is None else backend
        self.circuit = None

        self.num_qbm_qubits = qbm_data.num_qbm_qubits
        self.num_layer_connections = qbm_data.num_layer_connections
        self.num_all_qubits = qbm_data.num_all_qubits
        self.num_ancilla_qubits = qbm_data.num_ancilla_qubits
        self.num_params = qbm_data.num_params
        self.num_c_params = qbm_data.num_c_params
        self.num_q_params = qbm_data.num_q_params

    def prepare_empty_qbm_circuit(self):
        '''
        Gives out an empty circuit based on the parameters defined in qbm_data.
        Args:
            None
        Returns (q: qiskit)
            q.QuantumCircuit
        '''
        visible_register = q.QuantumRegister(self.num_visible_qubits,
                                             name='visible_register')
        hidden_register = q.QuantumRegister(self.num_hidden_qubits,
                                            name='hidden_register')
        ancilla_register = q.QuantumRegister(self.num_ancilla_qubits,
                                             name='ancilla_register')

        ancilla_result = q.ClassicalRegister(self.num_layer_connections,
                                             name='ancilla_result')
        qubit_result = q.ClassicalRegister(self.num_qbm_qubits,
                                           name='qubit_result')

        return q.QuantumCircuit(visible_register,
                                hidden_register,
                                ancilla_register,
                                qubit_result,
                                ancilla_result)

    @staticmethod
    def single_gate_rotations(qc, bias_params):
        ''' 
        Iterates over biases and applies gates on the qubit in q_reg with the same index.
        (seperate use for visible and hidden register)

        Args: (q: qiskit)
            qc: q.QuantumCircuit
                inital circuit with predefined registers
            q_reg: q.QuantumRegister
                Register where the gates should be applied
            biases: np.ndarray([Float])
                Biases which determine the rotation angles
            regulator: Float
                Parameter for maximizing probability of a successful sampling
        Returns:
            None
        '''
        for qubit_idx, bias_param in enumerate(bias_params):
            qc.ry(bias_param, qubit=qubit_idx)

    @staticmethod
    def doubly_controlled_rotation(qc, weight_param, qubits):
        '''
        Applies a rotational R_Y gate, which is controlled by two other qubits.
        The angle is determined by the variable weight

        Args: (q: qiskit)
            qc: q.QuantumCircuit
            weight: Float
                Strength of the connection between the control qubits
            Qubits: List([q.Qubit])
                First two entries are the control qubits in the visible/hidden register, 
                third entry corresponds to target qubit
            regulator: Float
                Parameter for maximizing the probability of a successful sampling
        Returns:
            None
        '''
        cq0, cq1 = qubits[0], qubits[1]
        ancilla = qubits[2]

        qc.cry(theta=weight_param/2, control_qubit=cq1, target_qubit=ancilla)
        qc.cx(control_qubit=cq0, target_qubit=cq1)
        qc.cry(theta=-weight_param/2, control_qubit=cq1, target_qubit=ancilla)
        qc.cx(control_qubit=cq0, target_qubit=cq1)
        qc.cry(theta=weight_param/2, control_qubit=cq0, target_qubit=ancilla)

    @staticmethod
    def doubly_rotation_layer(qc, qubit_indices, weight_param_pair):
        '''
        Applies a sequence of 4 doubly-controlled rotational gates, which are controlled
        by all combinations of 0/1 of the two control qubits.

        Args: ( q: qiskit )
            qc: q.QuantumCircuit
            qubit_indices: List([int, int, q.Qubit])
                First two entries refer to indices of the control qubits in the visible/hidden
                register, third entry is the target qubit, where the rotation is performed on
            weights: np.ndarray 
                weight-matrix with dim(weights) = visible_qubits*hidden_qubits
            regulator: Float
                Prameter for maximizing the probability of a successful sampling
            test: Boolean
                If True, sets the weight to -1, ignoring the values given by weights
        Returns:
            None
        '''

        visible_register = [reg for reg in qc.qregs if reg.name == 'visible_register'][0]
        hidden_register = [reg for reg in qc.qregs if reg.name == 'hidden_register'][0]

        q0 = visible_register[qubit_indices[0]]
        q1 = hidden_register[qubit_indices[1]]
        ancilla_q = qubit_indices[2]

        QBMBuilder.doubly_controlled_rotation(qc=qc, weight_param=weight_param_pair[0],
                                              qubits=[q0, q1, ancilla_q])  # r1

        qc.x(qubit=q0)
        QBMBuilder.doubly_controlled_rotation(qc=qc, weight_param=weight_param_pair[1],
                                              qubits=[q0, q1, ancilla_q])  # r2
        qc.x(qubit=q0)

        qc.x(qubit=q1)
        QBMBuilder.doubly_controlled_rotation(qc=qc, weight_param=weight_param_pair[1],
                                              qubits=[q0, q1, ancilla_q])  # r3
        qc.x(qubit=q1)

        qc.x(qubit=q0)
        qc.x(qubit=q1)
        QBMBuilder.doubly_controlled_rotation(qc=qc, weight_param=weight_param_pair[0],
                                              qubits=[q0, q1, ancilla_q])  # r4
        qc.x(qubit=q0)
        qc.x(qubit=q1)

    @staticmethod
    def bind_q_params(params_dict):
        single_gate_angles = QBMBuilder.bind_single_gate_angles(params_dict)
        double_gate_angles_1 = QBMBuilder.bind_double_gate_angles(params_dict)
        double_gate_angles_2 = QBMBuilder.bind_double_gate_angles(params_dict, sign=True)
        return np.concatenate([single_gate_angles, double_gate_angles_1, double_gate_angles_2])

    @staticmethod
    def bind_single_gate_angles(params_dict):
        regulator = params_dict['regulator']
        biases = np.concatenate([params_dict['visible_biases'],
                                 params_dict['hidden_biases']])
        pos_exp = np.exp(-biases/regulator)
        neg_exp = 1/pos_exp
        arg = np.sqrt(pos_exp/(pos_exp + neg_exp))
        return 2*np.arcsin(arg)

    @staticmethod
    def bind_double_gate_angles(params_dict, sign=False):
        regulator = params_dict['regulator']
        weights = params_dict['weights'].flatten()
        if sign:
            weights = (-1)*weights
        abs_weight = np.abs(weights)
        arg = np.sqrt(np.exp(weights/regulator)/np.exp(abs_weight/regulator))
        return 2*np.arcsin(arg)

    def prepare_qbm(self, params_list=None):
        '''
        Prepares the whole QBM for the input circuit parameters. The circuit containing all instructions
        as well as the measurements for the ancillae is returned.

        Args: ( q: qiskit )
            params_list:
                list of all parameters in an (n,)-array
        Returns:
            circuit: q.QuantumCircuit 
                Finished qbm-circuit including all gates and measurements.
        '''

        if self.circuit is not None:
            circuit = self.circuit
            if params_list is not None:
                params_dict = self.get_params_dict(params_list)
                parameter_values = QBMBuilder.bind_q_params(params_dict)
                circuit = self.circuit.bind_parameters(parameter_values)
            return circuit

        theta = q.circuit.ParameterVector('θ', length=self.num_q_params
                                          + self.num_layer_connections)
        bias_params = theta[:self.num_qbm_qubits]
        weights_params = theta[self.num_qbm_qubits:]

        circuit = self.prepare_empty_qbm_circuit()

        ancilla_register = [reg for reg in circuit.qregs if reg.name == 'ancilla_register'][0]
        ancilla_result = [reg for reg in circuit.cregs if reg.name == 'ancilla_result'][0]

        QBMBuilder.single_gate_rotations(qc=circuit, bias_params=bias_params)

        ancilla_idx = 0
        for i in range(self.num_visible_qubits):
            for j in range(self.num_hidden_qubits):

                if self.reuse_ancilla:
                    ancilla_qubit = ancilla_register[0]
                else:
                    ancilla_qubit = ancilla_register[ancilla_idx]
                weight_param_pair = np.append(
                    weights_params[ancilla_idx],
                    weights_params[ancilla_idx + self.num_layer_connections])
                QBMBuilder.doubly_rotation_layer(
                    qc=circuit, qubit_indices=[i, j, ancilla_qubit],
                    weight_param_pair=weight_param_pair)

                circuit.barrier()
                circuit.measure(ancilla_qubit, ancilla_result[ancilla_idx])
                if self.reuse_ancilla:
                    circuit.reset(ancilla_register[0])
                circuit.barrier()
                ancilla_idx += 1

        circuit = q.transpile(circuit, self.backend)
        self.circuit = circuit

        if params_list is not None:
            params_dict = self.get_params_dict(params_list)
            parameter_values = QBMBuilder.bind_q_params(params_dict)
            circuit = circuit.bind_parameters(parameter_values)

        return circuit

    def get_qbm_results(self, params_list, shots=10_000):
        '''
        Returns a special format of the QBM results based on the parameters specified in
        params_list. Circuit gets transpiled for the chosen backend (if it is not a simulator)
        and gets executed via backend.run() method.
        Args:
            params_list: 
                List of parameters building the QBM.
            backend:
                q.Provider (qiskit backend which the experiment should run on)
            shots: 
                int (number of repetitions of the same experiment)
        Returns:
            qbm_results: Dict([...])
                Dict containing the experiment results (the actual results, as well as 
                the adapted results where the ancilla results are 1)
        '''
        circuit = self.prepare_qbm(params_list)

        visible_register = [reg for reg in circuit.qregs if reg.name == 'visible_register'][0]
        hidden_register = [reg for reg in circuit.qregs if reg.name == 'hidden_register'][0]

        c_reg = [reg for reg in circuit.cregs if reg.name == 'qubit_result'][0]
        relevant_qubits = list(visible_register) + list(hidden_register)

        circuit.measure(relevant_qubits, c_reg)

        experiment_results = self.backend.run(circuit, shots=shots).result().get_counts()

        valid_results = {key[self.num_layer_connections+1:]: experiment_results[key]
                         for key in experiment_results.keys()
                         if key[:self.num_layer_connections] ==
                         '1'*self.num_layer_connections}
        valid_shots = sum(valid_results.values())

        result_dict = {}
        result_dict['valid_results'] = valid_results
        result_dict['valid_results_ordered'] = {
            key[::-1]: valid_results[key] for key in valid_results.keys()
        }
        result_dict['actual_results'] = experiment_results
        result_dict['actual_shots'] = shots
        result_dict['valid_shots'] = valid_shots
        result_dict['success_rate'] = np.round(valid_shots/shots, 2)

        return result_dict

    def get_gibbs_distribution(self, params_list, shots=10_000):
        '''
        Calculates qbm_results based on params_list. Afterwards, the probabilities are expo-
        nentiated with the regulator to get the actual Boltzmann-/Gibbs-distribution (remember,
        the regulator was introduced to maximize the sampling probability but does not have any
        meaning for the wanted distribution). The distribution gets normed as well Also, the state
        of the hidden qubits are still within the keys of the output dict).

        Args:
            same as for QBMBuilder.get_qbm_results()

        Returns:
            Dict(state: probability)
        '''
        regulator = self.get_params_dict(params_list)['regulator']
        qbm_results = self.get_qbm_results(params_list, shots)
        valid_results_ordered = qbm_results['valid_results_ordered']
        valid_shots = qbm_results['valid_shots']

        if regulator != 1:
            state_probabilities = {key: (value/valid_shots)**regulator
                                   for key, value in valid_results_ordered.items()}
            norm = sum(state_probabilities.values())
            gibbs_distribution = {key: value/norm for key, value in state_probabilities.items()}
            return gibbs_distribution

        gibbs_distribution = {key: value/valid_shots
                              for key, value in valid_results_ordered.items()}
        return gibbs_distribution

    def get_state_amplitudes(self, params_list, shots=10_000, joined=False):
        '''
        Boltzmann-/Gibbs-Distribution is being calculated based on params_list. Afterwards, the
        states of the hidden qubits are being summed over in order to calculate the state
        amplitudes of the wave function of interest. IMPORTANT: State amplitudes are NOT normed
        since this is done within the ExpValClass (for expectation values, as well as the gradient
        calculation).
        '''
        whole_distribution = self.get_gibbs_distribution(params_list, shots)
        params_dict = self.get_params_dict(params_list)

        state_amplitudes = {}
        for key, value in whole_distribution.items():
            visible_state = key[:self.num_visible_qubits]

            if visible_state not in state_amplitudes:
                spin_state = np.array(binary_string_to_spin_list(visible_state))
                node_coeff = self.get_node_value(spin_state, params_dict=params_dict)

                state_amplitudes[visible_state] = [0, node_coeff]

            state_amplitudes[visible_state][0] += value

        for key, value in state_amplitudes.items():
            state_amplitudes[key][0] = np.sqrt(value[0])

        if not joined:
            new_state_amplitudes = {key: [value[0], value[1]]
                                    for key, value in state_amplitudes.items()}
        else:
            new_state_amplitudes = {key: value[0]*value[1]
                                    for key, value in state_amplitudes.items()}
        return new_state_amplitudes


def binary_string_to_spin_list(obj, reverse=False):
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


if __name__ == "__main__":
    circuit = get_circuit(num_visible_qubits=num_visible_qubits,
                          num_hidden_qubits=num_hidden_qubits,
                          reuse_ancilla=reuse_ancilla)
    print(circuit.qasm())
