import numpy as np

class QBMData():
    '''
    Class containing all important information for the execution of the QBM. Expects #visible
    qubits (which is determined by length of the Pauli-strings within the Hamiltonian of 
    interest) as well as #hidden qubits (which can be arbitrary, though it is recommended
    that #hidden ~ O(#visible)). 
    '''

    def __init__(self, num_visible_qubits, num_hidden_qubits):
        self.num_visible_qubits = num_visible_qubits
        self.num_hidden_qubits = num_hidden_qubits
        self.reuse_ancilla = True
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
            newshape = (self.num_visible_qubits, self.num_hidden_qubits))

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
            phase_biases = params_list[self.num_q_params + self.num_visible_qubits + 1:\
                                    self.num_q_params+ 2*self.num_visible_qubits +1]
            phase_shift = np.array(params_list[-1])
            
            params_dict['phase_biases'] = phase_biases
            params_dict['phase_shift'] = phase_shift

        abs_weight_sum = float(np.sum(np.abs(weights)))
        regulator = max(abs_weight_sum/2,1)
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