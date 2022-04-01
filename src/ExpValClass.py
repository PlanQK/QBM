import numpy as np

from .QBMBuilder import QBMBuilder
from .QBMData import QBMData
from .helpers import binary_string_to_spin_list


class ExpValClass():
    '''
    Class for calculating expectation values, as well as its gradient. Expectation values can be 
    calculated using the hamiltonian and state_amplitudes. For calculating the gradient the
    qbm_data and the hamiltonian are required.
    '''

    def __init__(self, state_amplitudes, qbm_data, hamilton):
        '''
        Args:
            state_amplitudes: Dict(String: Float)
                Not-normed state amplitudes describing the wave function of interest. Keys
                are binary states (e.g. '101').
            qbm_data: QBMData
                Underlying data model for the QBM Use-Case
            hamilton: Dict(String: Float)
                Hamiltonian of interest. Keys are Pauli-Strings (e.g. 'IXYZ') where I denotes the
                identity. Values are the coefficients in front of the corresponding Pauli-string.
        '''

        self.state_amplitudes = state_amplitudes
        self.qbm_data = qbm_data
        self.hamilton = hamilton
        self.state_amplitudes_braket = \
            sum(np.abs(ampl)**2 for ampl in self.state_amplitudes.values())

    def get_op_state_amplitudes(self, op_string):
        '''
        For the input op_string (e.g. 'IXYZ'), the state op_string*psi gets calculated.
        
        Args:
            op_string: String
                Operator of interest. E.g. 'IXYZ'
        
        Returns:
            op_state_amplitudes: Dict(String: Float)
                New state amplitudes after the action of the operator.
        '''
        example_state = list(self.state_amplitudes.keys())[0]
        assert len(example_state) == len(op_string), \
        "operator has not the same length as the visible register"

        op_state_amplitudes = {}
        for key, value in self.state_amplitudes.items():
            binary_list = [int(x) for x in list(key)]

            new_binary_list = []
            if isinstance(value, list):
                new_amplitude = value[1]
            elif isinstance(value, (complex, float)):
                new_amplitude = value
            else:
                print('whoopsie')

            for op, bit in zip(list(op_string), binary_list):
                if op == 'Z' and bit == 1:
                    new_amplitude *= -1
                if op == 'Y':
                    new_amplitude *= (2*bit - 1)*complex(0, 1)
                if op in ('X', 'Y'):
                    bit = (bit + 1)%2
                new_binary_list.append(str(bit))

            new_state = ''.join(new_binary_list)

            if isinstance(value, list):
                op_state_amplitudes[new_state] = [value[0], new_amplitude]
            elif isinstance(value, (complex, float)):
                op_state_amplitudes[new_state] = new_amplitude
            else:
                print('whoopsie')
        return op_state_amplitudes


    def get_exp_value(self, op_string, normed = True):
        '''
        For a given op_string, the expectation value of it is calculated, namely
        np.conj(state_amplitudes)*op_string_amplitudes/(np.conj(state_amplitudes)*
        state_amplitudes). It is being normed if the corresponding arg is True, otherwise it will 
        not be normed. This is helpful for the calculation of the hamilton expectation value
        since only one division is necessary like that.

        Args:
            op_string: String
                Pauli-string whose expectation value should be calculated.
            normed: Boolean
                determines whether the expectation value should be normed (if False, the 
                result will have no physical meaning besides in the context of using the
                get_hamilton_exp_value method).
        Returns:
            exp_val: Float
                Expectation value. If the calculation gives rise to an imaginary part > 1e-10
                the complex result will be returned, otherwise only the real part.
        '''
        op_state_amplitudes = self.get_op_state_amplitudes(op_string)

        exp_val = 0
        for key, ket in op_state_amplitudes.items():
            if self.state_amplitudes.get(key) is not None:
                if isinstance(self.state_amplitudes[key], list):
                    bra = np.conj(self.state_amplitudes[key][0]*self.state_amplitudes[key][1])
                    exp_val += bra*ket[0]*ket[1]
                elif isinstance(self.state_amplitudes[key], (complex, float)):
                    bra = np.conj(self.state_amplitudes[key])
                    exp_val += bra*ket
                else:
                    print('whoopsie')

        if normed: 
            exp_val /= self.state_amplitudes_braket
        if exp_val.imag > 1e-10:
            print('expectation value has non vanishing imaginary part.')
            return exp_val
        return exp_val.real

    def get_hamilton_exp_value(self, cache_exp_values = False):
        '''
        Iterates over the hamiltonian and calculates its expectation value.

        Args:
            cache_exp_values: Bool
                If set to True, the expectation values of the pauli strings making up the 
                Hamiltonian will be stored in a dictionary.
        
        Returns:
            Tuple(Float, Union[Dict, None])
                Second output will be a dict containing the expectation values of the strings
                making up the Hamiltonian for the given state_amplitude if cache_exp_values is
                set to True, otherwise it will be None 
        '''
        exp_val = 0
        op_string_exp_val_dict = {} if cache_exp_values else None
        for op_string, coeff in self.hamilton.items():
            op_string_exp_val = self.get_exp_value(op_string = op_string, normed = False)
            if cache_exp_values:
                op_string_exp_val_dict[op_string] =  op_string_exp_val/self.state_amplitudes_braket
            exp_val += coeff*op_string_exp_val
        exp_val /= self.state_amplitudes_braket
        return exp_val, op_string_exp_val_dict

    def get_local_energy(self, state):
        '''
        Returns <x|H|\psi>/<\psi|\psi>, namely the normed amplitude of the state after the hamilton
        acted on the wave function determined by state_amplitudes. Necessary for the gradient
        calculation.
        '''
        hamilton_state_amplitude = 0
        for op, coeff in self.hamilton.items():
            op_on_state = self.get_op_state_amplitudes(op_string = op)
            if state in op_on_state.keys():
                hamilton_state_amplitude += coeff*op_on_state[state]

        if self.state_amplitudes[state] == 0:
            print('At least one sampled amplitude is 0 (not enough samples).')
            return None

        return hamilton_state_amplitude/self.state_amplitudes[state]

    def get_normed_amplitude_gradient(self, state, params_list):
        '''
        Calculates a special derivative of the amplitudes in a vectorized notion. For reference
        see supplementary note 1 in https://arxiv.org/pdf/1803.10296.pdf

        Args:
            state: String
                State for which the derivative should be calculated. Denoted as x in the reference.
            params_list: np.ndarray
                Parameters for which the result in state_amplitudes were obtained with. Anything
                else wouldn't make any sense.

        '''
        params_dict = self.qbm_data.get_params_dict(params_list)
        hidden_biases = params_dict['hidden_biases']
        weights = params_dict['weights']
        
        normed_d = np.zeros(shape = (self.qbm_data.num_params))

        spin_state_list = np.array(binary_string_to_spin_list(state))
        theta = np.dot(spin_state_list, weights) + hidden_biases

        # visible biases
        normed_d[:self.qbm_data.num_visible_qubits] = np.array(spin_state_list)/2

        # hidden biases
        normed_d[self.qbm_data.num_visible_qubits: self.qbm_data.num_qbm_qubits] = np.tanh(theta)/2

        # weights
        for i in range(self.qbm_data.num_visible_qubits):
            normed_d[self.qbm_data.num_qbm_qubits + i*self.qbm_data.num_hidden_qubits: 
                    self.qbm_data.num_qbm_qubits + (i+1)*self.qbm_data.num_hidden_qubits] \
                        = theta*spin_state_list[i]/2

        # sign biases
        node_value = self.qbm_data.get_node_value(spin_state_list, params_dict)
        sign_shift_func = 1/node_value - node_value

        normed_d[self.qbm_data.num_q_params:
                self.qbm_data.num_q_params + self.qbm_data.num_visible_qubits] \
                    = sign_shift_func*spin_state_list

        # sign shift
        normed_d[self.qbm_data.num_q_params + self.qbm_data.num_visible_qubits:
                self.qbm_data.num_q_params + self.qbm_data.num_visible_qubits + 1] \
                    = sign_shift_func
        
        return normed_d

    def get_grad_h(self, params_list):
        '''
        Calculates the gradient of the hamiltonian in a vectorized form. Should be used as
        jacobian for an optimization scheme, where the cost_function is the get_hamilton_exp_value
        method.
        Should work for scipy minimize function, as long as the cost_function is callable and
        depends on params_list as well. 
        '''

        exp_val_1 = np.zeros(shape = (self.qbm_data.num_params), dtype = complex)
        exp_val_2 = np.zeros(shape = (self.qbm_data.num_params), dtype = complex)
        exp_val_3 = np.zeros(shape = (self.qbm_data.num_params), dtype = complex)
        state_probabilities = {
            key: np.abs(value)**2 for key, value in self.state_amplitudes.items()}
        
        for state, probability in state_probabilities.items():
            e_loc = self.get_local_energy(state)
            d_vec = self.get_normed_amplitude_gradient(state, params_list)

            exp_val_1 += e_loc*d_vec*probability
            exp_val_2 += e_loc*probability
            exp_val_3 += d_vec*probability

        grad_h = 2*(exp_val_1 - exp_val_2*exp_val_3/(self.state_amplitudes_braket))\
            /self.state_amplitudes_braket

        return grad_h.real