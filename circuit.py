from src.QBMBuilder import QBMBuilder
from src.QBMData import QBMData

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
    circuit = qbm_builder.prepare_qbm()
    return circuit

if __name__ == "__main__":
    get_circuit(num_visible_qubits=num_visible_qubits,
                num_hidden_qubits=num_hidden_qubits,
                reuse_ancilla=reuse_ancilla)
