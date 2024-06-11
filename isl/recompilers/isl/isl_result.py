"""Contains ISLResult"""

class ISLResult:
    def __init__(
        self,
        circuit,
        overlap,
        exact_overlap,
        num_1q_gates,
        num_2q_gates,
        cnot_depth_history,
        global_cost_history,
        local_cost_history,
        circuit_history,
        entanglement_measures_history,
        e_val_history,
        qubit_pair_history,
        method_history,
        time_taken,
        cost_evaluations,
        coupling_map,
        circuit_qasm,
    ):
        """
        :param circuit: Resulting circuit.
        :param overlap: 1 - final_global_cost.
        :param exact_overlap: Only computable with SV backend.
        :param num_1q_gates: Number of rotation gates in circuit.
        :param num_2q_gates: Number of entangling gates in circuit.
        :param cnot_depth_history: Depth of ansatz after each layer when only considering 2-qubit gates.
        :param global_cost_history: List of global costs after each layer.
        :param local_cost_history: List of local costs after each layer (if applicable).
        :param circuit_history: List of circuits as qasm strings after each layer (if applicable).
        :param entanglement_measures_history: List of pairwise entanglements after each layer.
        :param e_val_history: List of single-qubit sigma_z expectation values after each layer.
        :param qubit_pair_history: List of qubit pair acted on in each layer.
        :param method_history: List of methods used to select qubit pairs for each layer.
        :param time_taken: Total time taken for recompilation.
        :param cost_evaluations: Total number of cost evalutions during recompilation.
        :param coupling_map: List of allowed qubit connections.
        :param circuit_qasm: QASM string of the resulting circuit.
        """
        self.circuit = circuit
        self.overlap = overlap
        self.exact_overlap = exact_overlap
        self.num_1q_gates = num_1q_gates
        self.num_2q_gates = num_2q_gates
        self.cnot_depth_history = cnot_depth_history
        self.global_cost_history = global_cost_history
        self.local_cost_history = local_cost_history
        self.circuit_history = circuit_history
        self.entanglement_measures_history = entanglement_measures_history
        self.e_val_history = e_val_history
        self.qubit_pair_history = qubit_pair_history
        self.method_history = method_history
        self.time_taken = time_taken
        self.cost_evaluations = cost_evaluations
        self.coupling_map = coupling_map
        self.circuit_qasm = circuit_qasm
