"""Contains ISLConfig"""

import isl.utils.constants as vconstants

class ISLConfig:
    def __init__(
        self,
        max_layers: int = int(1e5),
        sufficient_cost=vconstants.DEFAULT_SUFFICIENT_COST,
        max_2q_gates=1e4,
        cost_improvement_num_layers=10,
        cost_improvement_tol=1e-2,
        max_layers_to_modify=100,
        method="ISL",
        bad_qubit_pair_memory=10,
        reuse_exponent=0,
        reuse_priority_mode="pair",
        rotosolve_frequency=1,
        rotoselect_tol=1e-5,
        rotosolve_tol=1e-3,
        entanglement_threshold=1e-8,
    ):
        """
        ISL termination criteria.
        :param max_layers: ISL will terminate if the number of ansatz layers reaches this value.
        :param sufficient_cost: ISL will terminate if the cost reaches below this value.
        :param max_2q_gates: ISL will terminate if the number of 2 qubit gates reaches this value.
        :param cost_improvement_num_layers: The number of layer costs to consider when evaluating
        if the cost is decreasing fast enough.
        :param cost_improvement_tol: ISL will terminate if in the last cost_improvement_num_layers,
        the cost has not decreased by this value on average per layer.

        Add layer criteria:
        :param max_layers_to_modify: The number of layers to modify, counting from the back of
        the ansatz, when Rotosolve is used.
        :param method: Method by which a qubit pair is prioritised for the next layer. One of:
         'ISL' - Largest pairwise entanglement as defined by ISLRecompiler.entanglement_measure
         'expectation' - Smallest combined σz expectation values (i.e., closest to min value of -2)
         'basic' - Pair not picked in the longest time
         'random' - Pair selected randomly
         'general_gradient' - Pair with largest Euclidean norm of the global cost gradient with
         respect to all parameters (θ) in the layer ansatz, evaluated at θ=0.
        :param bad_qubit_pair_memory: For the ISL method, if acting on a qubit pair leads to
        entanglement increasing, it is labelled a "bad pair". After this, for a number of layers
        corresponding to the bad_qubit_pair_memory, this pair will not be selected.
        :param reuse_exponent: For ISL, expectation or general_gradient method, this
        controls how much priority should be given to picking qubits not recently acted on. If 0,
        the priority system is turned off and all qubits have the same reuse priority when adding
        a new layer. Note ISL never reuses the same pair of qubits regardless of this setting.
        :param reuse_priority_mode: For the priority system, given qubit pair (q1, q2) has been used
        before, should priority be given to:
        (a) not reusing the same pair of qubits (q1, q2) (set param to "pair")
        (b) not reusing the qubits q1 OR q2 (set param to "qubit")

        Other parameters:
        :param rotosolve_frequency: How often Rotosolve is used (if n, rotosolve will be used after
        every n layers).
        :param rotoselect_tol: How much does the cost need to decrease by each iteration to continue
         Rotoselect.
        :param rotosolve_tol: How much does the cost need to decrease by each iteration to continue
         Rotosolve.
        :param entanglement_threshold: For the ISL method, entanglement below this value is treated
         as zero in terms of picking the next layer.
        """
        self.bad_qubit_pair_memory = bad_qubit_pair_memory
        self.max_layers = max_layers
        self.sufficient_cost = sufficient_cost
        self.max_2q_gates = max_2q_gates
        self.cost_improvement_tol = cost_improvement_tol
        self.cost_improvement_num_layers = int(cost_improvement_num_layers)
        self.max_layers_to_modify = max_layers_to_modify
        self.method = method
        self.rotosolve_frequency = rotosolve_frequency
        self.rotoselect_tol = rotoselect_tol
        self.rotosolve_tol = rotosolve_tol
        self.entanglement_threshold = entanglement_threshold
        self.reuse_exponent = reuse_exponent
        self.reuse_priority_mode = reuse_priority_mode.lower()

    def __repr__(self):
        representation_str = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            representation_str += f"{k}={v!r}, "
        representation_str += ")"
        return representation_str
