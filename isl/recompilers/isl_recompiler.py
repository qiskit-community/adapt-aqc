"""Contains ISLRecompiler"""

import logging
import timeit

import aqc_research.mps_operations as mpsops
import isl.utils.cuquantum_functions as cu
import numpy as np
from qiskit import QuantumCircuit, qasm2

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.utils import circuit_operations as co
from isl.utils import cuquantum_functions as cu
from isl.utils.constants import CMAP_FULL, generate_coupling_map
from isl.utils.entanglement_measures import (
    EM_TOMOGRAPHY_CONCURRENCE,
    calculate_entanglement_measure,
)
from isl.utils.utilityfunctions import (
    expectation_value_of_qubits,
    expectation_value_of_qubits_mps,
    has_stopped_improving,
    remove_permutations_from_coupling_map,
)

logger = logging.getLogger(__name__)


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
        rotosolve_frequency=1,
        rotoselect_tol=1e-5,
        rotosolve_tol=1e-3,
        entanglement_threshold=1e-8,
        entanglement_reuse_exponent=0,
        heuristic_reuse_exponent=1,
        reuse_priority_mode="pair"
    ):
        """
        Termination criteria:
        :param max_layers: Maximum number of layers where each layer has a
        thinly dressed cnot gate
        :param sufficient_cost: ISL will terminate if the cost (1-overlap)
        reaches below this value
        :param max_2q_gates: ISL will terminate if the number of 2 qubit
        gates reaches this value
        :param cost_improvement_num_layers: The number of layer costs to
        consider when evaluating cost improvement
        :param cost_improvement_tol: The minimum relative cost improvement
        to continue adding layers

        Add layer criteria:
        :param max_layers_to_modify: Only the last max_layers_to_modify
        layers will be modified when Rotosolve is called
        :param method: Method to choose qubit pair for 2-qubit gates.
            One of 'ISL', 'random', 'heuristic','basic'

        Other parameters:
        :param bad_qubit_pair_memory: If acting on a qubit pair leads to entanglement increasing,
        it is labelled a "bad pair". This argument controls how many bad pairs should be remembered.
        :param rotosolve_frequency: How often rotosolve is used (if n, rotosolve will be used after
        every n layers). NOTE When using Aer MPS simulator setting this to 0 leads to large performance improvement.
        :param rotoselect_tol: How much does the cost need to decrease by each iteration to continue
         Rotoselect.
        :param rotosolve_tol: How much does the cost need to decrease by each iteration to continue
         Rotosolve.
        :param entanglement_threshold: Entanglement below this value is treated as zero.

        :param entanglement_reuse_exponent: When :param method == "ISL",
        controls how much priority should be given to picking qubits not recently acted on. If 0,
        the priority system is turned off and all qubits have the same priority when adding a new
        layer. Note ISL never reuses the same pair of qubits regardless of this setting.

        :param heuristic_reuse_exponent: See above, but for when :param method == "heuristic".
        :param reuse_priority_mode: For the priority system, given qubit pair (a, b) has been used
        before, should priority be given to:
        (a) not reusing the same pair of qubits (a, b) (set param to "pair")
        (b) not reusing the qubits a OR b (set param to "qubit")
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
        self.entanglement_reuse_exponent = entanglement_reuse_exponent
        self.heuristic_reuse_exponent = heuristic_reuse_exponent
        self.reuse_priority_mode = reuse_priority_mode.lower()

    def __repr__(self):
        representation_str = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            representation_str += f"{k}={v!r}, "
        representation_str += ")"
        return representation_str


class ISLRecompiler(ApproximateRecompiler):
    """
    Structure learning algorithm that incrementally builds a circuit that
    has the same result when acting on |0> state
    (computational basis) as the given circuit.
    """

    def __init__(
        self,
        target,
        entanglement_measure=EM_TOMOGRAPHY_CONCURRENCE,
        backend=co.SV_SIM,
        execute_kwargs=None,
        coupling_map=None,
        isl_config: ISLConfig = None,
        general_initial_state=False,
        custom_layer_2q_gate=None,
        save_circuit_history=False,
        starting_circuit=None,
        use_roto_algos=True,
        perform_final_minimisation=False,
        local_cost_function=False,
        debug_log_full_ansatz=False,
        initial_single_qubit_layer=False,
        cu_algorithm=None,
    ):
        """
        :param target: Circuit or MPS that is to be recompiled
        :param entanglement_measure: The entanglement measurement method to
        use for quantifying local entanglement
        :param backend: Backend to run circuits on
        :param coupling_map: 2-qubit gate coupling map to use
        :param isl_config: ISLConfig object
        :param general_initial_state: Recompile circuit for an arbitrary
        initial state
        :param custom_layer_2q_gate: Entangling gate to use (default is
        thinly dressed CNOT)
        :param save_circuit_history: Option to regularly save circuit output as a QASM string to 
        results object each time a block is added and optimised
        :param starting_circuit: This circuit will be used as a set of initial fixed gates for the
        recompiled solution. This means that during ISL, the inverse of this circuit will be added
        to the end of V†. WARNING: Using an entangled circuit will lead to worse ISL performance
        because it disrupts the measurement of local entanglement between qubits.
        :param use_roto_algos: Whether to use rotoselect and rotosolve
        for cost minimisation.
            Disable if custom_layer_2q_gate does not support rotosolve
        :param perform_final_minimisation: Perform a final cost minimisation
        once ISL has ended
        :param local_cost_function: Use LLET cost function as defined in
        (arXiv:1908.04416)
        :param execute_kwargs: keyword arguments passed into circuit runs (
        excluding backend)
            e.g. {'noise_model:NoiseModel, 'shots':10000}
        :param cu_algorithm: If using the cuquantum backend, this specifies the contract and
        decompose algorithm to use for gate application. Can be either a `dict` or a
        `ContractDecomposeAlgorithm`. If None set, the default used is
        isl.utils.circuit_operations.DEFAULT_CU_ALGORITHM
        """
        super().__init__(
            target=target,
            initial_state=None,
            backend=backend,
            execute_kwargs=execute_kwargs,
            general_initial_state=general_initial_state,
            starting_circuit=starting_circuit,
            local_cost_function=local_cost_function,
            cu_algorithm=cu_algorithm
        )

        self.save_circuit_history = save_circuit_history
        self.entanglement_measure_method = entanglement_measure
        self.isl_config = isl_config if isl_config is not None else ISLConfig()

        if coupling_map is None:
            coupling_map = generate_coupling_map(
                self.total_num_qubits, CMAP_FULL, False, False
            )

        # If custom layer gate is provided, do not remove gate during ISL
        # because individual gates
        # might depend on each other.
        self.remove_unnecessary_gates = custom_layer_2q_gate is None
        self.use_roto_algos = use_roto_algos
        self.perform_final_minimisation = perform_final_minimisation
        self.layer_2q_gate = self.construct_layer_2q_gate(custom_layer_2q_gate)

        # Remove permutations so that ISL is not stuck on the same pair of
        # qubits
        self.coupling_map = remove_permutations_from_coupling_map(coupling_map)
        self.coupling_map = [
            (q1, q2)
            for (q1, q2) in self.coupling_map
            if q1 in self.qubit_subset_to_recompile
            and q2 in self.qubit_subset_to_recompile
        ]
        # Used to avoid adding thinly dressed CNOTs to the same qubit pair
        self.qubit_pair_history = []
        # Avoid adding CNOTs to these qubit pairs
        self.bad_qubit_pairs = []
        # Used to keep track of whether ISL/heuristic method was used
        self.pair_selection_method_history = []
        self.entanglement_measures_history = []
        self.e_val_history = []

        self.debug_log_full_ansatz = debug_log_full_ansatz

        self.initial_single_qubit_layer = initial_single_qubit_layer

        if self.is_aer_mps_backend and self.isl_config.rotosolve_frequency == 0:
            self.save_previous_layer_mps_aer = True
            # As variational gates will be absorbed into one large MPS instruction, we need to
            # separately keep track of ansatz gates to return a compiled solution.
            self.ref_circuit_as_gates = self.full_circuit.copy()
        else:
            self.save_previous_layer_mps_aer = False

        if self.is_cuquantum_backend and self.isl_config.rotosolve_frequency == 0:
            self.save_previous_layer_mps_cuquantum = True
        else:
            self.save_previous_layer_mps_cuquantum = False

    def construct_layer_2q_gate(self, custom_layer_2q_gate):
        if custom_layer_2q_gate is None:
            qc = QuantumCircuit(2)
            if self.general_initial_state:
                co.add_dressed_cnot(qc, 0, 1, True)
                co.add_dressed_cnot(qc, 0, 1, True, v1=False, v2=False)
            else:
                co.add_dressed_cnot(qc, 0, 1, True)
            return qc
        else:
            return custom_layer_2q_gate

    def get_layer_2q_gate(self, layer_index):
        qc = self.layer_2q_gate.copy()
        co.add_subscript_to_all_variables(qc, layer_index)
        return qc

    def recompile_using_initial_ansatz(
        self, ansatz: QuantumCircuit, modify_ansatz=True
    ):
        """
        Use the provided ansatz as a starting point for the recompilation
        :param modify_ansatz: If ansatz should be optimised during ISL
        :param ansatz: Quantum Circuit to use
        :return: Recompilation result
        """
        old_vcr = self.variational_circuit_range()
        vcr = self.variational_circuit_range
        ansatz_inv = co.circuit_by_inverting_circuit(ansatz)
        co.add_to_circuit(self.full_circuit, ansatz_inv, vcr()[0])
        if not modify_ansatz:
            self.lhs_gate_count = old_vcr[0] + len(ansatz_inv.data)

        self.minimizer.minimize_cost(
            algorithm_kind=vconstants.ALG_ROTOSOLVE,
            tol=1e-5,
            stop_val=self.isl_config.sufficient_cost,
        )

        res = self.recompile()
        self.lhs_gate_count = old_vcr[0]

        recompiled_circuit = self.get_recompiled_circuit()
        exact_overlap = co.calculate_overlap_between_circuits(
            self.circuit_to_recompile, recompiled_circuit
        )
        num_2q_gates, num_1q_gates = co.find_num_gates(
            self.full_circuit, gate_range=vcr()
        )

        res["circuit"] = recompiled_circuit
        res["exact_overlap"] = exact_overlap
        res["num_1q_gates"] = num_1q_gates
        res["num_2q_gates"] = num_2q_gates
        res["circuit_qasm"] = qasm2.dumps(recompiled_circuit)
        return res

    def recompile(self, initial_ansatz: QuantumCircuit = None):
        """
        Perform recompilation algorithm.
        :param initial_ansatz: A trial ansatz to start the recompilation
        with instead of starting from scratch
        Termination criteria: SUFFICIENT_COST reached; max_layers reached;
        std(last_5_costs)/avg(last_5_costs) < TOL
        :return: {'circuit':resulting circuit(Instruction),
        'overlap':overlap(float),
        'num_1q_gates':number of rotation gates in circuit(int),
        'num_2q_gates':number of entangling gates in circuit(int)}
        'circuit_progression': list of circuits as qasm strings after each block is added and optimised
        'cost_progression': list of costs after each layer is added
        'time_taken': total time taken for recompilation
        'circuit_qasm': QASM string of the resulting circuit
        """
        logger.info("ISL started")
        logger.debug(f"ISL coupling map {self.coupling_map}")
        start_time = timeit.default_timer()
        self.cost_evaluation_counter = 0
        cost, num_1q_gates, num_2q_gates = None, None, None

        cost_history = []
        circuit_history = []
        g_range = self.variational_circuit_range

        # If an initial ansatz has been provided, add that and run minimization
        if initial_ansatz is not None:
            co.add_to_circuit(
                self.full_circuit,
                co.circuit_by_inverting_circuit(initial_ansatz),
                g_range()[1],
                transpile_before_adding=True,
            )
            if self.use_roto_algos:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=1e-3,
                    stop_val=self.isl_config.sufficient_cost,
                    indexes_to_modify=g_range(),
                )
            else:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_PYBOBYQA,
                    alg_kwargs={"seek_global_minimum": True},
                )
            if cost < self.isl_config.sufficient_cost:
                self.already_successful = True
                logger.debug(
                    "ISL successfully found approximate circuit using provided ansatz only"
                )

        for layer_count in range(self.isl_config.max_layers):
            # Make sure recompilation already hasn't been completed using initial ansatz
            if self.already_successful:
                break

            logger.info(f"Cost before adding layer: {cost}")
            cost = self._add_layer(layer_count)
            cost_history.append(cost)

            if self.save_previous_layer_mps_aer:
                self._add_last_layer_to_ref_circuit(layer_count)
                self._debug_log_optimised_layer(layer_count)
                self._absorb_layer_into_mps_aer()
                # Don't call remove_unnecessary_gates_from_circuit because no gates to remove
                num_2q_gates, num_1q_gates = co.find_num_gates(
                    self.ref_circuit_as_gates, gate_range=g_range(self.ref_circuit_as_gates)
                )
            else:
                self._debug_log_optimised_layer(layer_count)
                if self.remove_unnecessary_gates:
                    co.remove_unnecessary_gates_from_circuit(self.full_circuit, False, False, gate_range=g_range())
                num_2q_gates, num_1q_gates = co.find_num_gates(
                    self.full_circuit, gate_range=g_range()
                )
            
            if self.save_previous_layer_mps_cuquantum:
                self._absorb_layer_into_mps_cuquantum(layer_count)

            if self.save_circuit_history:
                if not self.is_aer_mps_backend:
                    circuit_qasm_string = qasm2.dumps(self.full_circuit)
                else:
                    circuit_copy = self.full_circuit.copy()
                    del circuit_copy.data[0]
                    circuit_qasm_string = qasm2.dumps(circuit_copy)
                circuit_history.append(circuit_qasm_string)

            cinl = self.isl_config.cost_improvement_num_layers
            cit = self.isl_config.cost_improvement_tol
            if len(cost_history) >= cinl and has_stopped_improving(
                cost_history[-1 * cinl:], cit
            ):
                logger.warning("ISL stopped improving")
                break

            if cost < self.isl_config.sufficient_cost:
                logger.info("ISL successfully found approximate circuit")
                break
            elif num_2q_gates >= self.isl_config.max_2q_gates:
                logger.warning("ISL MAX_2Q_GATES reached. Using ROTOSOLVE one last time")
                self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    max_cycles=10,
                    tol=1e-5,
                    stop_val=self.isl_config.sufficient_cost,
                )
                break

        # Perform a final optimisation
        if self.perform_final_minimisation:
            self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": False},
            )

        if self.save_previous_layer_mps_aer:
            # Replace full_circuit with ref_circuit_as_gates, otherwise no way to remove unnecessary gates
            self.full_circuit = self.ref_circuit_as_gates

        if self.remove_unnecessary_gates:
            co.remove_unnecessary_gates_from_circuit(
                self.full_circuit, True, True, gate_range=g_range()
            )

        final_cost = self.evaluate_cost()
        end_time = timeit.default_timer()

        recompiled_circuit = self.get_recompiled_circuit()

        num_2q_gates, num_1q_gates = co.find_num_gates(recompiled_circuit)

        exact_overlap = "Not computable without SV backend"
        if self.is_statevector_backend:
            exact_overlap = co.calculate_overlap_between_circuits(
                self.circuit_to_recompile, co.make_quantum_only_circuit(recompiled_circuit)
            )
        result_dict = {
            "circuit": recompiled_circuit,
            "overlap": 1 - final_cost,
            "exact_overlap": exact_overlap,
            "num_1q_gates": num_1q_gates,
            "num_2q_gates": num_2q_gates,
            "cost_progression": cost_history,
            "circuit_progression": circuit_history,
            "entanglement_measures_progression": self.entanglement_measures_history,
            "e_val_history": self.e_val_history,
            "qubit_pair_history": self.qubit_pair_history,
            "method_history": self.pair_selection_method_history,
            "time_taken": end_time - start_time,
            "cost_evaluations": self.cost_evaluation_counter,
            "coupling_map": self.coupling_map,
            "circuit_qasm": qasm2.dumps(recompiled_circuit)
        }
        if self.save_circuit_history and self.is_aer_mps_backend:
            logger.warning("When using MPS backend, circuit history will not contain the"
                                   " set_matrix_product_state instruction at the start of the circuit")
        logger.info("ISL completed")
        return result_dict

    def _debug_log_optimised_layer(self, layer_count):
        if logger.getEffectiveLevel() == 10:
            logger.debug(f'Qubit pair history: \n{self.qubit_pair_history}')

            if self.debug_log_full_ansatz:
                if self.save_previous_layer_mps_aer:
                    ansatz = self.ref_circuit_as_gates.copy()
                else:
                    ansatz = self.full_circuit.copy()
                del ansatz.data[:len(self.circuit_to_recompile.data)]
                logger.debug(f'Optimised ansatz after layer added: \n{ansatz}')

            layer_added = self._get_layer_added(layer_count)
            if self.initial_single_qubit_layer == True and layer_count == 0:
                logger.debug(f'Optimised layer added: \n{layer_added}')
            else:
                # Remove all qubits apart from the pair acted on in the current layer
                for qubit in range(layer_added.num_qubits - 1, -1, -1):
                    if qubit not in self.qubit_pair_history[-1]:
                        del layer_added.qubits[qubit]
                if not (layer_added.data[2][0].name == 'cx'):
                    logging.error(
                        "Final ansatz layer logging not implemented for custom ansatz or functionalities "
                        "placing more gates after trainable ansatz")
                else:
                    try:
                        logger.debug(f'Optimised layer added: \n{layer_added}')
                    except ValueError:
                        logging.error(
                            "Final ansatz layer logging not implemented for custom ansatz or functionalities "
                            "placing more gates after trainable ansatz")

    def _add_last_layer_to_ref_circuit(self, layer_count):
        # Find the layer which has just been added, and add to ref_circuit_as_gates
        layer_added = self._get_layer_added(layer_count)
        if self.starting_circuit is not None:
            # Add the layer between the previous layer and starting_circuit
            co.add_to_circuit(self.ref_circuit_as_gates, layer_added,
                              location=len(self.ref_circuit_as_gates.data) - len(
                                  self.starting_circuit))
        else:
            # Add the layer to the end of the ansatz
            co.add_to_circuit(self.ref_circuit_as_gates, layer_added)

    def _add_layer(self, index):
        """
        Adds a dressed CNOT gate to the qubits with the highest local
        entanglement. If all qubit pairs have no
        local entanglement, adds a dressed CNOT gate to the qubit pair with
        the highest sum of expectation values
        (computational basis).
        :return: New cost
        """

        ansatz_start_index = self.variational_circuit_range()[0]
        # Define first layer differently when initial_single_qubit_layer=True
        if self.initial_single_qubit_layer and index == 0:
            logger.debug("Starting with first layer comprising of only single qubit rotations")
            rotoselect_gate_indexes = self._add_rotation_to_all_qubits(ansatz_start_index)
        else:
            rotoselect_gate_indexes = self._add_entangling_layer(index)

        if self.use_roto_algos:
            # Do Rotoselect
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_ROTOSELECT,
                tol=self.isl_config.rotoselect_tol,
                stop_val=self.isl_config.sufficient_cost,
                indexes_to_modify=rotoselect_gate_indexes,
            )
            # Do Rotosolve
            if self.isl_config.rotosolve_frequency != 0 and index > 0 and index % self.isl_config.rotosolve_frequency == 0:
                rotosolve_gate_indexes = self._calculate_rotosolve_indices(ansatz_start_index)

                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=self.isl_config.rotosolve_tol,
                    stop_val=self.isl_config.sufficient_cost,
                    indexes_to_modify=rotosolve_gate_indexes,
                )
        else:
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": True},
            )
        return cost

    def _calculate_rotosolve_indices(self, ansatz_start_index):
        num_entangling_layers = (
            self.isl_config.max_layers_to_modify - int(self.initial_single_qubit_layer))
        # This assumes first layer has n gates
        num_gates_in_non_entangling_layer = self.full_circuit.num_qubits * int(
            self.initial_single_qubit_layer)
        # The earliest layer Rotosolve acts on is defined by the user. Calculating the
        # index requires taking into account the first layer potentially being different
        rotosolve_gate_start_index = max(
            ansatz_start_index,
            self.variational_circuit_range()[1]
            - len(self.layer_2q_gate.data) * num_entangling_layers
            - num_gates_in_non_entangling_layer,
        )
        # Don't modify only a fraction of the first layer gates
        first_layer_end_index = ansatz_start_index + num_gates_in_non_entangling_layer
        if ansatz_start_index < rotosolve_gate_start_index < first_layer_end_index:
            rotosolve_gate_start_index = first_layer_end_index
        rotosolve_gate_indexes = (
            rotosolve_gate_start_index,
            (self.variational_circuit_range()[1]),
        )
        return rotosolve_gate_indexes

    def _add_entangling_layer(self, index):
        logger.debug("Finding best qubit pair")
        control, target = self._find_appropriate_qubit_pair()
        logger.debug(f"Best qubit pair found {(control, target)}")
        co.add_to_circuit(
            self.full_circuit,
            self.get_layer_2q_gate(index),
            self.variational_circuit_range()[1],
            qubit_subset=[control, target],
        )
        self.qubit_pair_history.append((control, target))
        # Rotoselect is applied to most recent layer
        rotoselect_gate_indexes = (
            self.variational_circuit_range()[1] - len(self.layer_2q_gate.data),
            (self.variational_circuit_range()[1]),
        )
        return rotoselect_gate_indexes

    def _add_rotation_to_all_qubits(self, ansatz_start_index):
        first_layer = QuantumCircuit(self.full_circuit.num_qubits)
        first_layer.ry(0, range(self.full_circuit.num_qubits))
        co.add_to_circuit(self.full_circuit, first_layer, self.variational_circuit_range()[1])
        self._first_layer_increment_results_dict()
        # Gate indices in the initial layer
        rotoselect_gate_indexes = (
            ansatz_start_index,
            (self.variational_circuit_range()[1]),
        )
        return rotoselect_gate_indexes

    def _find_appropriate_qubit_pair(self):
        if self.isl_config.method == "random":
            rand_index = np.random.randint(len(self.coupling_map))
            self.pair_selection_method_history.append(f"random")
            return self.coupling_map[rand_index]

        if self.isl_config.method == "basic":
            # Choose the qubit pair with the highest reuse priority
            self.pair_selection_method_history.append(f"basic")
            reuse_priorities = self._get_all_qubit_pair_reuse_priorities(1)
            return self.coupling_map[np.argmax(reuse_priorities)]

        if self.isl_config.method == "heuristic":
            return self._find_best_heuristic_qubit_pair()

        if self.isl_config.method == "ISL":
            logger.debug("Computing entanglement of pairs")
            ems = self._get_all_qubit_pair_entanglement_measures()
            self.entanglement_measures_history.append(ems)
            return self._find_best_entanglement_qubit_pair(ems)

        raise ValueError(
            f"Invalid ISL method {self.isl_config.method}. "f"Method must be one of ISL,heuristic,random")

    def _find_best_entanglement_qubit_pair(self, entanglement_measures):
        """
        Returns the qubit pair with the largest entanglement multiplied by the reuse priority of
        that pair.
        """
        reuse_priorities = self._get_all_qubit_pair_reuse_priorities(self.isl_config.entanglement_reuse_exponent)

        # First check if the previous qubit pair was 'bad'
        if len(self.entanglement_measures_history) >= 2 + int(self.initial_single_qubit_layer):
            prev_qp_index = self.coupling_map.index(self.qubit_pair_history[-1])
            pre_em = self.entanglement_measures_history[-2][prev_qp_index]
            post_em = self.entanglement_measures_history[-1][prev_qp_index]
            if post_em >= pre_em:
                logger.debug(
                    f"Entanglement did not reduce for previous pair {self.coupling_map[prev_qp_index]}. "
                    f"Adding to bad qubit pairs list.")
                self.bad_qubit_pairs.append(self.coupling_map[prev_qp_index])
            if len(self.bad_qubit_pairs) > self.isl_config.bad_qubit_pair_memory:
                # Maintain max size of bad_qubit_pairs
                logger.debug(
                    f"Max size of bad qubit pairs reached. Removing {self.bad_qubit_pairs[0]} from list.")
                del self.bad_qubit_pairs[0]

        logger.debug(f"Entanglement of all pairs: {entanglement_measures}")

        # Combine entanglement value with reuse priority
        filtered_ems = [
            entanglement_measure * reuse_priority
            for (entanglement_measure, reuse_priority) in zip(entanglement_measures, reuse_priorities)
        ]

        for qp in set(self.bad_qubit_pairs):
            # Find the number of times this qubit pair has occurred recently
            reps = len(
                [x for x in self.qubit_pair_history[-1 * self.isl_config.bad_qubit_pair_memory:] if
                 x == qp]
            )
            if reps >= 1:
                filtered_ems[self.coupling_map.index(qp)] = -1

        logger.debug(f"Combined priority of all pairs: {filtered_ems}")
        if max(filtered_ems) <= self.isl_config.entanglement_threshold:
            # No local entanglement detected in non-bad qubit pairs;
            # defer to using 'basic' method
            logger.info("No local entanglement detected in non-bad qubit pairs")
            return self._find_best_heuristic_qubit_pair()
        else:
            self.pair_selection_method_history.append(f"ISL")
            # Add 'None' to e_val_history if no expectation values were needed
            self.e_val_history.append(None)
            return self.coupling_map[np.argmax(filtered_ems)]

    def _find_best_heuristic_qubit_pair(self):
        """
        Choose the qubit pair to be the one with the largest expectation value priority multiplied by the reuse 
        priority of that pair.
        @return: The pair of qubits with the highest multiplied e_val priority and reuse priority.
        """
        reuse_priorities = self._get_all_qubit_pair_reuse_priorities(self.isl_config.heuristic_reuse_exponent)

        e_vals = self._measure_qubit_expectation_values()
        self.e_val_history.append(e_vals)

        e_val_sums = self._get_all_qubit_pair_e_val_sums(e_vals)
        logger.debug(f"Summed σ_z expectation values of pairs {e_val_sums}")

        # Mapping from the σz expectation values {1, -1} to the range {0, 1} to make an expectation value based
        # priority. This ensures that the argmax of the list favours qubits close to the |1> state (eigenvalue -1)
        # to apply the next layer to.
        e_val_priorities = [2 - e_val for e_val in e_val_sums]

        logger.debug(f"σ_z expectation value priorities of pairs {e_val_priorities}")
        combined_priorities = [
            e_val_priority * reuse_priority
            for (e_val_priority, reuse_priority) in zip(e_val_priorities, reuse_priorities)
        ]
        logger.debug(f"Combined priorities of pairs {combined_priorities}")
        self.pair_selection_method_history.append(f"heuristic")
        return self.coupling_map[np.argmax(combined_priorities)]

    def _get_all_qubit_pair_entanglement_measures(self):
        entanglement_measures = []
        # Generate MPS from circuit once if using MPS backend
        if self.is_aer_mps_backend:
            circ = self.full_circuit.copy()
            self.circ_mps = mpsops.mps_from_circuit(circ, return_preprocessed=True, sim=self.backend)
        elif self.is_cuquantum_backend:
            if self.starting_circuit is not None:
                self.circ_mps = cu.mps_from_circuit_and_starting_mps(
                    self.starting_circuit, self.cu_cached_mps,
                    self.cu_algorithm)
                self.circ_mps = cu.cu_mps_to_aer_mps(self.circ_mps)
            else:
                self.circ_mps = cu.cu_mps_to_aer_mps(self.cu_cached_mps)
        else:
            self.circ_mps = None
        for control, target in self.coupling_map:
            if not self.is_statevector_backend:
                logger.debug(f"Computing entanglement for pair {(control, target)}")
            this_entanglement_measure = calculate_entanglement_measure(
                self.entanglement_measure_method,
                self.full_circuit,
                control,
                target,
                self.backend,
                self.backend_options,
                self.execute_kwargs,
                self.circ_mps,
            )
            entanglement_measures.append(this_entanglement_measure)
        return entanglement_measures

    def _get_all_qubit_pair_e_val_sums(self, e_vals):
        e_val_sums = []
        for control, target in self.coupling_map:
            e_val_sums.append(e_vals[control] + e_vals[target])
        return e_val_sums

    def _get_all_qubit_pair_reuse_priorities(self, k):
        if not len(self.qubit_pair_history):
            return [1 for _ in range(len(self.coupling_map))]
        priorities = []
        for qp in self.coupling_map:
            if self.isl_config.reuse_priority_mode == "pair":
                priorities.append(self._get_pair_reuse_priority(qp, k))
            elif self.isl_config.reuse_priority_mode == "qubit":
                priorities.append(self._get_qubit_reuse_priority(qp, k))
            else:
                raise ValueError(f"Reuse priority mode must be one of: {['pair','qubit']}")
        logger.debug(f"Reuse priorities of pairs: {priorities}")
        return priorities

    def _find_last_use_of_qubit(self, qubit_pairs, qubit):
        for index, tup in enumerate(qubit_pairs):
            if qubit in tup:
                return index
        return np.inf

    def _get_qubit_reuse_priority(self, qubit_pair, k):
        """
        Priority system based on how recently either of the qubits in a given pair were acted on.
        The priority of a qubit pair (a,b) is given by:
            1. -1 if (a,b) was the last pair acted on
            2. 1 if k=0
            3. 1 if a and b have never been acted on
            4. min[1-2^(-(la+1)/k), 1-2^(-(lb+1)/k)] where la (lb) is the number of layers since
            qubit a (b) has been acted on.

        @param qubit_pair: Tuple where each element is the index of a qubit
        @param k: Constant controlling how heavily recent pairs are disfavoured
        """
        # Hard code that previous pair has priority -1
        if (
            len(self.qubit_pair_history) > 0 + int(self.initial_single_qubit_layer)
            and qubit_pair == self.qubit_pair_history[-1]
        ):
            return -1
        # If not previous pair, then use exponential disfavouring
        elif k == 0:
            return 1
        else:
            qubit_pairs_reversed = self.qubit_pair_history[::-1]
            locs = [self._find_last_use_of_qubit(qubit_pairs_reversed, qubit) for qubit in qubit_pair]
            priorities = [1 - np.exp2(-(loc + 1) / k) for loc in locs]
            return np.min(priorities)

    def _get_pair_reuse_priority(self, qubit_pair, k):
        """
        Priority system based on how recently a specific pair of qubits were acted on.
        The priority of a qubit pair (a,b) is given by:
            1. -1 if (a,b) was the last pair acted on
            2. 1 if k=0
            3. 1 if (a,b) has never been acted on
            4. 1-2^(-l/k) l is the number of layers since the pair (a,b) has been acted on.

        @param qubit_pair: Tuple where each element is the index of a qubit
        @param k: Constant controlling how heavily recent pairs are disfavoured
        """
        # Hard code that previous pair has priority -1
        if (
            len(self.qubit_pair_history) > 0 + int(self.initial_single_qubit_layer)
            and qubit_pair == self.qubit_pair_history[-1]
        ):
            return -1
        # If not previous pair, then use exponential disfavouring
        elif k == 0:
            return 1
        else:
            qubit_pairs_reversed = self.qubit_pair_history[::-1]
            try:
                loc = qubit_pairs_reversed.index(qubit_pair)
                priority = 1 - np.exp2(-loc / k)
                return priority
            except ValueError:
                return 1

    def _measure_qubit_expectation_values(self):
        if self.is_aer_mps_backend:
            return expectation_value_of_qubits_mps(self.full_circuit, self.backend)
        if self.is_cuquantum_backend:
            if self.starting_circuit is not None:
                mps = cu.mps_from_circuit_and_starting_mps(
                    self.starting_circuit, self.cu_cached_mps,
                    self.cu_algorithm)
                mps = cu.cu_mps_to_aer_mps(mps)
            else:
                mps = cu.cu_mps_to_aer_mps(self.cu_cached_mps)
            return [(mpsops.mps_expectation(mps, 'Z', i, already_preprocessed=True))
                      for i in range(len(mps))]
        elif self.local_cost_function:

            output = self._run_full_circuit(add_measurements=not self.is_statevector_backend)

            rel_counts = {k[0: self.total_num_qubits]: v for k, v in output.items()}

            return expectation_value_of_qubits(rel_counts)
        else:
            output = self._run_full_circuit(return_statevector=self.is_statevector_backend)
            return expectation_value_of_qubits(output)

    def _first_layer_increment_results_dict(self):
        self.entanglement_measures_history.append([None])
        self.e_val_history.append(None)
        self.qubit_pair_history.append((None, None))
        self.pair_selection_method_history.append(None)

    def _absorb_layer_into_mps_aer(self):
        """
        Takes the circuit:
        -|0>--|mps(V†(n-1)U|0>)|--|layer_n_gates|--|starting_circuit_inverse|-
        and returns:
        -|0>--|mps(V†(n)U|0>)|--|starting_circuit_inverse|-

        Where mps(V†(k)U|0>) is the set_matrix_product_state instruction representing the state after k layers
        have been added, and layer_n_gates are the optimised gates added in the nth layer.
        """
        # Get full_circuit without starting_circuit on the end
        full_without_starting_circuit = self.full_circuit.copy()
        if self.starting_circuit is not None:
            del full_without_starting_circuit.data[-len(self.starting_circuit):]

        # Get MPS of full_circuit without starting_circuit
        full_circuit_mps = mpsops.mps_from_circuit(full_without_starting_circuit, sim=self.backend)

        # Create circuit with MPS instruction found above, with same registers as full_circuit
        mps_circuit = QuantumCircuit(self.full_circuit.qregs[0])
        mps_circuit.set_matrix_product_state(full_circuit_mps)

        # Replace non-starting_circuit part of full_circuit with its MPS instruction
        if self.starting_circuit is not None:
            del self.full_circuit.data[:-len(self.starting_circuit)]
        else:
            del self.full_circuit.data[:]
        self.full_circuit.data.insert(0, mps_circuit.data[0])

    def _absorb_layer_into_mps_cuquantum(self, index):
        """
        Takes the circuit:
        -|0>--|mps(V†(n-1)U|0>)|--|layer_n_gates|--|starting_circuit_inverse|-
        and returns:
        -|0>--|mps(V†(n)U|0>)|--|starting_circuit_inverse|-

        Overwrites self.cu_cached_mps with MPS calculated using 
        cuquantum with new optimised layer.

        """
        # Get MPS of full_circuit without starting_circuit
        layer_added = self._get_layer_added(index)
        full_circuit_mps = cu.mps_from_circuit_and_starting_mps(layer_added, self.cu_cached_mps, self.cu_algorithm)
        # Replace self.cu_cached_mps with new layer added
        self.cu_cached_mps = full_circuit_mps

    def _get_layer_added(self, layer_count):
        layer_added = self.full_circuit.copy()
        len_layer_added = len(self.layer_2q_gate)
        # Remove starting_circuit from end of ansatz, if there is one
        if self.starting_circuit is not None:
            del layer_added.data[-len(self.starting_circuit.data):]  
        if self.initial_single_qubit_layer == True and layer_count == 0:
            del layer_added.data[:-layer_added.num_qubits]
            return layer_added
        else:
            # Delete all gates apart from the 5 from the added layer
            del layer_added.data[:-len_layer_added]
            return layer_added
