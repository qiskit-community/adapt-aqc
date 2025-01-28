"""Contains RotoselectRecompiler"""
import logging
import timeit

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer import Aer

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.utils.constants import QiskitMPS
from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.backends.aer_sv_backend import AerSVBackend, AQCBackend

logger = logging.getLogger(__name__)


class FixedAnsatzResult:
    def __init__(
        self,
        circuit,
        overlap,
        exact_overlap,
        num_1q_gates,
        num_2q_gates,
        params,
        time_taken,
        cost_evaluations,
    ):
        """
        :param circuit: Resulting circuit.
        :param overlap: 1 - final_global_cost.
        :param exact_overlap: Only computable with SV backend.
        :param num_1q_gates: Number of rotation gates in circuit.
        :param num_2q_gates: Number of entangling gates in circuit.
        :param params: Final parameters in the resulting circuit.
        :param time_taken: Total time taken for recompilation.
        :param cost_evaluations: Total number of cost evalutions during recompilation.
        """
        self.circuit = circuit
        self.overlap = overlap
        self.exact_overlap = exact_overlap
        self.num_1q_gates = num_1q_gates
        self.num_2q_gates = num_2q_gates
        self.params = params
        self.time_taken = time_taken
        self.cost_evaluations = cost_evaluations


class FixedAnsatzRecompiler(ApproximateRecompiler):
    """
    Recompiling algorithm that performs recompilation using a fixed ansatz
    """

    def __init__(
        self,
        target: QuantumCircuit | QiskitMPS,
        ansatz_circuit: QuantumCircuit,
        initial_state=None,
        qubit_indexes=None,
        initial_params=None,
        optimization_algorithm_kind=vconstants.ALG_ROTOSOLVE,
        optimization_algorithm_identifier=None,
        optimization_algorithm_kwargs=None,
        backend: AQCBackend = AerSVBackend(),
        execute_kwargs=None,
        general_initial_state=False,
        starting_circuit=None,
        optimise_local_cost=False,
    ):
        """
        :param ansatz_circuit: The ansatz circuit that is to be recompiled to.
        Rotation gates with label FIXED_GATE_LABEL will not be varied
        :param initial_params: (list/np.ndarray) Initial ansatz rotation
        angles.
        Ordering must be same as in FixedAnsatzCircuit and list must not
        include angles for gates that are fixed.
        :param optimization_algorithm_kind: One of the ALG_ constants from
        variationalalgorithms.constants
        :param optimization_algorithm_identifier: Relevant optimization
        algorithm method identifier
        :param general_initial_state: Whether recompilation should be for a
        general initial state
        """
        super().__init__(
            target=target,
            initial_state=initial_state,
            backend=backend,
            qubit_subset=qubit_indexes,
            execute_kwargs=execute_kwargs,
            general_initial_state=general_initial_state,
            starting_circuit=starting_circuit,
            optimise_local_cost=optimise_local_cost,
        )
        self.optimization_algorithm_kwargs = optimization_algorithm_kwargs
        self.optimization_algorithm_kind = optimization_algorithm_kind
        self.optimization_algorithm_identifier = optimization_algorithm_identifier
        if len(ansatz_circuit.qubits) != len(self.qubit_subset_to_recompile):
            raise Exception(
                "Number of qubits in ansatz must equal the number of qubits "
                "in qubit_indexes (if provided) "
                "or else number of qubits in target"
            )
        g_range = self.variational_circuit_range

        ansatz_inv = co.circuit_by_inverting_circuit(ansatz_circuit.copy())
        co.add_to_circuit(
            self.full_circuit,
            ansatz_inv,
            location=g_range()[1],
            transpile_before_adding=True,
            transpile_kwargs={"backend": self.backend.simulator, "optimization_level": 0},
            qubit_subset=self.qubit_subset_to_recompile,
        )
        if initial_params is not None:
            ip_inv = -1 * np.array(initial_params[::-1])
            co.update_angles_in_circuit(self.full_circuit, ip_inv, gate_range=g_range())

    def recompile(self):
        """
        Perform recompilation algorithm
        :return: FixedAnsatzResult object
        """
        logger.info("Fixed ansatz recompilation started")
        start_time = timeit.default_timer()
        self.cost_evaluation_counter = 0

        self.minimizer.minimize_cost(
            algorithm_kind=self.optimization_algorithm_kind,
            algorithm_identifier=self.optimization_algorithm_identifier,
            alg_kwargs=self.optimization_algorithm_kwargs,
            tol=1e-10,
            stop_val=1e-5,
        )
        g_range = self.variational_circuit_range
        num_2q_gates, num_1q_gates = co.find_num_gates(
            self.full_circuit, gate_range=g_range()
        )
        final_angles = co.find_angles_in_circuit(self.full_circuit, g_range())
        final_angles_inv = -1 * np.array(final_angles[::-1])

        final_cost = self.evaluate_cost()

        end_time = timeit.default_timer()
        recompiled_circuit = self.get_recompiled_circuit()
        exact_overlap = co.calculate_overlap_between_circuits(
            self.circuit_to_recompile.to_instruction(),
            recompiled_circuit.to_instruction(),
            initial_state=self.initial_state_circuit,
            qubit_subset=self.qubit_subset_to_recompile,
        )

        result = FixedAnsatzResult(
            circuit=recompiled_circuit,
            overlap=1 - final_cost,
            exact_overlap=exact_overlap,
            num_1q_gates=num_1q_gates,
            num_2q_gates=num_2q_gates,
            params=final_angles_inv,
            time_taken=end_time - start_time,
            cost_evaluations=self.cost_evaluation_counter,
        )

        logger.info("Fixed ansatz recompilation completed")
        return result
