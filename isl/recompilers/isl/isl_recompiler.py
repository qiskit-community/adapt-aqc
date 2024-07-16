"""Contains ISLRecompiler"""

import logging
import pickle
import timeit
import os
from pathlib import Path

import aqc_research.mps_operations as mpsops
import numpy as np
from qiskit import QuantumCircuit, qasm2

from src.simulation.simulator import Simulator

import isl.utils.constants as vconstants
from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.recompilers.isl.isl_config import ISLConfig
from isl.recompilers.isl.isl_result import ISLResult
from isl.utils.utilityfunctions import tenpy_to_qiskit_mps
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
    multi_qubit_gate_depth,
)
import isl.utils.ansatzes as ans

logger = logging.getLogger(__name__)

tenpy_sim = Simulator()


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
        use_rotoselect=True,
        perform_final_minimisation=False,
        optimise_local_cost=False,
        debug_log_full_ansatz=False,
        initial_single_qubit_layer=False,
        cu_algorithm=None,
        tenpy_cut_off=None,
    ):
        """
        :param target: Circuit or MPS that is to be recompiled
        :param entanglement_measure: The entanglement measurement method to
        use for quantifying local entanglement. Valid options are defined in
        entanglement_measures.py.
        :param backend: Backend to run circuits on. Valid options are defined in
        circuit_operations_running.py.
        :param execute_kwargs: keyword arguments passed onto AerBackend.run
        :param coupling_map: 2-qubit gate coupling map to use
        :param isl_config: ISLConfig object
        :param general_initial_state: Recompile circuit for an arbitrary
        initial state.
        :param custom_layer_2q_gate: A two-qubit QuantumCircuit which will be used as the ansatz
        layers.
        :param save_circuit_history: Option to regularly save circuit output as a QASM string to
        results object each time a block is added and optimised
        :param starting_circuit: This circuit will be used as a set of initial fixed gates for the
        recompiled solution. This means that during ISL, the inverse of this circuit will be added
        to the end of V†. WARNING: Using an entangled circuit will lead to worse ISL performance
        because it disrupts the measurement of local entanglement between qubits.
        :param use_roto_algos: Whether to use rotoselect and rotosolve for cost minimisation.
        Disable if custom_layer_2q_gate does not support rotosolve
        :param use_rotoselect: Whether to use rotoselect for cost minimisation. Disable if 
        not appropriate for chosen ansatz.
        :param perform_final_minimisation: Perform a final cost minimisation
        once ISL has ended
        :param optimise_local_cost: Optimise layers with respect to the LLET cost function (arXiv:1908.04416). ISL will still use the global cost function when deciding if compiling is completed.
        :param debug_log_full_ansatz: When True, debug logging will print the entire ansatz at
        every step, as opposed to just the most recently optimised layer.
        :param initial_single_qubit_layer: When True, the first layer of the ISL ansatz will be
        a trainable single-qubit rotation on each qubit.
        :param cu_algorithm: If using the cuquantum backend, this specifies the contract and
        decompose algorithm to use for gate application. Can be either a `dict` or a
        `ContractDecomposeAlgorithm`. If None set, the default used is
        isl.utils.circuit_operations.DEFAULT_CU_ALGORITHM
        :param tenpy_cut_off: Cutoff used by tenpy for singular values if gate acts on more than one
        site.
        """
        super().__init__(
            target=target,
            initial_state=None,
            backend=backend,
            execute_kwargs=execute_kwargs,
            general_initial_state=general_initial_state,
            starting_circuit=starting_circuit,
            optimise_local_cost=optimise_local_cost,
            cu_algorithm=cu_algorithm,
            tenpy_cut_off=tenpy_cut_off
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
        self.remove_unnecessary_gates_during_isl = custom_layer_2q_gate is None
        self.use_roto_algos = use_roto_algos
        self.use_rotoselect = use_rotoselect
        if self.use_rotoselect and custom_layer_2q_gate in [ans.u4(), ans.fully_dressed_cnot(), ans.heisenberg()]:
            logger.warning("For ansatz designed to perform physically motivated or universal operations Rotoselect may "
                           "cause change from expected behaviour")
        if not self.use_rotoselect and (custom_layer_2q_gate == ans.thinly_dressed_cnot() or
                                        custom_layer_2q_gate == ans.identity_resolvable() or
                                        custom_layer_2q_gate is None):
            logger.warning("Rotoselect is necessary for convergence of chosen ansatz")
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
        # Used to keep track of whether ISL/expectation method was used
        self.pair_selection_method_history = []
        self.entanglement_measures_history = []
        self.e_val_history = []
        self.time_taken = None
        self.debug_log_full_ansatz = debug_log_full_ansatz

        self.initial_single_qubit_layer = initial_single_qubit_layer

        if self.is_aer_mps_backend:
            # As variational gates will be absorbed into one large MPS instruction, we need to
            # separately keep track of ansatz gates to return a compiled solution.
            self.layers_saved_to_mps = self.full_circuit.copy()
            del self.layers_saved_to_mps.data[1:]
        
        # Keep track of which layers have not been absorbed into the MPS
        self.layers_as_gates = []

        if self.is_cuquantum_backend:
            self.next_gate_to_cache_index = self.lhs_gate_count

        if self.is_tenpy_backend:
            for (q0, q1) in self.coupling_map:
                if abs(q0 - q1) != 1:
                    logger.warning("When using TenPy backend, a linear coupling map is recommended")
                    break

        self.resume_from_layer = None
        self.prev_checkpoint_time_taken = None

        self.lhs_moved_by_initial_ansatz = 0

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
            for gate in custom_layer_2q_gate:
                if gate[0].label is None and gate[0].name in co.SUPPORTED_1Q_GATES:
                    gate[0].label = gate[0].name
            return custom_layer_2q_gate

    def get_layer_2q_gate(self, layer_index):
        qc = self.layer_2q_gate.copy()
        co.add_subscript_to_all_variables(qc, layer_index)
        return qc

    def recompile(self, initial_ansatz: QuantumCircuit = None, optimise_initial_ansatz=True,
                checkpoint_every=0, checkpoint_dir='checkpoint/', delete_prev_chkpt=False):
        """
        Perform recompilation algorithm.
        :param initial_ansatz: A trial ansatz to start the recompilation
        with instead of starting from scratch
        :param modify_initial_ansatz: If True, optimise the parameters of initial_ansatz when
        intially adding it to the circuit. NOTE: the parameters of initial ansatz will be fixed for
        the rest of recompilation.
        :param checkpoint_every: If checkpoint_every = n != 0, recompiler object will be saved to a
        file after layers 0, n, 2n, ... have been added.
        :param checkpoint_dir: Directory to place checkpoints in. Will be created if not already
        existing.
        :param delete_prev_chkpt: Delete the last checkpoint each time a new one is made.
        Termination criteria: SUFFICIENT_COST reached; max_layers reached;
        std(last_5_costs)/avg(last_5_costs) < TOL
        :return: ISLResult object
        """

        start_time = timeit.default_timer()
        if self.resume_from_layer is None:
            self.time_taken = 0
            start_point = 0
            logger.info("ISL started")
            logger.debug(f"ISL coupling map {self.coupling_map}")
            self.cost_evaluation_counter = 0
            self.global_cost, self.local_cost, num_1q_gates, num_2q_gates, self.cnot_depth = None, None, None, None, None

            self.global_cost_history = []
            if self.optimise_local_cost:
                self.local_cost_history = []
            self.circuit_history = []
            self.cnot_depth_history = []
            self.g_range = self.variational_circuit_range

            # If an initial ansatz has been provided, add that and run minimization
            self.initial_ansatz_already_successful = False
            if initial_ansatz is not None:
                self._add_initial_ansatz(initial_ansatz, optimise_initial_ansatz)

        else:
            start_point = self.resume_from_layer
            self.time_taken = self.prev_checkpoint_time_taken
            logger.info(f"ISL resuming from layer: {start_point}")
            if initial_ansatz is not None:
                logger.warning("An initial ansatz will be ignored when resuming recompilation from a checkpoint")
        
        if checkpoint_every > 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for layer_count in range(start_point, self.isl_config.max_layers):
            if self.initial_ansatz_already_successful:
                break

            logger.info(f"Global cost before adding layer: {self.global_cost}")
            logger.info(f"CNOT depth before adding layer: {self.cnot_depth}")
            if self.optimise_local_cost:
                logger.info(f"Local cost before adding layer: {self.local_cost}")
                self.local_cost = self._add_layer(layer_count)
                self.global_cost = self._evaluate_global_cost()
                self.local_cost_history.append(self.local_cost)
            else:
                self.global_cost = self._add_layer(layer_count)
            self.global_cost_history.append(self.global_cost)
            self.record_cnot_depth()

            # Caching layers as MPS requires that the number of gates remain constant
            if self.remove_unnecessary_gates_during_isl and not (
                self.is_aer_mps_backend or self.is_cuquantum_backend):
                co.remove_unnecessary_gates_from_circuit(self.full_circuit, False, False, gate_range=self.g_range())

            num_2q_gates, num_1q_gates = co.find_num_gates(
                circuit=self.ref_circuit_as_gates if self.is_aer_mps_backend else self.full_circuit,
                gate_range=self.g_range(self.ref_circuit_as_gates if self.is_aer_mps_backend else None)
            )

            if self.save_circuit_history:
                if not self.is_aer_mps_backend:
                    circuit_qasm_string = qasm2.dumps(self.full_circuit)
                else:
                    circuit_copy = self.full_circuit.copy()
                    del circuit_copy.data[0]
                    circuit_qasm_string = qasm2.dumps(circuit_copy)
                self.circuit_history.append(circuit_qasm_string)

            cinl = self.isl_config.cost_improvement_num_layers
            cit = self.isl_config.cost_improvement_tol
            if len(self.global_cost_history) >= cinl and has_stopped_improving(
                self.global_cost_history[-1 * cinl:], cit
            ):
                logger.warning("ISL stopped improving")
                self.compiling_finished = True
                break

            if self.global_cost < self.isl_config.sufficient_cost:
                logger.info("ISL successfully found approximate circuit")
                self.compiling_finished = True
                break
            elif num_2q_gates >= self.isl_config.max_2q_gates:
                logger.warning("ISL MAX_2Q_GATES reached. Using ROTOSOLVE one last time")
                # NOTE this may need changing to use a different stop_val when using local cost
                self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    max_cycles=10,
                    tol=1e-5,
                    stop_val=self.isl_config.sufficient_cost,
                )
                self.compiling_finished = True
                break

            if checkpoint_every > 0 and layer_count % checkpoint_every == 0:
                self.checkpoint(checkpoint_every, checkpoint_dir, delete_prev_chkpt, layer_count,
                                start_time)

        # Perform a final optimisation
        if self.perform_final_minimisation:
            self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": False},
            )

        if self.is_aer_mps_backend:
            # Replace full_circuit with ref_circuit_as_gates, otherwise no way to remove unnecessary gates
            self.full_circuit = self.ref_circuit_as_gates

        # Add initial_ansatz indices back into variational_circuit_range(), if any
        self.lhs_gate_count -= self.lhs_moved_by_initial_ansatz

        co.remove_unnecessary_gates_from_circuit(
            self.full_circuit, True, True, gate_range=self.g_range()
        )

        final_global_cost = self._evaluate_global_cost()
        logger.info(f"Final global cost: {final_global_cost}")
        self.global_cost_history.append(final_global_cost)
        if checkpoint_every > 0:
            self.checkpoint(checkpoint_every, checkpoint_dir, delete_prev_chkpt,
                            len(self.qubit_pair_history) - 1, start_time)
        recompiled_circuit = self.get_recompiled_circuit()

        num_2q_gates, num_1q_gates = co.find_num_gates(recompiled_circuit)
        final_cnot_depth = multi_qubit_gate_depth(recompiled_circuit)
        logger.info(f"Final CNOT depth: {final_cnot_depth}")
        self.cnot_depth_history.append(final_cnot_depth)

        exact_overlap = "Not computable without SV backend"
        if self.is_statevector_backend:
            exact_overlap = co.calculate_overlap_between_circuits(
                self.circuit_to_recompile, co.make_quantum_only_circuit(recompiled_circuit)
            )

        result = ISLResult(
            circuit=recompiled_circuit,
            overlap=1 - final_global_cost,
            exact_overlap=exact_overlap,
            num_1q_gates=num_1q_gates,
            num_2q_gates=num_2q_gates,
            cnot_depth_history=self.cnot_depth_history,
            global_cost_history=self.global_cost_history,
            local_cost_history=self.local_cost_history if self.optimise_local_cost else None,
            circuit_history=self.circuit_history,
            entanglement_measures_history=self.entanglement_measures_history,
            e_val_history=self.e_val_history,
            qubit_pair_history=self.qubit_pair_history,
            method_history=self.pair_selection_method_history,
            time_taken=self.time_taken + (timeit.default_timer() - start_time),
            cost_evaluations=self.cost_evaluation_counter,
            coupling_map=self.coupling_map,
            circuit_qasm=qasm2.dumps(recompiled_circuit),
        )

        if self.save_circuit_history and self.is_aer_mps_backend:
            logger.warning("When using MPS backend, circuit history will not contain the"
                           " set_matrix_product_state instruction at the start of the circuit")
        logger.info("ISL completed")
        return result

    def checkpoint(self, checkpoint_every, checkpoint_dir, delete_prev_chkpt, layer_count,
                   start_time):
        self.resume_from_layer = layer_count + 1
        current_chkpt_time_taken = timeit.default_timer() - start_time
        self.prev_checkpoint_time_taken = self.time_taken + current_chkpt_time_taken
        file_name = str(layer_count)
        with open(os.path.join(checkpoint_dir, file_name), 'wb') as f:
            pickle.dump(self, f)
        if delete_prev_chkpt:
            try:
                os.remove(os.path.join(checkpoint_dir, str(layer_count - checkpoint_every)))
            except FileNotFoundError:
                pass

    def _debug_log_optimised_layer(self, layer_count):
        if logger.getEffectiveLevel() == 10:
            logger.debug(f'Qubit pair history: \n{self.qubit_pair_history}')

            if self.debug_log_full_ansatz:
                if self.is_aer_mps_backend:
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
                try:
                    logger.debug(f'Optimised layer added: \n{layer_added}')
                except ValueError:
                    logging.error(
                        "Final ansatz layer logging not implemented for custom ansatz or functionalities "
                        "placing more gates after trainable ansatz")

    def _add_initial_ansatz(self, initial_ansatz, optimise_initial_ansatz):
        # Label ansatz gates to work with rotosolve
        for gate in initial_ansatz:
            if gate[0].label is None and gate[0].name in co.SUPPORTED_1Q_GATES:
                gate[0].label = gate[0].name

        co.add_to_circuit(
            self.full_circuit,
            co.circuit_by_inverting_circuit(initial_ansatz),
            self.variational_circuit_range()[1],
        )
        if optimise_initial_ansatz:
            if self.use_roto_algos:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=1e-3,
                    stop_val=0 if self.optimise_local_cost else self.isl_config.sufficient_cost,
                    indexes_to_modify=self.variational_circuit_range(),
                )
            else:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_PYBOBYQA,
                    alg_kwargs={"seek_global_minimum": True},
                )
        else:
            cost = self.evaluate_cost()

        self.global_cost = cost if not self.optimise_local_cost else self._evaluate_global_cost()
        self.cnot_depth = multi_qubit_gate_depth(initial_ansatz)

        if self.global_cost < self.isl_config.sufficient_cost:
            self.initial_ansatz_already_successful = True
            logger.debug(
                "ISL successfully found approximate circuit using provided ansatz only"
            )

        if self.is_cuquantum_backend:
            # Cache initial_ansatz
            self._cache_n_gates_mps_cuquantum(n=len(initial_ansatz))
            self.next_gate_to_cache_index += len(initial_ansatz)

        if self.is_aer_mps_backend:
            # Absorb optimised initial_ansatz into MPS and add gates to ref_circuit_as_gates
            gates_absorbed = self._absorb_n_gates_into_mps(n=len(initial_ansatz))
            co.add_to_circuit(self.layers_saved_to_mps, gates_absorbed)
            self._update_reference_circuit()
        else:
            # Ensure initial_ansatz is not modified again
            self.lhs_moved_by_initial_ansatz = len(initial_ansatz)
            self.lhs_gate_count += self.lhs_moved_by_initial_ansatz

    def _add_layer(self, index):
        """
        Adds a dressed CNOT gate or other ansatz layer to the qubits with the 
        highest local entanglement. If all qubit pairs have no
        local entanglement, adds a dressed CNOT gate to the qubit pair with
        the highest sum of expectation values
        (computational basis).
        :return: New cost
        """

        ansatz_start_index = self.variational_circuit_range()[0]
        # Define first layer differently when initial_single_qubit_layer=True
        if self.initial_single_qubit_layer and index == 0:
            logger.debug("Starting with first layer comprising of only single qubit rotations")
            layer_added_optimisation_indexes = self._add_rotation_to_all_qubits()
        else:
            layer_added_optimisation_indexes = self._add_entangling_layer(index)

        if self.optimise_local_cost:
            stop_val = 0
        else:
            stop_val = self.isl_config.sufficient_cost

        if self.use_roto_algos:
            # Optimise layer currently being added
            # For normal layers, use Rotoselect/Rotosolve if self.use_rotoselect=True/False
            # For the initial_single_qubit_layer, use Rotoselect

            if self.use_rotoselect or (self.initial_single_qubit_layer and index == 0):
                ALG = vconstants.ALG_ROTOSELECT
            else:
                ALG = vconstants.ALG_ROTOSOLVE

            cost = self.minimizer.minimize_cost(
                algorithm_kind=ALG,
                tol=self.isl_config.rotoselect_tol,
                stop_val=stop_val,
                indexes_to_modify=layer_added_optimisation_indexes,
            )
            # Do Rotosolve on previous max_layers_to_modify layers, when appropriate
            if self.isl_config.rotosolve_frequency != 0 and index > 0 and index % self.isl_config.rotosolve_frequency == 0:
                multi_layer_optimisation_indexes = (
                    self._calculate_multi_layer_optimisation_indices(
                        ansatz_start_index
                    )
                )
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=self.isl_config.rotosolve_tol,
                    stop_val=stop_val,
                    indexes_to_modify=multi_layer_optimisation_indexes,
                )
        else:
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": True},
            )

        if self.is_aer_mps_backend or self.is_cuquantum_backend:
            self.layers_as_gates.append(index)

            num_layers_to_absorb = self._calculate_num_layers_to_absorb(index)

            # Absorb appropriate layers into MPS, and add their gates to layers_saved_to_mps
            if num_layers_to_absorb > 0:
                includes_isql = self.layers_as_gates[0] == 0 and self.initial_single_qubit_layer

                # Absorb layers into MPS, then add those layers to layers_saved_to_mps
                num_gates_to_absorb = self._get_num_gates_to_cache(n=num_layers_to_absorb, includes_isql=includes_isql)
                if self.is_aer_mps_backend:
                    gates_absorbed = self._absorb_n_gates_into_mps(num_gates_to_absorb)
                    co.add_to_circuit(self.layers_saved_to_mps, gates_absorbed)
                if self.is_cuquantum_backend:
                    self._cache_n_gates_mps_cuquantum(num_gates_to_absorb)
                    self.next_gate_to_cache_index += num_gates_to_absorb

                # Update layers_as_gates
                del self.layers_as_gates[:num_layers_to_absorb]

            if self.is_aer_mps_backend:
                self._update_reference_circuit()

        self._debug_log_optimised_layer(index)

        return cost

    def _calculate_num_layers_to_absorb(self, index):

        layers_since_solve = index % self.isl_config.rotosolve_frequency
        layers_to_next_solve = self.isl_config.rotosolve_frequency - layers_since_solve
        next_rotosolve_layer = index + layers_to_next_solve

        # Compute the index of the leftmost layer to be modified in the next Rotosolve
        lowest_index = next_rotosolve_layer - self.isl_config.max_layers_to_modify + 1

        # All layers with indices below lowest_index can be absorbed
        num_layers_to_absorb = len([i for i in self.layers_as_gates if i < lowest_index])

        return num_layers_to_absorb

    def _update_reference_circuit(self):
        # These are the layers now in circuit form, which are needed to update the reference circuit
        layers_not_saved_to_mps = self.full_circuit.copy()
        del layers_not_saved_to_mps.data[0]

        # Update ref_circuit_as_gates = layers_saved_to_mps + layers_not_saved_to_mps
        self.ref_circuit_as_gates = self.layers_saved_to_mps.copy()
        co.add_to_circuit(self.ref_circuit_as_gates, layers_not_saved_to_mps)

    def _calculate_multi_layer_optimisation_indices(self, ansatz_start_index):
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
        multi_layer_optimisation_indexes = (
            rotosolve_gate_start_index,
            (self.variational_circuit_range()[1]),
        )
        return multi_layer_optimisation_indexes

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
        # Rotoselect or Rotosolve is applied to most recent layer
        layer_added_optimisation_indexes = (
            self.variational_circuit_range()[1] - len(self.layer_2q_gate.data),
            (self.variational_circuit_range()[1]),
        )
        return layer_added_optimisation_indexes

    def _add_rotation_to_all_qubits(self):
        first_layer = QuantumCircuit(self.full_circuit.num_qubits)
        first_layer.ry(0, range(self.full_circuit.num_qubits))
        co.add_to_circuit(self.full_circuit, first_layer, self.variational_circuit_range()[1])
        self._first_layer_increment_results_dict()
        # Gate indices in the initial layer
        initial_layer_optimisation_indexes = (
            self.variational_circuit_range()[1] - self.full_circuit.num_qubits,
            (self.variational_circuit_range()[1]),
        )
        return initial_layer_optimisation_indexes

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

        if self.isl_config.method == "expectation":
            return self._find_best_expectation_qubit_pair()

        if self.isl_config.method == "ISL":
            logger.debug("Computing entanglement of pairs")
            ems = self._get_all_qubit_pair_entanglement_measures()
            self.entanglement_measures_history.append(ems)
            return self._find_best_entanglement_qubit_pair(ems)

        raise ValueError(
            f"Invalid ISL method {self.isl_config.method}. "f"Method must be one of ISL, expectation, random, basic")

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
            return self._find_best_expectation_qubit_pair()
        else:
            self.pair_selection_method_history.append(f"ISL")
            # Add 'None' to e_val_history if no expectation values were needed
            self.e_val_history.append(None)
            return self.coupling_map[np.argmax(filtered_ems)]

    def _find_best_expectation_qubit_pair(self):
        """
        Choose the qubit pair to be the one with the largest expectation value priority multiplied by the reuse
        priority of that pair.
        @return: The pair of qubits with the highest multiplied e_val priority and reuse priority.
        """
        reuse_priorities = self._get_all_qubit_pair_reuse_priorities(self.isl_config.expectation_reuse_exponent)

        e_vals = self._measure_qubit_expectation_values()
        self.e_val_history.append(e_vals)

        e_val_sums = self._get_all_qubit_pair_e_val_sums(e_vals)
        logger.debug(f"Summed σ_z expectation values of pairs {e_val_sums}")

        # Mapping from the σz expectation values {1, -1} to the range {0, 2} to make an expectation value based
        # priority. This ensures that the argmax of the list favours qubits close to the |1> state (eigenvalue -1)
        # to apply the next layer to.
        e_val_priorities = [2 - e_val for e_val in e_val_sums]

        logger.debug(f"σ_z expectation value priorities of pairs {e_val_priorities}")
        combined_priorities = [
            e_val_priority * reuse_priority
            for (e_val_priority, reuse_priority) in zip(e_val_priorities, reuse_priorities)
        ]
        logger.debug(f"Combined priorities of pairs {combined_priorities}")
        self.pair_selection_method_history.append(f"expectation")
        return self.coupling_map[np.argmax(combined_priorities)]

    def _get_all_qubit_pair_entanglement_measures(self):
        entanglement_measures = []
        # Generate MPS from circuit once if using MPS backend
        if self.is_aer_mps_backend:
            circ = self.full_circuit.copy()
            self.circ_mps = mpsops.mps_from_circuit(circ, return_preprocessed=True, sim=self.backend)
        elif self.is_cuquantum_backend:
            self.circ_mps = self._get_full_circ_mps_using_cu()
        elif self.is_tenpy_backend:
            self.circ_mps = self._get_full_circ_mps_using_tenpy()
        else:
            self.circ_mps = None
        for control, target in self.coupling_map:
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
                raise ValueError(f"Reuse priority mode must be one of: {['pair', 'qubit']}")
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
            mps = self._get_full_circ_mps_using_cu()
            return [(mpsops.mps_expectation(mps, 'Z', i, already_preprocessed=True))
                      for i in range(len(mps))]
        if self.is_tenpy_backend:
            mps = self._get_full_circ_mps_using_tenpy()
            return [(mpsops.mps_expectation(mps, 'Z', i, already_preprocessed=True))
                      for i in range(len(mps))]
        elif self.optimise_local_cost:

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

    def _cache_n_gates_mps_cuquantum(self, n):
        """
        Takes the circuit:
        -|0>--|mps(V†(n-1)U|0>)|--|layer_n_gates|--|starting_circuit_inverse|-
        and returns:
        -|0>--|mps(V†(n)U|0>)|--|starting_circuit_inverse|-

        Overwrites self.cu_cached_mps with MPS calculated using
        cuquantum with new optimised layer.

        """
        # Convert n gates to index range
        index_range_to_cache = (self.next_gate_to_cache_index, self.next_gate_to_cache_index + n)
        gates_to_cache = co.extract_inner_circuit(self.full_circuit, index_range_to_cache)
        new_cached_mps = cu.mps_from_circuit_and_starting_mps(gates_to_cache, self.cu_cached_mps, self.cu_algorithm)
        # Replace self.cu_cached_mps with new layer added
        self.cu_cached_mps = new_cached_mps

    def _get_layer_added(self, layer_count):
        layer_added = self.ref_circuit_as_gates.copy() if self.is_aer_mps_backend else self.full_circuit.copy()
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

    def _get_num_gates_to_cache(self, n, includes_isql=False):
        return len(self.layer_2q_gate) * (n - int(includes_isql)) + self.full_circuit.num_qubits * int(includes_isql)
    
    def _absorb_n_gates_into_mps(self, n):
        """
        Takes full_circuit, which consists of a set_matrix_product_state instruction, followed by some number of ISL
        gates and absorbs the first n of these gates (immediately after set_matrix_product_state) into the
        set_matrix_product_state instruction. Also returns a copy of the gates absorbed as a QuantumCircuit.

        In other words it converts full_circuit from this:
        -|0>--|mps(V†(k)U|0>)|--|N ISL gates |--|starting_circuit_inverse|-
        To this:
        -|0>--|mps(V†(k+n)U|0>)|--|N-n ISL gates|--|starting_circuit_inverse|-

        Where mps(V†(k)U|0>) is the set_matrix_product_state instruction representing the state after k gates
        have been added.

        :param n: Number of gates to absorb.
        :return: QuantumCircuit containing a copy of the gates which were absorbed.

        :param n: Number of gates to absorb.
        :return: QuantumCircuit containing a copy of the gates which were absorbed.
        """
        # +1 to include the initial set_matrix_product_state
        num_gates_to_absorb = n + 1

        # Get full_circuit up to and including gates to be absorbed
        circ_to_absorb = self.full_circuit.copy()
        del circ_to_absorb.data[num_gates_to_absorb:]

        # Keep a copy of what was absorbed to add to the reference circuit
        gates_absorbed = circ_to_absorb.copy()
        del gates_absorbed.data[0]

        # Get MPS of circ_to_absorb
        circ_to_absorb_mps = mpsops.mps_from_circuit(circ_to_absorb, sim=self.backend)

        # Create circuit with MPS instruction found above, with same registers as full_circuit
        mps_circuit = QuantumCircuit(self.full_circuit.qregs[0])
        mps_circuit.set_matrix_product_state(circ_to_absorb_mps)

        # Replace absorbed part of full_circuit with its MPS instruction
        num_gates_not_absorbed = len(self.full_circuit.data) - num_gates_to_absorb
        if num_gates_not_absorbed != 0:
            del self.full_circuit.data[:-num_gates_not_absorbed]
        else:
            del self.full_circuit.data[:]
        self.full_circuit.data.insert(0, mps_circuit.data[0])

        return gates_absorbed

    def record_cnot_depth(self):
        if self.is_aer_mps_backend:
            ansatz_circ = co.extract_inner_circuit(self.ref_circuit_as_gates, gate_range=(1, len(self.ref_circuit_as_gates)))
        else:
            start, end = self.variational_circuit_range()
            # Shift start to include initial ansatz
            ansatz_circ = co.extract_inner_circuit(self.full_circuit, gate_range=(start - self.lhs_moved_by_initial_ansatz, end))
        self.cnot_depth = multi_qubit_gate_depth(ansatz_circ)
        self.cnot_depth_history.append(self.cnot_depth)
