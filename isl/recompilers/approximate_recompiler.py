"""
Contains ApproximateRecompiler
"""
import logging
import multiprocessing
import os
import timeit
from abc import ABC, abstractmethod
from typing import Union

import aqc_research.mps_operations as mpsops
import numpy as np
from aqc_research.mps_operations import mps_from_circuit, check_mps
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_aer import Aer
from qiskit_aer.backends.compatibility import Statevector

import isl.utils.cuquantum_functions as cu
from isl.utils import circuit_operations as co
from isl.utils.circuit_operations import QASM_SIM, DEFAULT_CU_ALGORITHM
from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    remove_classical_operations,
)
from isl.utils.constants import QiskitMPS
from isl.utils.cost_minimiser import CostMinimiser
from isl.utils.utilityfunctions import is_statevector_backend, is_aer_mps_backend, \
    is_cuquantum_backend, expectation_value_of_qubits, expectation_value_of_qubits_mps

logger = logging.getLogger(__name__)


class RecompileInPartsResult:
    def __init__(
        self,
        circuit,
        overlap,
        individual_results,
        time_taken,
    ):
        """
        :param circuit: Resulting circuit.
        :param overlap: 1 - final_global_cost.
        :param individual_results: List of result objects for each sub-recompilation.
        :param time_taken: Total time taken for recompilation.
        """
        self.circuit = circuit
        self.overlap = overlap
        self.individual_results = individual_results
        self.time_taken = time_taken


class ApproximateRecompiler(ABC):
    """
    Variational hybrid quantum-classical algorithm that recompiles a given
    circuit into another circuit. The new circuit
    has the same result when acting on the given input state as the given
    circuit.
    """
    full_circuit: QuantumCircuit

    def __init__(
            self,
            target: QuantumCircuit | QiskitMPS,
            backend,
            execute_kwargs=None,
            initial_state=None,
            qubit_subset=None,
            general_initial_state=False,
            starting_circuit=None,
            optimise_local_cost=False,
            cu_algorithm=None,
    ):
        """
        :param target: Circuit or MPS that is to be recompiled
        :param backend: Backend that is to be used
        :param execute_kwargs: keyword arguments passed down to Qiskit AerBackend.run
        e.g. {'noise_model:NoiseModel, shots=10000}

        :param initial_state: Can be used to define an initial state to compile with respect to
        (as opposed to the default of the |0..0> state). Effectively redefines the cost function as
        C = 1 - |<init|V†U|init>|^2. Similar functionality can be achieved for ISLRecompiler with
        the `starting_circuit` param, but here the solution won't prepare the initial state - it
        assumes the initial state is already prepared. Can be a circuit (QuantumCircuit/Instruction)
        or vector (list/np.ndarray) or None

        :param qubit_subset: The subset of qubits (relative to initial state
        circuit) that target acts
        on. If None, it will be assumed that target and
        initial_state circuit have the same qubits
        :param general_initial_state: Whether recompilation should be for a
        general initial state
        """
        self.target = target
        self.original_circuit_classical_ops = None
        self.backend = (
            backend if backend is not None else Aer.get_backend("qasm_simulator")
        )
        self.cu_algorithm = cu_algorithm if cu_algorithm is not None else DEFAULT_CU_ALGORITHM
        self.is_statevector_backend = is_statevector_backend(self.backend)
        self.is_aer_mps_backend = is_aer_mps_backend(self.backend)
        self.is_cuquantum_backend = is_cuquantum_backend(self.backend)
        if self.is_cuquantum_backend:
            try:
                import cupy
                import cuquantum
                self.cu_target_mps = None
                self.cu_cached_mps = None
            except ModuleNotFoundError as e:
                logger.error(e)
                raise ModuleNotFoundError("cuquantum not installed. Try a different backend.")
        if check_mps(self.target) and not self.is_aer_mps_backend:
            raise Exception("An MPS backend must be used when target is an MPS")
        self.circuit_to_recompile = self.prepare_circuit()
        if hasattr(self.backend, "num_qubits") and self.circuit_to_recompile.num_qubits > self.backend.num_qubits:
            raise ValueError(f"Number of qubits is too large for backend chosen. Maximum is {self.backend.num_qubits}.")
        self.execute_kwargs = self.parse_default_execute_kwargs(execute_kwargs)
        self.backend_options = self.parse_default_backend_options()
        self.initial_state_circuit = co.initial_state_to_circuit(initial_state)
        self.total_num_qubits = self.calculate_total_num_qubits()
        self.qubit_subset_to_recompile = (
            qubit_subset if qubit_subset else list(range(self.total_num_qubits))
        )
        self.general_initial_state = general_initial_state
        self.starting_circuit = starting_circuit
        self.zero_mps = None
        self.optimise_local_cost = optimise_local_cost

        if initial_state is not None and general_initial_state:
            raise ValueError(
                "Can't recompile for general initial state when specific "
                "initial state is provided"
            )

        self.full_circuit, self.lhs_gate_count, self.rhs_gate_count, = self._prepare_full_circuit()
        self.minimizer = CostMinimiser(self.evaluate_cost, self.variational_circuit_range, self.full_circuit)

        # Count number of cost evaluations
        self.cost_evaluation_counter = 0

        self.compiling_finished = False

    def prepare_circuit(self):
        """
        Constructs a circuit from the target which will then be recompiled. This is composed of four
        possible parts:
        1. Remove classical operations from circuit
        2. Transpile circuit to BASIS_GATES
        3. Find MPS representation of target circuit
        4. Create circuit with set_matrix_product_state instruction generating the MPS found in 3.

        For the four combinations of (target, backend), prepare_circuit performs:
        (circuit, non-mps): 1 -> 2 -> return circuit
        (circuit, mps): 1 -> 2 -> 3 -> 4 -> return circuit
        (mps, non-mps): exception will have been raised already
        (mps, mps): 4 -> return circuit
        """
        # Check if target is Aer MPS
        if check_mps(self.target):
            target_mps = self.target
            target_mps_circuit = QuantumCircuit(len(target_mps[0]))
            target_mps_circuit.set_matrix_product_state(target_mps)
            return target_mps_circuit
        else:
            target_copy = self.target.copy()
            self.original_circuit_classical_ops = remove_classical_operations(target_copy)
            qc2 = QuantumCircuit(len(self.target.qubits))
            qc2.append(co.make_quantum_only_circuit(target_copy).to_instruction(), qc2.qregs[0])
            prepared_circuit = co.unroll_to_basis_gates(qc2)
            if self.is_aer_mps_backend:
                logger.info("Pre-computing target circuit as MPS")
                target_mps = mps_from_circuit(prepared_circuit, sim=self.backend)
                target_mps_circuit = QuantumCircuit(prepared_circuit.num_qubits)
                target_mps_circuit.set_matrix_product_state(target_mps)
                # Return a circuit with the target MPS embedded inside
                return target_mps_circuit
            if self.is_cuquantum_backend:
                logger.info("Pre-computing target circuit as MPS")
                # Here we cache the target MPS but don't return a circuit since we can't smuggle a
                # CuQuantum MPS inside a Qiskit QuantumCircuit
                self.cu_target_mps = (
                    cu.mps_from_circuit(prepared_circuit, algorithm=self.cu_algorithm))
                self.cu_cached_mps = self.cu_target_mps.copy()
            return prepared_circuit

    def parse_default_execute_kwargs(self, execute_kwargs):
        kwargs = {} if execute_kwargs is None else dict(execute_kwargs)
        if "shots" not in kwargs:
            if self.backend == "qulacs":
                kwargs["shots"] = 2 ** 30
            elif not self.is_statevector_backend:
                kwargs["shots"] = 8192
            else:
                kwargs["shots"] = 1
        if "optimization_level" not in kwargs:
            kwargs["optimization_level"] = 0
        return kwargs

    def parse_default_backend_options(self):
        backend_options = {}
        if (
                "noise_model" in self.execute_kwargs
                and self.execute_kwargs["noise_model"] is not None
        ):
            backend_options["method"] = "automatic"
        else:
            backend_options["method"] = "automatic"

        try:
            if os.environ["QISKIT_IN_PARALLEL"] == "TRUE":
                # Already in parallel
                backend_options["max_parallel_experiments"] = 1
            else:
                num_threads = multiprocessing.cpu_count()
                backend_options["max_parallel_experiments"] = num_threads
                logger.debug(
                    "Circuits will be evaluated with {} experiments in "
                    "parallel".format(num_threads)
                )
                os.environ["KMP_WARNINGS"] = "0"

        except KeyError:
            logger.debug(
                "No OMP number of threads defined. Qiskit will autodiscover "
                "the number of parallel shots to run"
            )
        return backend_options

    def calculate_total_num_qubits(self):
        if self.initial_state_circuit is None:
            total_num_qubits = self.circuit_to_recompile.num_qubits
        else:
            total_num_qubits = self.initial_state_circuit.num_qubits
        return total_num_qubits

    def variational_circuit_range(self, circuit=None):
        if circuit == None:
            circuit = self.full_circuit
        return self.lhs_gate_count, len(circuit.data) - self.rhs_gate_count

    def rhs_range(self):
        return self.lhs_gate_count, len(self.full_circuit.data)
    
    def layer_added_and_starting_circuit_range(self):
        ansatz = co.extract_inner_circuit(self.full_circuit, self.variational_circuit_range())
        if ansatz.depth() == 1 and len(ansatz) == ansatz.width():
            gates_in_last_layer = ansatz.width()
        else:
            gates_in_last_layer = len(self.layer_2q_gate)
        end = len(self.full_circuit)
        start = end - gates_in_last_layer - self.rhs_gate_count
        return start, end

    def _starting_circuit_range(self):
        end = len(self.full_circuit.data)
        start = end - self.rhs_gate_count
        return start, end

    @abstractmethod
    def recompile(self):
        """
        Run the recompilation algorithm
        :return: Result object (ISLResult, FixedAnsatzResult, RotoselectResult) containing the
        resulting circuit, the overlap between original and resulting circuit, and other optional
        entries (such as circuit parameters).
        """
        raise NotImplementedError(
            "A recompiler must provide implementation for the recompile() " "method"
        )

    def recompile_in_parts(self, max_depth_per_block=10):
        """
        Recompiles the circuit using the following procedure: First break
        the circuit into n subcircuits.
        Then iteratively find an approximation recompilation for the first
        m+1 subcircuits by finding an approximate
        of (approx_circuit_for_first_m_subcircuits + (m+1)th subcircuit)
        :param max_depth_per_block: The maximum allowed depth of each of the
        n subcircuits
        :return: RecompileInPartsResult object
        """
        logger.info("Started partial recompilation")
        start_time = timeit.default_timer()

        all_subcircuits = co.vertically_divide_circuit(
            self.circuit_to_recompile.copy(), max_depth_per_block
        )

        logger.info(f"Circuit was split into {len(all_subcircuits)} parts to compile sequentially")

        last_recompiled_subcircuit = None
        individual_results = []
        for subcircuit in all_subcircuits:
            co.replace_inner_circuit(
                self.full_circuit,
                last_recompiled_subcircuit,
                self.variational_circuit_range(),
                True,
                {"backend": self.backend},
            )
            co.add_to_circuit(
                self.full_circuit,
                subcircuit,
                self.variational_circuit_range()[1],
                True,
                {"backend": self.backend},
            )
            partial_recompilation_result = self.recompile()
            last_recompiled_subcircuit = partial_recompilation_result.circuit
            partial_recompilation_result.circuit = None
            individual_results.append(partial_recompilation_result)
            percentage = (
                    100 * (1 + all_subcircuits.index(subcircuit)) / len(all_subcircuits)
            )
            logger.info(f"Completed {percentage}%  of recompilation")

        end_time = timeit.default_timer()

        result = RecompileInPartsResult(
            circuit=last_recompiled_subcircuit,
            overlap=co.calculate_overlap_between_circuits(
                last_recompiled_subcircuit,
                self.circuit_to_recompile,
                self.initial_state_circuit,
                self.qubit_subset_to_recompile,
            ),
            individual_results=individual_results,
            time_taken=end_time - start_time,
        )

        return result

    def get_recompiled_circuit(self):
        recompiled_circuit = co.circuit_by_inverting_circuit(
            co.extract_inner_circuit(
                self.full_circuit, self.variational_circuit_range()
            )
        )
        if self.starting_circuit is not None:
            transpile_kwargs = {"backend": self.backend} if (
                isinstance(self.backend, Backend)) else None
            co.add_to_circuit(
                recompiled_circuit,
                self.starting_circuit,
                0,
                transpile_before_adding=True,
                transpile_kwargs=transpile_kwargs,
            )
        final_circuit = QuantumCircuit(
            *self.circuit_to_recompile.qregs, *self.circuit_to_recompile.cregs
        )
        qubit_map = {
            full_circ_index: subset_index
            for subset_index, full_circ_index in enumerate(
                self.qubit_subset_to_recompile
            )
        }
        co.add_to_circuit(final_circuit, recompiled_circuit, qubit_subset=qubit_map)

        # If self.target is a QuantumCircuit object, this ensures the quantum and classical registers of the recompiled
        # circuit are the same as those of the target. If self.target is an MPS, there were no registers in the first
        # place, so this makes a QuantumCircuit with the default register names
        if isinstance(self.target, QuantumCircuit):
            final_circuit_original_regs = QuantumCircuit(
                *self.target.qregs, *self.target.cregs
            )
        else:
            final_circuit_original_regs = QuantumCircuit(self.circuit_to_recompile.num_qubits)

        final_circuit_original_regs.append(
            final_circuit.to_instruction(), final_circuit_original_regs.qubits
        )
        circuit_no_classical_ops = co.unroll_to_basis_gates(final_circuit_original_regs)
        if self.original_circuit_classical_ops is not None:
            co.add_classical_operations(
                circuit_no_classical_ops, self.original_circuit_classical_ops
            )
        return circuit_no_classical_ops

    def _prepare_full_circuit(self):
        """Circuit is of form:
        -|0>--|initial_state|--|circuit_to_recompile
        |--|variational_circuit|--|initial_state_inverse|--|(measure)|
        With this circuit, the overlap between circuit_to_recompile and
        inverse of full_circuit
        w.r.t initial_state is just the probability of resulting state
        being in all zero |00...00> state
        If self.general_initial_state is true, circuit takes a different
        form described in the papers below.
        (refer to arXiv:1811.03147, arXiv:1908.04416)
        """
        total_qubits = (
            2 * self.total_num_qubits
            if self.general_initial_state
            else self.total_num_qubits
        )
        qr = QuantumRegister(total_qubits)
        qc = QuantumCircuit(qr)

        # only populate transpile_kwargs is not cuquantum backend
        transpile_kwargs = {"backend": self.backend} if (
            isinstance(self.backend, Backend)) else None

        if self.initial_state_circuit is not None:
            co.add_to_circuit(
                qc,
                self.initial_state_circuit, 
                transpile_before_adding=True,
                transpile_kwargs=transpile_kwargs,
            )
        elif self.general_initial_state:
            for qubit in range(self.total_num_qubits):
                qc.h(qubit)
                qc.cx(qubit, qubit + self.total_num_qubits)

        co.add_to_circuit(
            qc,
            self.circuit_to_recompile,
            transpile_before_adding=False,
            qubit_subset=self.qubit_subset_to_recompile,
        )

        lhs_gate_count = len(qc.data)

        if self.initial_state_circuit is not None:
            isc = co.unroll_to_basis_gates(self.initial_state_circuit)
            co.remove_reset_gates(isc)
            co.add_to_circuit(
                qc,
                isc.inverse(),
                transpile_before_adding=True,
                transpile_kwargs=transpile_kwargs,
            )
        if self.starting_circuit is not None:
            co.add_to_circuit(
                qc,
                self.starting_circuit.inverse(),
                transpile_before_adding=True,
                transpile_kwargs=transpile_kwargs,
            )
        elif self.general_initial_state:
            for qubit in range(self.total_num_qubits - 1, -1, -1):
                qc.cx(qubit, qubit + self.total_num_qubits)
                qc.h(qubit)

        if self.backend == QASM_SIM:
            if self.optimise_local_cost:
                register_size = 2 if self.general_initial_state else 1
                qc.add_register(
                    ClassicalRegister(register_size, name="recompiler_creg")
                )
            else:
                qc.add_register(ClassicalRegister(total_qubits, name="recompiler_creg"))
                [qc.measure(i, i) for i in range(total_qubits)]

        rhs_gate_count = len(qc.data) - lhs_gate_count

        return qc, lhs_gate_count, rhs_gate_count

    def evaluate_cost(self):
        """
        Run the full circuit and evaluate the overlap.
        The cost function is the Loschmidt Echo Test as defined in arXiv:1908.04416.
        "Global" and "local" cost functions refer to equations 9 and 11 respectively,
        (also illustrated in Figure 2 (a) and (b) respectively)
        :return: Cost (float)
        """
        self.cost_evaluation_counter += 1

        if self.optimise_local_cost:
            return self._evaluate_local_cost()
        else:
            return self._evaluate_global_cost()
        
    def _evaluate_local_cost(self):
        if self.is_aer_mps_backend or self.is_cuquantum_backend:
            return self._evaluate_local_cost_mps()
        elif self.is_statevector_backend:
            return self._evaluate_local_cost_sv()
        else:
            return self._evaluate_local_cost_counts()

    def _evaluate_local_cost_mps(self):
        if self.is_cuquantum_backend:
            mps = self._get_full_circ_mps_using_cu()
            e_vals = [(mpsops.mps_expectation(mps, 'Z', i, already_preprocessed=True))
                      for i in range(len(mps))]
        else:
            circ = self.full_circuit.copy()
            e_vals = expectation_value_of_qubits_mps(circ, self.backend)

        cost = 0.5 * (1 - np.mean(e_vals))
        return cost

    def _evaluate_local_cost_sv(self):
        sv = self._run_full_circuit(return_statevector=True)
        e_vals = expectation_value_of_qubits(sv)
        cost = 0.5 * (1 - np.mean(e_vals))
        return cost
    
    def _evaluate_local_cost_counts(self):
        qubit_costs = np.zeros(self.total_num_qubits)
        for i in range(self.total_num_qubits):
            if self.general_initial_state:
                self.full_circuit.measure(i, 0)
                self.full_circuit.measure(i + self.total_num_qubits, 1)
                counts = self._run_full_circuit()
                del self.full_circuit.data[-1]
                del self.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "00" in counts:
                    overlap = counts["00"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
            else:
                self.full_circuit.measure(i, 0)
                counts = self._run_full_circuit()
                del self.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "0" in counts:
                    overlap = counts["0"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
        cost = np.mean(qubit_costs)
        return cost
    
    def _evaluate_global_cost(self):
        if self.is_aer_mps_backend or self.is_cuquantum_backend:
            return self._evaluate_global_cost_mps()
        elif self.is_statevector_backend:
            return self._evaluate_global_cost_sv()
        else:
            return self._evaluate_global_cost_counts()

    def _evaluate_global_cost_mps(self):
        if self.is_cuquantum_backend:
            circ_mps = self._get_full_circ_mps_using_cu()
        else:
            circ = self.full_circuit.copy()
            circ_mps = mpsops.mps_from_circuit(circ, return_preprocessed=True, sim=self.backend)
        if self.zero_mps is None:
            sim = self.backend if self.is_aer_mps_backend else None
            self.zero_mps = mpsops.mps_from_circuit(QuantumCircuit(self.total_num_qubits),
                                                    return_preprocessed=True, sim=sim)

        cost = 1 - np.absolute(
            mpsops.mps_dot(circ_mps, self.zero_mps, already_preprocessed=True))**2
        return cost

    def _get_full_circ_mps_using_cu(self):
        # TODO We use ISL specific logic, so this and where it's called should be in ISLRecompiler
        if self.isl_config.rotosolve_frequency == 0 and not self.compiling_finished:
            # Contract the most recent layer and starting circuit (if using)
            gates_to_contract = co.extract_inner_circuit(self.full_circuit, self.layer_added_and_starting_circuit_range())
            circ_mps = cu.mps_from_circuit_and_starting_mps(gates_to_contract, self.cu_cached_mps,
                                                            self.cu_algorithm)
        else:
            ansatz_circ = co.extract_inner_circuit(self.full_circuit, self.rhs_range())
            circ_mps = cu.mps_from_circuit_and_starting_mps(ansatz_circ, self.cu_target_mps,
                                                            self.cu_algorithm)
        return cu.cu_mps_to_aer_mps(circ_mps)

    def _evaluate_global_cost_sv(self):
        sv1 = self._run_full_circuit(return_statevector=True)
        cost = 1 - (np.absolute(sv1[0])) ** 2
        return cost
    
    def _evaluate_global_cost_counts(self):
        counts = self._run_full_circuit()
        total_qubits = (
            2 * self.total_num_qubits
            if self.general_initial_state
            else self.total_num_qubits
        )
        all_zero_string = "".join(str(int(e)) for e in np.zeros(total_qubits))
        total_shots = sum([each_count for _, each_count in counts.items()])
        # '00...00' might not be present in counts if no shot resulted in
        # the ground state
        if all_zero_string in counts:
            overlap = counts[all_zero_string] / total_shots
        else:
            overlap = 0
        cost = 1 - overlap
        return cost
    
    def _run_full_circuit(self, return_statevector=None, add_measurements=False) -> Union[Counts, Statevector]:
        """
        Run the full circuit
        :rtype: dict or Statevector
        :return: statevector or counts_data or [counts_data] (e.g. counts_data = ['000':10,
        '010':31,'011':20,'110':40])
        """

        # Don't parallelise shots if ISl is already being run in parallel
        already_in_parallel = os.environ["QISKIT_IN_PARALLEL"] == "TRUE"
        backend_options = None if already_in_parallel else self.backend_options

        return_sv = self.is_statevector_backend if None else return_statevector

        output = co.run_circuit_without_transpilation(
            self.full_circuit, self.backend, backend_options, self.execute_kwargs, return_sv
        )

        return output
