"""
Contains ApproximateRecompiler
"""
import logging
import multiprocessing
import os
import timeit
from abc import ABC, abstractmethod

from aqc_research.mps_operations import mps_from_circuit, check_mps
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers import Backend
from qiskit_aer import Aer

from isl.backends.aer_mps_backend import AerMPSBackend
from isl.backends.aqc_backend import AQCBackend
from isl.backends.itensor_backend import ITensorBackend
from isl.backends.qiskit_sampling_backend import QiskitSamplingBackend
from isl.utils import circuit_operations as co
from isl.backends.python_default_backends import QASM_SIM
from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    remove_classical_operations,
)
from isl.utils.constants import QiskitMPS
from isl.utils.cost_minimiser import CostMinimiser
from isl.utils.utilityfunctions import is_statevector_backend, qiskit_to_tenpy_mps, tenpy_chi_1_mps_to_circuit

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
            backend: AQCBackend,
            execute_kwargs=None,
            initial_state=None,
            qubit_subset=None,
            general_initial_state=False,
            starting_circuit=None,
            optimise_local_cost=False,
            soften_global_cost=False,
            itensor_chi=None,
            itensor_cutoff=None,

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
            backend if backend is not None else QASM_SIM
        )
        self.is_statevector_backend = is_statevector_backend(self.backend)
        self.is_aer_mps_backend = isinstance(self.backend, AerMPSBackend)
        if isinstance(self.backend, ITensorBackend):
            logger.warning("ITensor is an experimental backend with many missing features")
            self.itensor_target = None
            self.itensor_chi = itensor_chi
            self.itensor_cutoff = itensor_cutoff
        if check_mps(self.target) and not self.is_aer_mps_backend:
            raise Exception("Aer MPS backend must be used when target is an Aer MPS")
        self.circuit_to_recompile = self.prepare_circuit()
        self.execute_kwargs = self.parse_default_execute_kwargs(execute_kwargs)
        self.backend_options = self.parse_default_backend_options()
        self.initial_state_circuit = co.initial_state_to_circuit(initial_state)
        self.total_num_qubits = self.calculate_total_num_qubits()
        self.qubit_subset_to_recompile = (
            qubit_subset if qubit_subset else list(range(self.total_num_qubits))
        )
        self.general_initial_state = general_initial_state
        self.starting_circuit = self.prepare_starting_circuit(starting_circuit)
        self.zero_mps = mps_from_circuit(QuantumCircuit(self.total_num_qubits),
                                         return_preprocessed=True)
        self.optimise_local_cost = optimise_local_cost
        self.soften_global_cost = soften_global_cost

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
                logger.info("Pre-computing target circuit as MPS using AerSimulator")
                target_mps = mps_from_circuit(prepared_circuit, sim=self.backend.simulator)
                target_mps_circuit = QuantumCircuit(prepared_circuit.num_qubits)
                target_mps_circuit.set_matrix_product_state(target_mps)
                # Return a circuit with the target MPS embedded inside
                return target_mps_circuit
            if isinstance(self.backend, ITensorBackend):
                from itensornetworks_qiskit.utils import qiskit_circ_to_it_circ
                from juliacall import Main as jl
                jl.seval("using ITensorNetworksQiskit")
                logger.info("Pre-computing target circuit as MPS using ITensor")
                gates = qiskit_circ_to_it_circ(prepared_circuit)
                n = self.target.num_qubits
                self.itensor_sites = jl.generate_siteindices_itensors(n)
                self.itensor_target = jl.mps_from_circuit_itensors(n, gates, self.itensor_chi,
                                                                   self.itensor_cutoff,
                                                                   self.itensor_sites)
            return prepared_circuit

    def prepare_starting_circuit(self, starting_circuit):
        if starting_circuit is None or isinstance(starting_circuit, QuantumCircuit):
            return starting_circuit
        elif starting_circuit == "tenpy_product_state":
            if isinstance(self.backend, AerMPSBackend):
                trunc_thr = self.backend.simulator.options.matrix_product_state_truncation_threshold
            else:
                trunc_thr = 1e-8
            tenpy_mps = qiskit_to_tenpy_mps(
                mps_from_circuit(self.circuit_to_recompile.copy(), trunc_thr=trunc_thr)
            )

            compression_options = {
                "compression_method": "variational",
                "trunc_params": {"chi_max": 1},
                "max_trunc_err": 1,
                "max_sweeps": 50,
                "min_sweeps": 10,
            }
            tenpy_mps.compress(compression_options)

            return tenpy_chi_1_mps_to_circuit(tenpy_mps)
        else:
            raise ValueError(
                "starting_circuit must be a QuantumCircuit, None, or string: 'tenpy_product_state'"
            )

    def parse_default_execute_kwargs(self, execute_kwargs):
        kwargs = {} if execute_kwargs is None else dict(execute_kwargs)
        if "shots" not in kwargs:
            if isinstance(self.backend, QiskitSamplingBackend):
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

    def ansatz_range(self):
        return self.lhs_gate_count, len(self.full_circuit.data)

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

        # TODO update this to use new custom backend class
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
            return self.backend.evaluate_local_cost(self)
        else:
            return self.backend.evaluate_global_cost(self)
