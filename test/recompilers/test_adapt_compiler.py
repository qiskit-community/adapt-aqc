import copy
import logging
import os
import pickle
import random
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from aqc_research.model_sp_lhs.trotter.trotter import trotter_circuit
from aqc_research.mps_operations import mps_from_circuit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from tenpy.algorithms import tebd
from tenpy.models import XXZChain
from tenpy.networks.mps import MPS

import adaptaqc.utils.ansatzes as ans
import adaptaqc.utils.circuit_operations as co
from adaptaqc.backends.python_default_backends import SV_SIM, MPS_SIM, QASM_SIM
from adaptaqc.compilers import AdaptConfig, AdaptCompiler
from adaptaqc.utils.constants import DEFAULT_SUFFICIENT_COST
from adaptaqc.utils.entanglement_measures import EM_TOMOGRAPHY_NEGATIVITY
from adaptaqc.utils.utilityfunctions import multi_qubit_gate_depth, tenpy_to_qiskit_mps


def create_initial_ansatz():
    initial_ansatz = QuantumCircuit(4)
    initial_ansatz.ry(0, [0, 1, 2, 3])
    initial_ansatz.cx(0, 1)
    initial_ansatz.cx(1, 2)
    initial_ansatz.cx(2, 3)
    initial_ansatz.rx(0, [0, 1, 2, 3])

    return initial_ansatz


class TestAdapt(TestCase):
    def test_adapt_procedure_sv(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        adapt_compiler = AdaptCompiler(
            qc, backend=SV_SIM, adapt_config=AdaptConfig(sufficient_cost=1e-2)
        )

        result = adapt_compiler.compile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_adapt_procedure_qasm(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        adapt_compiler_qasm = AdaptCompiler(
            qc, backend=QASM_SIM, execute_kwargs={"shots": shots}
        )

        result_qasm = adapt_compiler_qasm.compile()
        approx_circuit_qasm = result_qasm.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit_qasm, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_adapt_procedure_mps(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        adapt_compiler_mps = AdaptCompiler(
            qc, backend=MPS_SIM, execute_kwargs={"shots": shots}
        )

        result_mps = adapt_compiler_mps.compile()
        approx_circuit_mps = result_mps.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit_mps, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_adapt_procedure_when_input_mps_directly(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        mps = mps_from_circuit(qc.copy(), sim=MPS_SIM.simulator)

        # Input MPS for recompilation rather than QuantumCircuit
        compiler = AdaptCompiler(mps, backend=MPS_SIM)

        result = compiler.compile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_GHZ(self):
        qc = QuantumCircuit(5)

        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc = co.unroll_to_basis_gates(qc)

        adapt_compiler = AdaptCompiler(
            qc, backend=SV_SIM, adapt_config=AdaptConfig(sufficient_cost=1e-2)
        )

        result = adapt_compiler.compile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_exact_overlap_close_to_approx_overlap(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        adapt_compiler = AdaptCompiler(qc)

        result = adapt_compiler.compile()
        approx_circuit = result.circuit
        approx_overlap = result.overlap
        exact_overlap = result.exact_overlap
        self.assertAlmostEqual(approx_overlap, exact_overlap, delta=1e-2)

    def test_exact_overlap_calculated_correctly(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        adapt_compiler = AdaptCompiler(qc)

        result = adapt_compiler.compile()
        approx_circuit = result.circuit
        exact_overlap1 = result.exact_overlap
        exact_overlap2 = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertAlmostEqual(exact_overlap1, exact_overlap2, delta=1e-2)

    def test_local_cost_sv(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        adapt_config = AdaptConfig(cost_improvement_num_layers=10)

        adapt_compiler = AdaptCompiler(
            qc, optimise_local_cost=True, backend=SV_SIM, adapt_config=adapt_config
        )
        result = adapt_compiler.compile()
        cost = adapt_compiler.evaluate_cost()
        assert cost < DEFAULT_SUFFICIENT_COST

    def test_custom_layer_gate(self):
        from qiskit import QuantumCircuit

        from adaptaqc.utils.fixed_ansatz_circuits import number_preserving_ansatz

        # Initialize to a supervision of states with bit sum 2
        statevector = [
            0,
            0,
            0,
            -((1 / 3) ** 0.5),
            0,
            1j * (1 / 3) ** 0.5,
            -1 * (1 / 3) ** 0.5,
            0,
        ]
        qc = co.initial_state_to_circuit(statevector)

        initial_circuit = QuantumCircuit(3)
        initial_circuit.x(0)
        initial_circuit.x(1)

        adapt_compiler = AdaptCompiler(
            qc,
            custom_layer_2q_gate=number_preserving_ansatz(2, 1),
            starting_circuit=initial_circuit,
        )

        result = adapt_compiler.compile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_with_initial_ansatz(self):
        from adaptaqc.utils.fixed_ansatz_circuits import hardware_efficient_circuit

        qc = hardware_efficient_circuit(3, "rxrz", 3)

        qc_mod = qc.copy()
        qc_mod.cx(0, 1)
        qc_mod.h(1)
        qc_mod.cx(1, 2)

        adapt_compiler = AdaptCompiler(qc_mod)

        result = adapt_compiler.compile(initial_ansatz=qc)
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc_mod)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_expectation_method(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = AdaptConfig(method="expectation")

        adapt_compiler = AdaptCompiler(qc, adapt_config=config)
        result = adapt_compiler.compile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_basic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = AdaptConfig(method="basic")

        adapt_compiler = AdaptCompiler(qc, adapt_config=config)
        result = adapt_compiler.compile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_random_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = AdaptConfig(method="random")

        adapt_compiler = AdaptCompiler(qc, adapt_config=config)
        result = adapt_compiler.compile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_circuit_with_non_basis_gates_when_recompiling_then_no_error(self):
        qc1 = QuantumCircuit(2)
        qc1.h([0, 1])
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.append(qc1.to_instruction(), qc2.qregs[0])
        compiler = AdaptCompiler(qc2)
        compiler.compile()

    def test_given_starting_circuit_when_compile_with_debug_logging_then_happy(self):
        logging.basicConfig()
        logging.getLogger("adaptaqc").setLevel(logging.DEBUG)

        n = 3

        starting_ansatz_circuit = QuantumCircuit(n)
        starting_ansatz_circuit.x(range(0, n, 2))

        qc = co.create_random_initial_state_circuit(n)

        compiler = AdaptCompiler(qc, starting_circuit=starting_ansatz_circuit)

        compiler.compile()
        logging.getLogger("adaptaqc").setLevel(logging.WARNING)

    def test_given_starting_circuit_when_compile_then_solution_starts_with_it(self):
        n = 2
        starting_ansatz_circuit = QuantumCircuit(n)
        starting_ansatz_circuit.x(0)

        qc = co.create_random_initial_state_circuit(n)

        for boolean in [False, True]:
            compiler = AdaptCompiler(
                qc,
                starting_circuit=starting_ansatz_circuit,
                initial_single_qubit_layer=boolean,
            )

            result = compiler.compile()
            compiled_qc: QuantumCircuit = result.circuit
            del compiled_qc.data[1:]
            overlap = (
                np.abs(
                    np.dot(
                        Statevector(compiled_qc).conjugate(),
                        Statevector(starting_ansatz_circuit),
                    )
                )
                ** 2
            )
            self.assertAlmostEqual(overlap, 1)

    def test_given_two_registers_when_recompiling_then_no_error(self):
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        compiler = AdaptCompiler(qc)
        result = compiler.compile()

    def test_given_two_registers_when_recompiling_then_register_names_preserved(self):
        qr1 = QuantumRegister(2, "reg1")
        qr2 = QuantumRegister(2, "reg2")
        qc = QuantumCircuit(qr1, qr2)
        qc.h(1)
        qc.cx(1, 2)
        qc.x(3)
        compiler = AdaptCompiler(qc)
        result = compiler.compile()
        final_circuit = result.circuit
        assert final_circuit.qregs == qc.qregs

    def test_given_circuit_with_cregs_when_recompiling_then_no_error(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        compiler = AdaptCompiler(qc)
        compiler.compile()

    def test_given_circuit_with_cregs_when_recompiling_then_register_names_preserved(
        self,
    ):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        compiler = AdaptCompiler(qc)
        result = compiler.compile()
        final_circuit = result.circuit
        assert final_circuit.cregs == qc.cregs

    # TODO this test fails when if setting initial_single_qubit_layer=True
    # TODO Not priority fix as unusual case of |00..0> target state and circ with measurements.
    def test_given_circuit_with_measurements_when_recompiling_then_no_error(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=False)
        compiler.compile()

    def test_circuit_output_regularly_saved(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        adapt_compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            execute_kwargs={"shots": shots},
            save_circuit_history=True,
        )

        result = adapt_compiler.compile()
        self.assertTrue(
            len(result.circuit_history) == len(result.global_cost_history) - 1
        )
        self.assertTrue(
            len(result.circuit_history[-1]) > len(result.circuit_history[-2])
        )

    def test_circuit_output_not_saved_when_not_flagged(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        adapt_compiler = AdaptCompiler(
            qc, backend=MPS_SIM, execute_kwargs={"shots": shots}
        )

        result = adapt_compiler.compile()
        self.assertFalse(len(result.circuit_history))

    # TODO See above
    def test_given_circuit_with_one_measurement_when_recompiling_then_preserve_measurement(
        self,
    ):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=False)
        result = compiler.compile()
        assert result.circuit.data[-1] == qc.data[-1]

    # TODO See above
    def test_given_circuit_with_multi_measurement_when_recompiling_then_preserve_measurement(
        self,
    ):
        num_measurements = 3
        qreg = QuantumRegister(num_measurements + 2)
        creg = ClassicalRegister(num_measurements + 2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        for i in range(num_measurements):
            qc.measure(i, i)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=False)
        result = compiler.compile()
        assert result.circuit.data[-num_measurements:] == qc.data[-num_measurements:]

    def test_given_compiler_when_float_cost_improvement_num_layers_then_no_error(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        config = AdaptConfig(cost_improvement_num_layers=4.0, cost_improvement_tol=1)
        compiler = AdaptCompiler(qc, adapt_config=config)
        compiler.compile()

    def test_given_initial_single_qubit_layer_when_compiling_then_then_good_solution(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=True)
        result = compiler.compile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertTrue(overlap > 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_isql_when_compiling_zero_state_then_zero_depth_solution(self):
        qc = QuantumCircuit(3)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=True)
        result = compiler.compile()
        approx_circuit = result.circuit
        self.assertEqual(approx_circuit.depth(), 0, "Depth of solution should be zero")

    def test_given_isql_when_compiling_then_ansatz_starts_with_n_single_qubit_gates(
        self,
    ):
        n = 3
        qc = co.create_random_initial_state_circuit(n)
        config = AdaptConfig(max_layers=2)
        compiler = AdaptCompiler(
            qc, adapt_config=config, initial_single_qubit_layer=True
        )
        result = compiler.compile()

        ansatz_start, ansatz_end = compiler.variational_circuit_range()
        ansatz = compiler.full_circuit[ansatz_start:ansatz_end]
        for instr in ansatz[:n]:
            self.assertIn(instr[0].name, ["rx", "ry", "rz"])

    def test_given_isql_when_compiling_then_results_object_elements_correct_length(
        self,
    ):
        qc = QuantumCircuit(3)
        compiler = AdaptCompiler(qc, initial_single_qubit_layer=True)
        result = compiler.compile()
        self.assertTrue(
            len(result.global_cost_history) - 1
            == len(result.entanglement_measures_history)
            == len(result.e_val_history)
            == len(result.qubit_pair_history)
            == len(result.method_history)
        )

    def test_given_adapt_mode_when_compile_circuit_with_very_small_entanglement_then_expectation_method_used(
        self,
    ):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.crx(1e-15, 0, 1)

        compiler = AdaptCompiler(qc, entanglement_measure=EM_TOMOGRAPHY_NEGATIVITY)
        result = compiler.compile()
        self.assertTrue("expectation" in result.method_history)

    @patch.object(SV_SIM, "measure_qubit_expectation_values")
    def test_given_entanglement_when_find_highest_entanglement_pair_then_evals_not_evaluated(
        self, mock_get_evals
    ):
        compiler = AdaptCompiler(QuantumCircuit(2))
        compiler._find_best_entanglement_qubit_pair([0.5])
        mock_get_evals.assert_not_called()

    @patch.object(SV_SIM, "measure_qubit_expectation_values")
    def test_given_entanglement_when_find_appropriate_pair_then_evals_not_evaluated(
        self, mock_get_evals
    ):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        compiler = AdaptCompiler(qc)
        compiler._find_appropriate_qubit_pair()
        mock_get_evals.assert_not_called()

    def test_given_compiling_with_isql_when_add_layer_then_correct_indices_modified(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3, seed=0)
        num_gates_u = len(qc.data)
        config = AdaptConfig(rotosolve_frequency=1e5)
        compiler = AdaptCompiler(
            qc,
            initial_single_qubit_layer=True,
            adapt_config=config,
        )
        compiler._add_layer(0)
        full_circuit = compiler.full_circuit.copy()
        layer_1_data = full_circuit[num_gates_u:]
        for gate in layer_1_data:
            self.assertTrue(
                gate[0].params[0] != 0, "Added layer should have modified angles"
            )

        compiler._add_layer(1)
        full_circuit = compiler.full_circuit.copy()
        layer_1_and_2_data = full_circuit[num_gates_u:]
        self.assertEqual(
            layer_1_data,
            layer_1_and_2_data[: len(layer_1_data)],
            "Adding next layer and using Rotoselect should not modify previous layer",
        )

        layer_2_data = layer_1_and_2_data[len(layer_1_data) :]
        for gate in layer_2_data:
            if gate[0].name != "cx":
                self.assertTrue(
                    gate[0].params[0] != 0, "Added layer should have modified angles"
                )

    def test_given_random_circuit_and_starting_circuit_True_when_count_gates_in_solution_then_correct(
        self,
    ):
        qc = co.create_random_circuit(3)
        starting_circuit = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, starting_circuit=starting_circuit)
        result = compiler.compile()

        num_1q_gates = 0
        num_2q_gates = 0
        for instr in result.circuit.data:
            if instr.operation.name == "cx":
                num_2q_gates += 1
            elif instr.operation.name == "rx" or "ry" or "rz":
                num_1q_gates += 1

        self.assertEqual(
            (num_1q_gates, num_2q_gates), (result.num_1q_gates, result.num_2q_gates)
        )

    def test_given_wrong_reuse_prio_mode_when_compile_then_error(self):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(reuse_priority_mode="foo")
        compiler = AdaptCompiler(qc, adapt_config=config)
        with self.assertRaises(ValueError):
            compiler.compile()

    def test_when_add_layer_then_previous_pair_reuse_priority_minus_1(self):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(rotosolve_frequency=1e5)
        compiler = AdaptCompiler(
            qc,
            adapt_config=config,
        )
        compiler._add_layer(0)

        pair_acted_on = compiler.qubit_pair_history[0]
        priority = compiler._get_qubit_reuse_priority(pair_acted_on, k=0)

        self.assertEqual(priority, -1)

    def test_given_exponent_equal_to_zero_when_find_reuse_priorities_then_correct(self):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(rotosolve_frequency=1e5)
        compiler = AdaptCompiler(
            qc,
            adapt_config=config,
        )
        compiler._add_layer(0)

        pair_acted_on = compiler.qubit_pair_history[0]
        priorities = compiler._get_all_qubit_pair_reuse_priorities(k=0)

        for pair in compiler.coupling_map:
            if pair != pair_acted_on:
                self.assertEqual(priorities[compiler.coupling_map.index(pair)], 1)

    def test_given_exponent_equal_to_one_when_find_qubit_reuse_priorities_then_correct(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(
            rotosolve_frequency=1e5, reuse_exponent=1, reuse_priority_mode="qubit"
        )
        compiler = AdaptCompiler(
            qc,
            adapt_config=config,
        )
        compiler._add_layer(0)

        pair_acted_on = compiler.qubit_pair_history[0]
        priorities = compiler._get_all_qubit_pair_reuse_priorities(k=1)

        for pair in compiler.coupling_map:
            if pair != pair_acted_on:
                if pair[0] in pair_acted_on or pair[1] in pair_acted_on:
                    self.assertEqual(priorities[compiler.coupling_map.index(pair)], 0.5)
                else:
                    self.assertEqual(priorities[compiler.coupling_map.index(pair)], 1)

    def test_given_random_exponents_when_add_layer_then_same_qubit_pair_never_acted_on_twice_in_a_row(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(
            rotosolve_frequency=1e5,
            reuse_exponent=np.random.rand() * 2,
        )
        compiler = AdaptCompiler(
            qc,
            adapt_config=config,
        )
        compiler._add_layer(0)
        for i in range(10):
            compiler._add_layer(i + 1)
            self.assertTrue(
                compiler.qubit_pair_history[-1] != compiler.qubit_pair_history[-2],
                "Same pair should not be acted on twice",
            )

    def test_given_circuit_when_manually_find_correct_pair_to_act_on_then_pair_acted_on_by_add_layer(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(rotosolve_frequency=1e5, reuse_exponent=1)
        compiler = AdaptCompiler(
            qc,
            adapt_config=config,
        )
        compiler._add_layer(0)

        # Manually find pair which should be acted on when add_layer() is called
        reuse_priorities = compiler._get_all_qubit_pair_reuse_priorities(k=1)
        entanglements = compiler._get_all_qubit_pair_entanglement_measures()
        priorities = [
            reuse_priorities[i] * entanglements[i] for i in range(len(reuse_priorities))
        ]
        correct_pair = compiler.coupling_map[priorities.index(max(priorities))]

        compiler._add_layer(1)

        self.assertTrue(compiler.qubit_pair_history[-1] == correct_pair)

    def test_given_random_rotosolve_frequency_and_max_layers_to_modify_values_when_compile_mps_then_works(
        self,
    ):
        n = 3
        starting_circuit = QuantumCircuit(n)
        starting_circuit.x(range(0, n, 2))
        for isql in [True, False]:
            for sc in [starting_circuit, None, "tenpy_product_state"]:
                qc = co.create_random_initial_state_circuit(n)
                rotosolve_frequency = np.random.randint(1, 101)
                max_layers_to_modify = np.random.randint(1, 101)
                config = AdaptConfig(
                    rotosolve_frequency=rotosolve_frequency,
                    max_layers_to_modify=max_layers_to_modify,
                    cost_improvement_num_layers=100,
                )
                compiler = AdaptCompiler(
                    qc,
                    backend=MPS_SIM,
                    adapt_config=config,
                    starting_circuit=sc,
                    initial_single_qubit_layer=isql,
                )
                result = compiler.compile()
                overlap = co.calculate_overlap_between_circuits(qc, result.circuit)

                self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_mps_backend_when_add_layer_then_num_gates_not_in_mps_is_as_expected(
        self,
    ):
        # Test both cases of using/not using an initial ansatz
        initial_ansatz_circ = create_initial_ansatz()

        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(rotosolve_frequency=4, max_layers_to_modify=3)
        # Rotosolve happens on layers 4, 8, 12...
        # Add layer 0: absorb layer -> 0 gates left in ansatz
        # Add layer 1: absorb layer -> 0 gates
        # Add layer 2: don't absorb layer -> 5 gates
        # Add layer 3: don't absorb layer -> 10 gates
        # Add layer 4: absorb layers 2, 3, 4 -> 0 gates
        # Etc.
        expected_num_gates = [0, 0, 5, 10, 0, 0, 5, 10, 0, 0, 5, 10, 0]
        for initial_ansatz in [initial_ansatz_circ, None]:
            compiler = AdaptCompiler(qc, backend=MPS_SIM, adapt_config=config)
            actual_num_gates = []
            if initial_ansatz is not None:
                # initial_ansatz should be absorbed into MPS and added to ref_circuit_as_gates
                compiler._add_initial_ansatz(
                    initial_ansatz, optimise_initial_ansatz=True
                )
                self.assertEqual(len(compiler.full_circuit), 1)
                self.assertEqual(len(compiler.ref_circuit_as_gates), 12)
            for i in range(13):
                compiler._add_layer(i)
                actual_num_gates.append(len(compiler.full_circuit.data) - 1)

            np.testing.assert_equal(actual_num_gates, expected_num_gates)

    def test_given_max_layers_larger_than_freq_when_add_layer_then_num_gates_not_in_mps_as_expected(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        config = AdaptConfig(rotosolve_frequency=4, max_layers_to_modify=5)
        compiler = AdaptCompiler(qc, backend=MPS_SIM, adapt_config=config)
        # layer counter       0  1   2   3   4  5   6   7   8   9  10  11  12
        expected_num_gates = [5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5]
        actual_num_gates = []
        for i in range(13):
            compiler._add_layer(i)
            actual_num_gates.append(len(compiler.full_circuit.data) - 1)

        np.testing.assert_equal(actual_num_gates, expected_num_gates)

    def test_given_optimise_local_cost_when_compile_then_global_cost_converged(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, optimise_local_cost=True)
        result = compiler.compile()
        circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(qc, circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_optimise_local_cost_when_compile_then_global_and_local_cost_histories_correct(
        self,
    ):
        # Tests that:
        # a) global_cost_history has one extra element (final cost)
        # b) every global cost is greater than its corresponding local cost
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, optimise_local_cost=True)
        result = compiler.compile()
        self.assertEqual(
            len(result.global_cost_history), len(result.local_cost_history) + 1
        )
        for global_cost, local_cost in zip(
            result.global_cost_history[:-1], result.local_cost_history
        ):
            self.assertGreaterEqual(np.round(global_cost, 15), np.round(local_cost, 15))

    def test_given_initial_ansatz_and_starting_circuit_and_isql_and_layer_caching_then_solution_has_correct_gates(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)

        starting_circuit = QuantumCircuit(4)
        starting_circuit.x([0, 1, 2, 3])

        initial_ansatz = create_initial_ansatz()

        config = AdaptConfig(rotosolve_frequency=4, max_layers_to_modify=2)
        compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            adapt_config=config,
            starting_circuit=starting_circuit,
            initial_single_qubit_layer=True,
        )

        compiler._add_initial_ansatz(
            initial_ansatz=initial_ansatz, optimise_initial_ansatz=True
        )
        [compiler._add_layer(i) for i in range(5)]

        # Delete set_matrix_product_state instruction
        del compiler.ref_circuit_as_gates.data[0]
        full_circuit = compiler.ref_circuit_as_gates.copy()

        # First 11 gates should be the same type as in initial_ansatz inverse
        initial_ansatz_part = [gate for gate in full_circuit[:11]]
        self.assertEqual(
            [gate.operation.name for gate in initial_ansatz_part],
            ["rx", "rx", "rx", "rx", "cx", "cx", "cx", "ry", "ry", "ry", "ry"],
        )

        # Next 4 gates should be initial single qubit layer
        isql_part = [gate for gate in full_circuit[11:15]]
        self.assertTrue(
            all(gate.operation.name in ["rx", "ry", "rz"] for gate in isql_part)
        )

        # Everything in between should be thinly-dressed CNOTs
        middle_part = [gate for gate in full_circuit[15:-4]]
        for i, gate in enumerate(middle_part):
            if i % 5 == 2:
                self.assertEqual(gate.operation.name, "cx")
            else:
                self.assertTrue(gate.operation.name in ["rx", "ry", "rz"])
        self.assertEqual(len(middle_part) % 5, 0)

        # Final 4 gates should be starting_circuit inverse
        starting_circuit_part = [gate for gate in full_circuit[-4:]]
        self.assertTrue(
            all(gate.operation.name == "rx" for gate in starting_circuit_part)
        )
        self.assertTrue(
            all(gate.operation.params[0] == np.pi for gate in starting_circuit_part)
        )

        # Make sure the circuit has been partitioned correctly
        reconstruct_circuit = (
            initial_ansatz_part + isql_part + middle_part + starting_circuit_part
        )
        self.assertEqual(compiler.ref_circuit_as_gates.data, reconstruct_circuit)

    def test_given_optimise_initial_ansatz_false_then_initial_ansatz_gates_unchanged(
        self,
    ):
        qc = co.create_random_initial_state_circuit(2)

        initial_ansatz = QuantumCircuit(2)
        initial_ansatz.cz(0, 1)
        initial_ansatz.ry(2.67, 0)
        initial_ansatz.rx(0.53, 1)

        compiler = AdaptCompiler(qc)
        result = compiler.compile(
            initial_ansatz=initial_ansatz, optimise_initial_ansatz=False
        )

        self.assertEqual(result.circuit.data[-3:], initial_ansatz.data)

    def test_given_initial_ansatz_when_add_layer_then_initial_ansatz_unchanged(self):
        qc = co.create_random_initial_state_circuit(4)
        target_gates = len(qc)
        initial_ansatz = create_initial_ansatz()

        config = AdaptConfig(rotosolve_frequency=1, max_layers_to_modify=10)
        compiler = AdaptCompiler(qc, adapt_config=config)

        compiler._add_initial_ansatz(initial_ansatz, optimise_initial_ansatz=True)
        ia_gates_before = [
            gate for gate in compiler.full_circuit[target_gates : (target_gates + 11)]
        ]

        # Rotosolve will occur during layer 1, not layer 0
        compiler._add_layer(0)
        compiler._add_layer(1)
        ia_gates_after = [
            gate for gate in compiler.full_circuit[target_gates : (target_gates + 11)]
        ]

        self.assertEqual(ia_gates_before, ia_gates_after)

    def test_cnot_depth_in_adapt_result_correct(self):
        qc = co.create_random_initial_state_circuit(4, seed=1)
        compiler = AdaptCompiler(qc)
        result = compiler.compile()
        circuit = result.circuit
        self.assertEqual(multi_qubit_gate_depth(circuit), result.cnot_depth_history[-1])

    def test_recompiling_from_tenpy_mps_works(self):
        n = 3
        num_trotter_steps = 5
        dt = 0.4
        # Target from tenpy
        # NOTE: our Hamiltonian with delta and field is equivalent to tenpy's XXZChain model with
        # Jxx = -1, Jz = -delta, hz = -field.
        model = XXZChain(
            {
                "L": n,
                "Jxx": -1.0,
                "Jz": -1.0,
                "hz": 0.0,
                "bc_MPS": "finite",
            }
        )
        neel_state = ["down", "up", "down"]
        psi = MPS.from_product_state(
            model.lat.mps_sites(), neel_state, bc=model.lat.bc_MPS
        )

        tebd_params = {
            "N_steps": num_trotter_steps,
            "dt": dt,
            "order": 2,
            "trunc_params": {"chi_max": 100, "svd_min": 1.0e-12},
        }

        eng = tebd.TEBDEngine(psi, model, tebd_params)
        eng.run()
        target_mps = tenpy_to_qiskit_mps(psi)

        # Compile
        starting_circuit = QuantumCircuit(n)
        starting_circuit.x(range(0, n, 2))

        compiler = AdaptCompiler(
            target_mps, backend=MPS_SIM, starting_circuit=starting_circuit
        )
        result = compiler.compile()

        # Target circuit created independently for comparison
        qc = QuantumCircuit(n)
        qc.x(range(0, n, 2))

        trotter_circuit(
            qc,
            dt=dt,
            delta=1.0,
            field=0.0,
            num_trotter_steps=num_trotter_steps,
            second_order=True,
        )

        overlap = co.calculate_overlap_between_circuits(result.circuit, qc)

        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_general_gradient_method_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)

        config = AdaptConfig(method="general_gradient")
        compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            adapt_config=config,
            custom_layer_2q_gate=ans.identity_resolvable(),
        )
        result = compiler.compile()

        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_general_gradient_when_compile_with_reuse_exponent_then_works(self):
        qc = co.create_random_initial_state_circuit(3)

        config = AdaptConfig(method="general_gradient", reuse_exponent=1)
        compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            adapt_config=config,
            custom_layer_2q_gate=ans.identity_resolvable(),
        )
        result = compiler.compile()

        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_soften_global_cost_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)

        compiler = AdaptCompiler(qc, backend=MPS_SIM, soften_global_cost=True)
        result = compiler.compile()
        self.assertLessEqual(compiler.evaluate_cost(), DEFAULT_SUFFICIENT_COST)

    @patch(
        "adaptaqc.backends.aer_mps_backend.AerMPSBackend.evaluate_hamming_weight_one_overlaps"
    )
    def test_given_soften_global_cost_true_or_false_when_evaluate_cost_then_appropriate_logic_executed(
        self, mock
    ):
        # This test checks that when evaluate_cost is called, the Hamming-weight-one overlaps are
        # calculated if soften_global_cost=True and not calculated if soften_global_cost=False.
        compiler = AdaptCompiler(
            QuantumCircuit(3),
            backend=MPS_SIM,
            soften_global_cost=False,
        )
        compiler.global_cost_history = []
        compiler.evaluate_cost()
        mock.assert_not_called()

        compiler = AdaptCompiler(
            QuantumCircuit(3),
            backend=MPS_SIM,
            soften_global_cost=True,
        )
        compiler.global_cost_history = []
        compiler.evaluate_cost()
        mock.assert_called()

    def test_given_soften_global_cost_and_aer_sv_backend_then_error(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(
            qc,
            backend=SV_SIM,
            soften_global_cost=True,
        )
        with self.assertRaises(NotImplementedError):
            compiler.compile()

    def test_given_soften_global_cost_and_qiskit_sampling_backend_then_error(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(
            qc,
            backend=QASM_SIM,
            soften_global_cost=True,
        )
        with self.assertRaises(NotImplementedError):
            compiler.compile()

    def test_given_tenpy_starting_circuit_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, starting_circuit="tenpy_product_state")
        result = compiler.compile()

        overlap = co.calculate_overlap_between_circuits(result.circuit, qc)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_tenpy_starting_circuit_then_solution_starts_with_rzryrz_on_each_qubit(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, starting_circuit="tenpy_product_state")
        result = compiler.compile()

        qubit_0_gates = []
        qubit_1_gates = []
        qubit_2_gates = []

        for instruction in result.circuit.data:
            if min([len(qubit_0_gates), len(qubit_1_gates), len(qubit_2_gates)]) >= 3:
                break
            elif (instruction.qubits[0] == result.circuit.qubits[0]) and (
                len(qubit_0_gates) < 3
            ):
                qubit_0_gates.append(instruction.operation.name)
            elif (instruction.qubits[0] == result.circuit.qubits[1]) and (
                len(qubit_1_gates) < 3
            ):
                qubit_1_gates.append(instruction.operation.name)
            elif (instruction.qubits[0] == result.circuit.qubits[2]) and (
                len(qubit_2_gates) < 3
            ):
                qubit_2_gates.append(instruction.operation.name)

        self.assertEqual(qubit_0_gates, ["rz", "ry", "rz"])
        self.assertEqual(qubit_1_gates, ["rz", "ry", "rz"])
        self.assertEqual(qubit_2_gates, ["rz", "ry", "rz"])

    def test_given_tenpy_starting_circuit_then_better_starting_cost(self):
        qc = co.create_random_initial_state_circuit(5)
        compiler_1 = AdaptCompiler(qc)
        compiler_2 = AdaptCompiler(qc, starting_circuit="tenpy_product_state")

        cost_1 = compiler_1.evaluate_cost()
        cost_2 = compiler_2.evaluate_cost()

        self.assertGreater(cost_1, cost_2)

    def test_given_advanced_transpilation_option_passed_then_compiled_circuits_equivalent(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        compiler = AdaptCompiler(
            qc,
            use_advanced_transpilation=True,
        )

        result = compiler.compile()
        circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(qc, circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_advanced_transpilation_option_passed_then_reference_circuit_updated_correctly(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            use_advanced_transpilation=True,
        )
        for i in range(3):
            compiler._add_layer(i)
        full_circuit = compiler.full_circuit.copy()
        self.assertEqual(compiler.ref_circuit_as_gates.data, full_circuit.data)


class TestAdaptCheckpointing(TestCase):
    def test_given_checkpoint_every_1_when_compile_then_n_layer_number_of_checkpoints(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            self.assertEqual(len(os.listdir(d)), len(result.qubit_pair_history))

    def test_given_delete_prev_chkpt_when_compile_then_1_checkpoint(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(
                checkpoint_every=1, checkpoint_dir=d, delete_prev_chkpt=True
            )
            self.assertEqual(len(os.listdir(d)), 1)

    def test_given_delete_prev_chkpt_when_save_then_load_then_load_checkpoint_deleted(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, adapt_config=AdaptConfig(max_layers=2))
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(
                checkpoint_every=1, checkpoint_dir=d, delete_prev_chkpt=True
            )
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                loaded_compiler = pickle.load(myfile)
            loaded_compiler.compile(
                checkpoint_every=1, checkpoint_dir=d, delete_prev_chkpt=True
            )
            self.assertEqual(len(os.listdir(d)), 1)

    def test_given_checkpoint_every_large_when_compile_then_2_checkpoints(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=100, checkpoint_dir=d)
            self.assertEqual(len(os.listdir(d)), 2)

    def test_given_checkpoint_every_0_when_compile_then_no_dir_created(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            shutil.rmtree(d)
            compiler.compile(checkpoint_every=0, checkpoint_dir=d)
            self.assertFalse(os.path.isdir(d))

    def test_given_checkpointing_when_compile_then_dir_created(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            shutil.rmtree(d)
            compiler.compile(checkpoint_every=100, checkpoint_dir=d)
            self.assertTrue(os.path.isdir(d))

    def test_given_save_and_resume_from_different_points_then_non_time_results_equal(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            for c in ["0", "1"]:
                with open(os.path.join(d, c + ".pkl"), "rb") as myfile:
                    loaded_compiler = pickle.load(myfile)
                result_1 = loaded_compiler.compile()
                for key in result.__dict__.keys():
                    if key != "time_taken":
                        self.assertEqual(result.__dict__[key], result_1.__dict__[key])

    def test_given_save_and_resume_from_any_point_then_time_taken_within_100ms(self):
        qc = co.create_random_initial_state_circuit(3, seed=3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            int_cn = [int(cn[:-4]) for cn in os.listdir(d)]
            for c in [str(i) for i in range(max(int_cn) + 1)]:
                with open(os.path.join(d, c + ".pkl"), "rb") as myfile:
                    loaded_compiler = pickle.load(myfile)
                result_1 = loaded_compiler.compile()
                self.assertAlmostEqual(
                    result.time_taken, result_1.time_taken, delta=0.1
                )
                self.assertLess(result_1.time_taken, 100)

    def test_given_save_and_resume_and_save_and_resume_then_overwrites(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "0.pkl"), "rb") as myfile:
                loaded_compiler = pickle.load(myfile)
            loaded_compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                loaded_compiler = pickle.load(myfile)
            result = loaded_compiler.compile()
            self.assertEqual(len(os.listdir(d)), len(result.qubit_pair_history))

    def test_given_resume_and_freeze_layers_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        for backend in [SV_SIM, MPS_SIM]:
            compiler = AdaptCompiler(qc, backend=backend)
            with tempfile.TemporaryDirectory() as d:
                compiler.compile(checkpoint_every=1, checkpoint_dir=d)
                with open(os.path.join(d, "2.pkl"), "rb") as myfile:
                    loaded_compiler = pickle.load(myfile)
                result = loaded_compiler.compile(freeze_prev_layers=True)
                overlap = co.calculate_overlap_between_circuits(result.circuit, qc)
                self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_resume_and_freeze_layers_multiple_times_when_compile_then_works(
        self,
    ):
        qc = co.create_random_initial_state_circuit(3)
        sc = QuantumCircuit(3)
        sc.h([0, 2])
        for backend in [SV_SIM, MPS_SIM]:
            compiler = AdaptCompiler(qc, backend=backend, starting_circuit=sc)
            with tempfile.TemporaryDirectory() as d:
                compiler.compile(checkpoint_every=1, checkpoint_dir=d)
                with open(os.path.join(d, "0.pkl"), "rb") as myfile:
                    # Load compiler after layer 0, freeze layer 0, compile
                    loaded_compiler_0 = pickle.load(myfile)

            with tempfile.TemporaryDirectory() as d:
                loaded_compiler_0.compile(
                    checkpoint_every=1, checkpoint_dir=d, freeze_prev_layers=True
                )
                with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                    # Load compiler after layer 1, freeze layers 0, 1, compile
                    loaded_compiler_1 = pickle.load(myfile)

            with tempfile.TemporaryDirectory() as d:
                loaded_compiler_1.compile(
                    checkpoint_every=1, checkpoint_dir=d, freeze_prev_layers=True
                )
                with open(os.path.join(d, "2.pkl"), "rb") as myfile:
                    # Load compiler after layer 2, freeze layers 0, 1, 2, compile
                    loaded_compiler_2 = pickle.load(myfile)

            result = loaded_compiler_2.compile(freeze_prev_layers=True)
            overlap = co.calculate_overlap_between_circuits(result.circuit, qc)
            self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_freeze_prev_layers_then_parameters_unchanged_sv(self):
        # This tests that given freeze_prev_layers=False(True), then the layers added before the
        # checkpoint are(are not) changed during the resumed recompilation.
        qc = co.create_random_initial_state_circuit(3, seed=0)
        target_length = len(qc)
        # We will load the compiler after two layers have been added, so if we freeze those layers,
        # these gates should be in the range:
        frozen_gate_range = (target_length, target_length + 10)
        compiler = AdaptCompiler(qc, backend=SV_SIM)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                compiler_freeze = pickle.load(myfile)
                compiler_no_freeze = copy.deepcopy(compiler_freeze)

            layers_added_before_checkpoint = co.extract_inner_circuit(
                compiler_freeze.full_circuit, frozen_gate_range
            )
            compiler_freeze.compile(freeze_prev_layers=True)
            compiler_no_freeze.compile(freeze_prev_layers=False)

            layers_after_recompiling_with_freezing = co.extract_inner_circuit(
                compiler_freeze.full_circuit, frozen_gate_range
            )
            layers_after_recompiling_without_freezing = co.extract_inner_circuit(
                compiler_no_freeze.full_circuit, frozen_gate_range
            )

            self.assertEqual(
                layers_added_before_checkpoint, layers_after_recompiling_with_freezing
            )
            self.assertNotEqual(
                layers_added_before_checkpoint,
                layers_after_recompiling_without_freezing,
            )

    def test_given_freeze_prev_layers_then_parameters_unchanged_mps(self):
        # This test is the same above, but for the aer mps backend.
        qc = co.create_random_initial_state_circuit(3)
        # For mps backend, the target is a set_matrix_product_state in compiler.ref_circuit_as_gates
        frozen_gate_range = (1, 11)
        compiler = AdaptCompiler(qc, backend=MPS_SIM)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                compiler_freeze = pickle.load(myfile)
                compiler_no_freeze = copy.deepcopy(compiler_freeze)

            layers_added_before_checkpoint = co.extract_inner_circuit(
                compiler_freeze.ref_circuit_as_gates, frozen_gate_range
            )
            compiler_freeze.compile(freeze_prev_layers=True)
            compiler_no_freeze.compile(freeze_prev_layers=False)

            layers_after_recompiling_with_freezing = co.extract_inner_circuit(
                compiler_freeze.ref_circuit_as_gates, frozen_gate_range
            )
            layers_after_recompiling_without_freezing = co.extract_inner_circuit(
                compiler_no_freeze.ref_circuit_as_gates, frozen_gate_range
            )

            self.assertEqual(
                layers_added_before_checkpoint, layers_after_recompiling_with_freezing
            )
            self.assertNotEqual(
                layers_added_before_checkpoint,
                layers_after_recompiling_without_freezing,
            )

    def test_given_freeze_prev_layers_then_lhs_gate_count_different_from_orig_during_recompiling(
        self,
    ):
        # When doing rotosolve, AdaptCompiler._calculate_multi_layer_optimisation_indices is called
        # with "ansatz_start_index" as argument. This is equal to variational_circuit_range()[0],
        # which is equal to lhs_gate_count. We can check that this is different from
        # original_lhs_gate_count
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                loaded_compiler = pickle.load(myfile)

        # Since we loaded the compiler after two layers had been added, we expect the
        # lhs_gate_count used during recompilation to be: old_lhs + 10
        expected_input = loaded_compiler.original_lhs_gate_count + 10

        with patch.object(
            loaded_compiler, "_calculate_multi_layer_optimisation_indices"
        ) as mock:
            # Dummy return value
            mock.return_value = loaded_compiler.variational_circuit_range()
            # Compile and assert that all calls to the function use the right input
            loaded_compiler.compile(freeze_prev_layers=True)
            for call in mock.call_args_list:
                self.assertEqual(call.args[0], expected_input)

    def test_given_save_and_resume_then_rotosolve_fraction_is_not_overwritten(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(qc, rotosolve_fraction=0.5)

        rotosolve_fractions = []
        # Before recompilation
        rotosolve_fractions.append(compiler.minimizer.rotosolve_fraction)
        with tempfile.TemporaryDirectory() as d:
            compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            # After recompilation
            rotosolve_fractions.append(compiler.minimizer.rotosolve_fraction)
            with open(os.path.join(d, "1.pkl"), "rb") as myfile:
                loaded_compiler = pickle.load(myfile)

            # After loading, before resuming recompilation
            rotosolve_fractions.append(loaded_compiler.minimizer.rotosolve_fraction)
            loaded_compiler.compile(checkpoint_every=1, checkpoint_dir=d)
            # After loading, after resuming recompilation
            rotosolve_fractions.append(loaded_compiler.minimizer.rotosolve_fraction)

        self.assertEqual(rotosolve_fractions, [0.5, 0.5, 0.5, 0.5])


class TestAdaptRandomRotosolve(TestCase):
    def test_given_different_rotosolve_fractions_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)

        for rotosolve_fraction in [0.2, 0.5, 0.8]:
            compiler = AdaptCompiler(
                qc, backend=MPS_SIM, rotosolve_fraction=rotosolve_fraction
            )
            result = compiler.compile()

            overlap = co.calculate_overlap_between_circuits(qc, result.circuit)

            self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_rotosolve_fraction_then_results_reproducible(self):
        qc = co.create_random_initial_state_circuit(3)

        compiler_1 = AdaptCompiler(qc, backend=MPS_SIM, rotosolve_fraction=0.5)
        compiler_2 = AdaptCompiler(qc, backend=MPS_SIM, rotosolve_fraction=0.5)

        random.seed(1)
        result_1 = compiler_1.compile()

        random.seed(1)
        result_2 = compiler_2.compile()

        self.assertEqual(result_1.global_cost_history, result_2.global_cost_history)
        self.assertEqual(result_1.circuit, result_2.circuit)

    def test_given_invalid_or_valid_rotosolve_fraction_then_error_or_no_error(self):
        qc = co.create_random_initial_state_circuit(3)

        # Should error
        with self.assertRaises(ValueError):
            compiler = AdaptCompiler(qc, backend=MPS_SIM, rotosolve_fraction=0)

        with self.assertRaises(ValueError):
            compiler = AdaptCompiler(
                qc, backend=MPS_SIM, rotosolve_fraction=1.000000001
            )

        # Shouldn't error
        compiler = AdaptCompiler(qc, backend=MPS_SIM, rotosolve_fraction=1)
        compiler = AdaptCompiler(qc, backend=MPS_SIM, rotosolve_fraction=0.000000001)


try:
    from itensornetworks_qiskit.utils import qiskit_circ_to_it_circ
    from juliacall import Main as jl, JuliaError
    from adaptaqc.backends.julia_default_backends import ITENSOR_SIM

    jl.seval("using ITensorNetworksQiskit")
    jl.seval("using ITensors: siteinds")
    module_failed = False
except Exception:
    module_failed = True


class TestITensor(TestCase):
    def setUp(self):
        if module_failed:
            self.skipTest("Skipping as ITensor backend not set up")

    def test_given_itensor_backend_when_compile_with_basic_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = transpile(qc, basis_gates=["cx", "rx", "ry", "rz"])
        config = AdaptConfig(method="basic")
        compiler = AdaptCompiler(qc, backend=ITENSOR_SIM, adapt_config=config)
        result = compiler.compile()
        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_itensor_backend_when_compile_with_adapt_then_error(self):
        with self.assertRaises(NotImplementedError):
            AdaptCompiler(QuantumCircuit(1), backend=ITENSOR_SIM).compile()

    def test_given_itensor_backend_when_compile_with_expectation_then_error(self):
        with self.assertRaises(NotImplementedError):
            config = AdaptConfig(method="expectation")
            AdaptCompiler(
                QuantumCircuit(1), backend=ITENSOR_SIM, adapt_config=config
            ).compile()

    def test_given_itensor_backend_then_target_cached(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = transpile(qc, basis_gates=["cx", "rx", "ry", "rz"])
        compiler = AdaptCompiler(qc, backend=ITENSOR_SIM)

        s = compiler.itensor_sites
        cached_target = compiler.itensor_target
        gates = qiskit_circ_to_it_circ(qc)
        actual_target = jl.mps_from_circuit_itensors(3, gates, 10, s)

        overlap = jl.overlap_itensors(cached_target, actual_target)
        self.assertAlmostEqual(overlap, 1)

    def test_given_itensor_backend_then_cache_not_modified(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = transpile(qc, basis_gates=["cx", "rx", "ry", "rz"])
        config = AdaptConfig(method="basic")
        compiler = AdaptCompiler(qc, backend=ITENSOR_SIM, adapt_config=config)
        cached_target = compiler.itensor_target
        compiler._add_layer(0)
        compiler._add_layer(1)
        compiler._add_layer(2)

        cached_target_after_layers_added = compiler.itensor_target

        overlap = jl.overlap_itensors(cached_target, cached_target_after_layers_added)
        self.assertAlmostEqual(overlap, 1)

    def test_given_soften_global_cost_and_itensor_backend_then_error(self):
        qc = co.create_random_initial_state_circuit(3)
        compiler = AdaptCompiler(
            qc,
            backend=ITENSOR_SIM,
            soften_global_cost=True,
        )
        with self.assertRaises(NotImplementedError):
            compiler.compile()


class TestBrickwall(TestCase):
    def test_given_brickwall_pair_selection_method_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        config = AdaptConfig(method="brickwall")
        compiler = AdaptCompiler(qc, adapt_config=config)

        result = compiler.compile()

        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)

        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_brickwall_mode_and_all_options_when_compile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        starting_circuit = QuantumCircuit(3)
        starting_circuit.x([0, 2])
        initial_ansatz = QuantumCircuit(3)
        initial_ansatz.ry(0.5, 0)

        config = AdaptConfig(
            cost_improvement_num_layers=50,
            max_layers_to_modify=5,
            method="brickwall",
            rotosolve_frequency=3,
        )
        compiler = AdaptCompiler(
            qc,
            backend=MPS_SIM,
            adapt_config=config,
            custom_layer_2q_gate=ans.identity_resolvable(),
            starting_circuit=starting_circuit,
            rotosolve_fraction=0.8,
            soften_global_cost=True,
            initial_single_qubit_layer=True,
        )

        result = compiler.compile(
            initial_ansatz=initial_ansatz, optimise_initial_ansatz=True
        )

        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)

        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_brickwall_mode_then_qubit_pair_history_correct(self):
        # Odd number of qubits
        qc = QuantumCircuit(5)
        expected_order = [(0, 1), (2, 3), (1, 2), (3, 4)]
        config = AdaptConfig(max_layers=10, method="brickwall")
        compiler = AdaptCompiler(qc, adapt_config=config)
        [compiler._add_layer(i) for i in range(5 * len(expected_order))]
        for i, pair in enumerate(compiler.qubit_pair_history):
            expected_pair = expected_order[i % len(expected_order)]
            self.assertEqual(pair, expected_pair)

        # Even number of qubits
        qc = QuantumCircuit(4)
        expected_order = [(0, 1), (2, 3), (1, 2)]
        config = AdaptConfig(max_layers=10, method="brickwall")
        compiler = AdaptCompiler(qc, adapt_config=config)
        [compiler._add_layer(i) for i in range(5 * len(expected_order))]
        for i, pair in enumerate(compiler.qubit_pair_history):
            expected_pair = expected_order[i % len(expected_order)]
            self.assertEqual(pair, expected_pair)

    def test_given_two_qubits_and_brickwall_mode_then_works(self):
        qc = co.create_random_initial_state_circuit(2)
        config = AdaptConfig(method="brickwall")
        compiler = AdaptCompiler(qc, adapt_config=config)
        result = compiler.compile()
        for pair in result.qubit_pair_history:
            self.assertEqual(pair, (0, 1))

    def test_given_less_than_two_qubits_and_brickwall_mode_then_error(self):
        qc = QuantumCircuit(1)
        config = AdaptConfig(method="brickwall")
        compiler = AdaptCompiler(qc, adapt_config=config)
        with self.assertRaises(ValueError):
            result = compiler.compile()
