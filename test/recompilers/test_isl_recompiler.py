import logging
import os
import pickle
import shutil
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from aqc_research.mps_operations import mps_from_circuit, mps_dot
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import isl.utils.circuit_operations as co
from isl.recompilers import ISLConfig, ISLRecompiler
from isl.utils.circuit_operations import QASM_SIM, SV_SIM, MPS_SIM, CUQUANTUM_SIM
from isl.utils.constants import DEFAULT_SUFFICIENT_COST
from isl.utils.entanglement_measures import EM_TOMOGRAPHY_NEGATIVITY


class TestISL(TestCase):

    def test_isl_procedure_sv(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend=SV_SIM,
                                       isl_config=ISLConfig(sufficient_cost=1e-2))

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_isl_procedure_qasm(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler_qasm = ISLRecompiler(qc, backend=QASM_SIM, execute_kwargs={'shots': shots})

        result_qasm = isl_recompiler_qasm.recompile()
        approx_circuit_qasm = result_qasm.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit_qasm, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_isl_procedure_mps(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler_mps = ISLRecompiler(qc, backend=MPS_SIM, execute_kwargs={'shots': shots})

        result_mps = isl_recompiler_mps.recompile()
        approx_circuit_mps = result_mps.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit_mps, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_isl_procedure_when_input_mps_directly(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        mps = mps_from_circuit(qc.copy(), sim=MPS_SIM)

        # Input MPS for recompilation rather than QuantumCircuit
        recompiler = ISLRecompiler(mps, backend=MPS_SIM)

        result = recompiler.recompile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_GHZ(self):
        qc = QuantumCircuit(5)

        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend=SV_SIM,
                                       isl_config=ISLConfig(sufficient_cost=1e-2))

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_exact_overlap_close_to_approx_overlap(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit
        approx_overlap = result.overlap
        exact_overlap = result.exact_overlap
        self.assertAlmostEquals(approx_overlap, exact_overlap, delta=1e-2)

    def test_exact_overlap_calculated_correctly(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit
        exact_overlap1 = result.exact_overlap
        exact_overlap2 = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertAlmostEquals(exact_overlap1, exact_overlap2, delta=1e-2)

    def test_local_cost_sv(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        isl_config = ISLConfig(cost_improvement_num_layers=10)

        isl_recompiler = ISLRecompiler(
            qc, optimise_local_cost=True, backend=SV_SIM, isl_config=isl_config
        )
        result = isl_recompiler.recompile()
        cost = isl_recompiler.evaluate_cost()
        assert cost < DEFAULT_SUFFICIENT_COST

    def test_custom_layer_gate(self):
        from qiskit import QuantumCircuit

        from isl.utils.fixed_ansatz_circuits import number_preserving_ansatz

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

        isl_recompiler = ISLRecompiler(
            qc,
            custom_layer_2q_gate=number_preserving_ansatz(2, 1),
            starting_circuit=initial_circuit,
        )

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_with_initial_ansatz(self):
        from isl.utils.fixed_ansatz_circuits import hardware_efficient_circuit

        qc = hardware_efficient_circuit(3, "rxrz", 3)

        qc_mod = qc.copy()
        qc_mod.cx(0, 1)
        qc_mod.h(1)
        qc_mod.cx(1, 2)

        isl_recompiler = ISLRecompiler(qc_mod)

        result = isl_recompiler.recompile_using_initial_ansatz(qc)
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc_mod)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_heuristic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="heuristic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_basic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="basic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_random_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="random")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_given_circuit_with_non_basis_gates_when_recompiling_then_no_error(self):
        qc1 = QuantumCircuit(2)
        qc1.h([0, 1])
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.append(qc1.to_instruction(), qc2.qregs[0])
        recompiler = ISLRecompiler(qc2)
        recompiler.recompile()

    def test_given_starting_circuit_when_recompile_with_debug_logging_then_happy(self):
        logging.basicConfig()
        logging.getLogger('isl').setLevel(logging.DEBUG)

        n = 3

        starting_ansatz_circuit = QuantumCircuit(n)
        starting_ansatz_circuit.x(range(0, n, 2))

        qc = co.create_random_initial_state_circuit(n)

        recompiler = ISLRecompiler(qc, starting_circuit=starting_ansatz_circuit)

        recompiler.recompile()
        logging.getLogger('isl').setLevel(logging.WARNING)

    def test_given_starting_circuit_when_recompile_then_solution_starts_with_it(self):

        n = 2
        starting_ansatz_circuit = QuantumCircuit(n)
        starting_ansatz_circuit.x(0)

        qc = co.create_random_initial_state_circuit(n)

        for boolean in [False, True]:
            recompiler = ISLRecompiler(qc, starting_circuit=starting_ansatz_circuit, initial_single_qubit_layer=boolean)

            result = recompiler.recompile()
            compiled_qc: QuantumCircuit = result.circuit
            del compiled_qc.data[1:]
            overlap = np.abs(np.dot(Statevector(compiled_qc).conjugate(),
                                    Statevector(starting_ansatz_circuit))) ** 2
            self.assertAlmostEquals(overlap, 1)

    def test_given_two_registers_when_recompiling_then_no_error(self):
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()

    def test_given_two_registers_when_recompiling_then_register_names_preserved(self):
        qr1 = QuantumRegister(2, "reg1")
        qr2 = QuantumRegister(2, "reg2")
        qc = QuantumCircuit(qr1, qr2)
        qc.h(1)
        qc.cx(1, 2)
        qc.x(3)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        final_circuit = result.circuit
        assert final_circuit.qregs == qc.qregs

    def test_given_circuit_with_cregs_when_recompiling_then_no_error(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        recompiler = ISLRecompiler(qc)
        recompiler.recompile()

    def test_given_circuit_with_cregs_when_recompiling_then_register_names_preserved(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
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
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=False)
        recompiler.recompile()

    def test_circuit_output_regularly_saved(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler = ISLRecompiler(qc, backend=MPS_SIM, execute_kwargs={'shots': shots}, save_circuit_history=True)

        result = isl_recompiler.recompile()
        self.assertTrue(len(result.circuit_history) == len(result.global_cost_history))
        self.assertTrue(len(result.circuit_history[-1]) > len(result.circuit_history[-2]))

    def test_circuit_output_not_saved_when_not_flagged(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler = ISLRecompiler(qc, backend=MPS_SIM, execute_kwargs={'shots': shots})

        result = isl_recompiler.recompile()
        self.assertFalse(len(result.circuit_history))

    # TODO See above
    def test_given_circuit_with_one_measurement_when_recompiling_then_preserve_measurement(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=False)
        result = recompiler.recompile()
        assert result.circuit.data[-1] == qc.data[-1]

    # TODO See above
    def test_given_circuit_with_multi_measurement_when_recompiling_then_preserve_measurement(self):
        num_measurements = 3
        qreg = QuantumRegister(num_measurements + 2)
        creg = ClassicalRegister(num_measurements + 2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        for i in range(num_measurements):
            qc.measure(i, i)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=False)
        result = recompiler.recompile()
        assert result.circuit.data[-num_measurements:] == qc.data[-num_measurements:]

    def test_given_recompiler_when_float_cost_improvement_num_layers_then_no_error(self):
        qc = co.create_random_initial_state_circuit(3)
        config = ISLConfig(cost_improvement_num_layers=4.0, cost_improvement_tol=1)
        recompiler = ISLRecompiler(qc, isl_config=config)
        recompiler.recompile()

    def test_given_initial_single_qubit_layer_when_compiling_then_then_good_solution(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        approx_circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertTrue(overlap > 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_isql_when_compiling_zero_state_then_zero_depth_solution(self):
        qc = QuantumCircuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        approx_circuit = result.circuit
        self.assertEqual(approx_circuit.depth(), 0, "Depth of solution should be zero")

    def test_given_isql_when_compiling_then_ansatz_starts_with_n_single_qubit_gates(self):
        n = 3
        qc = co.create_random_initial_state_circuit(n)
        config = ISLConfig(max_layers=2)
        recompiler = ISLRecompiler(qc, isl_config=config, initial_single_qubit_layer=True)
        result = recompiler.recompile()

        ansatz_start, ansatz_end = recompiler.variational_circuit_range()
        ansatz = recompiler.full_circuit[ansatz_start:ansatz_end]
        for instr in ansatz[:n]:
            self.assertIn(instr[0].name, ["rx", "ry", "rz"])

    def test_given_isql_when_compiling_then_results_object_elements_correct_length(self):
        qc = QuantumCircuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        self.assertTrue(len(result.global_cost_history)
                        == len(result.entanglement_measures_history)
                        == len(result.e_val_history)
                        == len(result.qubit_pair_history)
                        == len(result.method_history))

    def test_given_isl_mode_when_compile_circuit_with_very_small_entanglement_then_heuristic_method_used(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.crx(1e-15, 0, 1)

        recompiler = ISLRecompiler(qc, entanglement_measure=EM_TOMOGRAPHY_NEGATIVITY)
        result = recompiler.recompile()
        self.assertTrue("heuristic" in result.method_history)

    @patch.object(ISLRecompiler, '_measure_qubit_expectation_values')
    def test_given_entanglement_when_find_highest_entanglement_pair_then_evals_not_evaluated(self, mock_get_evals):
        recompiler = ISLRecompiler(QuantumCircuit(2))
        recompiler._find_best_entanglement_qubit_pair([0.5])
        mock_get_evals.assert_not_called()

    @patch.object(ISLRecompiler, '_measure_qubit_expectation_values')
    def test_given_entanglement_when_find_appropriate_pair_then_evals_not_evaluated(self, mock_get_evals):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        recompiler = ISLRecompiler(qc)
        recompiler._find_appropriate_qubit_pair()
        mock_get_evals.assert_not_called()

    def test_given_compiling_with_isql_when_add_layer_then_correct_indices_modified(self):
        qc = co.create_random_initial_state_circuit(3, seed=0)
        num_gates_u = len(qc.data)
        config = ISLConfig(rotosolve_frequency=1e5)
        compiler = ISLRecompiler(
            qc,
            initial_single_qubit_layer=True,
            isl_config=config,
        )
        compiler._add_layer(0)
        full_circuit = compiler.full_circuit.copy()
        layer_1_data = full_circuit[num_gates_u:]
        for gate in layer_1_data:
            self.assertTrue(gate[0].params[0] != 0, "Added layer should have modified angles")

        compiler._add_layer(1)
        full_circuit = compiler.full_circuit.copy()
        layer_1_and_2_data = full_circuit[num_gates_u:]
        self.assertEqual(layer_1_data, layer_1_and_2_data[:len(layer_1_data)],
                         "Adding next layer and using Rotoselect should not modify previous layer")

        layer_2_data = layer_1_and_2_data[len(layer_1_data):]
        for gate in layer_2_data:
            if gate[0].name != "cx":
                self.assertTrue(gate[0].params[0] != 0, "Added layer should have modified angles")

    def test_given_random_circuit_and_starting_circuit_True_when_count_gates_in_solution_then_correct(self):
        qc = co.create_random_circuit(3)
        starting_circuit = co.create_random_initial_state_circuit(3)
        compiler = ISLRecompiler(qc, starting_circuit=starting_circuit)
        result = compiler.recompile()

        num_1q_gates = 0
        num_2q_gates = 0
        for instr in result.circuit.data:
            if instr.operation.name == 'cx':
                num_2q_gates += 1
            elif instr.operation.name == 'rx' or 'ry' or 'rz':
                num_1q_gates += 1

        self.assertEqual((num_1q_gates, num_2q_gates), (result.num_1q_gates, result.num_2q_gates))

    def test_given_wrong_reuse_prio_mode_when_compile_then_error(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(reuse_priority_mode="foo")
        compiler = ISLRecompiler(qc, isl_config=config)
        with self.assertRaises(ValueError):
            compiler.recompile()

    def test_when_add_layer_then_previous_pair_reuse_priority_minus_1(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=1e5)
        compiler = ISLRecompiler(
            qc,
            isl_config=config,
        )
        compiler._add_layer(0)

        pair_acted_on = compiler.qubit_pair_history[0]
        priority = compiler._get_qubit_reuse_priority(pair_acted_on, k=0)

        self.assertEqual(priority, -1)

    def test_given_exponent_equal_to_zero_when_find_reuse_priorities_then_correct(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=1e5)
        compiler = ISLRecompiler(
            qc,
            isl_config=config,
        )
        compiler._add_layer(0)

        pair_acted_on = compiler.qubit_pair_history[0]
        priorities = compiler._get_all_qubit_pair_reuse_priorities(k=0)

        for pair in compiler.coupling_map:
            if pair != pair_acted_on:
                self.assertEqual(priorities[compiler.coupling_map.index(pair)], 1)

    def test_given_exponent_equal_to_one_when_find_qubit_reuse_priorities_then_correct(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(
            rotosolve_frequency=1e5, entanglement_reuse_exponent=1, reuse_priority_mode="qubit"
        )
        compiler = ISLRecompiler(
            qc,
            isl_config=config,
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

    def test_given_random_exponents_when_add_layer_then_same_qubit_pair_never_acted_on_twice_in_a_row(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=1e5,
                           entanglement_reuse_exponent=np.random.rand() * 2,
                           heuristic_reuse_exponent=np.random.rand() * 2,
                           )
        compiler = ISLRecompiler(
            qc,
            isl_config=config,
        )
        compiler._add_layer(0)
        for i in range(10):
            compiler._add_layer(i + 1)
            self.assertTrue(
                compiler.qubit_pair_history[-1] != compiler.qubit_pair_history[-2],
                "Same pair should not be acted on twice")

    def test_given_circuit_when_manually_find_correct_pair_to_act_on_then_pair_acted_on_by_add_layer(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=1e5, entanglement_reuse_exponent=1)
        compiler = ISLRecompiler(
            qc,
            isl_config=config,
        )
        compiler._add_layer(0)

        # Manually find pair which should be acted on when add_layer() is called
        reuse_priorities = compiler._get_all_qubit_pair_reuse_priorities(k=1)
        entanglements = compiler._get_all_qubit_pair_entanglement_measures()
        priorities = [reuse_priorities[i] * entanglements[i] for i in range(len(reuse_priorities))]
        correct_pair = compiler.coupling_map[priorities.index(max(priorities))]

        compiler._add_layer(1)

        self.assertTrue(compiler.qubit_pair_history[-1] == correct_pair)

    def test_given_cuquantum_not_installed_when_cuquantum_backend_called_then_error(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        try:
            import cuquantum
        except:
            with self.assertRaises(ModuleNotFoundError):
                ISLRecompiler(qc, backend=CUQUANTUM_SIM)

    def test_given_random_rotosolve_frequency_and_max_layers_to_modify_values_when_recompile_mps_then_works(self):
        n = 4
        starting_circuit = QuantumCircuit(n)
        starting_circuit.x(range(0, n, 2))
        for isql in [True, False]:
            for sc in [starting_circuit, None]:
                qc = co.create_random_initial_state_circuit(n)
                rotosolve_frequency = np.random.randint(1, 101)
                max_layers_to_modify = np.random.randint(1, 101)
                config = ISLConfig(rotosolve_frequency=rotosolve_frequency, max_layers_to_modify=max_layers_to_modify)
                recompiler = ISLRecompiler(
                    qc,
                    backend=MPS_SIM,
                    isl_config=config,
                    starting_circuit=sc,
                    initial_single_qubit_layer=isql,
                )
                result = recompiler.recompile()
                overlap = co.calculate_overlap_between_circuits(qc, result.circuit)

                self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_mps_backend_when_add_layer_then_num_gates_not_in_mps_is_as_expected(self):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=4, max_layers_to_modify=3)
        recompiler = ISLRecompiler(qc, backend=MPS_SIM, isl_config=config)
        # Rotosolve happens on layers 4, 8, 12...
        # Add layer 0: absorb layer -> 0 gates left in ansatz
        # Add layer 1: absorb layer -> 0 gates
        # Add layer 2: don't absorb layer -> 5 gates
        # Add layer 3: don't absorb layer -> 10 gates
        # Add layer 4: absorb layers 2, 3, 4 -> 0 gates
        # Etc.
        expected_num_gates = [0, 0, 5, 10, 0, 0, 5, 10, 0, 0, 5, 10, 0]
        actual_num_gates = []
        for i in range(13):
            recompiler._add_layer(i)
            actual_num_gates.append(len(recompiler.full_circuit.data) - 1)
        
        np.testing.assert_equal(actual_num_gates, expected_num_gates)

    def test_given_max_layers_larger_than_freq_when_add_layer_then_num_gates_not_in_mps_as_expected(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)
        config = ISLConfig(rotosolve_frequency=4, max_layers_to_modify=5)
        recompiler = ISLRecompiler(qc, backend=MPS_SIM, isl_config=config)
        # layer counter       0  1   2   3   4  5   6   7   8   9  10  11  12
        expected_num_gates = [5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20, 5]
        actual_num_gates = []
        for i in range(13):
            recompiler._add_layer(i)
            actual_num_gates.append(len(recompiler.full_circuit.data) - 1)
    
        np.testing.assert_equal(actual_num_gates, expected_num_gates)

    def test_given_optimise_local_cost_when_recompile_then_global_cost_converged(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc, optimise_local_cost=True)
        result = recompiler.recompile()
        circuit = result.circuit
        overlap = co.calculate_overlap_between_circuits(qc, circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_optimise_local_cost_when_recompile_then_global_and_local_cost_histories_correct(self):
        # Tests that:
        # a) local and global cost histories are the same length
        # b) every global cost is greater than its corresponding local cost
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc, optimise_local_cost=True)
        result = recompiler.recompile()
        self.assertEqual(len(result.global_cost_history), len(result.local_cost_history))
        for global_cost, local_cost in zip(result.global_cost_history, result.local_cost_history):
            self.assertGreaterEqual(np.round(global_cost, 15), np.round(local_cost, 15))

class TestISLCheckpointing(TestCase):

    def test_given_checkpoint_every_1_when_recompile_then_n_layer_number_of_checkpoints(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = recompiler.recompile(checkpoint_every=1, checkpoint_dir=d)
            self.assertEqual(len(os.listdir(d)), len(result.qubit_pair_history))

    def test_given_delete_prev_chkpt_when_recompile_then_1_checkpoint(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            recompiler.recompile(checkpoint_every=1, checkpoint_dir=d, delete_prev_chkpt=True)
            self.assertEqual(len(os.listdir(d)), 1)

    def test_given_checkpoint_every_large_when_recompile_then_2_checkpoints(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            recompiler.recompile(checkpoint_every=100, checkpoint_dir=d)
            self.assertEqual(len(os.listdir(d)), 2)

    def test_given_checkpoint_every_0_when_recompile_then_no_dir_created(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            shutil.rmtree(d)
            recompiler.recompile(checkpoint_every=0, checkpoint_dir=d)
            self.assertFalse(os.path.isdir(d))

    def test_given_checkpointing_when_recompile_then_dir_created(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            shutil.rmtree(d)
            recompiler.recompile(checkpoint_every=100, checkpoint_dir=d)
            self.assertTrue(os.path.isdir(d))

    def test_given_save_and_resume_from_different_points_then_non_time_results_equal(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = recompiler.recompile(checkpoint_every=1, checkpoint_dir=d)
            for c in ["0", "1"]:
                with open(os.path.join(d, c), 'rb') as myfile:
                    loaded_recompiler = pickle.load(myfile)
                result_1 = loaded_recompiler.recompile()
                for key in result.__dict__.keys():
                    if key != "time_taken":
                        self.assertEqual(result.__dict__[key], result_1.__dict__[key])

    def test_given_save_and_resume_from_any_point_then_time_taken_within_100ms(self):
        qc = co.create_random_initial_state_circuit(3, seed=3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            result = recompiler.recompile(checkpoint_every=1, checkpoint_dir=d)
            for c in ["0", "1", "2", "3", "4"]:
                with open(os.path.join(d, c), 'rb') as myfile:
                    loaded_recompiler = pickle.load(myfile)
                result_1 = loaded_recompiler.recompile()
                self.assertAlmostEqual(result.time_taken, result_1.time_taken,
                                       delta=0.1)
                self.assertLess(result_1.time_taken, 100)

    def test_given_save_and_resume_and_save_and_resume_then_overwrites(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc)
        with tempfile.TemporaryDirectory() as d:
            recompiler.recompile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "0"), 'rb') as myfile:
                loaded_recompiler = pickle.load(myfile)
            loaded_recompiler.recompile(checkpoint_every=1, checkpoint_dir=d)
            with open(os.path.join(d, "1"), 'rb') as myfile:
                loaded_recompiler = pickle.load(myfile)
            result = loaded_recompiler.recompile()
            self.assertEqual(len(os.listdir(d)), len(result.qubit_pair_history))



try:
    import qulacs

    module_failed_qulacs = False
except ImportError:
    module_failed_qulacs = True


class TestISLQulacs(TestCase):

    def setUp(self):
        if module_failed_qulacs:
            self.skipTest('Skipping as qulacs is not installed')

    def test_qulacs_recompiler(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        config = ISLConfig(cost_improvement_num_layers=1e3)
        isl_recompiler = ISLRecompiler(qc, backend="qulacs", isl_config=config)

        result = isl_recompiler.recompile()
        approx_circuit = result.circuit

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_qulacs_recompiler_noise_give_error(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        nm = co.create_noisemodel(0.1, 0.1, False)
        isl_recompiler = ISLRecompiler(
            qc, backend="qulacs", execute_kwargs={"noise_model": nm}
        )
        with self.assertRaises(ValueError):
            isl_recompiler.recompile()

    def test_with_initial_ansatz(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        isl_recompiler = ISLRecompiler(qc, backend="qulacs")
        result = isl_recompiler.recompile(qc.copy())
        num_2q_before = co.find_num_gates(qc)[0]
        num_2q_after = co.find_num_gates(result.circuit)[0]
        self.assertLessEqual(num_2q_after, num_2q_before)

try:
    import cuquantum

    module_failed_cuquantum = False
except ImportError:
    module_failed_cuquantum = True

class TestISLCuquantum(TestCase):

    def setUp(self):
        if module_failed_cuquantum:
            self.skipTest('Skipping as cuquantum is not installed')

    @patch('isl.utils.cuquantum_functions.mps_from_circuit_and_starting_mps')
    def test_given_cuquantum_backend_when_get_entanglement_then_correct_mps_from_circuit_called(self, mock_cuquantum_mps_from_circuit):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        stubbed_mps = [np.random.rand(1, 2, 2), np.random.rand(2, 2, 2), np.random.rand(2, 2, 1)]
        mock_cuquantum_mps_from_circuit.return_value = stubbed_mps
        starting_circuit = QuantumCircuit(3)
        starting_circuit.x([0, 1])
        recompiler = ISLRecompiler(qc, backend=CUQUANTUM_SIM, starting_circuit=starting_circuit)
        recompiler._get_all_qubit_pair_entanglement_measures()
        mock_cuquantum_mps_from_circuit.assert_called_once()

    def test_given_cuquantum_backend_when_recompile_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        config = ISLConfig(rotosolve_tol=1e-1)
        recompiler = ISLRecompiler(qc, backend=CUQUANTUM_SIM, isl_config=config)
        result = recompiler.recompile()
        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_cuquantum_backend_when_recompile_with_starting_circuit_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        starting_circuit = QuantumCircuit(3)
        starting_circuit.x([0, 1])
        config = ISLConfig(rotosolve_tol=1e-1)
        recompiler = ISLRecompiler(
            qc, backend=CUQUANTUM_SIM, starting_circuit=starting_circuit, isl_config=config
        )
        result = recompiler.recompile()
        overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
        self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_cuquantum_backend_when_recompile_and_save_previous_layer_mps_then_works(self):
        qc = co.create_random_initial_state_circuit(3)
        starting_circuit = QuantumCircuit(3)
        starting_circuit.x([0, 1])
        config = ISLConfig(rotosolve_frequency=0)
        for sc in [starting_circuit, None]:
            for isql in [True, False]:
                recompiler = ISLRecompiler(
                    qc, backend=CUQUANTUM_SIM, isl_config=config, starting_circuit=sc,
                    initial_single_qubit_layer=isql)
                result = recompiler.recompile()
                overlap = co.calculate_overlap_between_circuits(qc, result.circuit)
                self.assertGreater(overlap, 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_cuquantum_when_add_layers_then_mps_same_with_and_without_layer_caching(self):
        qc = co.create_random_initial_state_circuit(3)
        starting_circuit = QuantumCircuit(3)
        starting_circuit.x([0, 1])
        config1 = ISLConfig(rotosolve_frequency=1e5)
        config2 = ISLConfig(rotosolve_frequency=0)
        for sc in [starting_circuit, None]:
            recompiler1 = ISLRecompiler(
                qc, backend=CUQUANTUM_SIM, starting_circuit=sc, isl_config=config1
            )
            recompiler2 = ISLRecompiler(
                qc, backend=CUQUANTUM_SIM, starting_circuit=sc, isl_config=config2
            )

            recompiler1.recompile()
            recompiler2.recompile()

            mps_1 = recompiler1._get_full_circ_mps_using_cu()
            mps_2 = recompiler2._get_full_circ_mps_using_cu()
            self.assertAlmostEqual(abs(mps_dot(mps_1, mps_2, already_preprocessed=True))**2, 1, 1)

    def test_given_cuquantum_when_caching_previous_layers_then_faster(self):
        qc = co.create_random_initial_state_circuit(3)
        config1 = ISLConfig(rotosolve_frequency=1e5)
        config2 = ISLConfig(rotosolve_frequency=0)

        recompiler1 = ISLRecompiler(qc, backend=CUQUANTUM_SIM, isl_config=config1)
        recompiler2 = ISLRecompiler(qc, backend=CUQUANTUM_SIM, isl_config=config2)

        result1 = recompiler1.recompile()
        result2 = recompiler2.recompile()

        self.assertLess(result2['time_taken'], result1['time_taken'])
