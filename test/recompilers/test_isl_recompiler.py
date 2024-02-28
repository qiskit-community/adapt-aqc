import logging
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import isl.utils.circuit_operations as co
from isl.recompilers import ISLConfig, ISLRecompiler
from isl.utils.circuit_operations import QASM_SIM, SV_SIM, MPS_SIM
from isl.utils.constants import DEFAULT_SUFFICIENT_COST
from isl.utils.entanglement_measures import EM_TOMOGRAPHY_NEGATIVITY


class TestISL(TestCase):

    def test_isl_procedure_sv(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend=SV_SIM,
                                       isl_config=ISLConfig(sufficient_cost=1e-2))

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_isl_procedure_qasm(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler_qasm = ISLRecompiler(qc, backend=QASM_SIM, execute_kwargs={'shots': shots})

        result_qasm = isl_recompiler_qasm.recompile()
        approx_circuit_qasm = result_qasm["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit_qasm, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_isl_procedure_mps(self):
        qc = co.create_random_initial_state_circuit(3, seed=1)
        qc = co.unroll_to_basis_gates(qc)

        shots = 1e4
        isl_recompiler_mps = ISLRecompiler(qc, backend=MPS_SIM, execute_kwargs={'shots': shots})

        result_mps = isl_recompiler_mps.recompile()
        approx_circuit_mps = result_mps["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit_mps, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST - 5 / np.sqrt(shots)

    def test_GHZ(self):
        qc = QuantumCircuit(5)

        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend=SV_SIM,
                                       isl_config=ISLConfig(sufficient_cost=1e-2))

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_exact_overlap_close_to_approx_overlap(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        approx_overlap = result["overlap"]
        exact_overlap = result["exact_overlap"]
        self.assertAlmostEquals(approx_overlap, exact_overlap, delta=1e-2)

    def test_exact_overlap_calculated_correctly(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        exact_overlap1 = result["exact_overlap"]
        exact_overlap2 = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertAlmostEquals(exact_overlap1, exact_overlap2, delta=1e-2)

    def test_local_measurements_sv(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        isl_config = ISLConfig(cost_improvement_num_layers=10)

        isl_recompiler = ISLRecompiler(
            qc, local_measurements_only=True, backend=SV_SIM, isl_config=isl_config
        )
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_custom_layer_gate(self):
        from qiskit import QuantumCircuit

        from isl.utils.fixed_ansatz_circuits import number_preserving_ansatz

        # Initialize to a supervision of states with bit sum 2
        statevector = [
            0,
            0,
            0,
            -((1 / 3)**0.5),
            0,
            1j * (1 / 3)**0.5,
            -1 * (1 / 3)**0.5,
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
        approx_circuit = result["circuit"]

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
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc_mod)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_heuristic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="heuristic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_basic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="basic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 1 - DEFAULT_SUFFICIENT_COST

    def test_random_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="random")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
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

    def test_given_starting_circuit_when_recompile_then_solution_starts_with_it(self):

        n = 2
        starting_ansatz_circuit = QuantumCircuit(n)
        starting_ansatz_circuit.x(0)

        qc = co.create_random_initial_state_circuit(n)

        for boolean in [False, True]:
            recompiler = ISLRecompiler(qc, starting_circuit=starting_ansatz_circuit, initial_single_qubit_layer=boolean)

            result = recompiler.recompile()
            compiled_qc: QuantumCircuit = result.get("circuit")
            del compiled_qc.data[1:]
            overlap = np.abs(np.dot(Statevector(compiled_qc).conjugate(),
                                    Statevector(starting_ansatz_circuit)))**2
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
        final_circuit = result.get("circuit")
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
        final_circuit = result.get("circuit")
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

    # TODO See above
    def test_given_circuit_with_one_measurement_when_recompiling_then_preserve_measurement(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=False)
        result = recompiler.recompile()
        assert result["circuit"].data[-1] == qc.data[-1]

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
        assert result["circuit"].data[-num_measurements:] == qc.data[-num_measurements:]

    def test_given_recompiler_when_float_cost_improvement_num_layers_then_no_error(self):
        qc = co.create_random_initial_state_circuit(3)
        config = ISLConfig(cost_improvement_num_layers=4.0, cost_improvement_tol=1)
        recompiler = ISLRecompiler(qc, isl_config=config)
        recompiler.recompile()

    def test_given_initial_single_qubit_layer_when_compiling_then_then_good_solution(self):
        qc = co.create_random_initial_state_circuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertTrue(overlap > 1 - DEFAULT_SUFFICIENT_COST)

    def test_given_isql_when_compiling_zero_state_then_zero_depth_solution(self):
        qc = QuantumCircuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        approx_circuit = result["circuit"]
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

    def test_given_isql_when_compiling_then_results_dict_elements_correct_length(self):
        qc = QuantumCircuit(3)
        recompiler = ISLRecompiler(qc, initial_single_qubit_layer=True)
        result = recompiler.recompile()
        self.assertTrue(len(result.get("cost_progression"))
                        == len(result.get("entanglement_measures_progression"))
                        == len(result.get("e_val_history"))
                        == len(result.get("qubit_pair_history"))
                        == len(result.get("method_history")))



    def test_given_isl_mode_when_compile_circuit_with_very_small_entanglement_then_heuristic_method_used(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.crx(1e-15, 0, 1)

        recompiler = ISLRecompiler(qc, entanglement_measure=EM_TOMOGRAPHY_NEGATIVITY)
        result = recompiler.recompile()
        self.assertTrue("heuristic" in result.get("method_history"))

    @patch.object(ISLRecompiler, '_measure_qubit_expectation_values')
    def test_given_entanglement_when_find_highest_entanglement_pair_then_evals_not_evaluated(self, mock_get_evals):
        recompiler = ISLRecompiler(QuantumCircuit(2))
        recompiler._find_highest_entanglement_qubit_pair([0.5], [1.0])
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



try:
    import qulacs

    module_failed = False
except ImportError:
    module_failed = True


class TestISLQulacs(TestCase):

    def setUp(self):
        if module_failed:
            self.skipTest('Skipping as qulacs is not installed')

    def test_qulacs_recompiler(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        config = ISLConfig(cost_improvement_num_layers=1e3)
        isl_recompiler = ISLRecompiler(qc, backend="qulacs", isl_config=config)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

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
        num_2q_after = co.find_num_gates(result["circuit"])[0]
        self.assertLessEqual(num_2q_after, num_2q_before)
