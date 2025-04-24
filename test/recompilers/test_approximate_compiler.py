from unittest import TestCase
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit

import adaptaqc.utils.circuit_operations as co
from adaptaqc.backends.python_default_backends import QASM_SIM, SV_SIM, MPS_SIM
from adaptaqc.compilers.adapt.adapt_compiler import AdaptCompiler
from adaptaqc.compilers.approximate_compiler import ApproximateCompiler


@patch.multiple(ApproximateCompiler, __abstractmethods__=set())
class TestApproximateCompiler(TestCase):
    def setUp(self) -> None:
        self.qc = QuantumCircuit(1)

    def test_when_init_with_mps_backend_then_mps_backend_flag_true(self):
        self.assertTrue(ApproximateCompiler(self.qc, MPS_SIM).is_aer_mps_backend)

    def test_when_init_with_sv_backend_then_sv_backend_flag_true(self):
        self.assertTrue(ApproximateCompiler(self.qc, SV_SIM).is_statevector_backend)

    def test_given_global_cost_and_SV_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(QuantumCircuit(1), SV_SIM)
        with patch.object(compiler.backend, "evaluate_global_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_global_cost_and_MPS_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(QuantumCircuit(1), MPS_SIM)
        with patch.object(compiler.backend, "evaluate_global_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_global_cost_and_QASM_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(QuantumCircuit(1), QASM_SIM)
        with patch.object(compiler.backend, "evaluate_global_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_local_cost_and_SV_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(
            QuantumCircuit(1), SV_SIM, optimise_local_cost=True
        )
        with patch.object(compiler.backend, "evaluate_local_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_local_cost_and_MPS_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(
            QuantumCircuit(1), MPS_SIM, optimise_local_cost=True
        )
        with patch.object(compiler.backend, "evaluate_local_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_local_cost_and_QASM_backend_when_evaluate_cost_then_correct_function_called(
        self,
    ):
        compiler = ApproximateCompiler(
            QuantumCircuit(1), QASM_SIM, optimise_local_cost=True
        )
        with patch.object(compiler.backend, "evaluate_local_cost") as mock:
            compiler.evaluate_cost()
        mock.assert_called_once()

    def test_given_random_circuit_when_evaluate_local_cost_all_three_methods_return_same_cost(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)

        compiler_sv = AdaptCompiler(qc, backend=SV_SIM, optimise_local_cost=True)
        compiler_mps = AdaptCompiler(qc, backend=MPS_SIM, optimise_local_cost=True)
        compiler_qasm = AdaptCompiler(qc, backend=QASM_SIM, optimise_local_cost=True)

        cost_sv = compiler_sv.evaluate_cost()
        cost_mps = compiler_mps.evaluate_cost()
        cost_qasm = compiler_qasm.evaluate_cost()

        # Looser pass threshold for qasm because it includes some form of noise
        np.testing.assert_almost_equal(cost_sv, cost_mps, decimal=5)
        np.testing.assert_almost_equal(cost_sv, cost_qasm, decimal=2)
        np.testing.assert_almost_equal(cost_mps, cost_qasm, decimal=2)

    def test_given_random_circuit_when_evaluate_global_cost_all_three_methods_return_same_cost(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)

        compiler_sv = AdaptCompiler(qc, backend=SV_SIM)
        compiler_mps = AdaptCompiler(qc, backend=MPS_SIM)
        compiler_qasm = AdaptCompiler(qc, backend=QASM_SIM)

        cost_sv = compiler_sv.evaluate_cost()
        cost_mps = compiler_mps.evaluate_cost()
        cost_qasm = compiler_qasm.evaluate_cost()

        # Looser pass threshold for qasm because it includes some form of noise
        np.testing.assert_almost_equal(cost_sv, cost_mps, decimal=5)
        np.testing.assert_almost_equal(cost_sv, cost_qasm, decimal=2)
        np.testing.assert_almost_equal(cost_mps, cost_qasm, decimal=2)

    def test_given_simple_states_when_evaluate_global_and_local_costs_then_correct_value(
        self,
    ):
        # Analytically calculable costs:
        # |0000> global=0, local=0
        # |1010> global=1, local=1/2
        # 4-qubit GHZ global=1/2, local=1/2
        # |++++> global=15/16, local=1/2
        # Using equations 9 and 11 from arXiv:1908.04416

        analytic_costs = [0, 0, 1, 1 / 2, 1 / 2, 1 / 2, 15 / 16, 1 / 2]
        adapt_costs = []

        zero = QuantumCircuit(4)

        neel = QuantumCircuit(4)
        neel.x([0, 2])

        ghz = QuantumCircuit(4)
        ghz.h(0)
        for i in range(3):
            ghz.cx(0, i + 1)

        hadamard = QuantumCircuit(4)
        hadamard.h([0, 1, 2, 3])

        circuits = [zero, neel, ghz, hadamard]

        for circuit in circuits:
            for optimise_local_cost in [False, True]:
                compiler = AdaptCompiler(
                    circuit, backend=SV_SIM, optimise_local_cost=optimise_local_cost
                )
                cost = compiler.evaluate_cost()
                adapt_costs.append(cost)

        np.testing.assert_allclose(adapt_costs, analytic_costs)

    def test_given_random_circuit_when_calculate_cost_local_less_or_equal_to_global(
        self,
    ):
        qc = co.create_random_initial_state_circuit(4)

        compiler_local = AdaptCompiler(qc, backend=SV_SIM, optimise_local_cost=True)
        compiler_global = AdaptCompiler(qc, backend=SV_SIM)

        cost_local = compiler_local.evaluate_cost()
        cost_global = compiler_global.evaluate_cost()

        self.assertLessEqual(cost_local, cost_global)

    @patch("qiskit.QuantumCircuit.set_matrix_product_state")
    def test_set_matrix_product_state_used_when_mps_backend(
        self, mock_set_matrix_product_state
    ):
        ApproximateCompiler(QuantumCircuit(1), MPS_SIM)
        mock_set_matrix_product_state.assert_called_once()
