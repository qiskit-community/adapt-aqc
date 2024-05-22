from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co
from isl.utils.utilityfunctions import (
    _expectation_value_of_qubit,
    expectation_value_of_qubits,
    expectation_value_of_qubits_mps,
    multi_qubit_gate_depth,
)


class TestUtilityFunctions(TestCase):
    def test_qasm_when_zero_state_then_sigmaz_expectation_is_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()

        job = co.QASM_SIM.run(qc, shots=int(1e4))
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1)

    def test_qasm_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = co.QASM_SIM.run(qc, shots=int(1e4))
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1)

    def test_qasm_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        qc.measure_all()
        job = co.QASM_SIM.run(qc, shots=int(1e4))
        counts = job.result().get_counts()
        eval_one_plus_zero_state = expectation_value_of_qubits(counts)
        five_sigma_error_range = 5 / np.sqrt(1e4)
        equivalent_power_of_10 = -np.log10(five_sigma_error_range)
        assert_array_almost_equal(
            eval_one_plus_zero_state, [-1.0, 0.0, 1.0], decimal=equivalent_power_of_10
        )

    def test_sv_when_zero_state_then_sigmaz_expectation_is_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = co.SV_SIM.run(qc)
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1.0)

    def test_sv_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = co.SV_SIM.run(qc)
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1.0)

    def test_sv_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        job = co.SV_SIM.run(qc)
        sv = job.result().get_statevector()
        eval_one_plus_zero_state = expectation_value_of_qubits(sv)
        assert_array_almost_equal(
            eval_one_plus_zero_state, [-1.0, 0.0, 1.0], decimal=15
        )


class TestExpectationValueOfQubitsMPS(TestCase):

    def test_given_circuit_when_mps_expectation_value_then_callable_twice(self):
        """
        Qiskit cannot create a MPS for the same circuit object twice. This test checks that a copy
        of the input circuit is made before generating the MPS
        """
        qc = QuantumCircuit(4)
        expectation_value_of_qubits_mps(qc)
        expectation_value_of_qubits_mps(qc)

    def test_given_n_qubit_circuit_when_mps_expectation_value_then_n_output_values(self):
        qc = QuantumCircuit(3)
        self.assertEqual(len(expectation_value_of_qubits_mps(qc)), 3)

    def test_given_zero_state_mps_when_pauli_expectation_then_is_correct(self):
        qc = QuantumCircuit(4)
        expectation = expectation_value_of_qubits_mps(qc)
        np.testing.assert_allclose(expectation, [1, 1, 1, 1])

    def test_given_hadamard_state_mps_when_pauli_expectation_then_is_correct(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        expectation = expectation_value_of_qubits_mps(qc)
        np.testing.assert_allclose(expectation, [0, 0, 0, 0], atol=1e-07)


class TestMultiQubitGateDepth(TestCase):

    def test_given_no_gates_then_zero(self):
        qc = QuantumCircuit(1)
        self.assertEqual(multi_qubit_gate_depth(qc), 0)

    def test_given_single_qubit_gates_then_zero(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        self.assertEqual(multi_qubit_gate_depth(qc), 0)

    def test_given_single_cnot_then_one(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        self.assertEqual(multi_qubit_gate_depth(qc), 1)

    def test_given_multiple_cnots_overlapping_qubits_then_two(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        self.assertEqual(multi_qubit_gate_depth(qc), 2)

    def test_given_multiple_cnots_different_qubits_then_one(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        self.assertEqual(multi_qubit_gate_depth(qc), 1)

    def test_given_cnot_and_single_qubit_gates_then_one(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        self.assertEqual(multi_qubit_gate_depth(qc), 1)

    def test_given_nested_cnots_then_three(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        self.assertEqual(multi_qubit_gate_depth(qc), 3)
