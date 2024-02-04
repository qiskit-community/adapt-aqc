from unittest import TestCase
import numpy as np

from numpy.testing import assert_array_almost_equal
from qiskit import QuantumCircuit, execute

import isl.utils.circuit_operations as co
from isl.utils.utilityfunctions import _expectation_value_of_qubit, expectation_value_of_qubits, \
    expectation_value_of_qubits_mps


class TestUtilityFunctions(TestCase):
    def test_qasm_when_zero_state_then_sigmaz_expectation_is_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = execute(qc, backend=co.QASM_SIM, shots=int(1e4))
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1)

    def test_qasm_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = execute(qc, backend=co.QASM_SIM, shots=int(1e4))
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1)

    def test_qasm_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        qc.measure_all()
        job = execute(qc, backend=co.QASM_SIM, shots=int(1e6))
        counts = job.result().get_counts()
        eval_one_plus_zero_state = expectation_value_of_qubits(counts)
        assert_array_almost_equal(eval_one_plus_zero_state, [-1., 0., 1.], decimal=2)

    def test_sv_when_zero_state_then_sigmaz_expectation_is_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = execute(qc, backend=co.SV_SIM)
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1.)

    def test_sv_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = execute(qc, backend=co.SV_SIM)
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1.)

    def test_sv_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        job = execute(qc, backend=co.SV_SIM)
        sv = job.result().get_statevector()
        eval_one_plus_zero_state = expectation_value_of_qubits(sv)
        assert_array_almost_equal(eval_one_plus_zero_state, [-1., 0., 1.], decimal=15)


class TestExpectationValueOfQubitsMPS(TestCase):

    def test_given_circuit_when_mps_expectation_value_then_callable_twice(self):
        """
        Qiskit cannot create a MPS for the same circuit object twice. This test checks that a copy of the input
        circuit is made before generating the MPS
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
