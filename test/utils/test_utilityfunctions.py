from unittest import TestCase

from numpy.testing import assert_array_almost_equal
from qiskit import QuantumCircuit, execute

import isl.utils.circuit_operations as co
from isl.utils.utilityfunctions import _expectation_value_of_qubit, expectation_value_of_qubits


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
