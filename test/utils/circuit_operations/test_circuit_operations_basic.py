from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate

import adaptaqc.utils.circuit_operations as co
import adaptaqc.utils.constants as vconstants


def create_circuit_with_dep_and_indep_params():
    qc = QuantumCircuit(1)
    gate_1 = co.create_independent_parameterised_gate("rx", "theta_1", 0.2)
    gate_2 = co.create_independent_parameterised_gate("ry", "theta_2", 0.6)
    gate_3 = co.create_independent_parameterised_gate("rz", "theta_3", -0.1)
    # create a gate which depends on gate_1
    gate_4 = co.create_dependent_parameterised_gate("rz", "-2*theta_1+0.3", 0)
    co.add_gate(qc, gate_1, qubit_indexes=[0])
    co.add_gate(qc, gate_2, qubit_indexes=[0])
    co.add_gate(qc, gate_3, qubit_indexes=[0])
    co.add_gate(qc, gate_4, qubit_indexes=[0])
    # Find independent parameter values and update gate_4
    vals = co.calculate_independent_variable_values(qc)
    co.reevaluate_dependent_parameterised_gates(qc, vals)
    return qc


class TestCircuitOperationsBasic(TestCase):
    def test_create_1q_gate(self):
        rx_gate = co.create_1q_gate("rx", 0.5)
        ry_gate = co.create_1q_gate("ry", -0.5)
        rz_gate = co.create_1q_gate("rz", 0.23)

        self.assertEqual(rx_gate.name, "rx")
        self.assertEqual(rx_gate.params[0], 0.5)
        self.assertEqual(rx_gate.label, "rx")

        self.assertEqual(ry_gate.name, "ry")
        self.assertEqual(ry_gate.params[0], -0.5)
        self.assertEqual(ry_gate.label, "ry")

        self.assertEqual(rz_gate.name, "rz")
        self.assertEqual(rz_gate.params[0], 0.23)
        self.assertEqual(rz_gate.label, "rz")

    def test_create_2q_gate(self):
        cx_gate = co.create_2q_gate("cx")
        cz_gate = co.create_2q_gate("cz")
        self.assertEqual(cx_gate.name, "cx")
        self.assertEqual(cz_gate.name, "cz")

    def test_add_gate(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(0)
        qc.h(1)

        gate = RXGate(0.1, label="test_gate")
        co.add_gate(qc, gate, 1, [0])

        self.assertEqual(qc.data[1][0].label, "test_gate")

    def test_replace_1q_gate(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.3, 1)
        qc.z(2)
        co.replace_1q_gate(qc, 2, "rz", 1.2)
        self.assertEqual(qc.data[2][0].label, "rz")
        self.assertEqual(qc.data[2][0].params[0], 1.2)

    def test_replace_2q_gate(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 1)
        qc.z(2)
        co.replace_2q_gate(qc, 1, 1, 2, "cz")
        # qc.data has form [(gate,qargs,cargs)] where qargs,cargs have form
        # [Qubit]

        self.assertEqual(qc.data[1][0].name, "cz")
        self.assertEqual(qc.data[1][1][0]._index, 1)
        self.assertEqual(qc.data[1][1][1]._index, 2)

    def test_is_supported_1q_gate(self):
        assert co.is_supported_1q_gate(Gate("rx", 1, [0.5], "rx")) is True
        assert (
            co.is_supported_1q_gate(Gate("rx", 1, [0.5], vconstants.FIXED_GATE_LABEL))
            is False
        )
        assert co.is_supported_1q_gate(Gate("cx", 2, [])) is False
        assert co.is_supported_1q_gate(Gate("ZZ", 1, [0.5])) is False

    def test_add_appropritate_gates(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)
        qc.h(0)
        qc.s(1)

        # Test thinly dressed case
        circ = qc.copy()
        co.add_appropriate_gates(circ, 0, True, 2)
        self.assertEqual(circ.data[2][0].name, "rz")
        self.assertEqual(len(circ.data), 5)

        # Test fully dressed case
        circ = qc.copy()
        co.add_appropriate_gates(circ, 0, False, 2)
        self.assertEqual(circ.data[2][0].name, "rz")
        self.assertEqual(circ.data[3][0].name, "ry")
        self.assertEqual(circ.data[4][0].name, "rz")
        self.assertEqual(len(circ.data), 7)

    def test_add_dressed_cnot(self):
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 1)
        # Add dressed CNOT here
        qc.h(1)
        qc.z(2)

        expected_qc = QuantumCircuit(qr)
        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.rx(0.5, 1)
        # Before control rzryrz decomposition
        expected_qc.rz(0, 1)
        expected_qc.ry(0, 1)
        expected_qc.rz(0, 1)
        # CNOT
        expected_qc.cx(1, 2)
        # After target rzryrz decomposition
        expected_qc.rz(0, 2)
        expected_qc.ry(0, 2)
        expected_qc.rz(0, 2)
        expected_qc.h(1)
        expected_qc.z(2)

        co.add_dressed_cnot(qc, 1, 2, gate_index=3, v2=False, v3=False)
        assert co.are_circuits_identical(qc, expected_qc)

        # Test thinly dressed CNOT
        expected_qc.rz(0, 2)
        expected_qc.rz(0, 0)
        expected_qc.cx(2, 0)
        expected_qc.rz(0, 2)
        expected_qc.rz(0, 0)

        co.add_dressed_cnot(qc, 2, 0, thinly_dressed=True)
        assert co.are_circuits_identical(qc, expected_qc)

    def test_create_independent_parametrised_gate(self):
        gate = co.create_independent_parameterised_gate("rx", "theta_0", 0.2)

        self.assertEqual(gate.name, "rx")
        self.assertEqual(gate.label, "rx#theta_0")
        self.assertEqual(gate.params, [0.2])

    def test_create_dependent_parametrised_gate(self):
        gate = co.create_dependent_parameterised_gate("rx", "-theta_0", 0.2)

        self.assertEqual(gate.name, "rx")
        self.assertEqual(gate.label, "rx@-theta_0")
        self.assertEqual(gate.params, [0.2])

    def test_calculate_independent_variable_values(self):
        qc = QuantumCircuit(1)
        gate_1 = co.create_independent_parameterised_gate("rx", "theta_1", 0.2)
        gate_2 = co.create_independent_parameterised_gate("ry", "theta_2", 0.6)
        gate_3 = co.create_independent_parameterised_gate("rz", "theta_3", -0.1)
        # dependent gates should not be counted
        gate_4 = co.create_dependent_parameterised_gate("rz", "theta_4", 0.7)
        co.add_gate(qc, gate_1, qubit_indexes=[0])
        co.add_gate(qc, gate_2, qubit_indexes=[0])
        co.add_gate(qc, gate_3, qubit_indexes=[0])
        co.add_gate(qc, gate_4, qubit_indexes=[0])

        expected = {"theta_1": 0.2, "theta_2": 0.6, "theta_3": -0.1}
        self.assertEqual(co.calculate_independent_variable_values(qc), expected)

    def test_reevaluate_dependent_parameterised_gates(self):
        qc = create_circuit_with_dep_and_indep_params()

        # Now angle of gate_4 should be -2*0.2+0.3 = -0.1
        np.testing.assert_almost_equal(qc.data[3][0].params, -0.1, 10)

    def test_add_subscript_to_all_variables(self):
        qc = create_circuit_with_dep_and_indep_params()

        # Add a subscript "A" to all gate variables
        co.add_subscript_to_all_variables(qc, "A")

        expected = [
            "rx#theta_1_A",
            "ry#theta_2_A",
            "rz#theta_3_A",
            "rz@-2*theta_1_A+0.3",
        ]
        for i in range(len(qc.data)):
            self.assertEqual(qc.data[i][0].label, expected[i])
