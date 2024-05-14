from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co


class TestCircuitOperationsOptimisation(TestCase):
    def test_remove_unnecessary_gates_from_circuit(self):
        original_circuit = QuantumCircuit(3)
        original_circuit.cx(0, 2)
        original_circuit.cz(0, 2)
        original_circuit.cz(0, 2)
        original_circuit.h(0)
        original_circuit.cx(2, 1)
        original_circuit.cx(2, 1)
        original_circuit.rx(2.3, 1)
        original_circuit.rx(2.3, 1)
        original_circuit.rx(2.3, 1)
        original_circuit.rx(2.3, 1)
        original_circuit.x(0)

        expected_circuit = QuantumCircuit(3)
        expected_circuit.cx(0, 2)
        expected_circuit.h(0)
        expected_circuit.rz((np.pi / 2), 1)
        expected_circuit.ry(2.9168146928204126, 1)
        expected_circuit.rz((3 / 2) * np.pi, 1)
        expected_circuit.x(0)

        co.remove_unnecessary_gates_from_circuit(original_circuit)
        self.assertIsInstance(original_circuit, QuantumCircuit)
        self.assertEqual(original_circuit, expected_circuit)
