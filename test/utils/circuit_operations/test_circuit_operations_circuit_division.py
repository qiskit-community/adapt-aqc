from unittest import TestCase

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate, CXGate, CZGate

import isl.utils.circuit_operations as co


class TestCircuitOperationsCircuitDivision(TestCase):
    def test_find_previous_gate_on_qubit(self):
        qc = QuantumCircuit(3)
        qc.h(0)  # index 0
        qc.ry(0.4, 1)  # index 1
        qc.x(2)  # index 2
        qc.cx(0, 1)  # index 3
        qc.cz(1, 2)  # index 4
        qc.h(1)  # index 5
        expected = [
            (None, None),
            (None, None),
            (None, None),
            (RYGate(0.4), 1),
            (CXGate(), 3),
            (CZGate(), 4),
        ]
        for i in range(len(qc.data)):
            self.assertEqual(co.find_previous_gate_on_qubit(qc, i), expected[i])

    def test_index_of_bit_in_circuit(self):
        qr = QuantumRegister(8)
        cr = ClassicalRegister(4)

        qc = QuantumCircuit(qr, cr)

        indexes = []
        for qubit in qr:
            indexes.append(co.index_of_bit_in_circuit(qubit, qc))
        for clbit in cr:
            indexes.append(co.index_of_bit_in_circuit(clbit, qc))

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]

        self.assertEqual(indexes, expected)

    def test_vertically_divide_circuit(self):
        qc = co.create_random_initial_state_circuit(5)

        sub_circuits = co.vertically_divide_circuit(qc, 3)

        reconstructed_qc = QuantumCircuit(5)
        for circuit in sub_circuits:
            co.add_to_circuit(reconstructed_qc, circuit)
            self.assertLessEqual(circuit.depth(), 3)

        self.assertEqual(qc.data, reconstructed_qc.data)
