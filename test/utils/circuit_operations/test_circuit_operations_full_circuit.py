from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit import Instruction
from qiskit.compiler import transpile

import isl.utils.circuit_operations as co


def create_test_circuit_1(qr):
    full_circuit = QuantumCircuit(qr)
    full_circuit.h(0)
    full_circuit.cx(0, 1)
    full_circuit.rx(-1.5, 2)
    full_circuit.cx(2, 1)
    full_circuit.rz(2.1, 1)
    full_circuit.cx(1, 2)
    full_circuit.ry(0.2, 1)
    full_circuit.rx(0.23, 0)
    full_circuit.cx(1, 2)
    full_circuit.h(1)
    return full_circuit


def create_test_circuit_2():
    qr = QuantumRegister(3)
    left_circuit = QuantumCircuit(qr)
    left_circuit.h(0)
    left_circuit.cx(0, 1)
    # Add right circuit will be added at this location
    left_circuit.rx(0.23, 0)
    left_circuit.cx(1, 2)
    left_circuit.h(1)
    return left_circuit, qr


def create_test_circuit_3():
    qr = QuantumRegister(3)
    full_circuit = create_test_circuit_1(qr)
    partial_circuit = QuantumCircuit(3)
    partial_circuit.rx(-1.5, 2)
    partial_circuit.cx(2, 1)
    partial_circuit.rz(2.1, 1)
    partial_circuit.cx(1, 2)
    return full_circuit, partial_circuit


def create_test_circuit_4():
    qreg = QuantumRegister(3)
    creg = ClassicalRegister(1)
    quantum_circuit = QuantumCircuit(qreg, creg)
    quantum_circuit.cx(0, 2)
    quantum_circuit.h(0)
    quantum_circuit.cx(2, 1)
    quantum_circuit.rx(2.3, 1)
    return creg, qreg, quantum_circuit


class TestOperationsFullCircuit(TestCase):
    def test_find_register(self):
        qreg = QuantumRegister(2, "q1")
        creg = ClassicalRegister(2, "c0")
        qc = QuantumCircuit(qreg, creg)

        self.assertEqual(
            co.find_register(qc, qc.qubits[0]), QuantumRegister(2, "q1")
        )
        self.assertEqual(
            co.find_register(qc, qc.clbits[0]), ClassicalRegister(2, "c0")
        )

    def test_find_bit_index(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(1)
        qc = QuantumCircuit(qreg, creg)

        self.assertEqual(co.find_bit_index(qc.qregs[0], qc.qubits[0]), 0)
        self.assertEqual(co.find_bit_index(qc.qregs[0], qc.qubits[1]), 1)
        self.assertEqual(co.find_bit_index(qc.cregs[0], qc.clbits[0]), 0)
        self.assertEqual(co.find_bit_index(qc.qregs[0], qc.clbits[0]), None)

    def test_create_random_circuit(self):
        qc_default = co.create_random_circuit(2)
        qc_custom = co.create_random_circuit(2, 6, ["rx"], ["cx"])

        self.assertIsInstance(qc_default, QuantumCircuit)
        self.assertIsInstance(qc_custom, QuantumCircuit)
        self.assertEqual(qc_default.depth(), 5)
        self.assertEqual(qc_custom.depth(), 6)
        self.assertTrue(any(gate[0].name == "rx" for gate in qc_custom.data))
        self.assertTrue(any(gate[0].name == "cx" for gate in qc_custom.data))
        self.assertTrue(any(gate[0].name != "cz" for gate in qc_custom.data))

    def test_change_circuit_register(self):
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(3)
        qc = QuantumCircuit(qr2)
        qc.cx(0, 2)
        qc.h(0)
        qc.cx(2, 1)
        qc.rx(2.3, 1)
        qc.x(0)

        expected_qc = QuantumCircuit(qr1)
        expected_qc.cx(3, 1)
        expected_qc.h(3)
        expected_qc.cx(1, 2)
        expected_qc.rx(2.3, 2)
        expected_qc.x(3)
        qubit_mapping = {0: 3, 1: 2, 2: 1}
        co.change_circuit_register(qc, qr1, qubit_mapping)

        self.assertTrue(
            all(
                gate[1] == expected_gate[1]
                for gate, expected_gate in zip(qc.data, expected_qc.data)
            )
        )
        # Make sure old register was removed from circuit
        self.assertTrue(qr2 not in qc.qregs)
        self.assertTrue(co.are_circuits_identical(qc, expected_qc))

    def test_add_to_circuit_and_unroll_to_basis_gates_and_transpile(self):
        left_circuit, qr = create_test_circuit_2()

        right_circuit = QuantumCircuit(2)
        right_circuit.rx(-1.5, 1)
        right_circuit.cx(1, 0)
        right_circuit.rz(2.1, 0)
        # The following two gates should cancel each other out when transpiling
        right_circuit.cx(0, 1)
        right_circuit.cx(0, 1)
        right_circuit.cx(0, 1)
        right_circuit.ry(0.2, 0)

        expected_full_circuit = create_test_circuit_1(qr)

        co.add_to_circuit(
            left_circuit,
            right_circuit,
            location=2,
            transpile_before_adding=True,
            transpile_kwargs={"optimization_level": 1},
            qubit_subset=[1, 2],
        )
        self.assertAlmostEqual(
            co.calculate_overlap_between_circuits(
                left_circuit, expected_full_circuit
            ),
            1,
        )
        self.assertEqual(
            len(expected_full_circuit.data), len(left_circuit.data)
        )
        self.assertTrue(
            co.are_circuits_identical(left_circuit, expected_full_circuit)
        )

    def test_remove_inner_circuit(self):
        partial_circuit, qr = create_test_circuit_2()

        full_circuit = create_test_circuit_1(qr)

        gate_range_to_remove = (2, 7)
        co.remove_inner_circuit(full_circuit, gate_range_to_remove)
        self.assertEqual(len(full_circuit.data), len(partial_circuit.data))
        self.assertTrue(
            co.are_circuits_identical(partial_circuit, full_circuit)
        )

    def test_extract_inner_circuit(self):
        full_circuit, partial_circuit = create_test_circuit_3()
        partial_circuit.ry(0.2, 1)

        gate_range_to_extract = (2, 7)
        inner_circuit = co.extract_inner_circuit(
            full_circuit, gate_range_to_extract
        )
        self.assertLess(len(inner_circuit.data), len(full_circuit.data))
        self.assertTrue(
            co.are_circuits_identical(inner_circuit, partial_circuit)
        )

    def test_replace_inner_circuit(self):
        full_circuit, partial_circuit = create_test_circuit_3()
        partial_circuit.ry(1.5, 2)

        gate_range_to_replace = (2, 7)
        new_circuit = full_circuit.copy()
        co.replace_inner_circuit(
            new_circuit, partial_circuit, gate_range_to_replace
        )
        self.assertIsInstance(new_circuit, QuantumCircuit)
        self.assertFalse(co.are_circuits_identical(new_circuit, full_circuit))

    def test_replace_inner_circuit_with_transpile_level_3(self):
        full_circuit, partial_circuit = create_test_circuit_3()
        partial_circuit.ry(1.5, 2)

        gate_range_to_replace = (2, 7)
        replaced_circuit_no_transpilation = full_circuit.copy()
        replaced_circuit_with_transpilation = full_circuit.copy()
        co.replace_inner_circuit(
            replaced_circuit_no_transpilation,
            partial_circuit,
            gate_range_to_replace,
        )
        transpiled_partial_circuit = transpile(
            partial_circuit,
            basis_gates=["cx", "rx", "ry", "rz"],
            optimization_level=2,
        )
        co.replace_inner_circuit(
            replaced_circuit_with_transpilation,
            transpiled_partial_circuit,
            gate_range_to_replace,
        )
        self.assertAlmostEqual(
            co.calculate_overlap_between_circuits(
                replaced_circuit_no_transpilation,
                replaced_circuit_with_transpilation,
            ),
            1,
        )

    def test_find_num_gates(self):
        qc = QuantumCircuit(3)
        qc.rx(0.6, 0)
        qc.cx(0, 1)
        # Counting start
        qc.rx(0.3, 1)
        # Next 2 cx gates should cancel each other if transpiling
        qc.ry(1.3, 0)
        qc.cx(1, 0)
        qc.cx(1, 0)
        qc.cx(1, 0)
        qc.rz(-2.3, 2)
        qc.cz(2, 0)
        # Counting end
        qc.cx(0, 1)
        qc.rx(0.3, 1)

        self.assertEqual(co.find_num_gates(None), (0, 0))
        self.assertEqual(
            co.find_num_gates(qc, False, gate_range=(2, 9)), (4, 3)
        )
        self.assertEqual(
            co.find_num_gates(qc, True, {"optimization_level": 1}), (4, 5)
        )

    def test_append_to_instruction(self):
        full_circuit, qr = create_test_circuit_2()
        full_circuit_instruction = full_circuit.to_instruction()

        partial_circuit = QuantumCircuit(qr)
        partial_circuit.h(0)
        partial_circuit.cx(0, 1)
        partial_circuit.rx(0.23, 0)
        partial_circuit.cx(1, 2)

        h_gate = QuantumCircuit(qr)
        h_gate.h(1)
        final_circuit_instruction = co.append_to_instruction(
            partial_circuit.to_instruction(), h_gate.to_instruction()
        )
        self.assertIsInstance(final_circuit_instruction, Instruction)
        self.assertAlmostEqual(
            co.calculate_overlap_between_circuits(
                final_circuit_instruction.definition,
                full_circuit_instruction.definition,
            ),
            1,
        )

    def test_remove_classical_operations_and_add_classical_operations(self):
        creg, qreg, quantum_circuit = create_test_circuit_4()

        classical_and_quantum_circuit = QuantumCircuit(qreg, creg)
        classical_and_quantum_circuit.cx(0, 2)
        classical_and_quantum_circuit.h(0)
        classical_and_quantum_circuit.cx(2, 1)
        classical_and_quantum_circuit.rx(2.3, 1)
        classical_and_quantum_circuit.measure(0, 0)

        classical_gates = classical_and_quantum_circuit.copy()
        classical_gates = co.remove_classical_operations(classical_gates)
        co.add_classical_operations(quantum_circuit, classical_gates)
        self.assertTrue(
            any(gate[0].name == "measure" for gate in quantum_circuit.data)
        )
        self.assertTrue(
            co.are_circuits_identical(
                quantum_circuit, classical_and_quantum_circuit
            )
        )

    def test_make_quantum_only_circuit(self):
        creg, qreg, classical_and_quantum_circuit = create_test_circuit_4()
        classical_and_quantum_circuit.measure(0, 0)

        classical_circuit = co.make_quantum_only_circuit(
            classical_and_quantum_circuit
        )
        self.assertIsInstance(classical_circuit, QuantumCircuit)
        self.assertLess(
            len(classical_circuit.data), len(classical_and_quantum_circuit.data)
        )
        self.assertFalse(
            any(gate[0].name == "measure" for gate in classical_circuit.data)
        )

    def test_circuit_by_inverting_circuit(self):
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        circuit.h(0)
        circuit.cx(2, 1)
        circuit.rx(2.3, 1, label="my_label")

        inverted_circuit = co.circuit_by_inverting_circuit(circuit.copy())
        double_inverted_circuit = co.circuit_by_inverting_circuit(
            inverted_circuit
        )
        self.assertTrue(
            co.are_circuits_identical(circuit, double_inverted_circuit)
        )

    def test_initial_state_to_circuit_and_unroll_to_basis_gates_and_remove_reset_gates(
        self,
    ):
        # Test None
        self.assertEqual(co.initial_state_to_circuit(None), None)

        # Test Vector
        qubits = 3
        x = np.random.RandomState().standard_normal(2**qubits)
        rand_state = x / np.linalg.norm(x)

        qc = QuantumCircuit(qubits)
        qc.append(co.initial_state_to_circuit(rand_state), qc.qubits)
        sv = Statevector(qc)
        overlap_minus_1 = np.abs(np.abs(np.vdot(rand_state, sv)) - 1)
        self.assertLess(overlap_minus_1, 1e-3)

    def test_create_random_initial_state_circuit(self):
        qc, sv = co.create_random_initial_state_circuit(
            3, return_statevector=True
        )

        self.assertIsInstance(qc, QuantumCircuit)
        self.assertIsInstance(sv, np.ndarray)
        self.assertEqual(
            co.find_register(qc, qc.qubits[0]), QuantumRegister(3, "q")
        )

    def test_are_circuits_identical(self):
        qc1 = co.create_random_initial_state_circuit(3)
        qc2 = qc1.copy()
        self.assertTrue(co.are_circuits_identical(qc1, qc2))
