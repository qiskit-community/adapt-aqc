from unittest import TestCase

from qiskit import QuantumCircuit, QuantumRegister

import adaptaqc.utils.circuit_operations as co
import adaptaqc.utils.constants as vconstants


class TestCircuitOperationsVariational(TestCase):
    def test_find_angles_in_circuit(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.23, 0)
        qc.cx(1, 0)
        qc.rz(3.1, 2)
        qc.rx(2.9, 2)
        instr = qc.data[-1]
        instr.operation.label = vconstants.FIXED_GATE_LABEL
        qc.data[-1] = instr
        qc.cx(1, 2)
        qc.ry(-1.4, 1)
        qc.measure_all()

        self.assertEqual(co.find_angles_in_circuit(qc), [0.23, 3.1, -1.4])
        self.assertEqual(co.find_angles_in_circuit(qc, (4, 5)), [3.1])

    def test_update_angles_in_circuit(self):
        fixed_gate = co.create_1q_gate("rx", 2.3)
        fixed_gate.label = vconstants.FIXED_GATE_LABEL
        parameterized_gate1 = co.create_1q_gate("rz", 1.0)
        parameterized_gate2 = co.create_1q_gate("ry", 1.0)
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.append(parameterized_gate1.copy(), [qr[0]])
        qc.cx(0, 1)
        qc.append(parameterized_gate1.copy(), [qr[0]])
        qc.append(fixed_gate, [qr[2]])
        qc.append(parameterized_gate1.copy(), [qr[1]])
        qc.z(2)
        qc.append(parameterized_gate2.copy(), [qr[2]])

        new_angles = [-1, 0, 0.5, 0.23]
        co.update_angles_in_circuit(qc, new_angles)

        for index, angle in zip([1, 3, 5, 7], new_angles):
            self.assertEqual(qc.data[index][0].params[0], angle)

        new_angles = [0.5, 0.23]
        co.update_angles_in_circuit(qc, new_angles, (4, 8))

        for index, angle in zip([5, 7], new_angles):
            self.assertEqual(qc.data[index][0].params[0], angle)
