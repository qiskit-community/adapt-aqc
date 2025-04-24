from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector

import adaptaqc.backends.python_default_backends
import adaptaqc.utils.ansatzes as ans
import adaptaqc.utils.gradients as gr


class TestGeneralGradOfPairs(TestCase):
    def test_given_no_ansatz_then_no_gradient(self):
        qc = transpile(random_circuit(5, 5), basis_gates=["cx", "ry", "rz", "rx"])
        starting_circuit = transpile(
            random_circuit(5, 1), basis_gates=["cx", "ry", "rz", "rx"]
        )

        ansatz = QuantumCircuit(2)

        generators, degeneracies = gr.get_generators_and_degeneracies(ansatz)
        sim = adaptaqc.backends.python_default_backends.MPS_SIM
        gradients = gr.general_grad_of_pairs(
            qc,
            ansatz,
            generators,
            degeneracies,
            coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
            starting_circuit=starting_circuit,
            backend=sim,
        )

        expected_gradients = [0.0, 0.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(gradients, expected_gradients)

    def test_given_random_state_and_rx_ry_ansatz_when_general_grad_then_as_expected(
        self,
    ):
        """
        Given a random state [a,b,c,d] and an arbitrarily chosen ansatz: Rx(θ) on qubit 0 and Ry(ɸ)
        on qubit 1, the analytical partial derivatives w.r.t θ and ɸ around θ=ɸ=0 are:
            dC/dθ|θ,ɸ=0 = -Im(conj(a)b)
            dC/dɸ|θ,ɸ=0 = Re(conj(a)c)
        general_grad_of_pairs returns the Euclidean norm of this.
        """
        qc = transpile(random_circuit(2, 5), basis_gates=["cx", "ry", "rz", "rx"])
        sv = Statevector(qc)
        a = sv.data[0]
        b = sv.data[1]
        c = sv.data[2]

        # Calculate gradient analytically
        theta_grad = -1 * np.imag(np.conj(a) * b)
        phi_grad = np.real(np.conj(a) * c)
        expected_grad = np.sqrt(theta_grad**2 + phi_grad**2)

        # Calculate gradient using general_grad_of_pairs
        ansatz = QuantumCircuit(2)
        ansatz.rx(0, 0)
        ansatz.ry(0, 1)

        generators, degeneracies = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=False, inverse=True
        )
        inverse_zero_ansatz = transpile(ansatz.inverse())
        actual_grad = gr.general_grad_of_pairs(
            qc, inverse_zero_ansatz, generators, degeneracies, coupling_map=[(0, 1)]
        )[0]

        self.assertAlmostEqual(expected_grad, actual_grad, places=10)


class TestGetGenerators(TestCase):
    def test_given_random_ansatz_then_correct_sum_of_degeneracies(self):
        ansatz = transpile(random_circuit(2, 3), basis_gates=["cx", "ry", "rz", "rx"])
        ops = ansatz.count_ops()
        num_rotations = ops.get("rx", 0) + ops.get("ry", 0) + ops.get("rz", 0)

        _, degeneracies_no_rotoselect = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=False
        )
        _, degeneracies_rotoselect = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=True
        )

        self.assertEqual(sum(degeneracies_no_rotoselect), num_rotations)
        self.assertEqual(sum(degeneracies_rotoselect), 3 * num_rotations)

    def test_given_known_ansatz_then_correct_generators(self):
        ansatz = QuantumCircuit(2)
        ansatz.rx(0, 0)
        ansatz.cx(0, 1)

        # All possible generator circuits
        gen_0 = QuantumCircuit(2)
        gen_0.x(0)
        gen_0.cx(0, 1)
        gen_1 = QuantumCircuit(2)
        gen_1.y(0)
        gen_1.cx(0, 1)
        gen_2 = QuantumCircuit(2)
        gen_2.z(0)
        gen_2.cx(0, 1)
        gen_3 = QuantumCircuit(2)
        gen_3.cx(0, 1)
        gen_3.x(0)
        gen_4 = QuantumCircuit(2)
        gen_4.cx(0, 1)
        gen_4.y(0)
        gen_5 = QuantumCircuit(2)
        gen_5.cx(0, 1)
        gen_5.z(0)

        generators_no_rotoselect, _ = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=False, inverse=False
        )
        generators_rotoselect, _ = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=True, inverse=False
        )
        inv_generators_no_rotoselect, _ = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=False, inverse=True
        )
        inv_generators_rotoselect, _ = gr.get_generators_and_degeneracies(
            ansatz, rotoselect=True, inverse=True
        )

        self.assertEqual(generators_no_rotoselect, [gen_0])
        self.assertEqual(generators_rotoselect, [gen_0, gen_1, gen_2])
        self.assertEqual(inv_generators_no_rotoselect, [gen_3])
        self.assertEqual(inv_generators_rotoselect, [gen_3, gen_4, gen_5])

    def test_given_specific_inputs_then_get_generator_returns_correct_generator(self):
        ansatz = QuantumCircuit(2)
        ansatz.rx(0, 0)
        ansatz.ry(0, 1)
        ansatz.cx(0, 1)
        ansatz.rz(0, 0)
        ansatz.rx(0, 1)
        ansatz.cx(1, 0)
        ansatz.ry(0, 0)
        ansatz.rz(0, 1)
        ansatz.cx(1, 0)

        generator = gr.get_generator(ansatz, index=3, op="ry")

        expected_generator = QuantumCircuit(2)
        expected_generator.cx(0, 1)
        expected_generator.y(0)

        self.assertEqual(generator, expected_generator)

    def test_given_ansatz_with_degenerate_generators_then_correct_generators_and_degeneracies(
        self,
    ):
        ansatz = QuantumCircuit(2)
        ansatz.rx(0, 0)
        ansatz.cx(0, 1)
        ansatz.ry(0, 1)
        ansatz.cx(0, 1)
        ansatz.rx(0, 0)

        gen_0 = QuantumCircuit(2)
        gen_0.x(0)
        gen_1 = QuantumCircuit(2)
        gen_1.cx(0, 1)
        gen_1.y(1)
        gen_1.cx(0, 1)

        generators, degeneracies = gr.get_generators_and_degeneracies(ansatz)

        self.assertEqual(generators, [gen_0, gen_1])
        self.assertEqual(degeneracies, [2, 1])

    def test_given_default_ansatzes_then_correct_number_of_generators(self):
        ansatzes = [
            ans.fully_dressed_cnot(),
            ans.heisenberg(),
            ans.identity_resolvable(),
            ans.thinly_dressed_cnot(),
            ans.u4(),
        ]

        num_distinct_generators_no_rotoselect = [8, 5, 4, 4, 11]
        total_num_generators_no_rotoselect = [12, 5, 6, 4, 15]
        num_distinct_generators_rotoselect = [12, 15, 12, 12, 21]
        total_num_generators_rotoselect = [36, 15, 18, 12, 45]

        for i, ansatz in enumerate(ansatzes):
            # No rotoselect
            generators, degeneracies = gr.get_generators_and_degeneracies(
                ansatz, rotoselect=False
            )
            self.assertEqual(len(generators), num_distinct_generators_no_rotoselect[i])
            self.assertEqual(sum(degeneracies), total_num_generators_no_rotoselect[i])

            # Rotoselect
            generators, degeneracies = gr.get_generators_and_degeneracies(
                ansatz, rotoselect=True
            )
            self.assertEqual(len(generators), num_distinct_generators_rotoselect[i])
            self.assertEqual(sum(degeneracies), total_num_generators_rotoselect[i])
