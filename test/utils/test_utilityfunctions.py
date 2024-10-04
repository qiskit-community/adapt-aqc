from unittest import TestCase

import aqc_research.mps_operations as mpsops
import numpy as np
from numpy.testing import assert_array_almost_equal
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from tenpy import MPS, SpinChain
from tenpy.models import XXZChain

import isl.utils.circuit_operations as co
from isl.backends.python_default_backends import QASM_SIM, SV_SIM
from isl.utils.utilityfunctions import (
    _expectation_value_of_qubit,
    expectation_value_of_qubits,
    expectation_value_of_qubits_mps,
    multi_qubit_gate_depth,
    remove_permutations_from_coupling_map,
    tenpy_to_qiskit_mps,
    tenpy_chi_1_mps_to_circuit,
    qiskit_to_tenpy_mps,
    find_rotation_indices,
)


def get_random_tenpy_mps(num_sites=4, chi=None):
    model_params = dict(L=num_sites, conserve="None")

    model = SpinChain(model_params)
    tenpy_chi = 2**(num_sites // 2) if chi is None else chi
    tenpy_mps = MPS.from_desired_bond_dimension(model.lat.mps_sites(), tenpy_chi)
    return tenpy_mps


class TestUtilityFunctions(TestCase):
    def test_qasm_when_zero_state_then_sigmaz_expectation_is_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()

        job = QASM_SIM.simulator.run(qc, shots=int(1e4))
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1)

    def test_qasm_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = QASM_SIM.simulator.run(qc, shots=int(1e4))
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1)

    def test_qasm_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        qc.measure_all()
        job = QASM_SIM.simulator.run(qc, shots=int(1e4))
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
        job = SV_SIM.simulator.run(qc)
        counts = job.result().get_counts()
        eval_zero_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_zero_state, 1.0)

    def test_sv_when_zero_state_then_sigmaz_expectation_is_minus_one(self):
        qc = QuantumCircuit(1)
        qc.measure_all()
        job = SV_SIM.simulator.run(qc)
        counts = job.result().get_counts()
        eval_one_state = _expectation_value_of_qubit(0, counts, 1)
        self.assertAlmostEqual(eval_one_state, 1.0)

    def test_sv_multi_qubit_sigma_z_expectations(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(1)
        job = SV_SIM.simulator.run(qc)
        sv = job.result().get_statevector()
        eval_one_plus_zero_state = expectation_value_of_qubits(sv)
        assert_array_almost_equal(
            eval_one_plus_zero_state, [-1.0, 0.0, 1.0], decimal=15
        )

    def test_given_unique_coupling_map_when_remove_permutation_then_returned_in_same_order(
        self,
    ):
        cmap = [(1, 2), (2, 3), (3, 4)]
        self.assertEqual(remove_permutations_from_coupling_map(cmap), cmap)

    def test_given_coupling_map_with_permutations_when_remove_permutation_then_unique(
        self,
    ):
        cmap = [(2, 1), (1, 2), (2, 1)]
        self.assertEqual(remove_permutations_from_coupling_map(cmap), [(2, 1)])

    def test_given_coupling_map_with_permutations_when_remove_permutation_then_same_order(
        self,
    ):
        cmap = [(1, 2), (1, 2), (2, 3), (2, 3), (3, 4)]
        self.assertEqual(
            remove_permutations_from_coupling_map(cmap), [(1, 2), (2, 3), (3, 4)]
        )

    def test_find_rotation_indices(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.y(1)
        qc.cx(0, 2)
        qc.rx(1.3, 0)
        qc.ry(0.7, 0)
        qc.cx(0, 1)
        qc.rx(1.1, 2)
        qc.rz(1.6, 2)

        indices = [0, 2, 3, 4, 5, 7]

        expected_rotation_indices = [3, 4, 7]
        rotation_indices = find_rotation_indices(qc, indices)

        self.assertEqual(expected_rotation_indices, rotation_indices)


class TestExpectationValueOfQubitsMPS(TestCase):

    def test_given_circuit_when_mps_expectation_value_then_callable_twice(self):
        """
        Qiskit cannot create a MPS for the same circuit object twice. This test checks that a copy
        of the input circuit is made before generating the MPS
        """
        qc = QuantumCircuit(4)
        expectation_value_of_qubits_mps(qc)
        expectation_value_of_qubits_mps(qc)

    def test_given_n_qubit_circuit_when_mps_expectation_value_then_n_output_values(
        self,
    ):
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


class TestTenpyToQiskitMPS(TestCase):

    def test_given_tenpy_mps_then_not_qiskit_mps(self):
        tenpy_mps = get_random_tenpy_mps()
        is_qiskit_mps = mpsops.check_mps(tenpy_mps)
        self.assertFalse(is_qiskit_mps)

    def test_given_tenpy_mps_when_convert_to_qiskit_mps_then_qiskit_mps(self):
        tenpy_mps = get_random_tenpy_mps()
        qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)
        is_qiskit_mps = mpsops.check_mps(qiskit_mps)
        self.assertTrue(is_qiskit_mps)

    def test_given_tenpy_to_qiskit_mps_then_mps_normalised(self):
        tenpy_mps = get_random_tenpy_mps()
        qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)
        norm = np.sqrt(np.real(mpsops.mps_dot(qiskit_mps, qiskit_mps)))
        self.assertAlmostEqual(norm, 1)

    def test_given_tenpy_mps_to_qiskit_mps_then_statevectors_equal(self):
        n = 4
        tenpy_mps = get_random_tenpy_mps(n)
        qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)

        # NOTE: tenpy uses the opposite notation to qiskit. I.e. the state q0 = |1>, q1 = |0> would
        # be |10> = [0, 0, 1, 0] in tenpy but |01> = [0, 1, 0, 0] in qiskit.
        tenpy_sv = tenpy_mps.get_theta(0, n).to_ndarray().reshape([2] * n)
        tenpy_sv = np.transpose(tenpy_sv, axes=range(n)[::-1])
        tenpy_sv = tenpy_sv.flatten()
        qiskit_sv = mpsops.mps_to_vector(qiskit_mps)

        np.testing.assert_allclose(tenpy_sv, qiskit_sv)

    def test_given_neel_state_then_mps_from_tenpy_and_mps_from_circuit_equal(self):
        n = 3
        # Neel state from tenpy
        # NOTE: tenpy "down" is the same as qiskit |0>. I.e. [1, 0]
        model = XXZChain(
            {
                "L": n,
                "Jxx": 1.0,
                "Jz": 1.0,
                "hz": 0.0,
                "bc_MPS": "finite",
            }
        )
        neel_state = ["up", "down", "up"]
        tenpy_mps = MPS.from_product_state(
            model.lat.mps_sites(), neel_state, bc=model.lat.bc_MPS
        )
        qiskit_mps_from_tenpy = tenpy_to_qiskit_mps(tenpy_mps)

        # Neel state from QuantumCircuit
        qc = QuantumCircuit(n)
        qc.x([0, 2])
        qiskit_mps_from_circuit = mpsops.mps_from_circuit(qc)

        overlap = (
            np.abs(mpsops.mps_dot(qiskit_mps_from_tenpy, qiskit_mps_from_circuit)) ** 2
        )

        self.assertEqual(overlap, 1)

    def test_given_tenpy_mps_when_initialise_qiskit_with_it_then_output_qiskit_mps_same(
        self,
    ):
        n = 4
        tenpy_mps = get_random_tenpy_mps(n)
        qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)

        mps_as_circuit = QuantumCircuit(n)
        mps_as_circuit.set_matrix_product_state(qiskit_mps)
        mps_after_circuit = mpsops.mps_from_circuit(mps_as_circuit)

        overlap = np.abs(mpsops.mps_dot(mps_after_circuit, qiskit_mps)) ** 2
        self.assertAlmostEqual(overlap, 1, places=10)

        for i in range(n):
            gamma_before = np.array(qiskit_mps[0][i])
            gamma_after = np.array(mps_after_circuit[0][i])
            np.testing.assert_allclose(gamma_before, gamma_after)
            if i < n - 1:
                lambda_before = np.array(qiskit_mps[1][i])
                lambda_after = np.array(mps_after_circuit[1][i])
                np.testing.assert_allclose(lambda_before, lambda_after)


class TestTenpyChi1MPSToCircuit(TestCase):

    def test_given_random_tenpy_mps_when_map_to_circuit_then_correct_number_of_gates(
        self,
    ):
        mps = get_random_tenpy_mps(chi=1)
        qc = tenpy_chi_1_mps_to_circuit(mps)
        self.assertLess(len(qc), 13)

    def test_given_mps_with_chi_greater_than_1_then_error(self):
        mps = get_random_tenpy_mps(chi=2)
        with self.assertRaises(Exception):
            tenpy_chi_1_mps_to_circuit(mps)

    def test_given_random_chi_1_tenpy_mps_when_map_to_circuit_then_fidelity_1_with_mps(
        self,
    ):
        mps = get_random_tenpy_mps(chi=1)
        qc = tenpy_chi_1_mps_to_circuit(mps)

        mps_from_tenpy = tenpy_to_qiskit_mps(mps)
        mps_from_qc = mpsops.mps_from_circuit(qc)

        fidelity = np.abs(mpsops.mps_dot(mps_from_qc, mps_from_tenpy)) ** 2
        self.assertAlmostEqual(fidelity, 1)

    def test_given_random_large_chi_1_tenpy_mps_when_map_to_circuit_then_fidelity_1_with_mps(
        self,
    ):
        mps = get_random_tenpy_mps(num_sites=100, chi=1)
        qc = tenpy_chi_1_mps_to_circuit(mps)

        mps_from_tenpy = tenpy_to_qiskit_mps(mps)
        mps_from_qc = mpsops.mps_from_circuit(qc)

        fidelity = np.abs(mpsops.mps_dot(mps_from_qc, mps_from_tenpy)) ** 2
        self.assertAlmostEqual(fidelity, 1)

    def test_given_random_compressed_chi_1_mps_when_map_to_circuit_then_fidelity_1_with_compress_mps(
        self,
    ):
        mps = get_random_tenpy_mps(chi=4)
        compression_options = {
            "compression_method": "variational",
            "trunc_params": {"chi_max": 1},
            "max_trunc_err": 1,
            "max_sweeps": 100,
            "min_sweeps": 50,}
        mps.compress(compression_options)
        qc = tenpy_chi_1_mps_to_circuit(mps)
        mps_from_tenpy = tenpy_to_qiskit_mps(mps)
        mps_from_qc = mpsops.mps_from_circuit(qc)

        fidelity = np.abs(mpsops.mps_dot(mps_from_qc, mps_from_tenpy))**2
        self.assertAlmostEqual(fidelity, 1)

    def test_given_neel_state_when_map_to_circuit_then_correct(self):
        model = XXZChain({"L": 3})
        neel_state = ["up", "down", "up"]
        mps = MPS.from_product_state(model.lat.mps_sites(), neel_state)
        qc = tenpy_chi_1_mps_to_circuit(mps)
        sv = Statevector(qc).data
        expected_sv = np.array([0, 0, 0, 0, 0, 1, 0, 0])

        np.testing.assert_allclose(sv, expected_sv, atol=1e-10)


class TestQiskitToTenpyMPS(TestCase):

    def test_given_qiskit_to_tenpy_mps_then_mps_normalised(self):
        for prep in [True, False]:
            qc = co.create_random_initial_state_circuit(3)
            qiskit_mps = mpsops.mps_from_circuit(qc, return_preprocessed=prep)
            tenpy_mps = qiskit_to_tenpy_mps(qiskit_mps)

            self.assertAlmostEqual(tenpy_mps.norm, 1)

    def test_given_qiskit_mps_to_tenpy_mps_then_statevectors_equal(self):
        for prep in [True, False]:
            n = 4
            qc = co.create_random_initial_state_circuit(n)
            qiskit_mps = mpsops.mps_from_circuit(qc, return_preprocessed=prep)
            tenpy_mps = qiskit_to_tenpy_mps(qiskit_mps)

            # NOTE: tenpy uses the opposite notation to qiskit. I.e. the state q0 = |1>, q1 = |0> would
            # be |10> = [0, 0, 1, 0] in tenpy but |01> = [0, 1, 0, 0] in qiskit.
            tenpy_sv = tenpy_mps.get_theta(0, n).to_ndarray().reshape([2] * n)
            tenpy_sv = np.transpose(tenpy_sv, axes=range(n)[::-1])
            tenpy_sv = tenpy_sv.flatten()
            qiskit_sv = mpsops.mps_to_vector(qiskit_mps, already_preprocessed=prep)

            np.testing.assert_allclose(tenpy_sv, qiskit_sv)

    def test_given_neel_state_then_tenpy_mps_as_expected(self):
        for prep in [True, False]:
            n = 3
            qc = QuantumCircuit(n)
            qc.x([0, 2])
            qiskit_mps = mpsops.mps_from_circuit(qc, return_preprocessed=prep)
            tenpy_mps = qiskit_to_tenpy_mps(qiskit_mps)

            # Convert to sv
            tenpy_sv = tenpy_mps.get_theta(0, n).to_ndarray().reshape([2] * n)
            tenpy_sv = np.transpose(tenpy_sv, axes=range(n)[::-1])
            tenpy_sv = tenpy_sv.flatten()

            expected_sv = [0, 0, 0, 0, 0, 1, 0, 0]

            np.testing.assert_allclose(tenpy_sv, expected_sv)

    def test_converter_functions_are_inverses(self):
        for prep in [True, False]:
            n = 4
            # Tenpy -> Qiskit -> Tenpy
            tenpy_mps = get_random_tenpy_mps(n)
            qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)
            if prep:
                qiskit_mps = mpsops._preprocess_mps(qiskit_mps)
            tenpy_mps_reconstructed = qiskit_to_tenpy_mps(qiskit_mps)

            fidelity = np.abs(tenpy_mps.overlap(tenpy_mps_reconstructed)) ** 2
            self.assertAlmostEqual(fidelity, 1)

            # Qiskit -> Tenpy -> Qiskit
            qc = co.create_random_initial_state_circuit(n)
            qiskit_mps = mpsops.mps_from_circuit(qc, return_preprocessed=prep)
            tenpy_mps = qiskit_to_tenpy_mps(qiskit_mps)
            qiskit_mps_reconstructed = tenpy_to_qiskit_mps(tenpy_mps)
            if prep:
                qiskit_mps_reconstructed = mpsops._preprocess_mps(qiskit_mps_reconstructed)

            fidelity = (
                np.abs(
                    mpsops.mps_dot(
                        qiskit_mps, qiskit_mps_reconstructed, already_preprocessed=prep
                    )
                )
                ** 2
            )
            self.assertAlmostEqual(fidelity, 1)
