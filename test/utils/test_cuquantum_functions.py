from unittest import TestCase

import numpy as np
from aqc_research import mps_operations as mpsops
from qiskit.quantum_info import Statevector

import isl.utils.circuit_operations as co
from isl.utils import cuquantum_functions as cu
from isl.utils.circuit_operations import MPS_SIM, CUQUANTUM_SIM, CUQUANTUM_SIM_EXPERIMENTAL
from isl.utils.utilityfunctions import expectation_value_of_qubits_mps

try:
    import cuquantum

    module_failed = False
except ImportError:
    module_failed = True

class TestCuquantumFunctions(TestCase):

    def setUp(self):
        if module_failed:
            self.skipTest('Skipping as cuquantum is not installed')

    def test_given_circuit_when_cuquantum_mps_then_same_as_aer_mps(self):
        qc = co.create_random_initial_state_circuit(4)
        qc = co.unroll_to_basis_gates(qc)
        cuquantum_mps = cu.cu_mps_from_circuit(qc)
        aer_mps = mpsops.mps_from_circuit(qc, return_preprocessed=True)
        dot_product = mpsops.mps_dot(cuquantum_mps, aer_mps, True)
        self.assertAlmostEqual(dot_product.real, 1.0)
        self.assertAlmostEqual(dot_product.imag, 0)

    def test_given_circuit_when_cuquantum_mps_50_times_then_same_as_aer_mps(self):
        for _ in range(50):
            qc = co.create_random_initial_state_circuit(4)
            cuquantum_mps = cu.cu_mps_from_circuit(qc)
            aer_mps = mpsops.mps_from_circuit(qc, return_preprocessed=True)
            dot_product = mpsops.mps_dot(cuquantum_mps, aer_mps, True)
            self.assertAlmostEqual(dot_product.real, 1.0)
            self.assertAlmostEqual(dot_product.imag, 0)

    def test_given_circuit_when_two_qubit_partial_trace_then_cuquantum_matches_aer(self):
        qc = co.create_random_initial_state_circuit(4)
        aer_mps = mpsops.mps_from_circuit(qc.copy())
        aer_rdm = mpsops.partial_trace(aer_mps, [0, 1])
        cu_rdm = cu.cu_two_qubit_rdm_from_circuit(qc, [0, 1])
        np.testing.assert_allclose(aer_rdm, cu_rdm, atol=1e-6)

    def test_given_circuit_when_compute_expectation_values_then_cuquantum_matches_aer(self):
        qc = co.create_random_initial_state_circuit(4)
        aer_evals = expectation_value_of_qubits_mps(qc, MPS_SIM)
        cu_evals = cu.cu_expectation_value_of_qubits(qc, CUQUANTUM_SIM)
        np.testing.assert_allclose(aer_evals, cu_evals, atol=1e-6)

    def test_when_compute_expectation_values_then_cuquantum_experimental_matches_aer(self):
        qc = co.create_random_initial_state_circuit(4)
        aer_evals = expectation_value_of_qubits_mps(qc, MPS_SIM)
        cu_evals = cu.cu_expectation_value_of_qubits(qc, CUQUANTUM_SIM_EXPERIMENTAL)
        np.testing.assert_allclose(aer_evals, cu_evals, atol=1e-6)

    def test_when_calculate_zero_amplitude_then_cuquantum_experimental_matches_aer(self):
        qc = co.create_random_initial_state_circuit(4)
        aer_ampltiude = Statevector(qc)[0]
        cu_amplitude = cu.cu_get_zero_amplitude(qc)
        np.testing.assert_allclose(aer_ampltiude, cu_amplitude, atol=1e-6)
