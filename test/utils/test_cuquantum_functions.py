from unittest import TestCase

from aqc_research import mps_operations as mpsops

import isl.utils.circuit_operations as co
from isl.utils import cuquantum_functions as cu
from qiskit import QuantumCircuit

try:
    import cuquantum
    import cupy as cp

    module_failed = False
except ImportError:
    module_failed = True

class TestCuquantumFunctions(TestCase):

    def setUp(self):
        if module_failed:
            self.skipTest('Skipping as cuquantum is not installed')

    def test_given_circuit_when_cuquantum_mps_then_same_as_aer_mps(self):
        qc = co.create_random_initial_state_circuit(4, seed=1)
        qc = co.unroll_to_basis_gates(qc)
        cuquantum_mps = cu.mps_from_circuit(qc)
        cuquantum_mps = cu.cu_mps_to_aer_mps(cuquantum_mps)
        aer_mps = mpsops.mps_from_circuit(qc, return_preprocessed=True)
        dot_product = mpsops.mps_dot(cuquantum_mps, aer_mps, True)
        self.assertAlmostEqual(dot_product.real, 1.0)
        self.assertAlmostEqual(dot_product.imag, 0)

    def test_given_circuit_when_cuquantum_mps_50_times_then_same_as_aer_mps(self):
        for _ in range(50):
            qc = co.create_random_initial_state_circuit(4, seed=1)
            cuquantum_mps = cu.mps_from_circuit(qc)
            cuquantum_mps = cu.cu_mps_to_aer_mps(cuquantum_mps)
            aer_mps = mpsops.mps_from_circuit(qc, return_preprocessed=True)
            dot_product = mpsops.mps_dot(cuquantum_mps, aer_mps, True)
            self.assertAlmostEqual(dot_product.real, 1.0)
            self.assertAlmostEqual(dot_product.imag, 0)

    def test_given_1_state_when_contract_bit_flip_circuit_then_0_state_mps(self):
        qc = QuantumCircuit(4)
        qc.x(range(4))
        state_tensor = cp.asarray([0, 1], dtype='complex128').reshape(1,2,1)
        mps_tensors = [state_tensor] * 4
        actual = cu.mps_from_circuit_and_starting_mps(qc, mps_tensors)
        actual_aer_mps = cu.cu_mps_to_aer_mps(actual)
        expected = cu._get_initial_mps(4)
        overlap = mpsops.mps_dot(actual_aer_mps, cu.cu_mps_to_aer_mps(expected), already_preprocessed=True)
        self.assertAlmostEqual(overlap, 1)
        