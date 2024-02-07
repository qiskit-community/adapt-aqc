from unittest import TestCase
from unittest import TestCase
from unittest.mock import patch

import aqc_research.mps_operations as mpsops
import numpy as np
from qiskit import Aer, QuantumCircuit

import isl.utils.circuit_operations as co
import isl.utils.entanglement_measures as em
from isl.recompilers import ISLRecompiler
from isl.utils.circuit_operations import MPS_SIM, SV_SIM
from isl.utils.entanglement_measures import perform_quantum_tomography, EM_TOMOGRAPHY_CONCURRENCE, \
    EM_TOMOGRAPHY_NEGATIVITY, EM_TOMOGRAPHY_EOF


class TestEntanglementMeasures(TestCase):
    def test_quantum_tomography(self):

        qc = co.create_random_initial_state_circuit(3)
        dm = perform_quantum_tomography(qc, 0, 1, Aer.get_backend("qasm_simulator"))
        assert isinstance(dm, np.ndarray)

    def test_tomography_entanglement_measures(self):

        qc = co.create_random_initial_state_circuit(3)
        for backend in [
            Aer.get_backend("qasm_simulator"),
            Aer.get_backend("statevector_simulator"),
        ]:
            for method in [
                em.EM_TOMOGRAPHY_CONCURRENCE,
                em.EM_TOMOGRAPHY_EOF,
                em.EM_TOMOGRAPHY_NEGATIVITY,
            ]:
                em.calculate_entanglement_measure(method, qc, 0, 1, backend)

    def test_observable_min_concurrence(self):
        qc = co.create_random_initial_state_circuit(3)
        em.measure_concurrence_lower_bound(qc, 0, 1, Aer.get_backend("qasm_simulator"))

    @patch('aqc_research.mps_operations.partial_trace')
    def test_given_mps_backend_when_calculate_em_measure_then_mps_operations_partial_trace_called(
            self, mock_mps_partial_trace):

        mock_mps_partial_trace.return_value = np.ones((4, 4))

        backend = MPS_SIM
        qc = co.create_random_initial_state_circuit(3)
        qc_mps = mpsops.mps_from_circuit(qc, print_log_data=False)
        em.calculate_entanglement_measure(em.EM_TOMOGRAPHY_CONCURRENCE, qc, 0, 1, backend, mps=qc_mps)

        mock_mps_partial_trace.assert_called_once_with(qc_mps, [0, 1])

    @patch('aqc_research.mps_operations.mps_from_circuit')
    def test_given_mps_backend_when_get_all_qubit_pair_em_measure_then_mps_from_circuit_called_exactly_once(
            self, mock_mps_from_circuit):

        from numpy import array
        qc = QuantumCircuit(2)
        circ_mps = ([(array([[1. + 0.j]]), array([[0. + 0.j]])),
                     (array([[1. + 0.j]]), array([[0. + 0.j]]))],
                    [array([1.])])
        mock_mps_from_circuit.return_value = circ_mps
        recompiler = ISLRecompiler(qc, backend=co.MPS_SIM)
        recompiler.circ_mps = circ_mps
        recompiler._get_all_qubit_pair_entanglement_measures()
        mock_mps_from_circuit.assert_called_once()

    def test_given_random_state_when_backend_mps_or_statevector_then_ent_measures_equal(self):
        qc = co.create_random_initial_state_circuit(3)

        entanglement_measures = [EM_TOMOGRAPHY_CONCURRENCE, EM_TOMOGRAPHY_NEGATIVITY, EM_TOMOGRAPHY_EOF]

        for i in entanglement_measures:
            sv_recompiler = ISLRecompiler(qc, entanglement_measure=i, backend=SV_SIM)
            mps_recompiler = ISLRecompiler(qc, entanglement_measure=i, backend=MPS_SIM)

            np.testing.assert_allclose(
                sv_recompiler._get_all_qubit_pair_entanglement_measures(),
                mps_recompiler._get_all_qubit_pair_entanglement_measures(),
                atol=1e-06)
