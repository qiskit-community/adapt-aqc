import logging

from isl.backends.aer_mps_backend import AerMPSBackend
from isl.backends.aer_sv_backend import AerSVBackend
from isl.backends.cuquantum_backend import CuQuantumBackend
from isl.backends.qiskit_sampling_backend import QiskitSamplingBackend
from isl.backends.qulacs_backend import QulacsBackend
from isl.backends.tenpy_backend import TenpyBackend

# These constants are generally used in testing and are a relic from before we had custom classes
# for each type of backend.
QASM_SIM = QiskitSamplingBackend()
SV_SIM = AerSVBackend()
CUQUANTUM_SIM = CuQuantumBackend()
MPS_SIM = AerMPSBackend()
QULACS = QulacsBackend()

# Under-development experimental backend
try:
    TENPY_SIM = TenpyBackend()
except ImportError:
    logging.debug('TenPy backend installation not found')
