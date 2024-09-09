import logging

from isl.backends.aer_mps_backend import AerMPSBackend
from isl.backends.aer_sv_backend import AerSVBackend
from isl.backends.qiskit_sampling_backend import QiskitSamplingBackend

# These constants are generally used in testing and are a relic from before we had custom classes
# for each type of backend.
QASM_SIM = QiskitSamplingBackend()
SV_SIM = AerSVBackend()
MPS_SIM = AerMPSBackend()
