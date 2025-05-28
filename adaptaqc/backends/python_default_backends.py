# (C) Copyright IBM 2025. 
# 
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# 
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from adaptaqc.backends.aer_mps_backend import AerMPSBackend
from adaptaqc.backends.aer_sv_backend import AerSVBackend
from adaptaqc.backends.qiskit_sampling_backend import QiskitSamplingBackend

# These constants are generally used in testing and are a relic from before we had custom classes
# for each type of backend.
QASM_SIM = QiskitSamplingBackend()
SV_SIM = AerSVBackend()
MPS_SIM = AerMPSBackend()
