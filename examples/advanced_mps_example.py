"""
Example script for running ADAPT-AQC recompilation using more advanced options
"""

import logging
import matplotlib.pyplot as plt

from tenpy import SpinChain, MPS
from tenpy.algorithms import dmrg

from adaptaqc.backends.aer_mps_backend import AerMPSBackend, mps_sim_with_args
from adaptaqc.compilers import AdaptCompiler, AdaptConfig
from adaptaqc.utils.ansatzes import identity_resolvable
from adaptaqc.utils.utilityfunctions import tenpy_to_qiskit_mps

logging.basicConfig()
logger = logging.getLogger("adaptaqc")
logger.setLevel(logging.INFO)

# Generate a ground state of the XXZ model using TenPy
l = 20
model_params = dict(
    S=0.5, L=l, Jx=1.0, Jy=1.0, Jz=5.0, hz=1.0, bc_MPS="finite", conserve="None"
)
model = SpinChain(model_params)

psi = MPS.from_product_state(
    model.lat.mps_sites(), ["up", "down"] * (l // 2), bc=model_params["bc_MPS"]
)

# Run the DMRG algorithm to obtain the ground state
dmrg_params = {"trunc_params": {"trunc_cut": 1e-4}}
dmrg_engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
E, psi = dmrg_engine.run()
logger.info(f"Ground state created with maximum bond dimension {max(psi.chi)}")

# Convert it to a format compatible with the Qiskit Aer MPS simulator
qiskit_mps = tenpy_to_qiskit_mps(psi)

# Set compiler to use the general gradient method as laid out in https://arxiv.org/abs/2503.09683
config = AdaptConfig(
    method="general_gradient", cost_improvement_num_layers=1e3, rotosolve_frequency=10
)

# Create an instance of Qiskit's MPS simulator with a specified truncation threshold
qiskit_mps_sim = mps_sim_with_args(mps_truncation_threshold=1e-8)

# Create an AQCBackend object
backend = AerMPSBackend(simulator=qiskit_mps_sim)

# Create a compiler with the target to be an MPS rather than a circuit
adapt_compiler = AdaptCompiler(
    target=qiskit_mps,
    backend=backend,
    adapt_config=config,
    starting_circuit="tenpy_product_state",  # Start compiling from best χ=1 compression of target
    custom_layer_2q_gate=identity_resolvable(),  # Use ansatz from https://arxiv.org/abs/2503.09683
)

result = adapt_compiler.compile()
approx_circuit = result.circuit
print(f"Overlap between circuits is {result.overlap}")

# Draw the circuit that prepares the target random MPS
approx_circuit.draw(output="mpl", fold=-1)
plt.show()
