import logging

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_aer.noise import thermal_relaxation_error, NoiseModel
from scipy.optimize import curve_fit

from isl.backends.aer_sv_backend import AerSVBackend
from isl.backends.python_default_backends import QASM_SIM
from isl.backends.qiskit_sampling_backend import QiskitSamplingBackend
from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    unroll_to_basis_gates,
)
from isl.utils.utilityfunctions import (
    counts_data_from_statevector,
    is_statevector_backend,
)

logger = logging.getLogger(__name__)


def run_circuit_with_transpilation(
        circuit: QuantumCircuit,
        backend=QASM_SIM,
        backend_options=None,
        execute_kwargs=None,
        return_statevector=False,
):
    transpiled_circuit = transpile(circuit, backend.simulator)
    return run_circuit_without_transpilation(
        transpiled_circuit, backend, backend_options, execute_kwargs, return_statevector
    )


def run_circuit_without_transpilation(
        circuit: QuantumCircuit,
        backend: QiskitSamplingBackend | AerSVBackend = QASM_SIM,
        backend_options=None,
        execute_kwargs=None,
        return_statevector=False,
):
    if execute_kwargs is None:
        execute_kwargs = {}

    # Backend options only supported for simulators
    if backend_options is None or not isinstance(backend, AerBackend):
        backend_options = {}
    # executing the circuits on the backend and returning the job
    job = backend.simulator.run(circuit, **backend_options, **execute_kwargs)

    result = job.result()
    if is_statevector_backend(backend):
        if return_statevector:
            output = result.get_statevector()
        else:
            output = counts_data_from_statevector(result.get_statevector())
    else:
        output = result.get_counts()

    return output


def create_noisemodel(t1, t2, log_fidelities=True):
    # Instruction times (in nanoseconds)
    time_u1 = 0  # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    t1 = t1 * 1e6
    t2 = t2 * 1e6

    # QuantumError objects
    error_reset = thermal_relaxation_error(t1, t2, time_reset)
    error_measure = thermal_relaxation_error(t1, t2, time_measure)
    error_u1 = thermal_relaxation_error(t1, t2, time_u1)
    error_u2 = thermal_relaxation_error(t1, t2, time_u2)
    error_u3 = thermal_relaxation_error(t1, t2, time_u3)
    error_cx = thermal_relaxation_error(t1, t2, time_cx).expand(
        thermal_relaxation_error(t1, t2, time_cx)
    )

    # Add errors to noise model
    noise_thermal = NoiseModel()
    noise_thermal.add_all_qubit_quantum_error(error_reset, "reset")
    noise_thermal.add_all_qubit_quantum_error(error_measure, "measure")
    noise_thermal.add_all_qubit_quantum_error(error_u1, "u1")
    noise_thermal.add_all_qubit_quantum_error(error_u2, "u2")
    noise_thermal.add_all_qubit_quantum_error(error_u3, "u3")
    noise_thermal.add_all_qubit_quantum_error(error_cx, "cx")

    if log_fidelities:
        logger.info("Noise model fidelities:")
        for qubit_error in noise_thermal.to_dict()["errors"]:
            logging.info(
                f"{qubit_error['operations']}: " f"{max(qubit_error['probabilities'])}"
            )
    return noise_thermal


def zero_noise_extrapolate(
        circuit: QuantumCircuit, measurement_function, num_points=10
):
    calculated_values = []
    probabilities = np.linspace(0, 1, num_points)
    for prob in probabilities:
        circuit_data_copy = circuit.data.copy()
        for i, (gate, qargs, cargs) in list(enumerate(circuit.data))[::-1]:
            if isinstance(gate, CXGate):
                if np.random.random() < prob:
                    circuit.data.insert(i, (gate, qargs, cargs))
                    circuit.data.insert(i, (gate, qargs, cargs))

        calculated_values.append(measurement_function())
        circuit.data = circuit_data_copy

    def exp_decay(x, intercept, amp, decay_rate):
        return intercept + amp * np.exp(-1 * x / decay_rate)

    try:
        popt, pcov = curve_fit(
            exp_decay, probabilities, calculated_values, [0, calculated_values[0], 1]
        )
        zne_val = exp_decay(-0.5, *popt)
        return zne_val
    except RuntimeError as e:
        logger.warning(f"Failed to zero-noise-extrapolate. Error was {e}")
        return measurement_function()
