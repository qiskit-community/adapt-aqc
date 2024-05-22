"""
Comparing between Qulacs and Qiskit for compiling random circuits
"""
import os.path
import platform

import numpy as np
import qiskit
import qiskit_aer
import qulacs
from matplotlib import pyplot as plt
from qiskit import transpile
from qiskit.circuit.random import random_circuit

from isl.recompilers import ISLRecompiler, ISLConfig
from isl.utils.circuit_operations import SV_SIM
from isl.utils.constants import DEFAULT_SUFFICIENT_COST


def run_random_circuits(n_max, depth, backend, n_repeats):
    times_per_qubit = []
    for n in range(2, n_max + 1):
        print(f"Running ISL for {n} qubit random circuit")
        times_per_repeat = []
        for i in range(n_repeats):
            print(f"Repeat {i + 1} of {n_repeats}")
            qc = random_circuit(n, depth, 2, seed=i)

            qc = transpile(qc, optimization_level=0, basis_gates=["cx", "u1", "u2", "u3"])

            # Abort ISL if the number of cnots is larger than original circuit
            config = ISLConfig(cost_improvement_num_layers=1e3, method="ISL", max_2q_gates=qc.count_ops().get("cx"))
            recompiler = ISLRecompiler(qc, isl_config=config,
                                       backend=backend, execute_kwargs={})
            result = recompiler.recompile()

            overlap = result.overlap
            if 1 - overlap > DEFAULT_SUFFICIENT_COST:
                # This is our marker for an aborted run
                time = -1
            else:
                time = result.time_taken
            times_per_repeat.append(time)
        times_per_qubit.append(times_per_repeat)

    return np.array(times_per_qubit)


def main():
    n_max = 15
    n_repeats = 10
    depth = 5

    fn = f"results_nmax_{n_max}_depth_{depth}_repeats_{n_repeats}.npy"

    # Load results if available
    if os.path.isfile(fn):
        data = np.load(fn, allow_pickle=True)
        num_qubits = data[0][0]
        qiskit_times = data[1]
        qulacs_times = data[2]

    else:
        qiskit_times = run_random_circuits(n_max, depth, SV_SIM, n_repeats)
        qulacs_times = run_random_circuits(n_max, depth, "qulacs", n_repeats)
        num_qubits = list(range(2, n_max + 1))
        np.save(fn, np.array([[num_qubits], qiskit_times, qulacs_times]))

    # Remove runs where ISL wasn't successful
    qiskit_times = np.ma.masked_equal(qiskit_times, -1)
    qulacs_times = np.ma.masked_equal(qulacs_times, -1)

    # Qulacs and Qiskit's statevector simulators produce stochastically different results. For a few runs, one backend
    # gets convergence and another doesn't. This code makes sure we don't use any results where this happens.
    qiskit_times = qiskit_times / qulacs_times * qulacs_times
    qulacs_times = qulacs_times / qiskit_times * qiskit_times

    plt.scatter(np.repeat(num_qubits, n_repeats), qiskit_times.flatten(), alpha=0.1)
    plt.scatter(np.repeat(num_qubits, n_repeats), qulacs_times.flatten(), alpha=0.1)

    qiskit_mean = np.ma.mean(qiskit_times, axis=1)
    qulacs_mean = np.ma.mean(qulacs_times, axis=1)

    # Absolute time plot
    plt.errorbar(num_qubits, qiskit_mean, fmt='x', label="Qiskit", markersize=10, markeredgewidth=2)
    plt.errorbar(num_qubits, qulacs_mean, fmt='x', label="Qulacs", markersize=10, markeredgewidth=2)

    details = (f"random circuit depth = {depth}\n"
               f"CPU: {platform.platform()}\n"
               f"qiskit v{qiskit.__version__}\n"
               f"aer v{qiskit_aer.__version__}\n"
               f"qulacs v%s.%s.%s" % qulacs.__version_tuple__)

    plt.text(.50, .25, details, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.yscale('log')
    plt.ylabel("Time /s")
    plt.xlabel("Number of qubits")
    plt.legend()
    plt.show()

    # Relative time plot
    relative_time = qiskit_times / qulacs_times
    relative_time_mean = np.ma.mean(relative_time, axis=-1)
    relative_time_stdev = np.ma.std(relative_time, axis=-1)

    plt.scatter(np.repeat(num_qubits, n_repeats), relative_time.flatten(), alpha=0.1)
    plt.errorbar(num_qubits, relative_time_mean, relative_time_stdev, capsize=4, markersize=10, fmt='x',
                 markeredgewidth=2, label="Qiskit")
    plt.errorbar(num_qubits, np.ones(len(num_qubits)), fmt='--', label="Qulacs")

    plt.text(.01, .25, details, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.yscale('linear')
    plt.ylabel("Time relative to Qulacs")
    plt.xlabel("Number of qubits")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
