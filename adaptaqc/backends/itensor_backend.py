import logging

from adaptaqc.backends.aqc_backend import AQCBackend
from adaptaqc.utils.circuit_operations import extract_inner_circuit


class ITensorBackend(AQCBackend):
    def __init__(self, chi=10_000, cutoff=1e-14):
        from itensornetworks_qiskit.utils import qiskit_circ_to_it_circ

        self.qiskit_circ_to_it_circ = qiskit_circ_to_it_circ
        try:
            from juliacall import Main as jl
            from juliacall import JuliaError

            self.jl = jl
            self.jl.seval("using ITensorNetworksQiskit")
        except JuliaError as e:
            logging.error("ITensor backend installation not found")
            raise e
        self.chi = chi
        self.cutoff = cutoff

    def evaluate_global_cost(self, compiler):
        if compiler.soften_global_cost:
            raise NotImplementedError(
                "soften_global_cost is currently only implemented for AerMPSBackend"
            )
        psi = self.evaluate_circuit(compiler)

        n = compiler.total_num_qubits
        return 1 - self.jl.overlap_with_zero_itensors(n, psi, compiler.itensor_sites)

    def evaluate_local_cost(self, compiler):
        raise NotImplementedError()

    def evaluate_circuit(self, compiler):
        ansatz_circ = extract_inner_circuit(
            compiler.full_circuit, compiler.ansatz_range()
        )
        gates = self.qiskit_circ_to_it_circ(ansatz_circ)
        psi = self.jl.mps_from_circuit_and_mps_itensors(
            compiler.itensor_target,
            gates,
            self.chi,
            self.cutoff,
            compiler.itensor_sites,
        )
        return psi

    def measure_qubit_expectation_values(self, compiler):
        raise NotImplementedError()
