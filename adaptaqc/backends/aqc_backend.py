import abc


class AQCBackend(abc.ABC):
    @abc.abstractmethod
    def evaluate_global_cost(self, compiler):
        pass

    @abc.abstractmethod
    def evaluate_local_cost(self, compiler):
        pass

    @abc.abstractmethod
    def evaluate_circuit(self, compiler):
        pass

    @abc.abstractmethod
    def measure_qubit_expectation_values(self, compiler):
        pass
