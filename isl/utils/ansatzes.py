from qiskit import QuantumCircuit


def u4():
    """
    U(4) ansatz from Fig. 6 of
    Vatan, Farrokh, and Colin Williams. "Optimal quantum circuits for general two-qubit gates."
    Physical Review A 69.3 (2004): 032315.
    """
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    qc.cx(1, 0)
    qc.rz(0, 0)
    qc.ry(0, 1)
    qc.cx(0, 1)
    qc.ry(0, 1)
    qc.cx(1, 0)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    return qc


def thinly_dressed_cnot():
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.cx(0, 1)
    qc.rz(0, 0)
    qc.rz(0, 1)
    return qc


def fully_dressed_cnot():
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    qc.cx(0, 1)
    qc.rz(0, 0)
    qc.ry(0, 0)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.ry(0, 1)
    qc.rz(0, 1)
    return qc


def identity_resolvable():
    qc = QuantumCircuit(2)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.cx(0, 1)
    qc.rz(0, 0)
    qc.rz(0, 1)
    qc.cx(0, 1)
    qc.rz(0, 0)
    qc.rz(0, 1)
    return qc
