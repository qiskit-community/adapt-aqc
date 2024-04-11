"""Contains functions for cuquantum backend operations"""
import logging

from aqc_research.mps_operations import mps_expectation

logger = logging.getLogger(__name__)

try: 
    import cupy as cp
    from cuquantum import contract, CircuitToEinsum
    from cuquantum import CircuitToEinsum, cutensornet as cutn
    from cuquantum.cutensornet.experimental import contract_decompose
except ModuleNotFoundError as e:
    logger.debug(e)

def _mps_site_right_swap(
    mps_tensors, 
    i, 
    algorithm=None, 
    options=None
):
    """
    Perform the swap operation between the ith and i+1th MPS tensors.
    """
    # contraction followed by QR decomposition
    a, _, b = contract_decompose('ipj,jqk->iqj,jpk', *mps_tensors[i:i+2], algorithm=algorithm, options=options)
    mps_tensors[i:i+2] = (a, b)
    return mps_tensors

def _apply_gate(
    mps_tensors, 
    gate, 
    qubits, 
    algorithm=None, 
    options=None
):
    """
    Apply the gate operand to the MPS tensors in-place.
    
    Args:
        mps_tensors: A list of rank-3 ndarray-like tensor objects. 
            The indices of the ith tensor are expected to be the bonding index to the i-1 tensor, 
            the physical mode, and then the bonding index to the i+1th tensor.
        gate: A ndarray-like tensor object representing the gate operand. 
            The modes of the gate is expected to be output qubits followed by input qubits, e.g, 
            ``A, B, a, b`` where ``a, b`` denotes the inputs and ``A, B`` denotes the outputs. 
        qubits: A sequence of integers denoting the qubits that the gate is applied onto.
        algorithm: The contract and decompose algorithm to use for gate application. 
            Can be either a `dict` or a `ContractDecomposeAlgorithm`.
        options: Specify the contract and decompose options. 
    
    Returns:
        The updated MPS tensors.
    """
    
    n_qubits = len(qubits)
    if n_qubits == 1:
        # single-qubit gate
        i = qubits[0]
        mps_tensors[i] = contract('ipj,qp->iqj', mps_tensors[i], gate, options=options) # in-place update
    elif n_qubits == 2:
        # two-qubit gate
        i, j = qubits
        if i > j:
            # swap qubits order
            return _apply_gate(mps_tensors, gate.transpose(1,0,3,2), (j, i), algorithm=algorithm, options=options)
        elif i+1 == j:
            # two adjacent qubits
            a, _, b = contract_decompose('ipj,jqk,rspq->irj,jsk', *mps_tensors[i:i+2], gate, algorithm=algorithm, options=options)
            mps_tensors[i:i+2] = (a, b) # in-place update
        else:
            # non-adjacent two-qubit gate
            # step 1: swap i with i+1
            _mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
            # step 2: apply gate to (i+1, j) pair. This amounts to a recursive swap until the two qubits are adjacent
            _apply_gate(mps_tensors, gate, (i+1, j), algorithm=algorithm, options=options) 
            # step 3: swap back i and i+1
            _mps_site_right_swap(mps_tensors, i, algorithm=algorithm, options=options)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")
    return mps_tensors

def _get_initial_mps(num_qubits, dtype='complex128'):
    """
    Generate the MPS with an initial state of |00...00> 
    """
    state_tensor = cp.asarray([1, 0], dtype=dtype).reshape(1,2,1)
    mps_tensors = [state_tensor] * num_qubits
    return mps_tensors

def cu_mps_from_circuit(circuit):
    # Start with initial MPS
    mps_tensors = _get_initial_mps(circuit.num_qubits, dtype='complex128')
    # We leverage ``cuquantum.CircuitToEinsum`` to obtain the gate operands.
    converter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
    gates = converter.gates
    gate_map = dict(zip(converter.qubits, range(circuit.num_qubits)))
    # We construct an exact MPS with algorithm below. 
    # For two-qubit gates, an SVD is performed with singular values partitioned onto the two MPS sites equally.
    # We also set a cutoff value of 1e-12 to filter out computational noise.
    exact_gate_algorithm = {'qr_method': False, 
                            'svd_method':{'partition': 'UV', 'abs_cutoff':1e-12}}
    
    # the same handle can be reused for further calls
    handle = cutn.create()
    # Constructing the final MPS
    for (gate, qubits) in gates:
        # mapping from qubits to qubit indices
        qubits = [gate_map[q] for q in qubits]
        # apply the gate in-place
        _apply_gate(mps_tensors, gate, qubits, algorithm=exact_gate_algorithm, options={'handle': handle})

    # when it's done, remember to destroy the handle
    cutn.destroy(handle)
    return [cp.asnumpy(cp.swapaxes(e, 0, 1)) for e in mps_tensors]


def cu_two_qubit_rdm_from_circuit(circuit, qubits_to_keep):
    """
    Returns specifically the two-qubit RDM using CuQuantum. Not generalised to more than 2 qubits
    due to the mismatched indexing between CuQuantum and Qiskit.
    @param circuit: Qiskit circuit
    @param qubits_to_keep: A list of two qubits to not be traced out
    """
    if len(qubits_to_keep) > 2:
        raise NotImplementedError("Only 2-qubit RDMs supported for CuQuantum backend")
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    where = [circuit.qubits[qubit] for qubit in qubits_to_keep]
    # we set lightcone=True to reduce the size of the tensor network
    expression, operands = myconverter.reduced_density_matrix(where, lightcone=True)
    rho = contract(expression, *operands)
    rho = cp.swapaxes(rho, 0, 1)
    rho = rho.reshape(2**len(qubits_to_keep), 2**len(qubits_to_keep))
    rho = rho[:, [0, 2, 1, 3]]
    return cp.asnumpy(rho)


def cu_expectation_value_of_qubits(circuit, backend):
    if backend == "cuquantum_experimental":
        mps = cu_mps_from_circuit(circuit)
        return [(mps_expectation(mps, 'Z', i, already_preprocessed=True))
                for i in range(len(mps))]
    else:
        n = circuit.num_qubits
        myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
        pauli_strings = ["I" * i + "Z" + "I"*(n-i-1) for i in range(n)]

        expectations = []
        for pauli in pauli_strings:
            expression, operands = myconverter.expectation(pauli, lightcone=True)
            expectation = contract(expression, *operands)
            assert cp.abs(cp.imag(expectation)) <= 1e-10
            expectations.append(cp.asnumpy(cp.real(expectation)))

        return [float(elem) for elem in expectations]

def cu_get_zero_amplitude(circuit):
    myconverter = CircuitToEinsum(circuit, dtype="complex128", backend=cp)
    n = circuit.num_qubits
    bitstring = '0'*n
    expression, operands = myconverter.amplitude(bitstring)
    amplitude = contract(expression, *operands)
    return cp.asnumpy(amplitude)
