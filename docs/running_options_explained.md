This document is to give a further in-detail explanation about the different ADAPT-AQC options made
available through `AdaptConfig` and `AdaptCompiler`.

# AdaptConfig()

## method="general_gradient"

This is the core heuristics of ADAPT-AQC, whereby the next two-qubit unitary is placed on the 
pair of qubits which would give the largest cost gradient $\Vert\vec{\nabla} C\Vert$.
For more details please see Appendix A of https://arxiv.org/abs/2503.09683.

## method="ISL"

**What is it:** \
When using this method, the ansatz is adaptively constructed
by prioritising pairs of qubits which have larger pairwise entanglement. This is the original
heuristic from https://github.com/abhishekagarwal2301/isl which has been kept as the default here
as it is the only one which supports all backends.

When
`AdaptConfig.reuse_exponent = 0`, which is the default setting, the pair with the largest
entanglement is always picked, except for picking the same pair twice. For any other
value of `reuse_exponent`, the entanglement of the pair is weighted against how
recently a layer has been applied to it.

If the pairwise entanglement between all qubits in the coupling map is zero, this method falls
back to the `expectation` method defined below.

**What problem it aims to solve:** \
The goal here is to use the entanglement structure of the compilation target to inform the adaptive
ansatz. This is motivated by the fact that the compiling succeeds by finding a set of gates that
"undoes" the target circuit back to the $|00..0\rangle$ state. Since the $|00..0\rangle$ state has
no pairwise entanglement, it makes sense that we want to iteratively reduce this.

## method="expectation"

**What is it:** \
This is another mode of operation for compiling. When using this method, the ansatz is adaptively
constructed by prioritising pairs which have smallest summed $\hat{\sigma}_z$ expectation values (
i.e., the closest to the minimum value of -2)

For the default value of `AdaptConfig.expectation_reuse_exponent = 1` the pair with the smallest
expectation value is also weighted against how recently a layer has been applied to it.

**What problem it aims to solve:** \
In this case, we aim to use the qubit magnetisation of the target to inform the adaptive ansatz.
This has a similar motivation to above, in that each qubit in the $|00..0\rangle$ state has
an expectation value of $\langle0|\hat{\sigma}_z|0\rangle = 1$

## bad_qubit_pair_memory

**What is it:** \
For the ADAPT-AQC method, if acting on a qubit pair leads to entanglement increasing, it is labelled a
"bad pair". After this, for a number of layers corresponding to the bad_qubit_pair_memory,
this pair will not be selected.

**What problem it aims to solve:** \
Although pairwise entanglement is used to select a two-qubit pair to act on, the variational
ansatz is optimised with respect to the overlap with the $|00..0\rangle$ state. The algorithm thus
does
not directly minimise entanglement. This means that in certain situations, the optimal angles of an
ansatz layer could actually increase the entanglement. This would lead to a high priority for acting
on that pair of qubits in the near-future, despite the fact that it was recently optimised. This
can lead ADAPT-AQC to get stuck, particularly if there are several unconnected bad qubit pairs. The use
of `bad_qubit_pair_memory` is to make sure that by the time the pair is acted on again,
the state of connected qubits has sufficiently changed so that optimising the bad pairs will lead
to new optimal angles.

## reuse_exponent

**What is it:** \
For the ADAPT-AQC, expectation or general_gradient methods, this controls how much priority should be given to picking qubits not recently
acted on. Specifically, given a qubit pair has been last acted on $l$ layers ago, it is given a
reuse priority $P_r$ of

$$P_r = 1-2^{\frac{-l}{k}},$$

where $k$ is the value of `reuse_exponent`. This is then multiplied with the 
entanglement measure or gradient (for ADAPT-AQC and general_gradient respectively) to produce the 
combined priority $P_c$ = $E*P_r$. For expectation mode, the combined priority is calculated 
differently. Given a pair of qubits, the combined priority is calculated as

$$P_c = (2 - \langle Z_1 \rangle + \langle Z_2 \rangle) *P_r$$,

where $\langle Z_1 \rangle$ ($\langle Z_2 \rangle$) is the $\\hat{\sigma}_z$ expectation value of qubit
1 (2).

The qubit pair with the highest combined priority is then picked for the next layer.

This means that for larger $k$, more weighting is given to how recently the pair was used.
Conversely, if $k=0$ then no
weighting is given.

**What problem it aims to solve:** \
The goal of approximate quantum compiling (AQC) is to produce a circuit that approximately prepares
a target state **with less depth** than the original circuit. The aim of this heuristic is to make
ADAPT-AQC depth-aware, so that e.g., the same pairs of qubits are not repeatedly picked if they are only
marginally higher entanglement than other pairs that haven't been used. Ultimately, compiling
with a larger exponent produces shallower solutions, at the cost of longer compiling times.

## reuse_priority_mode

**What is it:** \
The reuse priority system is used to de-prioritise qubits that were recently acted on. When
`reuse_priority_mode="pair"`, the priority of a pair of qubits (a, b) is calculated as

$$P_r = 1-2^{\frac{-l}{k}},$$

where $l$ is the number of layers since that pair had a layer applied to it.

When `reuse_priority_mode="qubit"`, the priority is instead calculated as

$$P_r = \mathrm{min}\[1-2^{\frac{-(l_a + 1)}{k}}, 1-2^{\frac{-(l_b + 1)}{k}}\],$$

where $l_a$ or $l_b$ is the number of layers since qubit a or b has been acted on respectively. Note
that in both cases, the priority of the most recently used qubit pair is set to -1 so that it is
never chosen. Additionally, the priority of a pair that has never been used is manually set to 1, so
that it receives maximum priority.

**What problem it aims to solve:** \
This heuristic is meant to reflect that, given a pair of qubits (a, b) was recently acted on, the
depth of the compiled solution will increase if _either_ a or b are acted on in a new layer.
Previously, the "pair" option was the only type of reuse priority in ADAPT-AQC, leading often to
solutions where successive layers might act on pairs (a, a+1), (a+1, a+2), (a+2, a+3)... These
branch-like structures significantly increase the depth of the solution.

## rotosolve_frequency

**What is it:** \
The main optimisation algorithms used by ADAPT-AQC are the Rotoselect and Rotosolve algorithms, more
details of which can be found at https://quantum-journal.org/papers/q-2021-01-28-391/. Put simply,
the Roto algorithms use sequential optimisation. Given a set of $L$ parameterised gates, the procedure
works by fixing $L-1$ of the gates and varying the remaining one to minimise the cost function.
This is then repeated for the remaining $L-1$ gates, unfixing one at a time and fixing the others.
As the changing of later gates in the layer will affect the loss landscape of the first gates, we
repeatedly cycle over all rotation gates until a termination criteria is reached.

`rotosolve_frequency` defines how often the ansatz is optimised using specifically the Rotosolve
algorithm, which only changes the angles of parameterised gates. Specifically, Rotosolve is called
after every `rotosolve_frequency` number of layers have been added. In the context of ADAPT-AQC, it is
notable that _only rotosolve_ has the ability to modify previous layers. Specifically, the last
`AdaptConfig.max_layers_to_modify` layers will be optimised using Rotosolve. This makes it an
expensive step but often necessary to reach convergence.

NOTE Setting the value `rotosolve_frequency=0` will disable rotosolve. This can lead to a large
performance improvement when using the matrix product state (MPS) backends, since the guarantee
that previous layers won't be modified allows us to cache the state of the system during evolution.

**What problem it aims to solve:** \
The use of Rotosolve reflects the idea that after adding layers, the optimal parameters of
previous layers may have changed. Thus it may be more efficient (in terms of final circuit depth)
to attempt to re-optimise previous layers than to only add new layers. As such, when not using
Rotosolve, generally the solution will be deeper.

# AdaptCompiler()

## coupling_map

**What is it:** \
A user specified list of tuples, each of which represents a connection between qubits. ADAPT-AQC will be
restricted to only adding CNOT gates between pairs which are in this coupling map.

**What problem it aims to solve:** \
Often we want to run the ADAPT-AQC solution on a specific connectivity hardware (e.g., heavy hex). Without
this option, it would be possible to convert any solution to a connectivity via SWAP gates, however
this can be extremely expensive in terms of number of gates. This option exists to allow ADAPT-AQC to
restrict the solution space during compiling.

## custom_layer_2q_gate

**What is it:** \
A Qiskit `QuantumCircuit` to be used as the ansatz layers.

**What problem it aims to solve:** \
ADAPT-AQC uses by default a thinly-dressed CNOT gate (i.e., CNOT surrounded by 4 single qubit rotations).
This ansatz is not universal and has not been shown to be objectively better than others, but is a
heuristic that originally worked. As such it may be valuable to change what ansatz layer is used.

## starting_circuit

**What is it:** \
This is a `QuantumCircuit` that will be used as a set of initial fixed gates for the compiled
solution $\hat{V}$. Because during ADAPT-AQC we are variationally optimising $\hat{V}^\dagger$, the
inverse of `starting_circuit` will be placed at the end of the ansatz.

**What problem it aims to solve:** \
`starting_circuit` is a useful heuristic when we have some knowledge of the structure of the
solution. A good example of what this aims to solve is when a compilation target includes
a distinct state preparation step. For example, consider compiling the evolution of a spin-chain
starting in the Neel state. The compiling problem is much more efficient if ADAPT-AQC does not need to
learn to start by applying an X gate to every other qubit.

## local_cost_function

**What is it:**\
Normally, ADAPT-AQC uses a a cost $C_\mathrm{LET} = 1- |\langle 0 | V^\dagger U|0\rangle|^2$. The fidelity
term,
as defined in section IIIB of https://arxiv.org/abs/1908.04416, is generated using the Loschmidt
Echo Test (LET), which is the formal name for acting $U|0\rangle$ followed by $V^\dagger$ to get
the fidelity. We note that the cost is global with respect to the Hilbert space, since the overlap
with the $|00...0\rangle$ state spans the full state vector.

By contrast, when setting `local_cost_function=True`, ADAPT-AQC will use a cost derived from the Local
Loschmidt Echo Test (LLET). Specifically, the cost is defined as

$$C_\mathrm{LLET} = 1 - \frac{1}{n} \sum_{j=1}^{n} \langle 0|\rho^j|0\rangle,$$

where the second term can be
recognised as the sum of the probabilities that each qubit is in the $|0\rangle$ state. Since the
cost function does not span the entire Hilbert space, it is described as local.

**What problem it aims to solve:** \
The distinction between global and local cost functions is very important in the context of
trainability and barren plateaus (see https://www.nature.com/articles/s41467-021-21728-w), where
a global cost function is difficult to train for large numbers of qubits. By contrast the local
cost function is trainable. However, we note that $C_\mathrm{LLET} <= C_\mathrm{LET}$, meaning
that ADAPT-AQC may not have achieved the desired global fidelity just because the local cost converges.

## initial_single_qubit_layer

**What is it:** \
When `initial_single_qubit_layer=True`, the first layer of the ADAPT-AQC ansatz will be
a trainable single-qubit rotation on each qubit. Since this layer will be optimised by Rotoselect,
this means that both the angles and the bases of rotations can be modified. Note that since this
is the first layer of the ADAPT-AQC ansatz $V^\dagger$, it will end up being the final layer of the
returned solution $V$. So we can think of this feature as adding a trainable basis change before
measuring the cost function.

**What problem it aims to solve:** \
ADAPT-AQC only applies layers in two-qubit blocks, which means that in certain situations ADAPT-AQC won't be
able to find the optimal depth solution. A good example of this is when only a subset of the
qubits are entangled. To demonstrate why, for the extreme case of compiling the $n$ qubit
$|++..+\rangle$ state, ADAPT-AQC without this feature will need to apply $n$ CNOT gates. By contrast,
with `initial_single_qubit_layer=True`, a solution can be found in depth 1 with no CNOT gates.
It is possible the same issue can arise for any target state, if during compiling ADAPT-AQC is left
with an intermediate low-entangled state.

## AdaptCompiler.compile_in_parts()

**What is it:** \
Compiling in parts, (also called ladder-ADAPT-AQC), is the idea of splitting up a
circuit into chunks that we compile sequentially. For example, given a depth 50 circuit $U_
{50}|0\rangle$, we can compile the first 10 depth of gates $U_{0-10}|0\rangle$ to produce
$V_{0-10}^\dagger|0\rangle \approx U_{0-10}|0\rangle$. This can then be used to construct a new
target $U_{11-20}V_{0-10}^\dagger|0\rangle$.

A particularly good example of this is for time evolution circuits. Here, we start by compiling only
the first Trotter step. Once we have a solution $V^\dagger_1$, we append a Trotter step to it and
use it as the target state for compiling 2 Trotter steps worth of evolution. We can "ladder" this
all the way to the desired number of Trotter steps. This is shown in Fig.7
of https://arxiv.org/abs/2002.04612.

When applied to time dynamics, compiling in parts is also referred to as restarted quantum dynamics
(https://arxiv.org/abs/1910.06284), iterative variational Trotter
compression (https://arxiv.org/abs/2404.10044) and compressed quantum
circuits (https://arxiv.org/abs/2008.10322). There are inevitably more references using the same
idea.

**What problem it aims to solve:** \
There are two key benefits to compiling in parts. Firstly, compiling smaller chunks makes each
individual optimisation problem easier and less likely to suffer from trainability issues. If
we consider the extreme case of compiling one gate at a time, it is clear that compiling only
needs to learn the application of one more gate on the starting state.

Secondly, if running approximate quantum compiling on real quantum hardware, compiling in parts
allows one to limit the depth of any circuit executed. For example, if the target circuit has a
depth of 20, but a device is limited to depth 10 before noise ruins the computation, one can compile
in blocks of 5 depth at a time. If successful, $V^\dagger$ will never be deeper than $U$, meaning
that the compiling circuit $V^\dagger U |0\rangle$ will never be more than 10 depth.

There are two key drawbacks to compiling in parts. Firstly, we need to solve several sequential
compilation problems, which can take longer than compiling the entire circuit at once (if possible).
Secondly, the approximation error of each individual solution will multiply every time it is used
as the input for the next compiling. Thus the approximation error of the final solution grows
exponentially. For example, if we compile 18 sub-circuits one at a time, with a sufficient
overlap for each one of $0.99$, the overlap of the final solution would be $0.99^{18} = 0.83$. Thus,
one
would need to instead use a much higher sufficient overlap of 0.9995 for each sub-ciruit to get
the desired final overlap of $0.99$.