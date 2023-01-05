from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
import importlib
import functions, arithmetic
importlib.reload(functions)
importlib.reload(arithmetic)

def adderCircuit(a, b, reverse=False):
    """
    :param a: first addend
    :param b: second addend
    :return: quantum circuit for the addition of a and b
    """

    n = len(bin(max(a, b))[2:])
    # create registers
    reg_a = QuantumRegister(n, name="a")
    reg_b = QuantumRegister(n + 1, name="b")
    reg_anc = QuantumRegister(n, name="ancilla")
    reg_control = QuantumRegister(2, name="control")
    cr = ClassicalRegister(n + 1, name="c")

    # create circuit
    qc = QuantumCircuit(reg_a, reg_b, reg_anc, reg_control, cr, name="{}-bit adder".format(n))

    # handle cases where a or b is 0 with a try except block
    try:
        qc.x(reg_a[i] for i in functions.getOneIndices(a))
    except CircuitError:
        pass
    try:
        qc.x(reg_b[i] for i in functions.getOneIndices(b))
    except CircuitError:
        pass

    # adder
    qc.append(arithmetic.nBitAdder(n, reg_a, reg_b, reg_anc, reverse=reverse), reg_a[:] + reg_b[:] + reg_anc[:])

    qc.barrier()
    qc.measure(reg_b, cr)

    return qc

def modNAdderCircuit(a, b, N, reverse=False):
    """
    :param a: first addend
    :param b: second addend
    :param N: modulus
    :return: quantum circuit for the addition of a and b mod N
    """

    n = len(bin(max(a, b, N))[2:])
    # create registers
    reg_a = QuantumRegister(n, name="a")
    reg_b = QuantumRegister(n + 1, name="b")
    reg_anc = QuantumRegister(n, name="anc")
    reg_N = QuantumRegister(n, name="modN")
    reg_tmp_qubit = QuantumRegister(1, name="tmp_qubit")
    cr = ClassicalRegister(n + 1, name="output")

    # create circuit
    qc = QuantumCircuit(reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, cr)

    # handle cases where a or b is 0 with a try except block
    try:
        qc.x(reg_a[i] for i in functions.getOneIndices(a))
    except CircuitError:
        pass
    try:
        qc.x(reg_b[i] for i in functions.getOneIndices(b))
    except CircuitError:
        pass
    try:
        qc.x(reg_N[i] for i in functions.getOneIndices(N))
    except CircuitError:
        pass

    qc.append(arithmetic.nbitModNAdder(n, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, reverse=reverse),
              reg_a[:] + reg_b[:] + reg_anc[:] + reg_N[:] + reg_tmp_qubit[:])

    # revert N to zeros
    try:
        qc.x(reg_N[i] for i in functions.getOneIndices(N))
    except CircuitError:
        pass

    qc.barrier()
    qc.measure(reg_b, cr)

    return qc
