from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import numpy as np
import sys

def CRn(theta):
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="CRn({})".format(theta))
    """
        -- control
        -- target
    """
    control, target = [qr[i] for i in range(2)]
    qc.p(theta/2, control)
    qc.p(theta/2, target)
    qc.cnot(control, target)
    qc.p(-theta/2, target)
    qc.cnot(control, target)
    return qc.to_instruction()


def twoBitCarry():
    no_qubits = 4
    qr = QuantumRegister(no_qubits)
    qc = QuantumCircuit(qr, name="2-bit carry")
    """
        -- anc1
        -- a
        -- b
        -- anc2
    """
    anc1, a, b, anc2 = [qr[i] for i in range(no_qubits)]

    qc.toffoli(a, b, anc2)
    qc.cnot(a, b)
    qc.toffoli(anc1, b, anc2)
    return qc.to_instruction()


def reverseTwoBitCarry():
    no_qubits = 4
    qr = QuantumRegister(no_qubits)
    qc = QuantumCircuit(qr, name="2-bit reverse carry")
    """
        -- anc1
        -- a
        -- b
        -- anc2
    """
    anc1, a, b, anc2 = [qr[i] for i in range(no_qubits)]

    qc.toffoli(anc1, b, anc2)
    qc.cnot(a, b)
    qc.toffoli(a, b, anc2)

    return qc.to_instruction()


def twoBitSum():
    no_qubits = 3
    qr = QuantumRegister(no_qubits)
    qc = QuantumCircuit(qr, name="2-bit adder")
    """
        -- anc
        -- a
        -- b
    """
    anc, a, b = [qr[i] for i in range(no_qubits)]

    qc.cnot(a, b)
    qc.cnot(anc, b)
    return qc.to_instruction()


def nBitAdder(n, reg_a, reg_b, reg_anc, reverse=False):
    """
    :param n: register size
    :param reg_a: register holding one of the addends
    :param reg_b: register holding one of the addends
    :param reg_anc: register holding ancilla qubits
    :param reverse: parameter specifying whether to reverse the circuit
    :return:
    """

    qc = QuantumCircuit(reg_a, reg_b, reg_anc, name="{}-bit adder, {}".format(n, "rev" if reverse else ""))

    qc.barrier()

    for i in range(n):
        try:
            qc.append(twoBitCarry(), [reg_anc[i], reg_a[i], reg_b[i], reg_anc[i + 1]])
        except:
            qc.append(twoBitCarry(), [reg_anc[i], reg_a[i], reg_b[i], reg_b[i + 1]])

    qc.cnot(reg_a[-1], reg_b[-2])

    qc.append(twoBitSum(), [reg_anc[-1], reg_a[-1], reg_b[-2]])

    for i in range(n - 1):
        qc.append(reverseTwoBitCarry(), [reg_anc[-2 - i], reg_a[-2 - i], reg_b[-3 - i], reg_anc[-1 - i]])
        qc.append(twoBitSum(), [reg_anc[-2 - i], reg_a[-2 - i], reg_b[-3 - i]])

    if reverse: qc = qc.inverse()

    # qc.barrier()

    return qc.to_instruction()


def nbitModNAdder(n, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, reverse=False):
    """
     :param n: register size
     :param reg_a: register holding one of the addends
     :param reg_b: register holding one of the addends
     :param reg_anc: register holding ancilla qubits
     :param reg_N: register holding the divisor
     :param reverse: parameter specifying whether to reverse the circuit
     :return:
     """

    qc = QuantumCircuit(reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit,
                        name="{}-bitModNadder, {}".format(n, "^(-1)" if reverse else ""))

    qc.barrier()

    qc.append(nBitAdder(n, reg_a, reg_b, reg_anc), reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n])

    for i in range(n):
        qc.swap(reg_a[i], reg_N[i])

    qc.append(nBitAdder(n, reg_a, reg_b, reg_anc, reverse=True), reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n])

    qc.barrier()

    # anti control
    qc.x(reg_b[-1])
    qc.cnot(reg_b[-1], reg_tmp_qubit[0])
    qc.x(reg_b[-1])

    qc.barrier()

    # overflow control
    for i in range(n):
        qc.cnot(reg_tmp_qubit[0], reg_a[i])

    qc.append(nBitAdder(n, reg_a, reg_b, reg_anc), reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n])

    # reverse overflow control
    for i in range(n):
        qc.cnot(reg_tmp_qubit[0], reg_a[i])

    qc.barrier()

    for i in range(n):
        qc.swap(reg_a[i], reg_N[i])

    qc.append(nBitAdder(n, reg_a, reg_b, reg_anc, reverse=True), reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n])

    # flip tmp bit to 0 if overflow occurred
    qc.cnot(reg_b[-1], reg_tmp_qubit[0])

    qc.append(nBitAdder(n, reg_a, reg_b, reg_anc), reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n])

    qc.barrier()

    if reverse: qc = qc.inverse()

    return qc.to_instruction()


def getBinModN(a, n, N=sys.maxsize):
    """
    :param a: input integer
    :param n: number of bits to stretch the output to
    :param N: number to take modulus on
    :return: return a string holding the converted a
    """

    output = bin(a % N)[2:].zfill(n)
    assert len(output) == n
    return bin(a % N)[2:].zfill(n)


# not working as intended
def getBinListModN(a, n, N):
    """
    :param a: input number
    :param n: number of bits to stretch the output to
    :param N: number to take modulus on
    :return: reversed list of 0s and 1s corresponding to the binary form of input a (starts from the least significant bit)
    """
    return [a*b for a,b in zip(list(map(int, getBinModN(a, n, N))), list(np.arange(2**n)))]


def nbitModNMultiplier(n, g, N, reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, reverse=False):
    """
        circuit for performing g*x mod N

         :param n: register size
         :param g: constant factor
         :param N: divisor
         :param reg_c_qubit: a qubit controlling whether to perform the operation
         :param reg_x: register holding the variable factor
         :param reg_a: register holding one of the addends
         :param reg_b: register holding one of the addends
         :param reg_anc: register holding ancilla qubits
         :param reg_N: register holding the divisor
         :param reg_tmp_qubit: qubit for controlling overflow
         :param reverse: parameter specifying whether to reverse the circuit
         :return:
         """

    # set g to its multiplicative inverse if gate is reversed
    if reverse: g = pow(g, -1, N)

    name = "({}x_mod_{}){}".format(g, N, "^(-1)" if reverse else "")

    qc = QuantumCircuit(reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, name=name)

    qc.barrier()

    for i in range(n):
        # keep adding exponents of 2 times g
        a = 2 ** i * g

        # convert a to binary mod N
        bin_a_mod = getBinModN(a, n, N)
        print("a = {}, binMod = {}".format(a, bin_a_mod))

        # perform a*2^i for each x_k = 1 in x
        for j, bit in enumerate(bin_a_mod[::-1]):
            qc.toffoli(reg_c_qubit[0], reg_x[i], reg_a[j]) if int(bit) else None
        qc.append(nbitModNAdder(n, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit),
                  reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n] + reg_N[0:n] + reg_tmp_qubit[0:1])
        for j, bit in enumerate(bin_a_mod):
            qc.toffoli(reg_c_qubit[0], reg_x[i], reg_a[-j - 1]) if int(bit) else None

    # add only with x if c == 0
    qc.x(reg_c_qubit[0])
    for i in range(n):
        qc.toffoli(reg_c_qubit[0], reg_x[i], reg_b[i])
    qc.x(reg_c_qubit[0])

    qc.barrier()

    if reverse: qc = qc.inverse()

    return qc.to_instruction()

def nbitModExponentiation(n, g, N, reg_exp, reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit):
    """
        circuit for performing g^x mod N

         :param n: register size
         :param g: constant factor
         :param N: divisor
         :param reg_exp: register holding the exponent
         :param reg_c_qubit: a qubit controlling whether to perform the operation
         :param reg_x: register holding the variable factor
         :param reg_a: register holding one of the addends
         :param reg_b: register holding one of the addends
         :param reg_anc: register holding ancilla qubits
         :param reg_N: register holding the divisor
         :param reg_tmp_qubit: qubit for controlling overflow
         :param reverse: parameter specifying whether to reverse the circuit
         :return:
    """

    name = "{}^x_mod_{}".format(g, N)
    qc = QuantumCircuit(reg_exp, reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, name=name)

    for i in range(n):
        qc.barrier()
        # every succesive gate should perform the multiplication with respect to the remainders of higher powers mo
        g = g ** (2 ** i) % N
        qc.cnot(reg_exp[i], reg_c_qubit[0])
        qc.append(nbitModNMultiplier(n, g, N, reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit), reg_c_qubit[0:1] + reg_x[0:n] + reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n] + reg_N[0:n] + reg_tmp_qubit[0:1])
        for j in range(n):
            qc.swap(reg_x[j], reg_b[j])
        qc.append(nbitModNMultiplier(n, g, N, reg_c_qubit, reg_x, reg_a, reg_b, reg_anc, reg_N, reg_tmp_qubit, reverse=True), reg_c_qubit[0:1] + reg_x[0:n] + reg_a[0:n] + reg_b[0:n + 1] + reg_anc[0:n] + reg_N[0:n] + reg_tmp_qubit[0:1])
        qc.cnot(reg_exp[i], reg_c_qubit[0])

    qc.barrier()

    return qc.to_instruction()

def nbitQFT(n, reg, delta = sys.maxsize, reversed=False):
    """
    :param n: circuit width
    :param reg: register to add the circuit to
    :param delta: cutoff point for logarithm of available phase shift precision
    :param reversed: change theta to -theta to get the inverse Fourier transform
    :return:
    """

    name = "{}-bitQFT{}".format(n, "^(-1)" if reversed else "")
    qc = QuantumCircuit(reg, name=name)

    qc.barrier()

    for i in range(n):
        qc.h(reg[i])
        for j in range(1, n-i):
            if j < delta:
                theta = np.pi / 2 ** j
                if reversed: theta = -theta
                control = i+j
                target = i
                qc.cp(theta, control, target)
                # qc.p(theta/2, reg[control])
                # qc.p(theta/2, reg[target])
                # qc.cnot(reg[control], reg[target])
                # qc.p(-theta/2, reg[target])
                # qc.cnot(reg[control], reg[target])
            else:
                break

    qc.barrier()

    for i in range(n // 2):
        qc.swap(reg[i], reg[n-i-1])

    qc.barrier()

    return qc.to_instruction()