from qiskit import QuantumRegister, QuantumCircuit

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

def nBitAdder(n):
    qr = QuantumRegister(3*n+1)
    qc = QuantumCircuit(qr, name="{}-bit adder".format(n))
    carry_bitlist = [0,1,2,3]
    for i in range(n):
        qc.append(twoBitCarry(), [qr[x + 3 * i] for x in carry_bitlist])
    qc.cnot(qr[-3], qr[-2])

    sum_bitlist = [-4, -3, -2]
    rev_carry_bitlist = [-7, -6, -5, -4]

    qc.append(twoBitSum(), [qr[x] for x in sum_bitlist])
    for i in range(n-1):
        qc.append(reverseTwoBitCarry(), [qr[x - 3 * i] for x in rev_carry_bitlist])
        qc.append(twoBitSum(), [qr[x - 3 * i] for x in rev_carry_bitlist[:3]])

    qc.draw(output="mpl", fold=-1, filename="circuits/{}-bitAdder.png".format(n))
    # print(qc)
    return qc.to_instruction()


def nBitAdder(n):
    """
        -- register a of size n
        -- b (sum) register of size n + 1
        -- ancilla register of size n
    """

    reg_a = QuantumRegister(n)
    reg_b = QuantumRegister(n+1)
    reg_anc = QuantumRegister(n)

    qc = QuantumCircuit(reg_a, reg_b, reg_anc, name="{}-bit adder".format(n))


    # qr = QuantumRegister(3 * n + 1)
    # qc = QuantumCircuit(qr, name="{}-bit adder".format(n))
    carry_bitlist = [0, 1, 2, 3]
    for i in range(n):
        qc.append(twoBitCarry(), [qr[x + 3 * i] for x in carry_bitlist])
    qc.cnot(qr[-3], qr[-2])

    sum_bitlist = [-4, -3, -2]
    rev_carry_bitlist = [-7, -6, -5, -4]

    qc.append(twoBitSum(), [qr[x] for x in sum_bitlist])
    for i in range(n - 1):
        qc.append(reverseTwoBitCarry(), [qr[x - 3 * i] for x in rev_carry_bitlist])
        qc.append(twoBitSum(), [qr[x - 3 * i] for x in rev_carry_bitlist[:3]])

    qc.draw(output="mpl", fold=-1, filename="circuits/{}-bitAdder.png".format(n))
    # print(qc)
    return qc.to_instruction()