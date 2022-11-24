from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister


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

def nBitAdder(n, reg_a, reg_b, reg_anc):
    qc = QuantumCircuit(reg_a, reg_b, reg_anc, name="{}-bit adder".format(n))
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

    return qc.to_instruction()