import sys
import numpy as np
np.set_printoptions(suppress = True)
import qiskit as qk
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import time
from qiskit import QuantumCircuit
from numpy import genfromtxt
from bitstring import Bits


def ZeroAncCRn(circuit, control, target, theta):
    circuit.p(theta/2, control)
    circuit.p(theta/2, target)
    circuit.cnot(control, target)
    circuit.p(-theta/2, target)
    circuit.cnot(control, target)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def AFTerror(qc2, n, qft_probs, delta, backend="statevector", display_bool=False):
    # returns a correlation coefficient between the qft and aft signals
    aft_probs = simulateQFT(qc2, n, display_bool=display_bool, delta=delta, backend=backend)
    mae = np.abs(np.subtract(qft_probs, aft_probs)).mean()
    return mae

def compareAFT(n, max_delta, qft_counts, display_bool=False, backend="statevector"):
    errors = []
    deltas = []
    for d in range(1, max_delta, 1):
        deltas.append(d)
        print("progress: {:.2f}%".format(d/n*100))
        qcAFT = QuantumCircuit(n, n)
        qcAFT.h(n//2)
        QFT(qcAFT, n, delta=d)
        errors.append(AFTerror(qcAFT, n, qft_counts, d, backend=backend, display_bool=display_bool))

    assert len(deltas) == len(errors)
    return deltas, errors

def QFT(circuit, n, delta=10000):
    """
        adds a quantum fourier transform to a specified circuit
        n = number of qubits to perform the operation on
        delta = exponent cutoff for phase shift precision
    """
    swap_registers(circuit, n)
    for i in range(n):
        circuit.h(i)
        for j in range(1, n-i):
            if j < delta:
                ZeroAncCRn(circuit, i+j, i, np.pi/2**(j))
            else:
                break

def simulateQFT(qc, n, reps=2 ** 14, display_bool=False, delta=10000, backend="statevector",
                noise = 0.01):
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    error = depolarizing_error(noise, 1)
    noise_model.add_all_qubit_quantum_error(error, ['h', 't', 'cnot', 'tdg', 'p'])

    # Print noise model info
    # print(noise_model)

    if backend == "simulator":
        qc.measure(range(n), range(n))
        # simulator = Aer.get_backend('qasm_simulator')
        simulator = AerSimulator(noise_model=noise_model)
        job_sim = simulator.run(qk.transpile(qc, simulator), shots=reps)
        result_sim = job_sim.result()
        counts = (result_sim.get_counts(qc))
        if display_bool:
            # choose appropriate filename
            filename = "charts/n={},delta={},noise={}.png".format(n, delta if delta < 9000 else "infinite", noise)
            plot_histogram(counts, legend=[], bar_labels=False, title="n={}, delta={}".format(n, delta if delta < 9000 else "infinite"), filename=filename)
            hist = plot_histogram(counts, legend=[], bar_labels=False, title="n={}, delta={}".format(n, delta if delta < 9000 else "infinite"))
        return counts


    elif backend == "statevector":
        # qc.measure(range(n), range(n))
        backend = Aer.get_backend('statevector_simulator') # the device to run on
        result = backend.run(qk.transpile(qc, backend)).result()
        psi  = result.get_statevector(qc)

        probs = psi.probabilities()
        if display_bool:
            filename = "charts/probabilities,n={},delta={}.png".format(n, delta if delta < 9000 else "infinite")
            plt.bar([x for x in range(2**n)], probs, color="#c888e3")
            plt.title("n={}, delta={}".format(n, delta if delta < 9000 else "infinite"))
            plt.show()
            plt.savefig(filename)
        return probs[:2**n]

def compareFTs(n_low, n_high, backend="statevector", display_bool=False):

    filename = 'errors{}-{}.txt'.format(n_low, n_high)
    print(filename)
    f = open('txt/{}'.format(filename), 'w')
    f.write("n, delta, error\n")

    i = 0
    for n in range(n_low, n_high + 1, 1):
        print("n={}, i={}".format(n, i))
        qcQFT = QuantumCircuit(n, n)
        qcQFT.h(n // 2)
        QFT(qcQFT, n)
        qft_probs = simulateQFT(qcQFT, n, display_bool=display_bool, backend=backend)
        start_time = time.time()
        deltas, errors = compareAFT(n, n, qft_probs, backend=backend, display_bool=display_bool)
        print("--- {:.3f}s seconds ---".format(time.time() - start_time))
        for d in range(len(deltas)):
            f = open('txt/{}'.format(filename), 'a')
            f.write("{}, {}, {}\n".format(n, deltas[d], errors[d]))
            f.close()
        i -= - 1

    f.close()

def plotErrorsFromFile(filepath):
    my_data = genfromtxt(filepath, delimiter=',')
    my_data = my_data[1:, :]

    max_n = np.max(my_data[:, 0]).astype(int) + 1
    max_d = np.max(my_data[:, 1]).astype(int) + 1
    min_n = np.min(my_data[:, 0]).astype(int)
    min_d = np.min(my_data[:, 1]).astype(int)

    print("min_n = {}, max_n = {}, min_d = {}, max_d = {}".format(min_n, max_n, min_d, max_d))
    errors = np.zeros((max_n, max_d))  # + sys.float_info.min
    print(errors.shape)

    for datum in my_data:
        n = datum[0].astype(int)
        d = datum[1].astype(int)
        print("x = {}, y = {}, data = {}".format(n, d, datum[2]))
        errors[n][d] = datum[2]

    # errors = errors[min_n:,min_d:]
    ma = errors[errors != 0]
    errors += np.min(ma)
    print(errors)
    plt.imshow(np.log(errors))
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_ylim(min_n, max_n - 1)
    ax.set_ylabel("n")
    ax.set_xlim(min_d, max_d - 1)
    ax.set_xlabel("delta")

    plt.colorbar()

def getOneIndices(n):
    # convert integer to binary string
    binary_str = bin(n)[2:][::-1]

    # initialize empty list to store indices
    indices = []

    # iterate through the binary string and get the indices where the string is equal to 1
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            indices.append(i)

    return indices

# convert output dict's keys to integer and print out
def convertKeys(dict, reverse=False):
    # convert each key to integer
    output = {int(k, 2): v for k, v in dict.items()}

    return output

def squashDict(dict):
    assert(len(dict)) == 1
    outcome = list(dict.keys())[0]
    return outcome

def genBinStrings(n):
    binstrings = []

    def genBin(n, bs=''):
        if len(bs) == n:
            binstrings.append(bs)
        else:
            genBin(n, bs + '0')
            genBin(n, bs + '1')

    genBin(n)
    return binstrings

def plotShor(no_qubits, len_exp, counts, g, N):

    # for key in counts:
    #     b_reg = key[(-len_exp-1):(-1)]
        # print("b_reg = {}, int = {}".format(b_reg, int(b_reg, 2)))

    for binstring in genBinStrings(len_exp):
        padded = '0' * (no_qubits - len_exp) + binstring
        print(padded)
        counts[padded] = counts.get(padded, 0)

    int_counts = {}
    for count in counts:
        int_counts[(int(count, 2))] = counts[count]

    xs = [x for x in range(len(int_counts))]
    results = []
    for i in range(len(int_counts)):
        results.append(int_counts[i])

    plt.bar(xs, results, color="#c888e3")
    plt.title("{}^x mod {}".format(g, N))
    filename = "charts/factored{}".format(N)
    plt.show()
    plt.savefig(filename)