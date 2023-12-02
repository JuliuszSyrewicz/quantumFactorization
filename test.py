import unittest
import importlib
from bitstring import Bits
import numpy as np

from qiskit_aer import AerSimulator
from functions import convertKeys, squashDict
import functions
import circuits
from qiskit import transpile

importlib.reload(functions)
importlib.reload(circuits)


def testCircuit(qc):
    sim = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True)
    qc = transpile(qc, sim)
    result = sim.run(qc, shots=64, seed_simulator=12345).result()

    counts = result.get_counts()
    counts = convertKeys(counts)
    counts = squashDict(counts)
    return counts


# def testCircuit(qc):
#     result = functions.simulateGPU(qc, shots=1024, print_bool=False)
#     counts = functions.convertKeys(result.get_counts(qc))
#     counts = functions.squashDict(counts)
#     return counts


class TestFunctions(unittest.TestCase):
    def testAdd(self):
        for _ in range(6):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            qc = circuits.adderCircuit(a, b)
            counts = testCircuit(qc)
            if not isinstance(counts, int):
                print(f"a = {a}, b = {b}, counts = {counts}")
                self.fail("Adder circuit output gives multiple results")
            self.assertEqual(counts, a + b)

    def testSubtract(self):
        for _ in range(20):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            qc = circuits.adderCircuit(a, b, reverse=True)

            counts = testCircuit(qc)

            # handle overflow case (two complements form)
            if a > b:
                counts = Bits(bin(counts)).int

            print("a = {}, b = {}, b-a = {}, counts = {}".format(a, b, b - a, counts))

            if not isinstance(counts, int):
                self.fail("Subtract circuit output gives multiple results")
            self.assertEqual(counts, b - a)

    def testModAdd(self):
        for _ in range(20):
            a = np.random.randint(0, 62)
            b = np.random.randint(0, 62)

            N = np.random.randint(max(a, b) + 1, 63)
            qc = circuits.modNAdderCircuit(a, b, N)

            counts = testCircuit(qc)
            print(
                "a = {}, b = {}, N = {}, (a+b) % N = {}, counts = {}".format(
                    a, b, N, (a + b) % N, counts
                )
            )

            if not isinstance(counts, int):
                self.fail("Mod adder circuit output gives multiple results")
            self.assertEqual(counts, (a + b) % N)

    def testModMultiplier(self):
        for _ in range(10):
            a = np.random.randint(0, 30)
            b = np.random.randint(0, 30)

            N = np.random.randint(max(a, b) + 1, 31)
            qc = circuits.modNMultiplierCircuit(a, b, N)

            counts = testCircuit(qc)
            print(
                "a = {}, b = {}, N = {}, (a*b) % N = {}, counts = {}".format(
                    a, b, N, (a * b) % N, counts
                )
            )

            if not isinstance(counts, int):
                self.fail("Mod multiplier circuit output gives multiple results")
            self.assertEqual(counts, (a * b) % N)
