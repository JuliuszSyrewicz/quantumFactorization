import unittest, importlib
import qiskit as qk
from bitstring import Bits
from qiskit_aer import Aer
import numpy as np

import functions, circuits
importlib.reload(functions)
importlib.reload(circuits)

def testCircuit(qc):
    simulator = Aer.get_backend('qasm_simulator')
    job_sim = simulator.run(qk.transpile(qc, simulator), shots=1024)
    result_sim = job_sim.result()
    counts = functions.convertKeys(result_sim.get_counts(qc))
    counts = functions.squashDict(counts)
    return counts

class TestFunctions(unittest.TestCase):
    def testAdd(self):
        for i in range(100):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            qc = circuits.adderCircuit(a, b)
            counts = testCircuit(qc)
            if type(counts) != int:
                self.fail("Adder circuit output gives multiple results")
            self.assertEqual(counts, a + b)

    def testSubtract(self):
        for i in range(100):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            qc = circuits.adderCircuit(a, b, reverse=True)

            counts = testCircuit(qc)

            # handle overflow case (two complements form)
            if a > b:
                counts = Bits(bin(counts)).int

            print("a = {}, b = {}, b-a = {}, counts = {}".format(a, b, b - a, counts))
            if type(counts) != int:
                self.fail("Subtract circuit output gives multiple results")
            self.assertEqual(counts, b - a)

    def testModAdd(self):
        for i in range(20):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            N = np.random.randint(min(a, b) + 1, 63)
            qc = circuits.modNAdderCircuit(a, b, N)

            counts = testCircuit(qc)
            print("a = {}, b = {}, N = {}, (a+b) % N = {}, counts = {}".format(a, b, N, (a+b)%N, counts))

            if type(counts) != int:
                self.fail("Mod adder circuit output gives multiple results")
            self.assertEqual(counts, (a + b) % N)