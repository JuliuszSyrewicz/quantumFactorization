import unittest, importlib
import qiskit as qk
from qiskit_aer import Aer
import numpy as np

import functions, arithmetic
importlib.reload(functions)
importlib.reload(arithmetic)

def testCircuit(qc):
    simulator = Aer.get_backend('qasm_simulator')
    job_sim = simulator.run(qk.transpile(qc, simulator), shots=1)
    result_sim = job_sim.result()
    counts = functions.convertKeys(result_sim.get_counts(qc))
    counts = functions.squashDict(counts)
    return counts

class TestFunctions(unittest.TestCase):
    def testAdd(self):
        for i in range(10):
            a = np.random.randint(0, 63)
            b = np.random.randint(0, 63)
            qc = arithmetic.adderCircuit(a, b)
            counts = testCircuit(qc)
            if type(counts) != int:
                self.fail("Adder circuit output gives multiple results")
            self.assertEqual(counts, a + b)

    # def testModAdd(self):
    #     for i in range(10):
    #         a = np.random.randint(0, 63)
    #         b = np.random.randint(0, 63)
    #         n = len(bin(max(a, b))[2:])
    #         qc = arithmetic.modAdderCircuit(a, b, n)
    #         counts = testCircuit(qc)
    #         if type(counts) != int:
    #             self.fail("Mod adder circuit output gives multiple results")
    #         self.assertEqual(counts, (a + b) % 2**n)