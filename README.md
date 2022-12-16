# quantumFactorization
A project implementing gate-by-gate circuitry for semiprime factorization on simulated quantum computers in qiskit. This repository is based on the papers by Shor (1995), Vedral et al. (1995) and Beauregard (2013)

  -- arithmetic.py - holds functions for generating quantum circuits for arithmetic and the quantum Fourier transform
  -- functions.py - code for analysis of the quantum Fourier transform and error estimation for different levels of phase shift precision
  -- demo.ipynb - a Jupyter notebook with demonstrations of all the circuit generating functions and some charts showing the effects of changing precision on the output
  -- fourierSpaceShor.ipynb - a Jupyter notebook implementing an optimized arithmetic circuit (making use of Fourier space arithmetic as described by Beauregard (2013))
  
Roadmap:
  1. Implement more efficient ideas for arithmetic.
  2. Add noise mitigation to make the project useful for running on real quantum hardware.
  
  [1] Shor, Peter W. ‘Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer’. SIAM Journal on Computing, vol. 26, no. 5, Oct. 1997, pp. 1484–509. arXiv.org, https://doi.org/10.1137/S0097539795293172.
  
  [2] Vedral, V., et al. ‘Quantum Networks for Elementary Arithmetic Operations’. Physical Review A, vol. 54, no. 1, July 1996, pp. 147–53. arXiv.org, https://doi.org/10.1103/PhysRevA.54.147.
  
  [3] Beauregard, Stephane. Circuit for Shor’s Algorithm Using 2n+3 Qubits. arXiv:quant-ph/0205095, arXiv, 21 Feb. 2003. arXiv.org, http://arxiv.org/abs/quant-ph/0205095.
