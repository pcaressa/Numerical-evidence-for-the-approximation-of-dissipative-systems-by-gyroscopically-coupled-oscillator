# Numerical-evidence-for-the-approximation-of-dissipative-systems-by-gyroscopically-coupled-oscillator

This repository contains the code used in the paper Bersani, Ciallella, Caressa
"Numerical evidence for the approximation of dissipative systems by gyroscopically coupled oscillators chains"
(to appear).

The code is contained in a single script [paper_code.py](poaper_code.py).

To perform numerical computations we used the `scipy` Python package under the Python 3.8 compiler.
We used the simulated annealing optimization function which implements a generalized annealing
by using its default parameters both for annealing temperature, number of iterations etc.
(see Xiang, Y. and Sun, D.Y. and Fan, W. and Gong, X.G.
"Generalized simulated annealing algorithm and its application to the Thomson model",
Physics Letters A, 233:3 (1997) 216-220.

The code can be executed in a Python 3 environment which provides standard libraries `numpy`, `scipy` and `matplotlib`,
easily installed via the `pip` tool.

The script optimizes parameters of a coupled system to approximate a damped one:
results are either written on png files or plotted interactively;
further information is printed.
The functions defined in this script may be easily engineered to run several classes of simulations and collect results.
