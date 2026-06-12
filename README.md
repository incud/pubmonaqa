# Achieving practical runtime advantage with quantum Markov chains

This repository contains the code and notebooks used to reproduce the numerical resource estimates for classical and quantum Markov-chain methods in the paper:

<center>
<i>Achieving practical runtime advantage with quantum Markov chains</i>, ...
</center><br/>

The project compares classical Metropolis proposals, quantum-enhanced proposal moves, and QSP/QSVT-based quantum-walk implementations for annealing toward low-temperature Gibbs distributions.

## Install

The reference configuration used for the Python analysis is:

```text
Python 3.14.4 (Linux, GCC 11.4.0)
```

Create and activate an empty Python 3.14 virtual environment:

```bash
python3.14 -m venv .monaqa_venv
source .monaqa_venv/bin/activate
```

If `pip` is not available inside the environment, install and upgrade it:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

Then install the project requirements:

```bash
python -m pip install -r requirements.txt
```

Reproducing the timing benchmarks requires additional software and hardware, such as the relevant C++/CUDA compilers and access to the CPU/GPU/FPGA systems used for the measurements. See the corresponding notebook and benchmark folders for further instructions.

## Reproduce

The main reproducibility entry point is the notebook:

```bash
python3.14 -m jupyter lab notebooks
```

Then open and run `notebooks/0_readme.ipynb`.

## Structure

* `notebooks/`: notebooks used for data generation, analysis, and plotting.
* `monaqa2/`: Python package containing MCMC, resource-estimation, plotting, and data utilities.
* `requirements.txt`: Python dependencies needed to run the notebooks.
* `data/`: generated and cached numerical data used by the notebooks, including spectral gaps, query counts, variance estimates, and fitted runtime data.
* `estimation_timing/`: CPU/GPU/FPGA timing benchmarks used to estimate the wall-clock cost of classical proposal moves.
* `estimation_variance/`: exact and approximate energy-variance estimation code used to build annealing schedules.
