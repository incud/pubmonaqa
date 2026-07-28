# Practical advantage beyond the quadratic speedup limit with fully-quantum walks

This repository contains the data, code, and notebooks used to reproduce the results presented in the paper:

<center>
<i><a href="https://arxiv.org/abs/2607.22818"> Practical advantage beyond the quadratic speedup limit with fully-quantum walks</a></i>, Incudini and Mazzola (2026), arXiv: 2607.22818
</center><br/>

## Install

The reference configuration used for the Python analysis is:

```text
Python 3.14.4 (Linux, GCC 11.4.0)
```

The Python executable is invoked below as `python3.14`. It may instead be named `python3` in your configuration, and other Python versions may also work.

Create and activate an empty Python 3.14 virtual environment:

```bash
python3.14 -m venv .monaqa_venv
source .monaqa_venv/bin/activate
```

If `pip` is not available in the environment, install and upgrade it:

```bash
python3.14 -m ensurepip --upgrade
python3.14 -m pip install --upgrade pip
```

Then install the project requirements:

```bash
python3.14 -m pip install -r requirements.txt
```

Reproducing the timing benchmarks requires additional software and hardware, including the relevant C++ and CUDA compilers and access to the CPU, GPU, and FPGA systems used for the measurements. See the corresponding notebook and benchmark folders for further instructions.

## Reproduce

The main entry point for reproducing the results is:

```bash
python3.14 -m jupyter lab notebooks
```

Then open and run `notebooks/0_readme.ipynb`.

## Structure

* `notebooks/`: notebooks illustrating the implementations and reproducing the plots
* `monaqa2/`: main Python package containing the code for numerical MCMC simulations, Szegedy quantum walks, and quantum-architecture cost models
* `requirements.txt`: Python dependencies required to run the notebooks
* `data/`: generated and cached numerical data used by the notebooks, including spectral gaps, query counts, variance estimates, and fitted runtime data
* `estimation_timing/`: CPU, GPU, and FPGA timing benchmarks used to estimate the wall-clock cost of classical proposal moves
* `estimation_variance/`: exact and approximate energy-variance estimation code used to construct the annealing schedules
