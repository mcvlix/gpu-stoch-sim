# sdegpu
## CUDA-accelerated Stochastic Differential Equation (SDE) Solver for PyTorch

A modular framework for simulating and benchmarking multiple runs of Itô SDEs on both CPU and GPU, developed by Maxwell Arcinas (B.S. in Applied Mathematics @ UCSC).  
Currently, it implements Euler–Maruyama integration in pure PyTorch and as a fused CUDA kernel for high-performance parallel stochastic simulation. 

A lightweight, conceptual notebook is viewable in `euler-maruyama_demo.ipynb`. I am still developing and learning myself!

---
## Features and Uses

---

## Installation

To properly run the CUDA SDE solver, you of course need an NVIDIA GPU. However, the CPU sanity check (in run by `test_gbm_moments.py`) is runnable regardless. 

0. Install CUDA driver and toolkit
https://developer.nvidia.com/cuda-downloads

1. Set up environment and install dependencies
```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
pip install -e .
```

2. Test the CUDA kernel on the GPU and GBM on the 
```
pytest -v 
```

3. Build the extension
```
python setup.py build_ext --inplace
```