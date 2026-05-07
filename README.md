# Efficient Calibration of the Heston Stochastic Volatility Model using Numerical Optimization

---

## Project Summary

This project focuses on calibrating the Heston stochastic volatility model to observed market option prices using numerical optimization techniques.

The Heston model allows volatility to evolve stochastically over time, making it more realistic than constant-volatility models. However, model parameters are not directly observable and must be determined through calibration.

The calibration problem is formulated as:

$$
\min_{v_0, \kappa, \theta, \eta, \rho}
\sum_{i=1}^{N}
\left(
C_{\text{model}}(K_i, T_i) - C_{\text{market}}(K_i, T_i)
\right)^2
$$

where $C_{\text{model}}$ represents the model-generated option price and $C_{\text{market}}$ denotes the observed market price for strike $K_i$ and maturity $T_i$.

The main challenge lies in the high computational cost and numerical instability of the calibration process. This project explores efficient numerical methods to improve both speed and stability.

---

## Team Members

- John Wang  
- Anish Reddy  
- William Qiu  
- Louie Tam  
- Michael Adegbite  

---

## Implemented Methodologies

The project implements and compares several numerical techniques to improve calibration performance:

- Continuous Characteristic Function  
- Analytical Gradient Formulation  
- Vectorized Gauss-Legendre Integration  
- Levenberg-Marquardt Optimization  

These methods are designed to reduce computational cost and improve convergence stability during optimization.

---

## Repository Structure

```text
heston_calibration_optimization_ipynb/
│
├── heston_calibration-FINAL.ipynb     # Main final notebook
├── heston_calibration.ipynb           # Earlier project notebook
├── heston_calibration-GPU.ipynb       # GPU-related experiment notebook
├── SPY_Complete.xlsx                  # Market option/input data
├── Part5_AnalyticalCalib/             # Analytical calibration package
├── pricerNew/                         # Baseline pricing code
├── part_6/                            # Final result/evaluation work
└── README.md                          # Project documentation
```

---

## Reference Paper Connection

This project is connected to the Heston stochastic volatility framework and the paper:

Cui, Y., del Baño Rollin, S., & Germano, G. (2017). *Full and Fast Calibration of the Heston Stochastic Volatility Model.*

The project follows similar ideas by improving calibration efficiency through characteristic-function pricing, analytical gradient formulation, efficient numerical integration, and optimization techniques.

---

## Final Project Notebook

The full implementation is contained in:

[View Final Notebook](https://github.com/maa2420-cloud/heston_calibration_optimization_ipynb/blob/main/notebooks/demo.ipynb)

---

## Notebook Structure: `heston_calibration-FINAL.ipynb`

The final notebook is organized into 6 main sections.

---

### 1. Introduction

The introduction presents the Heston stochastic volatility model and explains why calibration is needed.

The Heston model dynamics are written as:

$$
dS_t = \mu S_t dt + \sqrt{v_t}S_t dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)dt + \eta\sqrt{v_t}dW_t^v
$$

$$
dW_t^S dW_t^v = \rho dt
$$

This section motivates the calibration problem and highlights the main challenge: Heston calibration is computationally expensive, nonlinear, and sensitive to initial parameter values.

---

### 2. Heston Model and Pricing Framework

This section develops the Heston model pricing framework.

The model assumes that the underlying asset price and variance follow correlated stochastic processes:

$$
dS_t = (r - q)S_t dt + \sqrt{v_t}S_t dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)dt + \eta\sqrt{v_t}dW_t^v
$$

$$
dW_t^S dW_t^v = \rho dt
$$

This section establishes the pricing foundation used later in the calibration process.

---

### 3. Calibration Problem Formulation

This section defines the calibration setup used in the notebook.

The calibration process includes:

- setting initial parameter guesses  
- applying parameter bounds  
- loading market option data  
- constructing option prices across strikes and maturities  
- minimizing the difference between model prices and market prices  

This section prepares the data and optimization structure used in the baseline and improved methods.

---
### 4. Baseline Method (Naive Approach)

This section implements a baseline Heston calibration using a single start L-BFGS-B optimizer with simple box constraints. Market prices are derived from implied volatilities via the Black–Scholes formula, and the calibration minimizes the mean squared error across strikes and maturities.

While the method converges to a similar final error (MSE ≈ 0.111582), the computational effort varies across initial guesses, highlighting sensitivity to initialization and high computational cost. This serves as a benchmark for the improved methods in the next section.

---

### 5. Improved Calibration Method

This section presents the main contribution of the project.

The improved calibration framework incorporates:

- Continuous Characteristic Function  
- Analytical Gradient Formulation  
- Vectorized Gauss-Legendre Integration  
- Levenberg-Marquardt Optimization  

These enhancements improve both computational efficiency and convergence stability compared to the baseline method.

---

### 6. Results and Evaluation

This section evaluates calibration performance across different methods.

The notebook compares baseline calibration, improved calibration, the Cui et al. approach, and a GPU-based implementation.

The final method comparison reported in the notebook is:

| Method | Final MSE |
|---|---:|
| Baseline | 0.01581340 |
| Improved | 0.04707892 |
| Cui et al. | 0.07711890 |
| GPU | 6.87368833 |

The results show that calibration performance depends on the numerical methods used and highlight differences in accuracy and efficiency across implementations.

## Key Takeaways

- The Heston model captures stochastic volatility more realistically than constant-volatility models.
- Calibration is a nonlinear optimization problem.
- PyFeng provides the baseline Heston FFT pricing engine.
- The improved method focuses on characteristic-function pricing, analytical gradients, vectorized Gauss-Legendre integration, and Levenberg-Marquardt optimization.
- GPU/CuPy implementation is included as part of the performance comparison.
- The final framework emphasizes accuracy, stability, and computational efficiency.

---

## Demo Notebook

A demonstration of the full Heston calibration workflow is provided in the final notebook.

You can run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maa2420-cloud/heston_calibration_optimization_ipynb/blob/main/notebooks/demo.ipynb)

The notebook includes:

- Data loading and preprocessing  
- Construction of the volatility surface  
- Baseline calibration  
- Improved calibration methods  
- Performance comparison across methods  

This notebook serves as an end-to-end example of the calibration framework implemented in this project.

---

## API Reference

The final notebook uses the following libraries and project modules:

```python
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import pyfeng as pf
import cupy as cp

from scipy.optimize import minimize, least_squares
from scipy.stats import norm

from heston_package import HestonPricer, VolSurfaceBuilder, HestonCalibrator, HestonParams
from _baseline import runSimulation
```
### Core Libraries

| Library / Module | Usage |
|---|---|
| `numpy` | Numerical calculations, vectorization, exponentials, square roots, and arrays |
| `pandas` | Loading and organizing market data |
| `time` | Measuring runtime |
| `matplotlib.pyplot` | Plotting and visualization in the notebook |
| `pyfeng` | Heston FFT pricing through `pf.HestonFft` |
| `cupy` | GPU acceleration |
| `scipy.optimize.minimize` | Baseline local optimization |
| `scipy.optimize.least_squares` | Levenberg-Marquardt least-squares optimization |
| `scipy.stats.norm` | Black-Scholes call and put price construction |
| `heston_package` | Project calibration classes for the improved method |
| `_baseline.runSimulation` | Baseline simulation/calibration support |

### Important Functions

| Function / Class | Purpose |
|---|---|
| `tenor_to_years()` | Converts tenors such as `1M`, `3M`, and `6M` into year fractions |
| `black_scholes_prices()` | Converts implied volatility quotes into market call and put prices |
| `load_market_data()` | Loads the SPY dataset and builds the calibration dataset |
| `pf.HestonFft()` | Computes Heston model prices |
| `calculate_mse_baseline()` | Computes baseline calibration error |
| `minimize()` | Runs baseline optimization |
| `least_squares()` | Runs Levenberg-Marquardt least-squares optimization |
| `HestonPricer` | Pricing engine used in the improved method |
| `VolSurfaceBuilder` | Builds the option volatility surface |
| `HestonCalibrator` | Runs the improved calibration method |
| `HestonParams` | Stores Heston model parameters |
| `runSimulation()` | Runs baseline simulation/calibration support |