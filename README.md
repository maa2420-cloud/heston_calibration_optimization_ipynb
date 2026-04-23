# Efficient Calibration of the Heston Stochastic Volatility Model using Numerical Optimization

## Overview
This project focuses on calibrating the Heston stochastic volatility model to market option data using numerical optimization techniques. The goal is to improve the efficiency, stability, and accuracy of the calibration process.

## Team Members
- John Wang
- Michael Adegbite
- Anish Reddy
- William Qiu
- Louie Tam

## Project Outline
1. Introduction & Motivation
2. Heston Model & Pricing Framework
3. Calibration Problem Formulation
4. Baseline Optimization Method
5. Improved Calibration Method
6. Results & Performance Evaluation

## Project Notebook

Full implementation and analysis can be found here:

[heston_calibration.ipynb](./heston_calibration.ipynb)

# Part 5. Improved Heston Model Calibration

This part implements a fast and robust calibration method for the Heston stochastic volatility model. The core methodology and theoretical foundation are based on the paper:

> **"Full and fast calibration of the Heston stochastic volatility model"** > Yiran Cui, Sebastian del Baño Rollin, Guido Germano (2016).  
> *arXiv:1511.08718v2*

## Overview

Calibrating the Heston model to market implied volatility surfaces is traditionally a challenging inverse problem. Previous approaches often suffered from instability, heavy reliance on initial guesses, and high computational costs due to numerical gradient approximations (finite differences). Furthermore, the complex multi-valued logarithm in the original Heston characteristic function caused discontinuities (branch switching) for long-dated options.

By leveraging the methodologies introduced by Cui et al., this implementation achieves highly efficient, deterministic calibration that is approximately ten times faster than standard numerical gradient methods, making it suitable for real-time and high-frequency trading applications.

## Implemented Methodologies

### 1. Continuous Characteristic Function
We utilize the modified representation of the Heston characteristic function proposed in the paper. By algebraically rearranging the complex terms and replacing exponential functions with hyperbolic functions, this formulation completely eliminates the discontinuities caused by the branch switching of complex functions. This ensures numerical stability across the entire parameter space and for all option maturities.

### 2. Analytical Gradient Formulation
The primary breakthrough of the referenced paper is the derivation of the exact analytical gradient of a vanilla option's price with respect to the five Heston model parameters ($v_0$, $\overline{v}$, $\rho$, $\kappa$, $\sigma$). Because our characteristic function is continuous and easily differentiable, we can compute the Jacobian matrix analytically. This eliminates the need for computationally expensive and error-prone finite difference approximations.

### 3. Vectorized Gauss-Legendre Integration
Pricing an option and computing its gradient under the Heston model requires evaluating Fourier integrals. Because the components of the analytical gradient share many intermediate algebraic terms with the pricing function itself, we implement a vectorized Gauss-Legendre (GL) quadrature scheme. 
* **Efficiency:** All necessary integrals for the price and the gradient are computed simultaneously using shared nodes and weights.
* **Accuracy:** Gauss-Legendre quadrature converges significantly faster than the trapezoidal rule, achieving high precision ($10^{-8}$) with only around 60 nodes.

### 4. Levenberg-Marquardt Optimization
The calibration is formulated as a nonlinear least squares problem, aiming to minimize the difference between model prices and market prices. We use the **Levenberg-Marquardt (LM)** algorithm as our deterministic optimizer.
* **Adaptive Search:** The LM method intelligently transitions between the steepest descent method (when far from the optimal parameters) and the Gauss-Newton method (when close to the optimal parameters).
* **Global Convergence:** By pairing the LM algorithm with our exact analytical gradient, the objective function smoothly converges to the optimal parameter set. The paper demonstrates that the "multiple local minima" frequently reported in older literature are largely artifacts of poorly scaled numerical gradients, not an inherent property of the Heston calibration space.
