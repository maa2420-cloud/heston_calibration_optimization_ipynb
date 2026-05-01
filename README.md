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

[View Final Notebook](heston_calibration-FINAL.ipynb)

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

### 4. Baseline Method

This section implements the baseline calibration approach using PyFeng’s FFT-based Heston pricing and a local optimization routine.

The method repeatedly evaluates model prices and adjusts parameters to minimize pricing error. It serves as a reference point for comparing more advanced calibration techniques.

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