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