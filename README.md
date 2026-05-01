# Efficient Calibration of the Heston Stochastic Volatility Model

This project implements an efficient calibration framework for the Heston stochastic volatility model. The goal is to estimate Heston parameters that allow model-generated option prices to closely match observed market option prices while improving numerical speed, stability, and calibration accuracy.

---

## Team Members

- John Wang  
- Anish Reddy  
- William Qiu  
- Louie Tam  
- Michael Adegbite  

---

## Project Summary

The Heston model is useful because it allows volatility to evolve stochastically over time, making it more realistic than constant-volatility models such as Black-Scholes. This helps capture volatility smiles and skews observed in option markets.

The Heston parameter vector is:

$$
\Theta = (v_0, \kappa, \theta, \eta, \rho)
$$

where:

| Parameter | Meaning |
|---|---|
| $v_0$ | Initial variance |
| $\kappa$ | Mean reversion speed |
| $\theta$ | Long-run variance |
| $\eta$ | Volatility of volatility |
| $\rho$ | Correlation between asset price and variance |

The calibration problem is:

$$
\min_{\Theta} \sum_{i=1}^{N}
\left(C_i^{model}(\Theta) - C_i^{market}\right)^2
$$

---

## Implemented Methodologies

| Methodology | Purpose |
|---|---|
| Continuous Characteristic Function | Prices options using the Heston characteristic function |
| Analytical Gradient Formulation | Improves optimization stability and convergence |
| Vectorized Gauss-Legendre Integration | Speeds up numerical integration across strikes and maturities |
| Levenberg-Marquardt Optimization | Solves the nonlinear least-squares calibration problem |

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

The connection to the paper is that both focus on making Heston calibration faster and more stable. The project follows the same motivation by improving the calibration process through:

- characteristic function pricing,
- analytical gradient ideas,
- efficient numerical integration,
- and nonlinear least-squares optimization.

---

## Final Project Notebook

The full implementation is contained in:

[View Final Notebook](heston_calibration-FINAL.ipynb)

---



