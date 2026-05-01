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

where \( C_{\text{model}} \) represents the model-generated option price and \( C_{\text{market}} \) denotes the observed market price for strike \( K_i \) and maturity \( T_i \).

The main challenge lies in the high computational cost and numerical instability of the calibration process. This project explores efficient numerical methods to improve both speed and stability.

---