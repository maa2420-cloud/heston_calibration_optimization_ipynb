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


## Project Objective

The objective is to calibrate Heston model parameters so that model-implied option prices match observed market prices.

The calibration problem is:

$$
\min_{\Theta} \frac{1}{N} \sum_{i=1}^{N}
\left(C_i^{model}(\Theta) - C_i^{market}\right)^2
$$

where:

$$
\Theta = (v_0, \kappa, \theta, \eta, \rho)
$$

---

## Methodology

### 1. Heston Model Framework

The Heston model is defined as:

$$
dS_t = rS_tdt + \sqrt{v_t}S_tdW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)dt + \eta\sqrt{v_t}dW_t^v
$$

$$
dW_t^S dW_t^v = \rho dt
$$

### 2. Pricing Method

Option prices are computed using the characteristic function and Fourier-based pricing methods. This allows repeated option pricing during calibration to be performed more efficiently than relying only on direct simulation.

### 3. Calibration Approach

Calibration is performed using the L-BFGS-B optimization method with parameter bounds. The objective is to minimize the mean squared error between model-implied and market option prices.

### 4. Sensitivity Analysis

Multiple initial parameter guesses are tested to evaluate:

- Stability of the final calibration result
- Convergence speed of the optimization
- Dependence on initial parameter values

---

## Main Results

### 1. Calibration Accuracy

The calibrated Heston model achieves a close match between model-implied option prices and observed market prices. This confirms that the model is capable of capturing the structure of option prices across strikes and maturities.

---

### 2. Convergence Behavior

The calibration process was repeated using different initial parameter guesses. The results show:

- Final Mean Squared Error (MSE) converges to approximately **0.111582**
- Number of iterations ranges from **88 to 126**
- Runtime ranges from approximately **12 to 18 seconds**

These findings indicate that the calibration is stable in terms of the final solution, but sensitive in terms of convergence speed depending on the initial guess.

---

### 3. Performance Insights

Although different starting values lead to similar final errors, they result in different computational costs. This highlights an important practical consideration:

- Calibration accuracy alone is not sufficient
- Efficiency and robustness of the optimization process are equally important
- Better initialization and optimization strategies can improve practical performance

---

## Connection to the Reference Paper

This project is inspired by the paper:

**Jeng, S. W., & Kiliçman, A. (2021). _On Multilevel and Control Variate Monte Carlo Methods for Option Pricing under the Rough Heston Model_. Mathematics, 9(22), 2930.**

**Cui, Y., del Baño Rollin, S., & Germano, G. (2017). Full and fast calibration of the Heston stochastic volatility model. *European Journal of Operational Research*, 258(3), 939-949.**

While our project focuses on Heston model calibration, the paper provides important motivation for studying numerical efficiency, robustness, and performance improvements in stochastic volatility models.

---

## Key Takeaways

- The Heston model provides a strong fit to market option data after calibration.
- The final calibration result is stable across different initial parameter guesses.
- Optimization performance, measured by runtime and iterations, varies significantly.
- Calibration requires balancing accuracy with computational efficiency.
- Recent Rough Heston research motivates the importance of variance reduction and efficient numerical methods.
- Improved initialization and optimization strategies can enhance robustness in practical calibration tasks.

---

## Project Notebook

Full implementation and analysis can be found here:

[Heston Calibration Final Notebook](./heston_calibration-FINAL.ipynb)

---

