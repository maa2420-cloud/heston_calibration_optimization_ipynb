# Efficient Calibration of the Heston Stochastic Volatility Model using Numerical Optimization

## Overview

This project studies the calibration of the Heston stochastic volatility model to market option data using numerical optimization techniques. The goal is to estimate model parameters that minimize pricing error while improving computational efficiency, stability, and robustness.

The Heston model allows volatility to evolve stochastically over time, making it more realistic than constant-volatility models and capable of capturing volatility smiles observed in option markets.

This project is also motivated by recent research on computational efficiency in the Rough Heston model, especially the use of multilevel and control variate Monte Carlo methods to reduce simulation cost and variance.

---

## Team Members

- John Wang
- Michael Adegbite
- Anish Reddy
- William Qiu
- Louie Tam

---

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

The paper studies the computational challenges of option pricing under the Rough Heston model. Since the Rough Heston model involves stochastic Volterra equations, direct simulation can be computationally expensive. The authors propose multilevel and control variate Monte Carlo methods to improve efficiency and reduce variance.

The paper reports that combining multilevel Monte Carlo with control variates can achieve substantial cost-adjusted variance reduction. This motivates the efficiency-focused direction of our project, where we analyze not only calibration accuracy but also convergence stability and computational performance.

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

