# Efficient Calibration of the Heston Stochastic Volatility Model

## 1. Introduction

The Heston model is a widely used framework in quantitative finance for pricing derivative securities while accounting for stochastic volatility. Unlike the Black-Scholes model, which assumes constant volatility, the Heston model allows volatility to evolve randomly over time and revert to a long-term mean.

The dynamics of the Heston model are given by:

dS_t = μ S_t dt + √(v_t) S_t dW_t^S

dv_t = κ(θ - v_t) dt + σ √(v_t) dW_t^v

dW_t^S dW_t^v = ρ dt

A key challenge in applying the Heston model in practice is calibration. This involves determining model parameters such that model prices match observed market prices.

This can be formulated as:

min Σ (C_model - C_market)^2

This project focuses on developing an efficient and robust calibration framework by improving numerical optimization techniques.

This motivation is supported by the paper:
"On Multilevel and Control Variate Monte Carlo Methods for Option Pricing under the Rough Heston Model"
