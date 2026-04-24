import pandas as pd
import numpy as np
import pyfeng as pf
import math
import time
from scipy.stats import norm
from scipy.optimize import minimize

valid_strike = [30, 40, 60, 80, 90, 95, 97.5, 100, 105, 110, 120, 130, 150, 300]
valid_tenor = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

def get_texp(maturity):
    unit = "".join(c for c in maturity if c.isalpha())
    length = int("".join(c for c in maturity if c.isdigit()))
    if unit == "W": return length / 52
    if unit == "M": return length / 12
    if unit == "Y": return length
    raise ValueError(f"Unsupported tenor: {maturity}")

# pricing engines: B-S, Heston FFT, Heston Monte Carlo
def blackScholes(Spot, Strike, Volatility, TimeToExpiry, Riskfree, Dividend=0.0):
    r, q, sigma = Riskfree / 100, Dividend / 100, Volatility / 100
    texp = get_texp(TimeToExpiry)
    d1 = (math.log(Spot / Strike) + (r - q + 0.5 * sigma**2) * texp) / (sigma * math.sqrt(texp))
    d2 = d1 - sigma * math.sqrt(texp)
    call = norm.cdf(d1) * Spot * math.exp(-q * texp) - norm.cdf(d2) * Strike * math.exp(-r * texp)
    put = norm.cdf(-d2) * Strike * math.exp(-r * texp) - norm.cdf(-d1) * Spot * math.exp(-q * texp)
    return {"call_price": call, "put_price": put}

def heston_fft_price(spot, strike, maturity, r, q, params):
    texp = get_texp(maturity)
    v0, kappa, theta = params["initial variance"], params["mean reversion"], params["long-run variance"]
    eta, rho = params["vol of vol"], params["correlation"]
    m = pf.HestonFft(sigma=np.sqrt(v0), vov=eta, rho=rho, mr=kappa, theta=theta, intr=r, divr=q)
    return m.price(strike, spot, texp, cp=1), m.price(strike, spot, texp, cp=-1)

def heston_monte_carlo(spot, strike_price, maturity, r, q, params, paths=50000):
    texp = get_texp(maturity)
    dt = texp / 100
    v0, kappa, theta = params["initial variance"], params["mean reversion"], params["long-run variance"]
    eta, rho = params["vol of vol"], params["correlation"]
    
    Z1 = np.random.normal(0, 1, (paths, 100))
    Z2 = np.random.normal(0, 1, (paths, 100))
    W_v, W_s = Z1, rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    V, S = np.full(paths, v0), np.full(paths, spot)
    for t in range(100):
        v_curr = np.maximum(V, 0)
        S *= np.exp((r - q - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * W_s[:, t])
        V += kappa * (theta - v_curr) * dt + eta * np.sqrt(v_curr * dt) * W_v[:, t]
    return np.exp(-r * texp) * np.mean(np.maximum(S - strike_price, 0))

def calculate_mse_entire(model_chain, market_options):
    error, count = 0, 0
    for tenor in valid_tenor:
        for strike in valid_strike:
            m_call, m_put = market_options[tenor][strike]
            h_call, h_put = model_chain[tenor][strike]
            error += (m_call - h_call)**2 + (m_put - h_put)**2
            count += 2
    return error / count

def pricer():
    df = pd.read_excel("OrganizedData.xlsx", index_col=0)
    spot_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Underlying", index_col=0)
    other_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Other", index_col=0)
    
    print("\n[Section 6 Setup]")
    date_year = input("Year: ")
    date_month = input("Month: ").zfill(2)
    date_day = input("Day: ").zfill(2)
    agg_date = f"{date_year}-{date_month}-{date_day}"

    spot = spot_df.loc[agg_date]["Mid"]
    interest, dividend = other_df.loc[agg_date]["Interest"], other_df.loc[agg_date]["Dividend"]
    atm_vol = df.loc[agg_date][f"1M_100_Volatility"]

    option_prices = {}
    for tenor in valid_tenor:
        option_prices[tenor] = {}
        for strike in valid_strike:
            vol = df.loc[agg_date][f"{tenor}_{strike}_Volatility"]
            strike_p = strike / 100 * spot
            prices = blackScholes(spot, strike_p, vol, tenor, interest, dividend)
            option_prices[tenor][strike] = (prices["call_price"], prices["put_price"])
    return option_prices, spot, atm_vol, interest, dividend

def baseline_calibration(spot, interest, dividend, atm_vol, market_options):
    initial_guess = np.array([atm_vol**2, 2.0, atm_vol**2, 0.5, -0.5])
    bounds = [(1e-6, 2.0), (1e-4, 20.0), (1e-6, 2.0), (1e-4, 5.0), (-0.999, 0.999)]

    def objective(x):
        p = {"initial variance": x[0], "mean reversion": x[1], "long-run variance": x[2], "vol of vol": x[3], "correlation": x[4]}
        chain = {}
        for t in valid_tenor:
            chain[t] = {}
            for s in valid_strike:
                c, p_v = heston_fft_price(spot, (s/100)*spot, t, interest, dividend, p)
                chain[t][s] = (float(c), float(p_v))
        return calculate_mse_entire(chain, market_options)

    res = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
    return {"initial variance": res.x[0], "mean reversion": res.x[1], "long-run variance": res.x[2], "vol of vol": res.x[3], "correlation": res.x[4]}, res.fun

def transform_to_heston(x):
    """Latent space mapping for Improved Method"""
    v0, kappa, theta = np.exp(x[0]), np.exp(x[1]), np.exp(x[2])
    rho = np.tanh(x[3])
    eta = np.sqrt(2 * kappa * theta) * (1 / (1 + np.exp(-x[4])))
    return {"initial variance": v0, "mean reversion": kappa, "long-run variance": theta, "vol of vol": eta, "correlation": rho}

def improved_calibration(spot, interest, dividend, atm_vol, market_options):
    x0_latent = np.array([np.log(atm_vol**2), np.log(2.0), np.log(atm_vol**2), 0.0, 0.0])

    def objective(x):
        p = transform_to_heston(x)
        chain = {}
        for t in valid_tenor:
            chain[t] = {}
            for s in valid_strike:
                c, p_v = heston_fft_price(spot, (s/100)*spot, t, interest, dividend, p)
                chain[t][s] = (float(c), float(p_v))
        return calculate_mse_entire(chain, market_options)

    res = minimize(objective, x0_latent, method="L-BFGS-B")
    return transform_to_heston(res.x), res.fun
#eval
def run_evaluation():
    mkt_options, spot, atm_vol, r_pct, q_pct = pricer()
    r, q, atm_v = r_pct/100, q_pct/100, atm_vol/100

    print("===BASELINE CALIBRATION:===")
    start = time.time()
    p_base, mse_base = baseline_calibration(spot, r, q, atm_v, mkt_options)
    time_base = time.time() - start

    print("===IMPROVED CALIBRATION:===")
    start = time.time()
    p_imp, mse_imp = improved_calibration(spot, r, q, atm_v, mkt_options)
    time_imp = time.time() - start

    print("MONTE CARLO PRICING FOR OTM OPTION:")
    target_s = 1.20 * spot
    mkt_p = mkt_options["1M"][120][0]
    mc_base = heston_monte_carlo(spot, target_s, "1M", r, q, p_base)
    mc_imp = heston_monte_carlo(spot, target_s, "1M", r, q, p_imp)

    print("Results:")
    print("Metric: MSE")
    print("Baseline:", f"{mse_base:.8f}")
    print("Improved:", f"{mse_imp:.8f}")
    print()

    print("Metric: Time (s)")
    print("Baseline:", f"{time_base:.2f}")
    print("Improved:", f"{time_imp:.2f}")
    print()

    print("Metric: Feller Condition")
    print(
        "Baseline:",
        (2 * p_base['mean reversion'] * p_base['long-run variance'] > p_base['vol of vol']**2)
    )
    print("Improved:", "STRICTLY PASS")
    print()

    print("Metric: OTM MC Price Error")
    print("Baseline:", f"{abs(mc_base - mkt_p):.4f}")
    print("Improved:", f"{abs(mc_imp - mkt_p):.4f}")
    print()
if __name__ == "__main__":
    run_evaluation()
