import math
import cupy as cp
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from scipy.optimize import least_squares
import pyfeng as pf

valid_strike = [30, 40, 60, 80, 90, 95, 97.5, 100, 105, 110, 120, 130, 150, 300]
valid_tenor = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

def get_texp(maturity):
    unit = "".join(c for c in maturity if c.isalpha())
    length = int("".join(c for c in maturity if c.isdigit()))
    if unit == "W": return length / 52
    if unit == "M": return length / 12
    if unit == "Y": return length
    raise ValueError(f"Unsupported tenor: {maturity}")

def blackScholes(Spot, Strike, Volatility, TimeToExpiry, Riskfree, Dividend=0.0):
    r, q, sigma = Riskfree / 100, Dividend / 100, Volatility / 100
    texp = get_texp(TimeToExpiry)
    d1 = (math.log(Spot / Strike) + (r - q + 0.5 * sigma**2) * texp) / (sigma * math.sqrt(texp))
    d2 = d1 - sigma * math.sqrt(texp)
    call = norm.cdf(d1) * Spot * math.exp(-q * texp) - norm.cdf(d2) * Strike * math.exp(-r * texp)
    put = norm.cdf(-d2) * Strike * math.exp(-r * texp) - norm.cdf(-d1) * Spot * math.exp(-q * texp)
    return {"call_price": call, "put_price": put}

heston_kernel = cp.RawKernel(r'''
extern "C" __global__
void heston_paths(const float* Z1, const float* Z2, float* S_out,
                  float S0, float v0, float r, float q, float kappa,
                  float theta, float eta, float rho, float dt, int steps, int paths) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < paths) {
        float v = v0;
        float S = S0;
        float sqrt_1_rho2 = sqrtf(1.0f - rho * rho);

        for (int t = 0; t < steps; ++t) {
            float z1 = Z1[idx * steps + t];
            float z2 = Z2[idx * steps + t];

            float w_v = z1;
            float w_s = rho * z1 + sqrt_1_rho2 * z2;

            float v_curr = fmaxf(v, 0.0f);
            float sqrt_v_dt = sqrtf(v_curr * dt);

            S *= expf((r - q - 0.5f * v_curr) * dt + sqrt_v_dt * w_s);
            v += kappa * (theta - v_curr) * dt + eta * sqrt_v_dt * w_v;
        }
        S_out[idx] = S;
    }
}
''', 'heston_paths')

def heston_monte_carlo(spot, strike_price, maturity, r, q, params, paths=50000):
    texp = get_texp(maturity)
    steps = 100
    dt = texp / steps
    v0, kappa, theta = params["initial variance"], params["mean reversion"], params["long-run variance"]
    eta, rho = params["vol of vol"], params["correlation"]

    Z1 = cp.random.normal(0, 1, (paths, steps), dtype=cp.float32)
    Z2 = cp.random.normal(0, 1, (paths, steps), dtype=cp.float32)
    S_out = cp.empty(paths, dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid = (paths + threads_per_block - 1) // threads_per_block

    heston_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (Z1, Z2, S_out, cp.float32(spot), cp.float32(v0), cp.float32(r), cp.float32(q),
         cp.float32(kappa), cp.float32(theta), cp.float32(eta), cp.float32(rho), cp.float32(dt),
         cp.int32(steps), cp.int32(paths))
    )

    return float(math.exp(-r * texp) * cp.mean(cp.maximum(S_out - strike_price, 0.0)))

def pricer():
    df = pd.read_excel("OrganizedData.xlsx", index_col=0)
    spot_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Underlying", index_col=0)
    other_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Other", index_col=0)

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

def transform_to_heston(x):
    v0, kappa, theta = np.exp(x[0]), np.exp(x[1]), np.exp(x[2])
    rho = np.tanh(x[3])
    eta = np.sqrt(2 * kappa * theta) * (1 / (1 + np.exp(-x[4])))
    return {"initial variance": v0, "mean reversion": kappa, "long-run variance": theta, "vol of vol": eta, "correlation": rho}

GL_NODES = 256
x_np, w_np = np.polynomial.legendre.leggauss(GL_NODES)
U_MAX = 200.0
u_np = 0.5 * U_MAX * (x_np + 1.0)
w_np = 0.5 * U_MAX * w_np

u_cp = cp.array(u_np, dtype=cp.float64)[:, None]
w_cp = cp.array(w_np, dtype=cp.float64)[:, None]

def heston_surface_pricer_gpu(spot, strikes, texps, r, q, v0, kappa, theta, eta, rho):
    K = cp.asarray(strikes, dtype=cp.float64)[None, :]
    T = cp.asarray(texps, dtype=cp.float64)[None, :]
    
    z = u_cp - 0.5j
    iz = 1j * z
    
    d = cp.sqrt((rho * eta * iz - kappa)**2 + eta**2 * (z**2 + iz))
    g = (kappa - rho * eta * iz - d) / (kappa - rho * eta * iz + d)
    
    C = (kappa * theta / eta**2) * ((kappa - rho * eta * iz - d) * T - 2 * cp.log((1 - g * cp.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * eta * iz - d) / eta**2) * ((1 - cp.exp(-d * T)) / (1 - g * cp.exp(-d * T)))
    
    phi = cp.exp(C + D * v0 + iz * (r - q) * T)
    
    k = cp.log(spot / K)
    integrand = cp.real(cp.exp(-1j * u_cp * k) * phi / (u_cp**2 + 0.25))
    
    integral = cp.sum(w_cp * integrand, axis=0)
    
    call_prices = spot * cp.exp(-q * T) - (cp.sqrt(spot * K) / cp.pi) * cp.exp(-r * T) * integral
    put_prices = call_prices - spot * cp.exp(-q * T) + K * cp.exp(-r * T)
    
    return call_prices[0], put_prices[0]

def calibrate_heston_gpu(spot, r, q, atm_vol, market_options):
    strikes_list, texps_list, mkt_calls, mkt_puts = [], [], [], []
    for tenor in market_options:
        for strike in market_options[tenor]:
            strikes_list.append((strike / 100.0) * spot)
            texps_list.append(get_texp(tenor))
            c, p = market_options[tenor][strike]
            mkt_calls.append(c)
            mkt_puts.append(p)
            
    mkt_calls_cp = cp.array(mkt_calls, dtype=cp.float64)
    mkt_puts_cp = cp.array(mkt_puts, dtype=cp.float64)
    
    def objective_gpu(x):
        p = transform_to_heston(x)
        try:
            c_gpu, p_gpu = heston_surface_pricer_gpu(
                spot, strikes_list, texps_list, r, q, 
                p["initial variance"], p["mean reversion"], p["long-run variance"],
                p["vol of vol"], p["correlation"]
            )
            diff_calls = c_gpu - mkt_calls_cp
            diff_puts = p_gpu - mkt_puts_cp
            
            residuals = cp.asnumpy(cp.concatenate((diff_calls, diff_puts)))
            if np.any(np.isnan(residuals)):
                return np.full(len(strikes_list) * 2, 1e6)
            return residuals
        except:
            return np.full(len(strikes_list) * 2, 1e6)

    print("Starting GPU-accelerated calibration...")
    
    v0_guess = atm_vol**2
    kappa_guess = 2.0
    theta_guess = atm_vol**2
    eta_guess = 0.5
    rho_guess = -0.5
    
    x0 = np.log(v0_guess)
    x1 = np.log(kappa_guess)
    x2 = np.log(theta_guess)
    x3 = np.arctanh(rho_guess)
    
    feller_limit = np.sqrt(2 * kappa_guess * theta_guess)
    safe_eta = min(eta_guess, feller_limit * 0.999)
    ratio = safe_eta / feller_limit
    x4 = np.log(ratio / (1 - ratio))
    
    x0_latent = np.array([x0, x1, x2, x3, x4])
    
    res = least_squares(
        objective_gpu, 
        x0_latent, 
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        max_nfev=5000
    )
    
    print(f"Optimizer Status: {res.message}")
    
    final_mse = np.mean(res.fun**2)
    return transform_to_heston(res.x), float(final_mse)

def run_evaluation():
    mkt_options, spot, atm_vol, r_pct, q_pct = pricer()
    r, q, atm_v = r_pct/100, q_pct/100, atm_vol/100

    print("===GPU ACCELERATED CALIBRATION:===")
    start = time.time()
    p_gpu, mse_gpu = calibrate_heston_gpu(spot, r, q, atm_v, mkt_options)
    time_gpu = time.time() - start

    print("MONTE CARLO PRICING FOR OTM OPTION:")
    target_s = 1.20 * spot
    mkt_p = mkt_options["1M"][120][0]

    mc_gpu = heston_monte_carlo(spot, target_s, "1M", r, q, p_gpu)

    print("Results:")
    print("Metric: MSE")
    print("GPU Accelerated:", f"{mse_gpu:.8f}")
    print()

    print("Metric: Time (s)")
    print("GPU Accelerated:", f"{time_gpu:.2f}")
    print()

    print("Metric: Feller Condition")
    print("GPU Accelerated:", "STRICTLY PASS")
    print()

    print("Metric: OTM MC Price Error")
    print("GPU Accelerated:", f"{abs(mc_gpu - mkt_p):.4f}")
    print()

if __name__ == "__main__":
    run_evaluation()