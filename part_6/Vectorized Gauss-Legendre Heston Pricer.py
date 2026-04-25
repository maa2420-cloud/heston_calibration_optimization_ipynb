import cupy as cp
import numpy as np
from scipy.optimize import minimize
import time

# 1. Pre-compute Gauss-Legendre nodes and weights on the CPU, then move to GPU
GL_NODES = 128  # Increased nodes for higher precision
x_np, w_np = np.polynomial.legendre.leggauss(GL_NODES)

# Scale nodes from [-1, 1] to integration domain [0, U_MAX]
U_MAX = 200.0
u_np = 0.5 * U_MAX * (x_np + 1.0)
w_np = 0.5 * U_MAX * w_np

u_cp = cp.array(u_np, dtype=cp.float64)[:, None]  # Shape: (GL_NODES, 1)
w_cp = cp.array(w_np, dtype=cp.float64)[:, None]

def heston_surface_pricer_gpu(spot, strikes, texps, r, q, v0, kappa, theta, eta, rho):
    """
    Prices an entire surface of European Calls and Puts simultaneously using CuPy.
    Uses the Lewis (2000) integral representation for numerical stability.
    """
    K = cp.asarray(strikes, dtype=cp.float64)[None, :]
    T = cp.asarray(texps, dtype=cp.float64)[None, :]

    # Complex integration variable for Lewis formula: z = u - 0.5j
    z = u_cp - 0.5j
    iz = 1j * z

    # Heston CF components (Vectorized over (nodes, options_grid))
    d = cp.sqrt((rho * eta * iz - kappa)**2 + eta**2 * (z**2 + iz))
    g = (kappa - rho * eta * iz - d) / (kappa - rho * eta * iz + d)

    # Exponent terms
    C = (kappa * theta / eta**2) * ((kappa - rho * eta * iz - d) * T - 2 * cp.log((1 - g * cp.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * eta * iz - d) / eta**2) * ((1 - cp.exp(-d * T)) / (1 - g * cp.exp(-d * T)))

    # Characteristic Function (normalized S0=1)
    phi = cp.exp(C + D * v0 + iz * (r - q) * T)

    # Lewis Integrand (FIXED SIGN ERROR: k = ln(K/S))
    k = cp.log(K / spot)
    integrand = cp.real(cp.exp(-1j * u_cp * k) * phi / (u_cp**2 + 0.25))

    # Gauss-Legendre Integration
    integral = cp.sum(w_cp * integrand, axis=0)

    # Call Prices
    call_prices = spot * cp.exp(-q * T) - (cp.sqrt(spot * K) / cp.pi) * cp.exp(-r * T) * integral

    #
    
    put_prices = call_prices - spot * cp.exp(-q * T) + K * cp.exp(-r * T)

    return call_prices[0], put_prices[0]

def calibrate_heston_gpu(spot, r, q, atm_vol, market_options):
    """
    Optimization loop utilizing the GPU-vectorized surface pricer.
    """
    
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
            # Price the ENTIRE surface in one GPU call
            c_gpu, p_gpu = heston_surface_pricer_gpu(
                spot, strikes_list, texps_list, r, q,
                p["initial variance"], p["mean reversion"], p["long-run variance"],
                p["vol of vol"], p["correlation"]
            )
            diff_calls = c_gpu - mkt_calls_cp
            diff_puts = p_gpu - mkt_puts_cp

            residuals = cp.asnumpy(cp.concatenate((diff_calls, diff_puts)))
            if np.any(np.isnan(residuals)):
                return 1e6

            return float(np.mean(residuals**2))
        except:
            return 1e6

    print("Starting GPU-accelerated calibration with L-BFGS-B...")
    start = time.time()
    x0_latent = np.array([np.log(atm_vol**2), np.log(2.0), np.log(atm_vol**2), 0.0, 0.0])

    res = minimize(objective_gpu, x0_latent, method="L-BFGS-B")

    print(f"GPU Calibration finished in {time.time() - start:.2f} seconds.")
    return transform_to_heston(res.x), res.fun