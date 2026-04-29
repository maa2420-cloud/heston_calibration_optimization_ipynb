"""
heston_recovery_via_iv.py

End-to-end self-test of the calibration pipeline:
  1. Sample random "true" Heston parameters AND random market context
     (spot, interest rate, dividend yield).
  2. Price every (tenor, strike) cell with Heston (pf.HestonFft).
  3. Invert each price to a BS implied vol — the synthetic "vol surface."
  4. Save the IV grid to disk.
  5. Re-price the surface from the IV grid via Black-Scholes (the form
     market data takes when it enters your real calibration pipeline).
  6. Hand the option prices to the EXACT baseline calibrator
     (imported from main_baseline.hestonOptimization).
  7. Compare recovered params to the true ones.

If everything is correct, recovered params should be close to true and
final MSE should be near zero.
"""

import numpy as np
import pandas as pd
import pyfeng as pf
from scipy.optimize import brentq
import time
import os

# ── Import the EXACT calibration routine used by the real pipeline ──
# This guarantees the recovery test exercises the same code path as
# main_baseline.runSimulation. If you rename your baseline file, fix
# this import.
from _baseline import hestonOptimization, valid_strike, valid_tenor, TENOR_YEARS


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

N_TRIALS    = 10
SEED        = 42
OUTPUT_DIR  = "recovery_runs"
SUMMARY_XLSX = "heston_recovery_summary.xlsx"

# Random sampling ranges
TRUE_PARAM_RANGES = {
    "v0":    (0.01,  0.10),    # 10% – 31.6% short vol
    "kappa": (0.5,   5.0),
    "theta": (0.02,  0.10),    # 14% – 31.6% long-run vol
    "eta":   (0.2,   1.5),
    "rho":   (-0.9, -0.3),
}

MARKET_RANGES = {
    "spot":     (50.0,  500.0),    # broad equity range
    "interest": (0.00,  0.07),     # 0% – 7%, raw decimals
    "dividend": (0.00,  0.04),     # 0% – 4%, raw decimals
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Heston pricing — generate the "true" prices
# ─────────────────────────────────────────────────────────────────────────────

def heston_price_grid(spot, r, q, params):
    """Return dicts call_prices[tenor] = array, put_prices[tenor] = array."""
    pricer = pf.HestonFft(
        sigma=np.sqrt(params["v0"]),
        vov=params["eta"],
        rho=params["rho"],
        mr=params["kappa"],
        theta=params["theta"],
        intr=r,
        divr=q,
    )
    strikes_arr = np.array(valid_strike, dtype=float) / 100.0 * spot

    calls, puts = {}, {}
    for t in valid_tenor:
        texp = TENOR_YEARS[t]
        calls[t] = np.array(pricer.price(strikes_arr, spot, texp, cp=+1), dtype=float)
        puts[t]  = np.array(pricer.price(strikes_arr, spot, texp, cp=-1), dtype=float)
    return calls, puts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Invert price → BS implied vol (OTM side)
# ─────────────────────────────────────────────────────────────────────────────

def implied_vol_otm(target_price, spot, strike, texp, r, q, cp,
                    vol_lo=1e-4, vol_hi=5.0):
    """Solve BS(sigma, cp) = target_price via brentq. Always invert from
    the OTM side (put for K<S, call for K>=S)."""
    if not np.isfinite(target_price) or target_price <= 0:
        return np.nan

    def diff(sigma):
        bs = pf.Bsm(sigma=sigma, intr=r, divr=q)
        return bs.price(strike=strike, spot=spot, texp=texp, cp=cp) - target_price

    try:
        return brentq(diff, vol_lo, vol_hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def build_iv_surface(spot, r, q, calls, puts):
    """Return DataFrame of implied vols in vol points (e.g. 18.42)."""
    iv_grid = pd.DataFrame(index=valid_tenor, columns=valid_strike, dtype=float)
    strikes_arr = np.array(valid_strike, dtype=float) / 100.0 * spot
    for t in valid_tenor:
        texp = TENOR_YEARS[t]
        for k_pct, K, c, p in zip(valid_strike, strikes_arr, calls[t], puts[t]):
            if K < spot:
                target, cp = float(p), -1
            else:
                target, cp = float(c), +1
            iv = implied_vol_otm(target, spot, float(K), texp, r, q, cp)
            iv_grid.loc[t, k_pct] = iv * 100 if np.isfinite(iv) else np.nan
    return iv_grid


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Re-price from IV grid via BS — produces the dict the calibrator
# expects when reading market data: option_prices[tenor][strike] = (call, put)
# ─────────────────────────────────────────────────────────────────────────────

def bs_price_dict_from_iv(spot, r, q, iv_grid):
    """Return option_prices[tenor][strike] = (call, put), matching the
    structure the baseline pipeline builds via pricerSimulation."""
    strikes_arr = np.array(valid_strike, dtype=float) / 100.0 * spot
    option_prices = {}
    for t in valid_tenor:
        option_prices[t] = {}
        texp = TENOR_YEARS[t]
        for k_pct, K in zip(valid_strike, strikes_arr):
            sigma = iv_grid.loc[t, k_pct]
            if not np.isfinite(sigma):
                # Fill with NaN-safe placeholder; the calibrator will skip it
                option_prices[t][k_pct] = (np.nan, np.nan)
                continue
            bs = pf.Bsm(sigma=sigma / 100.0, intr=r, divr=q)
            c = float(bs.price(strike=K, spot=spot, texp=texp, cp=+1))
            p = float(bs.price(strike=K, spot=spot, texp=texp, cp=-1))
            option_prices[t][k_pct] = (c, p)
    return option_prices


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    print(f"Running {N_TRIALS} recovery trials (seed={SEED})\n")

    for i in range(1, N_TRIALS + 1):
        # 1a. Random market context
        spot     = float(rng.uniform(*MARKET_RANGES["spot"]))
        interest = float(rng.uniform(*MARKET_RANGES["interest"]))
        dividend = float(rng.uniform(*MARKET_RANGES["dividend"]))

        # 1b. Random true Heston params
        true_params = {
            p: float(rng.uniform(lo, hi)) for p, (lo, hi) in TRUE_PARAM_RANGES.items()
        }

        # 2. Heston-price the surface
        h_calls, h_puts = heston_price_grid(spot, interest, dividend, true_params)

        # 3. Invert prices → BS implied vol surface
        iv_grid = build_iv_surface(spot, interest, dividend, h_calls, h_puts)
        n_nan = int(iv_grid.isna().sum().sum())

        # 4. Save IV surface (and trial metadata) to Excel
        iv_path = os.path.join(OUTPUT_DIR, f"trial_{i:02d}_iv_surface.xlsx")
        with pd.ExcelWriter(iv_path) as w:
            iv_grid.to_excel(w, sheet_name="implied_vols")
            meta = pd.DataFrame({"value": [
                spot, interest, dividend,
                true_params["v0"], true_params["kappa"], true_params["theta"],
                true_params["eta"], true_params["rho"],
            ]}, index=[
                "spot", "interest", "dividend",
                "v0_true", "kappa_true", "theta_true", "eta_true", "rho_true",
            ])
            meta.to_excel(w, sheet_name="trial_meta")

        # 5. From IV grid → BS prices in the dict structure the calibrator expects
        option_prices = bs_price_dict_from_iv(spot, interest, dividend, iv_grid)

        # 6. Run the EXACT baseline calibrator. It expects:
        #      hestonOptimization(spot, interest, dividend, atm_vol, atm_vol_long, market_options)
        #    with atm_vol and atm_vol_long as DECIMALS (e.g. 0.18, not 18.42).
        atm_vol_1M  = iv_grid.loc["1M", 100] / 100.0
        atm_vol_2Y  = iv_grid.loc["2Y", 100] / 100.0

        t0 = time.perf_counter()
        best_params, final_error, initial_error, n_iter, n_fev, init_params = \
            hestonOptimization(
                spot, interest, dividend,
                atm_vol_1M, atm_vol_2Y,
                option_prices,
            )
        elapsed = time.perf_counter() - t0

        v0_hat    = best_params["initial variance"]
        kappa_hat = best_params["mean reversion"]
        theta_hat = best_params["long-run variance"]
        eta_hat   = best_params["vol of vol"]
        rho_hat   = best_params["correlation"]

        row = {
            "trial":            i,
            "iv_surface_file":  iv_path,
            "iv_nan_cells":     n_nan,
            "spot":             spot,
            "interest":         interest,
            "dividend":         dividend,
            "atm_1M":           atm_vol_1M * 100,
            "atm_2Y":           atm_vol_2Y * 100,
            "initial_mse":      initial_error,
            "final_mse":        final_error,
            "n_iterations":     int(n_iter),
            "n_function_evals": int(n_fev),
            "cpu_time_s":       elapsed,
            "v0_true":    true_params["v0"],
            "kappa_true": true_params["kappa"],
            "theta_true": true_params["theta"],
            "eta_true":   true_params["eta"],
            "rho_true":   true_params["rho"],
            "v0_hat":    v0_hat,
            "kappa_hat": kappa_hat,
            "theta_hat": theta_hat,
            "eta_hat":   eta_hat,
            "rho_hat":   rho_hat,
            "v0_err":    v0_hat    - true_params["v0"],
            "kappa_err": kappa_hat - true_params["kappa"],
            "theta_err": theta_hat - true_params["theta"],
            "eta_err":   eta_hat   - true_params["eta"],
            "rho_err":   rho_hat   - true_params["rho"],
        }
        rows.append(row)

        print(
            f"[{i:>2d}/{N_TRIALS}] "
            f"S={spot:6.2f} r={interest*100:.2f}% q={dividend*100:.2f}% | "
            f"MSE={final_error:.2e} nfev={n_fev:>3d} t={elapsed:5.2f}s nan={n_nan:>2d} | "
            f"true v0={true_params['v0']:.3f} k={true_params['kappa']:.2f} "
            f"th={true_params['theta']:.3f} eta={true_params['eta']:.2f} "
            f"rho={true_params['rho']:+.2f} | "
            f"hat  v0={v0_hat:.3f} k={kappa_hat:.2f} "
            f"th={theta_hat:.3f} eta={eta_hat:.2f} rho={rho_hat:+.2f}"
        )

    df = pd.DataFrame(rows)

    # Aggregate
    print("\n" + "=" * 72)
    print("Recovery summary")
    print("=" * 72)
    print(f"\nFinal MSE  median={df['final_mse'].median():.2e}, "
          f"max={df['final_mse'].max():.2e}")
    print(f"Iterations median={df['n_iterations'].median():.0f}")
    print(f"CPU time   median={df['cpu_time_s'].median():.2f}s")
    print(f"IV NaN cells median={df['iv_nan_cells'].median():.0f}, "
          f"max={df['iv_nan_cells'].max()}")

    print("\nPer-parameter recovery error (recovered − true):")
    print(f"{'param':<8} {'mean err':>12} {'median err':>12} "
          f"{'mean |err|':>12} {'max |err|':>12}")
    for p in ["v0", "kappa", "theta", "eta", "rho"]:
        e = df[f"{p}_err"]
        print(f"{p:<8} {e.mean():>+12.4f} {e.median():>+12.4f} "
              f"{e.abs().mean():>12.4f} {e.abs().max():>12.4f}")

    print("\nRelative error (|err| / param_range):")
    for p in ["v0", "kappa", "theta", "eta", "rho"]:
        lo, hi = TRUE_PARAM_RANGES[p]
        rel = df[f"{p}_err"].abs() / (hi - lo)
        print(f"  {p:<6}: median {rel.median()*100:.1f}% of range, "
              f"max {rel.max()*100:.1f}%")

    df.to_excel(SUMMARY_XLSX, index=False)
    print(f"\nSummary written to {SUMMARY_XLSX}")
    print(f"Per-trial IV surfaces in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()