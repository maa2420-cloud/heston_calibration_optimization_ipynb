import pandas as pd
import numpy as np
import pyfeng as pf
import math
from scipy.stats import norm
from scipy.optimize import minimize
import tracemalloc
import time
import os


valid_strike = [30, 40, 60, 80, 90, 95, 97.5, 100, 105, 110, 120, 130, 150, 300]
valid_tenor = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

# Pre-built tenor → year fraction lookup so we don't parse "1W"/"2M" strings
# 70,000+ times per date inside the optimizer.
TENOR_YEARS = {
    "1W": 1/52, "2W": 2/52, "3W": 3/52,
    "1M": 1/12, "2M": 2/12, "3M": 3/12,
    "6M": 6/12, "9M": 9/12,
    "1Y": 1.0, "18M": 1.5, "2Y": 2.0,
}


def calculate_mse(model_price, market_price):
    model_price, market_price = np.array(model_price), np.array(market_price)
    return np.mean((model_price - market_price) ** 2)


def calculate_mse_entire(model_price, market_price):
    error = 0
    count = 0
    for tenor in valid_tenor:
        for strike in valid_strike:
            market_call = market_price[tenor][strike][0]
            market_put = market_price[tenor][strike][1]
            heston_call = model_price[tenor][strike][0]
            heston_put = model_price[tenor][strike][1]

            # Changed to Relative Error bounded by 0.5
            error += ((market_call - heston_call) / max(market_call, 0.5)) ** 2 
            error += ((market_put - heston_put) / max(market_put, 0.5)) ** 2
            count += 2

    return error / count


def blackScholes(Spot, Strike, Volatility, TimeToExpiry, Riskfree, Dividend=0.0):
    Riskfree = Riskfree / 100
    Dividend = Dividend / 100
    Volatility = Volatility / 100

    unit = "".join(c for c in TimeToExpiry if c.isalpha())
    length = int("".join(c for c in TimeToExpiry if c.isdigit()))

    if unit == "W":
        texp = length / 52
    elif unit == "M":
        texp = length / 12
    elif unit == "Y":
        texp = length
    else:
        raise ValueError(f"Unsupported tenor: {TimeToExpiry}")

    model = pf.Bsm(
        sigma=Volatility,
        intr=Riskfree,
        divr=Dividend
    )

    call_price = model.price(strike=Strike, spot=Spot, texp=texp, cp=1)
    put_price  = model.price(strike=Strike, spot=Spot, texp=texp, cp=-1)

    return {
        "call_price": call_price,
        "put_price": put_price,
    }


def prepareData():
    sheet_list = [
        "1W Volatility", "2W Volatility", "3W Volatility", "1M Volatility",
        "2M Volatility", "3M Volatility", "6M Volatility", "9M Volatility",
        "1Y Volatility", "18M Volatility", "2Y Volatility"
    ]

    df_stock = pd.read_excel(
        "SPY_Complete.xlsx",
        sheet_name=sheet_list,
        engine="openpyxl",
        index_col=0,
        header=[0, 1]
    )

    dfs = df_stock

    for key in dfs.keys():
        dfs[key] = dfs[key].replace(r'^\s*$', np.nan, regex=True)
        dfs[key] = dfs[key].apply(pd.to_numeric, errors='coerce')
        dfs[key] = dfs[key].dropna(how="any")

    bid_df = pd.DataFrame()
    ask_df = pd.DataFrame()

    bid_cols = {}
    ask_cols = {}

    for key in dfs.keys():
        df = dfs[key]

        for level1, level2 in df.columns:
            col_name = f"{key}_{level2}"
            s = df[(level1, level2)]

            if level1 == "Ask":
                ask_cols[col_name] = s
            else:
                bid_cols[col_name] = s

    bid_df = pd.concat(bid_cols, axis=1)
    bid_df = bid_df.iloc[::-1]

    ask_df = pd.concat(ask_cols, axis=1)
    ask_df = ask_df.iloc[::-1]

    bid_df.index = pd.to_datetime(bid_df.index)
    ask_df.index = pd.to_datetime(ask_df.index)

    bid_df.sort_index(ascending=True, inplace=True)
    ask_df.sort_index(ascending=True, inplace=True)

    bid_df.columns = bid_df.columns.str.replace(" Volatility", "", n=1)
    ask_df.columns = ask_df.columns.str.replace(" Volatility", "", n=1)

    common_cols = bid_df.columns.intersection(ask_df.columns)
    mid_df = (bid_df[common_cols] + ask_df[common_cols]) / 2

    mid_df = mid_df.dropna(how="any")
    mid_df.to_excel("OrganizedData.xlsx")

    return {
        "mid_df": mid_df,
        "bid_df": bid_df,
        "ask_df": ask_df
    }


def pricer():
    df = pd.read_excel("OrganizedData.xlsx", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")

    spot_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Underlying", index_col=0)
    other_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Other", index_col=0)

    df.index = df.index.astype(str)
    spot_df.index = spot_df.index.astype(str)
    other_df.index = other_df.index.astype(str)

    print("Welcome to the pricer.")
    print("Please enter the required inputs below.")
    print()

    while True:
        print("Date Year options: [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012]")
        date_year = input("Enter date year: ")
        print()

        print("Date Month options: [01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12]")
        date_month = input("Enter date month: ")
        print()

        print("Date Day options: [01, 02, 03, ..., 31]")
        date_day = input("Enter date day: ")
        print()

        aggregate_date = f"{date_year}-{date_month.zfill(2)}-{date_day.zfill(2)}"

        if aggregate_date in df.index and aggregate_date in spot_df.index and aggregate_date in other_df.index:
            print(f"Valid Date! You selected {aggregate_date}")
            break
        else:
            print(f"Invalid Date: {aggregate_date}")
            print("Please re-enter.\n")

    option_prices = {}
    spot         = spot_df.loc[aggregate_date]["Mid"]
    interest     = other_df.loc[aggregate_date]["Interest"]
    dividend     = other_df.loc[aggregate_date]["Dividend"]
    atm_vol      = df.loc[aggregate_date][f"1M_{100}_Volatility"]
    atm_vol_long = df.loc[aggregate_date][f"2Y_{100}_Volatility"]

    for tenor in valid_tenor:
        option_prices[tenor] = {}
        for strike in valid_strike:
            volatility = df.loc[aggregate_date][f"{tenor}_{strike}_Volatility"]
            strike_price = strike / 100 * spot
            price = blackScholes(spot, strike_price, volatility, tenor, interest, dividend)
            option_prices[tenor][strike] = (price["call_price"], price["put_price"])

            print(
                f"For strike {strike} and tenor {tenor}, vol is {volatility}, "
                f"call price is {price['call_price']} and put price is {price['put_price']}"
            )

    return option_prices, spot, atm_vol, atm_vol_long, interest, dividend


def pricerSimulation(df, pricing_info):
    option_prices = {}
    spot     = pricing_info["spot"]
    interest = pricing_info["interest"]
    dividend = pricing_info["dividend"]
    atm_vol  = pricing_info['volatility']
    date     = pricing_info['date']

    for tenor in valid_tenor:
        option_prices[tenor] = {}
        for strike in valid_strike:
            volatility = df.loc[date][f"{tenor}_{strike}_Volatility"]
            strike_price = strike / 100 * spot
            price = blackScholes(spot, strike_price, volatility, tenor, interest, dividend)
            option_prices[tenor][strike] = (price["call_price"], price["put_price"])

    return option_prices


def heston(spot, strike, maturity, r, q, parameters):
    """Per-(tenor, strike) Heston pricer. Kept for backward compatibility."""
    texp = TENOR_YEARS[maturity]

    v0    = parameters["initial variance"]
    kappa = parameters["mean reversion"]
    theta = parameters["long-run variance"]
    eta   = parameters["vol of vol"]
    rho   = parameters["correlation"]

    m = pf.HestonFft(
        sigma=np.sqrt(v0),
        vov=eta,
        rho=rho,
        mr=kappa,
        theta=theta,
        intr=r,
        divr=q
    )

    call_prices = m.price(strike, spot, texp, cp=1)
    put_prices  = m.price(strike, spot, texp, cp=-1)
    return call_prices, put_prices


def hestonOptimization(spot, interest, dividend, atm_vol, atm_vol_long,
                       market_options, targetMSE=0.001):
    """
    atm_vol      : short-end ATM IV (1M),  used as v0 init           (decimal, e.g. 0.18)
    atm_vol_long : long-end  ATM IV (2Y),  used as theta (vbar) init (decimal, e.g. 0.22)
    """
    initial_params = {
        "initial variance":  atm_vol      ** 2,    # v0    from 1M ATM²
        "mean reversion":    2.0,
        "long-run variance": atm_vol_long ** 2,    # theta from 2Y ATM²
        "vol of vol":        0.8,
        "correlation":       -0.7,
    }

    initial_guess = np.array([
        initial_params["initial variance"],
        initial_params["mean reversion"],
        initial_params["long-run variance"],
        initial_params["vol of vol"],
        initial_params["correlation"],
    ])

    bounds = [
        (1e-6, 5.0),
        (1e-4, 20.0),
        (1e-6, 5.0),
        (1e-4, 6.0),
        (-0.999, 0.999)
    ]

    # ── Pre-compute static arrays / lookups (do this ONCE per date, not
    # ── once per objective call). The optimizer hits objective() many
    # ── hundreds of times — anything we can hoist out saves real wall time.
    strikes_arr = np.array(valid_strike, dtype=float) / 100.0 * spot
    n_cells     = len(valid_tenor) * len(valid_strike) * 2

    market_calls_by_tenor = {
        t: np.array([market_options[t][k][0] for k in valid_strike], dtype=float)
        for t in valid_tenor
    }
    market_puts_by_tenor = {
        t: np.array([market_options[t][k][1] for k in valid_strike], dtype=float)
        for t in valid_tenor
    }

    def objective(x):
        v0, kappa, theta, eta, rho = x
        try:
            m = pf.HestonFft(
                sigma=np.sqrt(v0),
                vov=eta,
                rho=rho,
                mr=kappa,
                theta=theta,
                intr=interest,
                divr=dividend,
            )

            error_sum = 0.0
            for tenor in valid_tenor:
                texp = TENOR_YEARS[tenor]
                heston_calls = m.price(strikes_arr, spot, texp, cp=1)
                heston_puts  = m.price(strikes_arr, spot, texp, cp=-1)

                if not (np.all(np.isfinite(heston_calls))
                        and np.all(np.isfinite(heston_puts))):
                    return 1e12

                err_c = market_calls_by_tenor[tenor] - heston_calls
                err_p = market_puts_by_tenor[tenor]  - heston_puts
                error_sum += np.dot(err_c, err_c) + np.dot(err_p, err_p)

            error = error_sum / n_cells
            return error if np.isfinite(error) else 1e12

        except Exception:
            return 1e12

    print("\nStarting Heston calibration...")
    print("Initial parameters guess:")
    for key, value in initial_params.items():
        print(f"  {key}: {float(value):.8f}")

    initial_error = objective(initial_guess)
    print(f"Initial MSE = {initial_error:.8f}")

    result = minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=bounds
    )

    fitted_v0, fitted_kappa, fitted_theta, fitted_eta, fitted_rho = result.x

    best_params = {
        "initial variance":  fitted_v0,
        "mean reversion":    fitted_kappa,
        "long-run variance": fitted_theta,
        "vol of vol":        fitted_eta,
        "correlation":       fitted_rho,
    }

    final_error  = result.fun
    n_iterations = result.nit
    n_func_evals = result.nfev

    print("\nOptimization complete.")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {n_iterations}, Function evals: {n_func_evals}")
    print(f"Final MSE = {final_error:.8f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {float(value):.8f}")

    return best_params, final_error, initial_error, n_iterations, n_func_evals, initial_params


def run():
    option_prices, spot, atm_vol, atm_vol_long, interest, dividend = pricer()

    atm_vol      = atm_vol      / 100
    atm_vol_long = atm_vol_long / 100
    interest     = interest     / 100
    dividend     = dividend     / 100

    start_time = time.perf_counter()
    tracemalloc.start()

    best_params, final_error, initial_error, n_iter, n_fev, initial_params = hestonOptimization(
        spot, interest, dividend, atm_vol, atm_vol_long, option_prices
    )

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()

    print("\nHeston calibration computational intensity:")
    print(f"  Runtime: {end_time - start_time:.4f} seconds")
    print(f"  Iterations: {n_iter}, Function evals: {n_fev}")
    print(f"  Current memory usage: {current_memory / 1024**2:.4f} MB")
    print(f"  Peak memory usage: {peak_memory / 1024**2:.4f} MB")

    print("\nReturned best parameters:")
    print(best_params)
    print(f"Returned initial MSE: {initial_error:.8f}")
    print(f"Returned final MSE: {final_error:.8f}")


def runSimulation(number, random_dates=None, output_file="simulation_results.xlsx"):
    df = pd.read_excel("OrganizedData.xlsx", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")

    spot_df  = pd.read_excel("SPY_Complete.xlsx", sheet_name="Underlying", index_col=0)
    other_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Other",      index_col=0)

    df.index       = df.index.astype(str)
    spot_df.index  = spot_df.index.astype(str)
    other_df.index = other_df.index.astype(str)

    runs = []

    if random_dates is not None:
        random_dates = [str(d) for d in random_dates]
        print(f"Using {len(random_dates)} dates supplied by caller")
    else:
        all_dates = df.index.intersection(spot_df.index.intersection(other_df.index))
        iv_cols = [f"{t}_{k}_Volatility" for t in valid_tenor for k in valid_strike]
        all_dates = all_dates[
            spot_df.loc[all_dates, "Mid"].notna()
            & other_df.loc[all_dates, "Interest"].notna()
            & other_df.loc[all_dates, "Dividend"].notna()
            & df.loc[all_dates, iv_cols].notna().all(axis=1)
        ]

        file = "random_date.csv"
        if os.path.exists(file):
            random_dates = pd.read_csv(file, index_col=0).index.astype(str).tolist()
            print(f"Loaded {len(random_dates)} cached dates from {file}")
        else:
            random_dates = np.random.choice(all_dates, size=number, replace=False)
            random_dates = sorted(random_dates)
            pd.DataFrame(index=random_dates).to_csv(file)
            print(f"Generated {len(random_dates)} new random dates, cached to {file}")

    total = len(random_dates)
    overall_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Starting simulation over {total} dates")
    print(f"{'='*60}")

    for i, dates in enumerate(random_dates, start=1):
        print(f"\n[{i}/{total}] === {dates} ===")

        spot         = spot_df.loc[dates]["Mid"]
        interest     = other_df.loc[dates]["Interest"]
        dividend     = other_df.loc[dates]["Dividend"]
        atm_vol      = df.loc[dates][f"1M_{100}_Volatility"]
        atm_vol_long = df.loc[dates][f"2Y_{100}_Volatility"]

        print(f"  spot={spot:.2f}, r={interest:.3f}%, q={dividend:.3f}%, "
              f"atm_1M={atm_vol:.3f}%, atm_2Y={atm_vol_long:.3f}%")

        pricing_info = {
            "date": dates,
            "spot": spot,
            "interest": interest,
            "dividend": dividend,
            "volatility": atm_vol,
            "volatility_long": atm_vol_long,
        }
        option_prices = pricerSimulation(df, pricing_info)

        atm_vol      = atm_vol      / 100
        atm_vol_long = atm_vol_long / 100
        interest     = interest     / 100
        dividend     = dividend     / 100

        start_time = time.process_time()
        tracemalloc.start()

        best_params, final_error, initial_error, n_iter, n_fev, initial_params = hestonOptimization(
            spot, interest, dividend, atm_vol, atm_vol_long, option_prices
        )

        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        run_time = time.process_time() - start_time

        info = {
            "date":               dates,
            "run_time":           run_time,
            "n_iterations":       n_iter,
            "n_function_evals":   n_fev,
            "peak_memory":        peak_memory,
            "current_memory":     current_memory,
            "initial_parameters": initial_params,
            "best_parameters":    best_params,
            "final_mse":          final_error,
            "initial_mse":        initial_error,
        }
        runs.append(info)

        elapsed = time.perf_counter() - overall_start
        avg_per_run = elapsed / i
        eta_seconds = avg_per_run * (total - i)
        print(
            f"  ✓ done in {run_time:.2f}s | nit={n_iter}, nfev={n_fev} | "
            f"initial MSE={initial_error:.6f} | final MSE={final_error:.6f} | "
            f"peak_mem={peak_memory/1024**2:.1f} MB"
        )
        print(
            f"  progress: {i}/{total} ({100*i/total:.0f}%) | "
            f"elapsed={elapsed:.0f}s | ETA≈{eta_seconds:.0f}s"
        )

    overall_runtime = time.perf_counter() - overall_start
    print(f"\n{'='*60}")
    print(f"Simulation complete: {total} runs in {overall_runtime:.1f}s "
          f"(avg {overall_runtime/total:.2f}s/run)")
    print(f"{'='*60}")

    rows = []
    for r in runs:
        ip = r["initial_parameters"]
        bp = r["best_parameters"]
        rows.append({
            "date":              r["date"],
            "final_mse":         float(r["final_mse"]),
            "initial_mse":       float(r["initial_mse"]),
            "n_iterations":      int(r["n_iterations"]),
            "n_function_evals":  int(r["n_function_evals"]),
            "cpu_time_s":        r["run_time"],
            "peak_memory_MB":    r["peak_memory"]    / 1024**2,
            "current_memory_MB": r["current_memory"] / 1024**2,
            "v0_init":           float(ip["initial variance"]),
            "kappa_init":        float(ip["mean reversion"]),
            "theta_init":        float(ip["long-run variance"]),
            "eta_init":          float(ip["vol of vol"]),
            "rho_init":          float(ip["correlation"]),
            "v0":                float(bp["initial variance"]),
            "kappa":             float(bp["mean reversion"]),
            "theta":             float(bp["long-run variance"]),
            "eta":               float(bp["vol of vol"]),
            "rho":               float(bp["correlation"]),
        })

    results_df = pd.DataFrame(rows).set_index("date").sort_index()

    results_df.to_excel(output_file)
    print(f"\nResults written to {output_file}")
    print(results_df.describe().T[["mean", "std", "min", "max"]])

    return runs


if __name__ == "__main__":
    # run()
    runSimulation(20)
