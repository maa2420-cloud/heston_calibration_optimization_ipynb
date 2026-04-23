import pandas as pd
import numpy as np
import pyfeng as pf
import math
from scipy.stats import norm
from scipy.optimize import minimize


valid_strike = [30, 40, 60, 80, 90, 95, 97.5, 100, 105, 110, 120, 130, 150, 300]
valid_tenor = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]


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

            error += (market_call - heston_call) ** 2 + (market_put - heston_put) ** 2
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

    TimeToExpiry = texp

    d1 = (
        math.log(Spot / Strike)
        + (Riskfree - Dividend + 0.5 * Volatility ** 2) * TimeToExpiry
    ) / (Volatility * math.sqrt(TimeToExpiry))
    d2 = d1 - Volatility * math.sqrt(TimeToExpiry)

    call_price = (
        norm.cdf(d1) * Spot * math.exp(-Dividend * TimeToExpiry)
        - norm.cdf(d2) * Strike * math.exp(-Riskfree * TimeToExpiry)
    )
    call_delta = math.exp(-Dividend * TimeToExpiry) * norm.cdf(d1)

    put_price = (
        norm.cdf(-d2) * Strike * math.exp(-Riskfree * TimeToExpiry)
        - norm.cdf(-d1) * Spot * math.exp(-Dividend * TimeToExpiry)
    )
    put_delta = -math.exp(-Dividend * TimeToExpiry) * norm.cdf(-d1)

    return {
        "call_price": call_price,
        "put_price": put_price,
        "call_delta": call_delta,
        "put_delta": put_delta,
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
    spot = spot_df.loc[aggregate_date]["Mid"]
    interest = other_df.loc[aggregate_date]["Interest"]
    dividend = other_df.loc[aggregate_date]["Dividend"]
    atm_vol = df.loc[aggregate_date][f"1M_{100}_Volatility"]

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

    return option_prices, spot, atm_vol, interest, dividend


def heston(spot, strike, maturity, r, q, parameters):
    unit = "".join(c for c in maturity if c.isalpha())
    length = int("".join(c for c in maturity if c.isdigit()))

    if unit == "W":
        texp = length / 52
    elif unit == "M":
        texp = length / 12
    elif unit == "Y":
        texp = length
    else:
        raise ValueError(f"Unsupported tenor: {maturity}")

    v0 = parameters["initial variance"]
    kappa = parameters["mean reversion"]
    theta = parameters["long-run variance"]
    eta = parameters["vol of vol"]
    rho = parameters["correlation"]

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
    put_prices = m.price(strike, spot, texp, cp=-1)

    return call_prices, put_prices

def hestonOptimization(spot, interest, dividend, atm_vol, market_options, targetMSE=0.001):
    initial_guess = np.array([
        atm_vol ** 2,   # initial variance
        2.0,            # mean reversion
        atm_vol ** 2,   # long-run variance
        0.5,            # vol of vol
        -0.5            # correlation
    ])

    bounds = [
        (1e-6, 2.0),    # initial variance
        (1e-4, 20.0),   # mean reversion
        (1e-6, 2.0),    # long-run variance
        (1e-4, 5.0),    # vol of vol
        (-0.999, 0.999) # correlation
    ]

    def objective(x):
        v0, kappa, theta, eta, rho = x

        params = {
            "initial variance": v0,
            "mean reversion": kappa,
            "long-run variance": theta,
            "vol of vol": eta,
            "correlation": rho
        }

        heston_chain = {}

        try:
            for tenor in valid_tenor:
                heston_chain[tenor] = {}
                for strike in valid_strike:
                    strike_price = strike / 100 * spot
                    heston_call, heston_put = heston(
                        spot, strike_price, tenor, interest, dividend, params
                    )

                    if not np.isfinite(heston_call) or not np.isfinite(heston_put):
                        return 1e12

                    heston_chain[tenor][strike] = (float(heston_call), float(heston_put))

            error = calculate_mse_entire(heston_chain, market_options)

            if not np.isfinite(error):
                return 1e12

            return error

        except Exception:
            return 1e12

    print("\nStarting Heston calibration...")
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
        "initial variance": fitted_v0,
        "mean reversion": fitted_kappa,
        "long-run variance": fitted_theta,
        "vol of vol": fitted_eta,
        "correlation": fitted_rho
    }
    print(f"Under Best-Param, Heston Option Prices are")
    if(best_params is not None):
        best_heston_chain = {}
        for tenor in valid_tenor:
            best_heston_chain[tenor] = {}
            for strike in valid_strike:
                strike_price = strike / 100 * spot
                heston_call, heston_put = heston(
                    spot, strike_price, tenor, interest, dividend, best_params
                )
                market_call, market_put = market_options[tenor][strike]
                best_heston_chain[tenor][strike] = (float(heston_call), float(heston_put))
                print(f"For strike {strike} and tenor {tenor}")
                print(f"market call is {market_call:4f} and heston call is {heston_call}")
                print(f"market put is {market_put:4f} and  heston put is {heston_put}")
                
    
    final_error = result.fun

    print("\nOptimization complete.")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final MSE = {final_error:.8f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.8f}")

    return best_params, final_error


def run():
    option_prices, spot, atm_vol, interest, dividend = pricer()

    atm_vol = atm_vol / 100
    interest = interest / 100
    dividend = dividend / 100

    best_params, final_error = hestonOptimization(
        spot, interest, dividend, atm_vol, option_prices
    )

    print("\nReturned best parameters:")
    print(best_params)
    print(f"Returned final MSE: {final_error:.8f}")


if __name__ == "__main__":
    run()
