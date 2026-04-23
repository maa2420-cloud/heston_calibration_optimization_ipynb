import pandas as pd
import numpy as np 
import pyfeng as pf 
import math 
from scipy.stats import norm
from scipy.optimize import minimize

def calculate_mse(model_price, market_price):
    model_price, market_price = np.array(model_price), np.array(market_price)
    return np.mean((model_price - market_price)**2)


def blackScholes(Spot,Strike,Volatility,TimeToExpiry,Riskfree,Dividend=0.0):
    Riskfree=Riskfree/100
    Dividend=Dividend/100
    Volatility=Volatility/100
    
    unit = "".join(c for c in TimeToExpiry if c.isalpha())
    length = int("".join(c for c in TimeToExpiry if c.isdigit()))

    if unit == "W":
        texp = length / 52
    elif unit == "M":
        texp = length / 12
    elif unit == "Y":
        texp = length
    TimeToExpiry=texp
    
    
    d1= (math.log(Spot/Strike)+(Riskfree - Dividend+ 0.5*Volatility**2)*(TimeToExpiry))/(Volatility*math.sqrt(TimeToExpiry))
    d2= d1 - Volatility*math.sqrt(TimeToExpiry)
    call_price= norm.cdf(d1)*Spot*math.exp(-Dividend*TimeToExpiry) - norm.cdf(d2)*Strike*math.exp(-Riskfree*TimeToExpiry)
    call_delta = math.exp(-Dividend*TimeToExpiry)*norm.cdf(d1)

    d1= (math.log(Spot/Strike)+(Riskfree - Dividend+ 0.5*Volatility**2)*(TimeToExpiry))/(Volatility*math.sqrt(TimeToExpiry))
    d2= d1 - Volatility*math.sqrt(TimeToExpiry)
    put_price= norm.cdf(-d2)*Strike*math.exp(-Riskfree*TimeToExpiry) - norm.cdf(-d1)*Spot*math.exp(-Dividend*TimeToExpiry)
    put_delta =-math.exp(-Dividend*TimeToExpiry)*norm.cdf(-d1)

  
    return({
        "call_price": call_price,
        "put_price": put_price,
        "call_delta": call_delta,
        "put_delta": put_delta
    })


def prepareData():
    
    sheet_list = ["1W Volatility", "2W Volatility", "3W Volatility","1M Volatility","2M Volatility","3M Volatility","6M Volatility","9M Volatility","1Y Volatility","18M Volatility","2Y Volatility"]
  
    df_stock=pd.read_excel("SPY_Complete.xlsx",sheet_name= sheet_list,engine="openpyxl",index_col=0, header=[0,1])

    dfs=df_stock

    for key in dfs.keys():
        dfs[key] = dfs[key].replace(r'^\s*$', np.nan, regex=True)
        dfs[key]= dfs[key].apply(pd.to_numeric, errors='coerce') 
        dfs[key]=dfs[key].dropna(how="any")
    

    bid_df=pd.DataFrame()
    ask_df=pd.DataFrame()
    
    bid_cols = {}
    ask_cols = {}

    # print(bid_df)
    for key in dfs.keys():
        df=dfs[key]

        for level1, level2 in df.columns:
            col_name = f"{key}_{level2}"
            s = df[(level1, level2)]

            if level1=="Ask":
                ask_cols[col_name]= s
            else:
                bid_cols[col_name]= s

    bid_df= pd.concat(bid_cols, axis =1)
    bid_df = bid_df.iloc[::-1]
    ask_df=pd.concat(ask_cols,axis=1)
    ask_df = ask_df.iloc[::-1]
    
    bid_df.index = pd.to_datetime(bid_df.index)
    ask_df.index = pd.to_datetime(ask_df.index)

    # Force Ascending Order (Oldest -> Newest)
    bid_df.sort_index(ascending=True, inplace=True)
    ask_df.sort_index(ascending=True, inplace=True)

    bid_df.columns = bid_df.columns.str.replace(" Volatility", "", n=1)
    ask_df.columns = ask_df.columns.str.replace(" Volatility", "", n=1)

    common_cols = bid_df.columns.intersection(ask_df.columns)
    mid_df = (bid_df[common_cols] + ask_df[common_cols]) / 2    
    
    # print(mid_df)
    
    mid_df=mid_df.dropna(how="any")
    mid_df.to_excel("OrganizedData.xlsx")
    
    return({
        "mid_df": mid_df,
        "bid_df": bid_df,
        "ask_df": ask_df
    })

def pricer():
    df=pd.read_excel("OrganizedData.xlsx",index_col=0)
    df=df.apply(pd.to_numeric,errors="coerce")
    df=df.dropna(axis=0,how="any")
    # print(df)
    
    spot_df= pd.read_excel("SPY_Complete.xlsx",sheet_name="Underlying",index_col=0)
    other_df= pd.read_excel("SPY_Complete.xlsx",sheet_name="Other",index_col=0)
    # print(spot_df)
    # print(other_df)
    
    valid_strike = [30,40,60,80,90,95,97.5,100,105,110,120,130,150,300]
    valid_tenor = ["1W","2W","3W","1M","2M","3M","6M","9M","1Y","18M","2Y"]

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
            
    while True:
        
        print(f"Strike options: {valid_strike}")
        strike = float(input("Enter strike: "))
        print()
        if(strike in valid_strike):
            print(f"Valid Strike! You selected {strike}")
            break
        else:
            print(f"Invalid Strike: {strike}")
            print("Please re-enter.\n")
    while True:
            
        print(f"Maturity options: {np.array(valid_tenor)}")
        maturity = input("Enter maturity: ")
        print()
        if(maturity in valid_tenor):
            print(f"Valid Maturity! You selected {maturity}")
            break
        else:
            print(f"Invalid Maturity: {maturity}")
            print("Please re-enter.\n")
    
    volatility = df.loc[aggregate_date][f"{maturity}_{int(strike)}_Volatility"]
    spot= spot_df.loc[aggregate_date]["Mid"]
    interest= other_df.loc[aggregate_date]["Interest"]
    dividend = other_df.loc[aggregate_date]["Dividend"]
    strike_price = strike/100 * spot
    print(f"Implied Volatility Under Chosen Parameters is {volatility:4f}, spot is {spot:4f} and interest rate is {interest:4f}, strike price is {strike_price:4f}, dividend is {dividend:4f}")
    
    option =blackScholes(spot,strike_price,volatility,maturity,interest,dividend)
    print(f"Call price is {option["call_price"]:4f} Put price is {option["put_price"]:4f}")
    print()
    
    market= [option["call_price"],option["put_price"]]
    
    return (spot,strike_price,maturity,interest,dividend,volatility,market)
       

def heston(spot, strike, maturity,r, q, parameters):

    unit = "".join(c for c in maturity if c.isalpha())
    length = int("".join(c for c in maturity if c.isdigit()))
    if unit == "W":
        texp = length / 52
    elif unit == "M":
        texp = length / 12
    elif unit == "Y":
        texp = length
    maturity=texp

    v0,kappa,theta,eta,rho= parameters.values()
    
    # pyfeng Heston FFT model
    m = pf.HestonFft(
        sigma=v0,
        vov=eta,
        rho=rho,
        mr=kappa,
        theta=theta,
        intr=r,
        divr=q
    )
    call_prices = m.price(strike, spot, texp, cp=1)
    

    put_prices = m.price(strike, spot, texp, cp=-1)
    # print("Heston Call prices:", call_prices)
    # print("Heston Put prices:", put_prices)
    
    return call_prices, put_prices


def hestonOptimization(spot, strike, maturity, implied_vol, r, q, objective, targetMSE=0.001):
    print(f"Market Call Prices is {objective[0]}")
    print(f"Market Put Prices is {objective[1]}")

    def loss(x):
        params = {
            "initial variance": x[0],
            "mean reversion": x[1],
            "long-run variance": x[2],
            "vol of vol": x[3],
            "correlation": x[4]
        }
        
        heston_call, heston_put = heston(spot, strike, maturity, r, q, params)
        model = [heston_call, heston_put]
        return calculate_mse(model, objective)

    x0 = [
        implied_vol**2,
        2.0,
        implied_vol**2,
        0.5,
        -0.5
    ]
    
    bounds = [
        (1e-6, 5.0),     # initial variance
        (1e-6, 10.0),    # mean reversion
        (1e-6, 5.0),     # long-run variance
        (1e-6, 5.0),     # vol of vol
        (-0.999, 0.999)  # correlation
    ]
    
    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    
    best_x = result.x
    best_params = {
        "initial variance": best_x[0],
        "mean reversion": best_x[1],
        "long-run variance": best_x[2],
        "vol of vol": best_x[3],
        "correlation": best_x[4]
    }
    
    heston_call, heston_put = heston(spot, strike, maturity, r, q, best_params)
    best_model = [heston_call, heston_put]
    best_error = calculate_mse(best_model, objective)

    print(f"Final best error = {best_error}")
    print(f"Best model call prices = {best_model[0]:4f} versus market call price {objective[0]:4f}")
    print(f"Best model put prices = {best_model[1]:4f} versus market put price {objective[1]:4f}")
 
    for k, v in best_params.items():
        print(f"{k}: {float(v):.6f}")
    
    return best_params, best_model, best_error
    
    
def run():
   spot,strike_price,maturity,interest,dividend,volatility,market= pricer()
   interest = interest/100
   dividend= dividend/100
   volatility = volatility/100
   hestonOptimization(spot, strike_price, maturity, volatility, interest,dividend, market)
   
   
if __name__ == "__main__":
    run()
    
