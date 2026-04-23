from heston_package import HestonPricer, VolSurfaceBuilder, HestonCalibrator, HestonParams

data   = prepareData()
mid_df = data['mid_df']

# --- build surface for a specific date ---
surface = VolSurfaceBuilder.build_surface(
    mid_df     = mid_df,
    date       = '2024-01-15',
    spot       = 475.0,        # SPY spot on that date
    r          = 0.05,         # risk-free rate
    q          = 0.013,        # SPY dividend yield
    mon_min    = 70.0,         # exclude strikes below 70% moneyness
    mon_max    = 150.0,        # exclude strikes above 150% moneyness
    maturities = ['1M','3M','6M','1Y'],  # optional: restrict maturities
)

# --- inspect what went into calibration ---
df = VolSurfaceBuilder.surface_to_dataframe(surface, spot=475.0)
print(df)

# --- calibrate ---
cal    = HestonCalibrator()
result = cal.calibrate(surface, spot=475.0, r=0.05, q=0.013)
print(result)

# --- price any option with calibrated params ---
call, put = HestonPricer.price(
    spot=475.0, strike=460.0, texp=1/12,
    r=0.05, q=0.013, params=result.params
)

# --- validate gradient before trusting calibration ---
grad_check = HestonPricer.validate_gradient(
    spot=475.0, strike=475.0, texp=1/12,
    r=0.05, q=0.013, params=result.params
)
print(grad_check)  # rel_error column should all be < 1e-4
