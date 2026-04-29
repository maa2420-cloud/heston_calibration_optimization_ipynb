
import math 
import numpy as np
import pandas as pd

# skew_factor=""
# smile_factor=""
# curve_factor=""

rho_default,   rho_dn,   rho_up   = -0.7, 0.2, 0.2    # ρ ∈ [-0.9, -0.1] # correlation, skew , z-score
eta_default,   eta_dn,   eta_up   =  0.5, 0.45, 1.0    # η ∈ [ 0.05, 1.5]    # vol of vol, smile, z_score
kappa_default, kappa_dn, kappa_up =  2.0, 1.5, 1.5    # κ ∈ [ 0.5, 3.5]    # mean reversion, curve, slope 


def sigmoid(x):
    return math.tanh(x / 4) 

def apply_pc(default, delta_neg, delta_pos, score, sign):
    """sign = +1 if PC ↑ should push parameter up, -1 if down."""
    s = sign * sigmoid(score)
    return default + (delta_pos if s >= 0 else delta_neg) * s

    
def _atm_iv(df_row,tenor):
    col = f"{tenor}_100_Volatility"
    return float(df_row[col]) / 100.0 if col in df_row.index else 20 / 100.0


def initialization(factors,df_row):
    # pc_1 = data["PC1"]   ## smile          → eta
    # pc_2 = data["PC2"]   ## skew           → rho
    # pc_5 = data["PC5"]   ## term structure → kappa
    
    curve=factors["curve"]
    smile=factors["smile"]
    skew=factors["skew"]
    

    # PC2 ↑ → put skew flattens → ρ less negative → ρ UP → sign = +1
    rho_initial   = apply_pc(rho_default,   rho_dn,   rho_up,  skew, sign=-1)
    # PC1 ↑ → smile stronger → η UP → sign = +1
    eta_initial   = apply_pc(eta_default,   eta_dn,   eta_up,  smile, sign=+1)
    # PC5 ↑ → upward term structure → κ DOWN (slow reversion) → sign = -1
    # higher value, higher reversion
    kappa_initial = apply_pc(kappa_default, kappa_dn, kappa_up, curve, sign=+1)
    # kappa_initial= kappa_default
    # rho_initial = rho_default

    # v0_initial   = _atm_iv(df_row,"1W") ** 2
    # vbar_initial = _atm_iv(df_row,"1Y") ** 2
    v0_initial   = _atm_iv(df_row,"1W") ** 2
    vbar_initial = _atm_iv(df_row,"1Y") ** 2

    v0_initial    = float(np.clip(v0_initial,    1e-6, 5.0))
    vbar_initial  = float(np.clip(vbar_initial,  1e-6, 5.0))
    eta_initial   = float(np.clip(eta_initial,   1e-4, 6.0))
    kappa_initial = float(np.clip(kappa_initial, 1e-4, 20.0))
    rho_initial   = float(np.clip(rho_initial,  -0.999, 0.999))

    initial_params = {
        "initial variance":  v0_initial,
        "mean reversion":    kappa_initial,
        "long-run variance": vbar_initial,
        "vol of vol":        eta_initial,
        "correlation":       rho_initial,
    }
    return initial_params



# ── Defaults: parameter values returned on a "typical" day ─────────────
# These are what the init produces when each raw factor sits at its
# historical median (i.e. today's surface looks normal). They roughly
# track median calibrated values from past Heston fits.
RHO_DEFAULT,   RHO_HALF   = -0.7, 0.20    # rho   ∈ ~[-0.95, -0.55]
ETA_DEFAULT,   ETA_HALF   =  0.8, 0.55    # eta   ∈ ~[0.30, 1.40]
KAPPA_DEFAULT, KAPPA_HALF =  2.0,  1.5     # kappa ∈ ~[0.5, 3.5]

# ── Anchors and scales for raw factor scores ───────────────────────────
# Raw factors are not centered at zero and live on different scales.
# We subtract the empirical median (so the "typical day" lands at the
# default parameter value) and divide by a robust scale so tanh saturates
# only on genuinely unusual days.
#
# Values pulled from the historical 2010+ distribution (n≈3900):
#   skew_score:   median 0.849, (q90-q10)/2 = 0.358, std 0.288   (ATM-scaled)
#   smile_score:  median 36.67, (q90-q10)/2 = 5.61,  std 5.47
#   pc5_score:    median  3.33, (q90-q10)/2 = 0.89,  std 0.71
#
# Using (q90-q10)/2 as SCALE: a q90 day → tanh(+1) ≈ +0.76,
# a q10 day → tanh(-1) ≈ -0.76. Saturation only above q95-ish days.
SKEW_MEDIAN,  SKEW_SCALE  = 0.849, 0.358
SMILE_MEDIAN, SMILE_SCALE = 36.67, 5.61
CURVE_MEDIAN, CURVE_SCALE =  3.33, 0.89


def initializationNew(factors, df_row):
    """Build a Heston initial-guess from raw factor scores.

    Parameters
    ----------
    factors : dict with keys 'skew', 'smile', 'curve'
        Today's RAW factor scores (not z-scored). Loaded from the
        skew_score / smile_score / pc5_score columns of the factor files.
    df_row : pandas Series
        Today's row from OrganizedData.xlsx. Used only for v0 and theta.

    Returns
    -------
    dict with keys 'initial variance', 'mean reversion',
                   'long-run variance', 'vol of vol', 'correlation'.
    """
    score_skew  = float(factors["skew"])
    score_smile = float(factors["smile"])
    score_curve = float(factors["curve"])

    # ── Center each score on its historical median, then scale ─────────
    # tanh maps the centered-and-scaled score into (-1, +1). The median
    # subtraction is what lets us use raw factors directly: a "typical"
    # day produces tanh ≈ 0, so the parameter sits at its default. There
    # is no implicit assumption that the input is mean-zero or unit-std.
    z_skew  = (score_skew  - SKEW_MEDIAN)  / SKEW_SCALE
    z_smile = (score_smile - SMILE_MEDIAN) / SMILE_SCALE
    z_curve = (score_curve - CURVE_MEDIAN) / CURVE_SCALE

    # ── Map to parameter deviations via tanh squashing ─────────────────
    # Sign conventions:
    #   skew  ↑  ⇒  steeper put skew         ⇒  rho more negative   (subtract)
    #   smile ↑  ⇒  more pronounced smile    ⇒  eta higher           (add)
    #   |curve|↑ ⇒  larger TS deviation      ⇒  stronger MR (kappa)  (add, |abs|)
    
    # rho_init   = RHO_DEFAULT   - RHO_HALF   * np.tanh(z_skew)
    # eta_init   = ETA_DEFAULT   + ETA_HALF   * np.tanh(z_smile)
    # # kappa_init = KAPPA_DEFAULT + KAPPA_HALF * np.tanh(abs(z_curve))
    # kappa_init = KAPPA_DEFAULT + KAPPA_HALF * np.tanh(z_curve)
    
    rho_init   = RHO_DEFAULT                              # = -0.75, no factor dependence
    eta_init   = ETA_DEFAULT + ETA_HALF * np.tanh(z_smile)  # KEEP — this works
    kappa_init = KAPPA_DEFAULT        

        # ── Variance levels: anchor on the observed term structure ─────────
    # v0 and theta are level quantities. Pull straight from observed ATM.
    def _atm_iv(tenor):
        col = f"{tenor}_100_Volatility"
        if col in df_row.index:
            return float(df_row[col]) / 100.0
        raise KeyError(f"Missing ATM column {col} in df_row")

    v0_init   = _atm_iv("1M") ** 2
    # vbar_init = _atm_iv("1Y") ** 2
    vbar_init = _atm_iv("2Y") ** 2

    # ── Clip to L-BFGS-B feasible region ───────────────────────────────
    v0_init    = float(np.clip(v0_init,    1e-6, 5.0))
    vbar_init  = float(np.clip(vbar_init,  1e-6, 5.0))
    eta_init   = float(np.clip(eta_init,   1e-4, 6.0))
    kappa_init = float(np.clip(kappa_init, 1e-4, 20.0))
    rho_init   = float(np.clip(rho_init,  -0.999, 0.999))

    return {
        "initial variance":  v0_init,
        "mean reversion":    kappa_init,
        "long-run variance": vbar_init,
        "vol of vol":        eta_init,
        "correlation":       rho_init,
    }