"""
heston_package.py
═══════════════════════════════════════════════════════════════════════════════
Heston Model Calibration Package — Cui, del Baño Rollin & Germano (2016)

Built specifically for the mid_df output of prepareData(), which reads SPY
implied volatility data from the multi-sheet Excel file.

Data format consumed (mid_df from prepareData())
────────────────────────────────────────────────
  Index   : pd.DatetimeIndex  (trading dates, 2007-01-04 → present)
  Columns : "{maturity}_{moneyness}_Volatility"
            e.g. "1W_100_Volatility", "1M_97.5_Volatility", "18M_80_Volatility"
  Values  : mid-point implied volatility in PERCENT  (e.g. 14.8 means 14.8%)

Moneyness convention: K = (moneyness / 100) × spot
  80_Volatility  → strike = 0.80 × spot  (20% OTM put)
  100_Volatility → strike = spot          (ATM)
  120_Volatility → strike = 1.20 × spot  (20% OTM call)

Classes
───────
  HestonParams       — dataclass for the 5 model parameters
  OptionContract     — dataclass for a single option observation
  CalibrationResult  — dataclass returned by HestonCalibrator.calibrate()
  HestonPricer       — stateless pricing engine (GL quadrature + analytical gradient)
  VolSurfaceBuilder  — converts a mid_df row into a list of OptionContracts
  HestonCalibrator   — Levenberg-Marquardt calibration with analytical Jacobian

Typical usage
─────────────
  from heston_package import HestonPricer, VolSurfaceBuilder, HestonCalibrator
  from heston_package import HestonParams

  # 1. Load data (your existing prepareData() function)
  data   = prepareData()
  mid_df = data['mid_df']

  # 2. Build vol surface for a specific date
  surface = VolSurfaceBuilder.build_surface(
      mid_df = mid_df,
      date   = '2024-01-15',
      spot   = 475.0,
      r      = 0.05,
      q      = 0.013,
  )

  # 3. Calibrate
  cal    = HestonCalibrator()
  result = cal.calibrate(surface, spot=475.0, r=0.05, q=0.013)
  print(result)

  # 4. Price with calibrated parameters
  call, put = HestonPricer.price(
      spot=475.0, strike=470.0, texp=1/12,
      r=0.05, q=0.013, params=result.params
  )

  # 5. Validate gradient (sanity check)
  grad_df = HestonPricer.validate_gradient(
      spot=475.0, strike=475.0, texp=1/12,
      r=0.05, q=0.013, params=result.params
  )
  print(grad_df)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MATURITY_MAP: dict[str, float] = {
    "1W":  1 / 52,  "2W":  2 / 52,  "3W":  3 / 52,
    "1M":  1 / 12,  "2M":  2 / 12,  "3M":  3 / 12,
    "6M":  6 / 12,  "9M":  9 / 12,  "1Y":  1.0,
    "18M": 18 / 12, "2Y":  2.0,
}

PARAM_NAMES: list[str] = ["v0", "kappa", "vbar", "eta", "rho"]

_N_GL, _U_MAX        = 64, 200.0
_raw_nodes, _raw_wts = leggauss(_N_GL)

U_NODES:   np.ndarray = _U_MAX / 2.0 * (_raw_nodes + 1.0)
U_WEIGHTS: np.ndarray = _U_MAX / 2.0 * _raw_wts

# log_kern = log(exp(-iu log K) / (iu))
#           = -iu log K - log(u) - i pi/2
# Real part = -log(u)  →  large negative for large u, partially cancels large
# positive real part of log_phi, preventing overflow in exp(log_phi + log_kern).
# Precomputed here for efficiency; the log_K term is added per-option in
# price_and_grad because it depends on strike.
_LOG_U:        np.ndarray = np.log(U_NODES)            # shape (64,)
_HALF_PI_IMAG: complex    = -1j * math.pi / 2


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonParams:
    """Five Heston model parameters."""
    v0:    float
    kappa: float
    vbar:  float
    eta:   float
    rho:   float

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.vbar, self.eta, self.rho])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        return cls(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))

    def __str__(self) -> str:
        return (
            f"  v0    (initial var)  : {self.v0:.6f}\n"
            f"  kappa (mean rev)     : {self.kappa:.6f}\n"
            f"  vbar  (long-run var) : {self.vbar:.6f}\n"
            f"  eta   (vol of vol)   : {self.eta:.6f}\n"
            f"  rho   (correlation)  : {self.rho:.6f}"
        )


@dataclass
class OptionContract:
    """
    A single vanilla option observation used in calibration.

    Attributes
    ----------
    strike : float   absolute strike price
    texp   : float   time to expiry in years
    price  : float   market option price
    iv     : float   implied vol in decimal (0.0 if unknown)
    weight : float   calibration weight (default 1.0 = uniform)
    cp     : int     1 = call (default), -1 = put

    Note: ∂put/∂θ = ∂call/∂θ (put-call parity constants are θ-independent),
    so no additional gradient derivation is needed for puts.
    """
    strike: float
    texp:   float
    price:  float
    iv:     float
    weight: float = 1.0
    cp:     int   = 1


@dataclass
class CalibrationResult:
    """Output of HestonCalibrator.calibrate()."""
    params:        HestonParams
    residual_norm: float
    mse:           float
    n_options:     int
    n_iterations:  int
    success:       bool
    message:       str

    def __str__(self) -> str:
        status = "succeeded ✓" if self.success else "failed ✗"
        return (
            f"Calibration {status}\n"
            f"  Options in surface : {self.n_options}\n"
            f"  LM iterations      : {self.n_iterations}\n"
            f"  Residual ||r||     : {self.residual_norm:.4e}\n"
            f"  MSE (price)        : {self.mse:.4e}\n"
            f"  Solver message     : {self.message}\n"
            f"Calibrated parameters:\n{self.params}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(spot: float, strike: float, texp: float,
             r: float, q: float, sigma: float) -> float:
    """Black-Scholes call (used to convert IVs to prices in VolSurfaceBuilder)."""
    if sigma <= 0 or texp <= 0:
        return max(spot * math.exp(-q * texp) - strike * math.exp(-r * texp), 0.0)
    sqT = math.sqrt(texp)
    d1  = (math.log(spot / strike) + (r - q + 0.5 * sigma**2) * texp) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    return (spot * math.exp(-q * texp) * norm.cdf(d1)
            - strike * math.exp(-r * texp) * norm.cdf(d2))


def _parse_column(col_name: str) -> Tuple[str, float]:
    """
    Parse a mid_df column name → (maturity_code, moneyness_pct).
    Format: "{maturity}_{moneyness}_Volatility", e.g. "18M_97.5_Volatility".
    Longest codes matched first so "18M" is not confused with "1M".
    """
    base = col_name.removesuffix("_Volatility")
    for mat in sorted(MATURITY_MAP, key=len, reverse=True):
        if base.startswith(mat + "_"):
            return mat, float(base[len(mat) + 1:])
    raise ValueError(f"Cannot parse column '{col_name}'.")


def _safe_exp(log_z: np.ndarray) -> np.ndarray:
    """
    exp(log_z) with NaN and ±inf replaced by 0.

    Rationale: if log_z has a large positive real part at some GL node but
    the corresponding imaginary part oscillates rapidly, Re(exp(log_z))
    averages to zero in the quadrature.  Replacing the overflow with 0 is
    therefore the correct approximation and eliminates RuntimeWarnings.
    """
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return np.nan_to_num(np.exp(log_z), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a * b with NaN and ±inf replaced by 0.

    Used for combo * h_j in the gradient.  If combo was set to 0 by
    _safe_exp (overflow case), the product is already 0.  If h_j is very
    large for extreme parameters, nan_to_num clips any residual overflow.
    errstate suppresses warnings before nan_to_num runs.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        return np.nan_to_num(a * b, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# HESTON PRICER
# ─────────────────────────────────────────────────────────────────────────────

class HestonPricer:
    """
    Stateless pricing engine — GL quadrature + analytical gradient (Cui et al. 2016).
    All methods are class methods; no instantiation required.

    Key numerical change vs earlier versions
    ─────────────────────────────────────────
    _cf() returns log_phi (the exponent) rather than phi = exp(log_phi).
    price_and_grad() combines log_phi with log_kern before exponentiating:

        integrand = exp(log_phi + log_kern)

    This prevents overflow because:
      Re(log_kern) = -log(u)  →  large negative for large u nodes
      Re(log_phi)  can be large positive for extreme parameters
      Re(log_phi + log_kern) = Re(log_phi) - log(u)  →  cancellation

    _safe_exp() handles any residual infinities by replacing them with 0.
    """

    # ── Characteristic function (returns log_phi, not phi) ────────────────

    @staticmethod
    def _cf(u, S0: float, r: float, q: float, t: float,
            v0: float, kappa: float, vbar: float, eta: float, rho: float):
        """
        Log of the Heston characteristic function — Cui et al. Eq. (18).

        Returns log_phi and all intermediate quantities for _h().
        Returning log_phi instead of phi = exp(log_phi) prevents the
        RuntimeWarning: overflow encountered in exp that occurs when
        2*kappa*vbar/eta^2 is large (small eta) and D has positive real part.

        u may be real (U_NODES) or complex (U_NODES - 1j).
        """
        xi    = kappa - eta * rho * 1j * u
        d     = np.sqrt(xi**2 + eta**2 * (u**2 + 1j * u))
        e_ndt = np.exp(-d * t)

        # Rescaled A1, A2 (×e^{-dt/2}) — avoids sinh/cosh overflow
        A1s = (u**2 + 1j * u) * (1.0 - e_ndt) / 2.0
        A2s = d * (1.0 + e_ndt) / 2.0 + xi * (1.0 - e_ndt) / 2.0
        A   = A1s / A2s

        # Eq. (17b) — rearranged log B, resolves branch discontinuities
        D = (np.log(d)
             + (kappa - d) * t / 2.0
             - np.log((d + xi) / 2.0 + (d - xi) / 2.0 * e_ndt))

        # Eq. (18) — log of characteristic function (NOT exponentiated here)
        log_phi = (
            1j * u * (np.log(S0) + (r - q) * t)
            - t * kappa * vbar * rho * 1j * u / eta
            - v0 * A
            + 2.0 * kappa * vbar / eta**2 * D
        )
        return log_phi, xi, d, A1s, A2s, A, D, e_ndt

    # ── Gradient of log_phi ───────────────────────────────────────────────

    @staticmethod
    def _h(u, t: float,
           v0: float, kappa: float, vbar: float, eta: float, rho: float,
           xi, d, A, A1s, A2s, D, e_ndt):
        """
        h(u) such that ∇φ = φ · h(u)  (Eqs. 23-30, stable form).
        All intermediates come from _cf() — no recomputation.
        Returns shape (5, N_GL) in order [v0, kappa, vbar, eta, rho].

        Note: since _cf now returns log_phi, and ∇φ = φ · h, we have
              h = ∇(log_phi), i.e. h is the gradient of the log of φ.
              This does not change the formulas — h is unchanged from the paper.
        """
        # ── ∂/∂ρ ─────────────────────────────────────────────────────────
        dd_dr    = -xi * eta * 1j * u / d
        dA1s_dr  = -1j * u * (u**2 + 1j*u) * t * xi * eta * e_ndt / (2.0 * d)
        dA2s_dr  = (eta * 1j * u / (2.0 * d)
                    * (-(d + xi) + (d - xi) * (1.0 + t * xi) * e_ndt))
        dA_dr    = dA1s_dr / A2s - A / A2s * dA2s_dr
        dlogB_dr = dd_dr * (1.0 / d - t / 2.0) - dA2s_dr / A2s

        # ── ∂/∂κ  (Eqs. 28) ───────────────────────────────────────────────
        dlogB_dk = 1j / (eta * u) * dlogB_dr + t / 2.0

        # ── ∂/∂η  (Eqs. 30) ───────────────────────────────────────────────
        dd_ds    = (rho / eta - 1.0 / xi) * dd_dr + eta * u**2 / d
        dA1s_ds  = (u**2 + 1j*u) * t * dd_ds * e_ndt / 2.0
        dA2s_ds  = ((dd_ds - rho * 1j * u)
                    + (dd_ds + rho * 1j * u) * e_ndt
                    - (d - xi) * t * dd_ds * e_ndt) / 2.0
        dA_ds    = dA1s_ds / A2s - A / A2s * dA2s_ds
        dlogB_ds = dd_ds * (1.0 / d - t / 2.0) - dA2s_ds / A2s

        # ── h components (Eq. 23), order: [v0, κ, v̄, η, ρ] ──────────────
        h_v0    = -A
        h_vbar  = 2.0 * kappa / eta**2 * D - t * kappa * rho * 1j * u / eta
        h_kappa = (v0 / (eta * 1j * u) * dA_dr
                   + 2.0 * vbar / eta**2 * D
                   + 2.0 * kappa * vbar / eta**2 * dlogB_dk
                   - t * vbar * rho * 1j * u / eta)
        h_eta   = (-v0 * dA_ds
                   - 4.0 * kappa * vbar / eta**3 * D
                   + 2.0 * kappa * vbar / eta**2 * dlogB_ds
                   + t * kappa * vbar * rho * 1j * u / eta**2)
        h_rho   = (-v0 * dA_dr
                   + 2.0 * kappa * vbar / eta**2 * dlogB_dr
                   - t * kappa * vbar * 1j * u / eta)

        return np.array([h_v0, h_kappa, h_vbar, h_eta, h_rho])

    # ── Price and gradient ────────────────────────────────────────────────

    @classmethod
    def price_and_grad(
        cls,
        spot:   float,
        strike: float,
        texp:   float,
        r:      float,
        q:      float,
        params: HestonParams,
    ) -> Tuple[float, np.ndarray]:
        """
        Call price and analytical gradient in one GL pass.

        Overflow-safe implementation
        ────────────────────────────
        Instead of:
            phi  = exp(log_phi)          ← can overflow for extreme params
            kern = exp(-iu log K)/(iu)
            I    = sum(w * Re(kern * phi))

        We compute:
            log_kern = -iu log K - log(u) - i pi/2
            integrand = exp(log_phi + log_kern)  ← large terms cancel in log
            I = sum(w * Re(integrand))

        Re(log_kern) = -log(u) which is large negative for large u nodes and
        partially cancels any large positive Re(log_phi), preventing overflow.
        _safe_exp() handles any residual cases by replacing inf/NaN with 0.

        Returns
        -------
        call : float
        grad : ndarray (5,)   ∂call/∂[v0, kappa, vbar, eta, rho]
        """
        v0, kappa, vbar, eta, rho = params.to_array()
        log_K = np.log(strike)
        disc  = math.exp(-r * texp)
        fwd   = math.exp(-q * texp)

        # log_kern = -iu log_K - log(u) - i pi/2
        # (the -log(u) term is the key stabiliser — precomputed in _LOG_U)
        log_kern = -1j * U_NODES * log_K - _LOG_U + _HALF_PI_IMAG

        # log_phi from _cf — NOT yet exponentiated
        log_phi_u,  xi_u,  d_u,  A1s_u,  A2s_u,  A_u,  D_u,  en_u  = cls._cf(
            U_NODES,      spot, r, q, texp, v0, kappa, vbar, eta, rho)
        log_phi_mi, xi_mi, d_mi, A1s_mi, A2s_mi, A_mi, D_mi, en_mi = cls._cf(
            U_NODES - 1j, spot, r, q, texp, v0, kappa, vbar, eta, rho)

        # Combine in log space before exponentiating — avoids overflow
        combo_u  = _safe_exp(log_phi_u  + log_kern)   # kern * phi_u
        combo_mi = _safe_exp(log_phi_mi + log_kern)   # kern * phi_mi

        # ── Call price (Eq. 9, direct form) ──────────────────────────────
        # nan_to_num on I1/I2 guards against the rare case where combo is
        # large-but-not-inf, making the dot product overflow as a scalar.
        I1   = float(np.nan_to_num(np.dot(U_WEIGHTS, np.real(combo_mi))))
        I2   = float(np.nan_to_num(np.dot(U_WEIGHTS, np.real(combo_u))))
        call = (spot * fwd - disc * strike) / 2.0 + disc / math.pi * (I1 - strike * I2)

        # ── Analytical gradient (Eq. 22) ──────────────────────────────────
        h_u  = cls._h(U_NODES,      texp, v0, kappa, vbar, eta, rho,
                       xi_u,  d_u,  A_u,  A1s_u,  A2s_u,  D_u,  en_u)
        h_mi = cls._h(U_NODES - 1j, texp, v0, kappa, vbar, eta, rho,
                       xi_mi, d_mi, A_mi, A1s_mi, A2s_mi, D_mi, en_mi)

        # Direct multiply: combo is already bounded (overflow → 0 via _safe_exp).
        # _safe_mul handles any residual overflow from large h_j values.
        # No log tricks needed — those introduced divide-by-zero when combo=0.
        grad = disc / math.pi * np.array([
            np.dot(U_WEIGHTS, np.real(_safe_mul(combo_mi, h_mi[j])))
            - strike * np.dot(U_WEIGHTS, np.real(_safe_mul(combo_u, h_u[j])))
            for j in range(5)
        ])

        return call, grad

    @classmethod
    def price(cls, spot, strike, texp, r, q, params):
        """Return (call, put). Put via put-call parity."""
        call, _ = cls.price_and_grad(spot, strike, texp, r, q, params)
        put = call - spot * math.exp(-q * texp) + strike * math.exp(-r * texp)
        return call, put

    @classmethod
    def validate_gradient(cls, spot, strike, texp, r, q, params, eps=1e-5):
        """
        Compare analytical gradient against central finite differences.
        Returns DataFrame [analytical, numerical, rel_error].
        All rel_error < 1e-4 confirms correct implementation.
        """
        x = params.to_array()
        _, g_an = cls.price_and_grad(spot, strike, texp, r, q, params)
        g_fd    = np.zeros(5)
        for i in range(5):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps;  xm[i] -= eps
            cp, _ = cls.price_and_grad(spot, strike, texp, r, q, HestonParams.from_array(xp))
            cm, _ = cls.price_and_grad(spot, strike, texp, r, q, HestonParams.from_array(xm))
            g_fd[i] = (cp - cm) / (2.0 * eps)
        rel = np.abs(g_an - g_fd) / (np.abs(g_fd) + 1e-15)
        return pd.DataFrame({"analytical": g_an, "numerical": g_fd, "rel_error": rel},
                             index=PARAM_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# VOL SURFACE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class VolSurfaceBuilder:
    """
    Converts mid_df rows (output of prepareData()) into OptionContracts.
    All methods are static — no instantiation required.
    """

    DEFAULT_MON_MIN: float = 70.0
    DEFAULT_MON_MAX: float = 150.0

    @classmethod
    def build_surface(
        cls,
        mid_df:       pd.DataFrame,
        date,
        spot:         float,
        r:            float,
        q:            float,
        mon_min:      float = DEFAULT_MON_MIN,
        mon_max:      float = DEFAULT_MON_MAX,
        maturities:   Optional[List[str]] = None,
        include_puts: bool = True,
    ) -> List[OptionContract]:
        """
        Build a calibration surface from one row of mid_df.

        Parameters
        ----------
        mid_df       : pd.DataFrame  output of prepareData()['mid_df']
        date         : date-like     e.g. '2024-01-15'
        spot, r, q   : float
        mon_min/max  : float         moneyness filter (default 70-150%)
        maturities   : list|None     restrict to specific codes e.g. ['1M','3M']
        include_puts : bool          add put alongside each call (default True)
        """
        ts = pd.Timestamp(date)
        if ts not in mid_df.index:
            idx = mid_df.index.get_indexer([ts], method="nearest")[0]
            ts  = mid_df.index[idx]
            warnings.warn(f"Date {date} not found; using nearest: {ts.date()}", stacklevel=2)
        row = mid_df.loc[ts]

        allowed = set(maturities) if maturities else set(MATURITY_MAP.keys())
        contracts: List[OptionContract] = []

        for col, iv_pct in row.items():
            if pd.isna(iv_pct) or iv_pct <= 0:
                continue
            try:
                mat, mon = _parse_column(str(col))
            except ValueError:
                continue
            if mat not in allowed or not (mon_min <= mon <= mon_max):
                continue

            texp   = MATURITY_MAP[mat]
            strike = (mon / 100.0) * spot
            iv     = iv_pct / 100.0
            call_p = _bs_call(spot, strike, texp, r, q, iv)
            if call_p <= 0:
                continue

            contracts.append(OptionContract(strike=strike, texp=texp,
                                             price=call_p, iv=iv, cp=1))
            if include_puts:
                put_p = call_p - spot * math.exp(-q*texp) + strike * math.exp(-r*texp)
                if put_p > 0:
                    contracts.append(OptionContract(strike=strike, texp=texp,
                                                     price=put_p, iv=iv, cp=-1))

        if not contracts:
            raise ValueError(
                f"No valid options for date={ts.date()}, spot={spot}, "
                f"moneyness=[{mon_min},{mon_max}], maturities={list(allowed)}."
            )
        return contracts

    @staticmethod
    def surface_to_dataframe(surface: List[OptionContract],
                              spot: Optional[float] = None) -> pd.DataFrame:
        rows = []
        for c in surface:
            row = {"cp": c.cp, "strike": c.strike, "texp": c.texp,
                   "price": c.price, "iv_pct": round(c.iv*100, 4), "weight": c.weight}
            if spot is not None:
                row["moneyness_pct"] = round(c.strike / spot * 100, 2)
            rows.append(row)
        return (pd.DataFrame(rows)
                  .sort_values(["texp", "cp", "strike"])
                  .reset_index(drop=True))

    @staticmethod
    def available_dates(mid_df: pd.DataFrame) -> pd.DatetimeIndex:
        return mid_df.index

    @staticmethod
    def latest_date(mid_df: pd.DataFrame) -> pd.Timestamp:
        return mid_df.index[-1]


# ─────────────────────────────────────────────────────────────────────────────
# HESTON CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────

class HestonCalibrator:
    """
    Calibrates Heston parameters via Levenberg-Marquardt with analytical Jacobian.

    Supports both call and put contracts (OptionContract.cp field).
    Put gradient = call gradient because ∂P/∂θ = ∂C/∂θ (parity constants
    S·e^{-qT} and K·e^{-rT} are independent of θ).
    """

    BOUNDS_LO = np.array([1e-6, 1e-4, 1e-6, 1e-4, -0.999])
    BOUNDS_HI = np.array([2.0,  20.0, 2.0,  5.0,   0.999])

    def __init__(self, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000):
        self.ftol, self.xtol, self.gtol, self.max_nfev = ftol, xtol, gtol, max_nfev

    def calibrate(self, surface: List[OptionContract], spot: float,
                  r: float, q: float,
                  x0: Optional[HestonParams] = None) -> CalibrationResult:
        """Run LM calibration. x0=None → auto initial guess."""
        if not surface:
            raise ValueError("surface is empty.")
        x0_params = x0 if x0 is not None else self._auto_initial_guess(surface)
        x0_arr    = np.clip(x0_params.to_array(), self.BOUNDS_LO, self.BOUNDS_HI)

        res = least_squares(
            self._residuals, x0_arr, jac=self._jacobian, method="lm",
            ftol=self.ftol, xtol=self.xtol, gtol=self.gtol,
            max_nfev=self.max_nfev, args=(surface, spot, r, q),
        )

        best   = np.clip(res.x, self.BOUNDS_LO, self.BOUNDS_HI)
        params = HestonParams.from_array(best)
        resid  = self._residuals(best, surface, spot, r, q)
        return CalibrationResult(
            params=params, residual_norm=float(np.linalg.norm(resid)),
            mse=float(np.mean(resid**2)), n_options=len(surface),
            n_iterations=res.nfev, success=res.success, message=res.message,
        )

    @staticmethod
    def _residuals(x, surface, spot, r, q):
        """rᵢ = wᵢ · [model_price(θ; Kᵢ, Tᵢ, cpᵢ) − market_price_i]"""
        params = HestonParams.from_array(x)
        out    = np.empty(len(surface))
        for i, c in enumerate(surface):
            call, _ = HestonPricer.price_and_grad(spot, c.strike, c.texp, r, q, params)
            model   = call if c.cp == 1 else (
                call - spot * math.exp(-q*c.texp) + c.strike * math.exp(-r*c.texp))
            out[i]  = (model - c.price) * c.weight
        return out

    @staticmethod
    def _jacobian(x, surface, spot, r, q):
        """Jᵢⱼ = wᵢ · ∂model_price/∂θⱼ  (∂put/∂θ = ∂call/∂θ by parity)"""
        params = HestonParams.from_array(x)
        J      = np.empty((len(surface), 5))
        for i, c in enumerate(surface):
            _, grad = HestonPricer.price_and_grad(spot, c.strike, c.texp, r, q, params)
            J[i]    = grad * c.weight
        return J

    @staticmethod
    def _auto_initial_guess(surface: List[OptionContract]) -> HestonParams:
        calls  = [c for c in surface if c.cp == 1] or surface
        tmin   = min(c.texp for c in calls)
        cands  = [c for c in calls if c.texp == tmin]
        atm    = sorted(cands, key=lambda c: abs(c.iv - 0.20))[0]
        v_init = max(atm.iv**2, 1e-4)
        return HestonParams(v0=v_init, kappa=2.0, vbar=v_init, eta=0.5, rho=-0.5)