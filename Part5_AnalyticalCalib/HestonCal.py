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

#: Maps maturity code → year fraction.  Covers all sheets in prepareData().
MATURITY_MAP: dict[str, float] = {
    "1W":  1 / 52,
    "2W":  2 / 52,
    "3W":  3 / 52,
    "1M":  1 / 12,
    "2M":  2 / 12,
    "3M":  3 / 12,
    "6M":  6 / 12,
    "9M":  9 / 12,
    "1Y":  1.0,
    "18M": 18 / 12,
    "2Y":  2.0,
}

#: Heston parameter names in calibration order.
PARAM_NAMES: list[str] = ["v0", "kappa", "vbar", "eta", "rho"]

# Gauss-Legendre quadrature nodes and weights — precomputed once at import.
# Paper uses N=64 nodes and upper truncation limit of 200.
_N_GL, _U_MAX = 64, 200.0
_raw_nodes, _raw_weights = leggauss(_N_GL)

#: GL nodes mapped from [-1, 1] to [0, U_MAX]
U_NODES: np.ndarray = _U_MAX / 2.0 * (_raw_nodes + 1.0)

#: Corresponding integration weights
U_WEIGHTS: np.ndarray = _U_MAX / 2.0 * _raw_weights


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonParams:
    """
    The five Heston model parameters.

    Attributes
    ----------
    v0    : initial variance              (v₀)
    kappa : mean reversion speed          (κ)
    vbar  : long-run (equilibrium) variance (v̄)
    eta   : vol of vol                    (σ in the paper)
    rho   : correlation BM₁ and BM₂      (ρ)
    """
    v0:    float
    kappa: float
    vbar:  float
    eta:   float
    rho:   float

    def to_array(self) -> np.ndarray:
        """Return [v0, kappa, vbar, eta, rho] as a numpy array."""
        return np.array([self.v0, self.kappa, self.vbar, self.eta, self.rho])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        """Construct from a length-5 array [v0, kappa, vbar, eta, rho]."""
        return cls(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))

    def __str__(self) -> str:
        return (
            f"  v0    (initial var)    : {self.v0:.6f}\n"
            f"  kappa (mean rev)       : {self.kappa:.6f}\n"
            f"  vbar  (long-run var)   : {self.vbar:.6f}\n"
            f"  eta   (vol of vol)     : {self.eta:.6f}\n"
            f"  rho   (correlation)    : {self.rho:.6f}"
        )


@dataclass
class OptionContract:
    """
    A single vanilla call option observation used in calibration.

    Attributes
    ----------
    strike : float   absolute strike price
    texp   : float   time to expiry in years
    price  : float   market call price (converted from implied vol via BS)
    iv     : float   market implied vol in decimal (e.g. 0.148)
    weight : float   calibration weight (default 1.0, uniform)
    """
    strike: float
    texp:   float
    price:  float
    iv:     float
    weight: float = 1.0


@dataclass
class CalibrationResult:
    """
    Output of HestonCalibrator.calibrate().

    Attributes
    ----------
    params        : HestonParams   calibrated parameter set
    residual_norm : float          ||r(θ†)||  (lower is better)
    mse           : float          mean squared price error
    n_options     : int            number of options in the surface
    n_iterations  : int            number of LM function evaluations
    success       : bool
    message       : str            LM solver message
    """
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
# INTERNAL UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(spot: float, strike: float, texp: float,
             r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes call price.
    Used internally to convert implied volatilities to prices.
    """
    if sigma <= 0 or texp <= 0:
        return max(spot * np.exp(-q * texp) - strike * np.exp(-r * texp), 0.0)
    sqT = np.sqrt(texp)
    d1  = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * texp) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    return (spot * np.exp(-q * texp) * norm.cdf(d1)
            - strike * np.exp(-r * texp) * norm.cdf(d2))


def _parse_column(col_name: str) -> Tuple[str, float]:
    """
    Parse a mid_df column name into (maturity_code, moneyness_pct).

    Expected format (output of prepareData()):
        "{maturity}_{moneyness}_Volatility"
    Examples:
        "1W_100_Volatility"   → ("1W",  100.0)
        "18M_97.5_Volatility" → ("18M",  97.5)
        "2Y_30_Volatility"    → ("2Y",   30.0)

    Sorts maturity codes longest-first so "18M" is matched before "1M".
    """
    # Strip trailing _Volatility
    base = col_name.removesuffix("_Volatility")

    for mat in sorted(MATURITY_MAP, key=len, reverse=True):
        if base.startswith(mat + "_"):
            moneyness = float(base[len(mat) + 1:])
            return mat, moneyness

    raise ValueError(
        f"Cannot parse column '{col_name}'. "
        f"Expected format: '{{maturity}}_{{moneyness}}_Volatility', "
        f"e.g. '1M_100_Volatility'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# HESTON PRICER
# ─────────────────────────────────────────────────────────────────────────────

class HestonPricer:
    """
    Stateless pricing engine for vanilla options under the Heston model.

    Implements the Cui et al. (2016) continuous characteristic-function
    representation (Eq. 18) and computes option prices and their analytical
    parameter gradients via Gauss-Legendre quadrature (Algorithm 3.1).

    All methods are class methods — no instantiation required.

    Key equations from the paper
    ────────────────────────────
    φ(θ; u, t)  — characteristic function, Eq. (18)
    ∇φ = φ · h  — gradient factorisation, Eqs. (23)–(30)
    C(θ; K, T)  — call price via P₁, P₂ integrals, Eq. (9)
    ∇C          — gradient of call price, Eq. (22)
    """

    # ── Characteristic function ───────────────────────────────────────────

    @staticmethod
    def _cf(u, S0: float, r: float, q: float, t: float,
            v0: float, kappa: float, vbar: float, eta: float, rho: float):
        """
        φ(θ; u, t) — Cui et al. Eq. (18).

        u may be real or complex (U_NODES or U_NODES − i).
        Returns φ and all intermediate quantities consumed by _h(),
        avoiding recomputation.
        """
        # ── Eq. (11) ─────────────────────────────────────────────────────
        xi = kappa - eta * rho * 1j * u                            # (11a)
        d  = np.sqrt(xi ** 2 + eta ** 2 * (u ** 2 + 1j * u))      # (11b)

        # ── Eq. (15) ─────────────────────────────────────────────────────
        hdt      = d * t / 2.0
        sinh_hdt = np.sinh(hdt)
        cosh_hdt = np.cosh(hdt)
        e_ndt    = np.exp(-d * t)

        A1 = (u ** 2 + 1j * u) * sinh_hdt                         # (15b)
        A2 = d * cosh_hdt + xi * sinh_hdt                         # (15c)
        A  = A1 / A2                                               # (15a)
        B  = d * np.exp(kappa * t / 2.0) / A2                     # (15d)

        # ── Eq. (17b) — rearranged log B, resolves branch discontinuities ─
        D = (np.log(d)
             + (kappa - d) * t / 2.0
             - np.log((d + xi) / 2.0 + (d - xi) / 2.0 * e_ndt))

        # ── Eq. (18) ─────────────────────────────────────────────────────
        log_phi = (
            1j * u * (np.log(S0) + (r - q) * t)
            - t * kappa * vbar * rho * 1j * u / eta
            - v0 * A
            + 2.0 * kappa * vbar / eta ** 2 * D
        )
        return np.exp(log_phi), xi, d, A1, A2, A, D, B, sinh_hdt, cosh_hdt, e_ndt

    # ── Gradient of φ ─────────────────────────────────────────────────────

    @staticmethod
    def _h(u, t: float,
           v0: float, kappa: float, vbar: float, eta: float, rho: float,
           xi, d, A, A1, A2, D, B, sinh_hdt, cosh_hdt, e_ndt):
        """
        h(u) such that ∇φ = φ · h(u)   (Eqs. 23–30).

        All intermediate arrays come from a prior _cf() call at the same u,
        so no redundant computation occurs.

        Returns
        -------
        h : complex ndarray  (5, N_GL)
            Row order: [v0, kappa, vbar, eta, rho]  (matches PARAM_NAMES)
        """
        # ── ∂/∂ρ  (primary building block, Eqs. 27) ──────────────────────

        dd_dr  = -xi * eta * 1j * u / d                                    # (27a)

        dA1_dr = (                                                          # (27d)
            -1j * u * (u ** 2 + 1j * u) * t * xi * eta / (2.0 * d)
            * cosh_hdt
        )
        dA2_dr = (                                                          # (27b)
            -eta * 1j * u * (2.0 + t * xi) / (2.0 * d)
            * (xi * cosh_hdt + d * sinh_hdt)
        )
        dA_dr  = dA1_dr / A2 - A / A2 * dA2_dr                            # (27e)
        dB_dr  = (                                                          # (27c)
            np.exp(kappa * t / 2.0) * (dd_dr / A2 - d / A2 ** 2 * dA2_dr)
        )

        # ── ∂/∂κ  (Eqs. 28 — reuse ρ-derivatives) ────────────────────────
        # ∂A/∂κ = i/(ηu) · ∂A/∂ρ                                           (28a)
        # ∂B/∂κ = i/(ηu) · ∂B/∂ρ + tB/2                                   (28b)
        dB_dk  = 1j / (eta * u) * dB_dr + t * B / 2.0

        # ── ∂/∂σ = ∂/∂eta  (Eqs. 30 — reuse ρ-derivatives) ─────────────

        dd_ds  = (rho / eta - 1.0 / xi) * dd_dr + eta * u ** 2 / d        # (30a)
        dA1_ds = (u ** 2 + 1j * u) * t / 2.0 * dd_ds * cosh_hdt          # (30b)
        dA2_ds = (                                                          # (30c)
            rho / eta * dA2_dr
            - (2.0 + t * xi) / (1j * u * t * xi) * dA1_dr
            + eta * t * A1 / 2.0
        )
        dA_ds  = dA1_ds / A2 - A / A2 * dA2_ds                            # (30d)

        # ── h components  (Eq. 23)  in user order [v0, κ, v̄, η, ρ] ──────

        h_v0 = -A                                                           # (23a)

        # Note: v0/(η·iu)·∂A/∂ρ = −v0·∂A/∂κ  (via Eq. 28a)
        h_kappa = (                                                         # (23d)
            v0 / (eta * 1j * u) * dA_dr
            + 2.0 * vbar / eta ** 2 * D
            + 2.0 * kappa * vbar / (eta ** 2 * B) * dB_dk
            - t * vbar * rho * 1j * u / eta
        )

        h_vbar = (                                                          # (23b)
            2.0 * kappa / eta ** 2 * D
            - t * kappa * rho * 1j * u / eta
        )

        h_eta = (                                                           # (23e)
            -v0 * dA_ds
            - 4.0 * kappa * vbar / eta ** 3 * D
            + 2.0 * kappa * vbar / (eta ** 2 * d) * (dd_ds - d / A2 * dA2_ds)
            + t * kappa * vbar * rho * 1j * u / eta ** 2
        )

        h_rho = (                                                           # (23c)
            -v0 * dA_dr
            + 2.0 * kappa * vbar / (eta ** 2 * d) * (dd_dr - d / A2 * dA2_dr)
            - t * kappa * vbar * 1j * u / eta
        )

        return np.array([h_v0, h_kappa, h_vbar, h_eta, h_rho])  # (5, N_GL)

    # ── Price and gradient  (Algorithm 3.1 — vectorised) ─────────────────

    @classmethod
    def price_and_grad(
        cls,
        spot: float,
        strike: float,
        texp: float,
        r: float,
        q: float,
        params: HestonParams,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the call price and its analytical gradient in a single GL pass.

        Both the price integrals (P₁ at u−i, P₂ at u) and all 5 gradient
        integrals share the same quadrature nodes and most intermediate
        quantities.  This is Algorithm 3.1 of the paper.

        Parameters
        ----------
        spot, strike : float
        texp         : float   time to expiry (years)
        r, q         : float   risk-free rate and continuous dividend yield
        params       : HestonParams

        Returns
        -------
        call : float
        grad : ndarray (5,)   ∂call/∂[v0, kappa, vbar, eta, rho]
        """
        v0, kappa, vbar, eta, rho = params.to_array()
        log_K = np.log(strike)
        disc  = np.exp(-r * texp)   # e^{−rT}
        fwd   = np.exp(-q * texp)   # e^{−qT}

        # Common kernel K^{−iu}/(iu) — same for both P₁ and P₂ integrands
        kern = np.exp(-1j * U_NODES * log_K) / (1j * U_NODES)   # (N_GL,)

        # Characteristic function at u and u−i over all 64 GL nodes
        phi_u,  *I_u  = cls._cf(U_NODES,      spot, r, q, texp, v0, kappa, vbar, eta, rho)
        phi_mi, *I_mi = cls._cf(U_NODES - 1j, spot, r, q, texp, v0, kappa, vbar, eta, rho)

        # ── Call price  (Eq. 9) ───────────────────────────────────────────
        P1   = 0.5 + np.dot(U_WEIGHTS, np.real(kern * phi_mi)) / np.pi
        P2   = 0.5 + np.dot(U_WEIGHTS, np.real(kern * phi_u )) / np.pi
        call = spot * fwd * P1 - disc * strike * P2

        # ── Gradient  (Eq. 22) ───────────────────────────────────────────
        xi_u,  d_u,  A1_u,  A2_u,  A_u,  D_u,  B_u,  sh_u,  ch_u,  en_u  = I_u
        xi_mi, d_mi, A1_mi, A2_mi, A_mi, D_mi, B_mi, sh_mi, ch_mi, en_mi  = I_mi

        h_u  = cls._h(U_NODES,      texp, v0, kappa, vbar, eta, rho,
                       xi_u,  d_u,  A_u,  A1_u,  A2_u,  D_u,  B_u,  sh_u,  ch_u,  en_u)
        h_mi = cls._h(U_NODES - 1j, texp, v0, kappa, vbar, eta, rho,
                       xi_mi, d_mi, A_mi, A1_mi, A2_mi, D_mi, B_mi, sh_mi, ch_mi, en_mi)

        # ∇C = (e^{−rT}/π) [∫ Re(kern·φ(u−i)·h_j(u−i)) du
        #                   − K ∫ Re(kern·φ(u)·h_j(u)) du ]
        grad = disc / np.pi * np.array([
            np.dot(U_WEIGHTS, np.real(kern * phi_mi * h_mi[j]))
            - strike * np.dot(U_WEIGHTS, np.real(kern * phi_u * h_u[j]))
            for j in range(5)
        ])
        return call, grad

    @classmethod
    def price(
        cls,
        spot: float,
        strike: float,
        texp: float,
        r: float,
        q: float,
        params: HestonParams,
    ) -> Tuple[float, float]:
        """
        Return (call_price, put_price).
        Put is computed via put-call parity: P = C − Se^{−qT} + Ke^{−rT}.
        """
        call, _ = cls.price_and_grad(spot, strike, texp, r, q, params)
        put = call - spot * np.exp(-q * texp) + strike * np.exp(-r * texp)
        return call, put

    @classmethod
    def validate_gradient(
        cls,
        spot: float,
        strike: float,
        texp: float,
        r: float,
        q: float,
        params: HestonParams,
        eps: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Compare the analytical gradient against central finite differences.

        Returns a DataFrame with columns [analytical, numerical, rel_error].
        Relative errors < 1e-4 confirm a correct implementation.
        Typical values are < 1e-6 for well-behaved parameter sets.

        Parameters
        ----------
        eps : float   finite-difference step size (default 1e-5)
        """
        x = params.to_array()
        _, g_analytic = cls.price_and_grad(spot, strike, texp, r, q, params)

        g_numeric = np.zeros(5)
        for i in range(5):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            cp, _ = cls.price_and_grad(spot, strike, texp, r, q,
                                        HestonParams.from_array(xp))
            cm, _ = cls.price_and_grad(spot, strike, texp, r, q,
                                        HestonParams.from_array(xm))
            g_numeric[i] = (cp - cm) / (2.0 * eps)

        rel_err = np.abs(g_analytic - g_numeric) / (np.abs(g_numeric) + 1e-15)
        return pd.DataFrame(
            {"analytical": g_analytic, "numerical": g_numeric, "rel_error": rel_err},
            index=PARAM_NAMES,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VOL SURFACE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class VolSurfaceBuilder:
    """
    Converts the mid_df output of prepareData() into a list of OptionContracts
    suitable for HestonCalibrator.

    All methods are static — no instantiation required.
    """

    #: Default moneyness filter.  Excludes very deep OTM / illiquid strikes.
    DEFAULT_MON_MIN: float = 70.0
    DEFAULT_MON_MAX: float = 150.0

    @classmethod
    def build_surface(
        cls,
        mid_df: pd.DataFrame,
        date,
        spot: float,
        r: float,
        q: float,
        mon_min: float = DEFAULT_MON_MIN,
        mon_max: float = DEFAULT_MON_MAX,
        maturities: Optional[List[str]] = None,
    ) -> List[OptionContract]:
        """
        Build a calibration surface from one row of mid_df.

        The method:
          1. Selects the row for `date` (nearest available if exact not found).
          2. Parses each column to extract maturity and moneyness.
          3. Converts K = (moneyness/100) × spot.
          4. Converts implied vol (%) → call price via Black-Scholes.
          5. Returns a filtered list of OptionContracts.

        Parameters
        ----------
        mid_df     : pd.DataFrame   output of prepareData()['mid_df']
        date       : date-like      e.g. '2024-01-15' or pd.Timestamp
        spot       : float          underlying spot price on that date
        r          : float          risk-free rate (decimal, e.g. 0.05)
        q          : float          continuous dividend yield (decimal, e.g. 0.013)
        mon_min    : float          minimum moneyness to include (default 70.0)
        mon_max    : float          maximum moneyness to include (default 150.0)
        maturities : list[str]|None restrict to specific maturity codes
                                    e.g. ['1M','2M','3M','6M','1Y']
                                    None means all maturities

        Returns
        -------
        List[OptionContract]

        Raises
        ------
        ValueError  if no valid options remain after filtering
        """
        # ── Locate row ────────────────────────────────────────────────────
        ts = pd.Timestamp(date)
        if ts not in mid_df.index:
            idx = mid_df.index.get_indexer([ts], method="nearest")[0]
            ts  = mid_df.index[idx]
            warnings.warn(
                f"Date {date} not found in mid_df; using nearest: {ts.date()}",
                stacklevel=2,
            )
        row = mid_df.loc[ts]

        # ── Filter maturities ─────────────────────────────────────────────
        allowed_mats = set(maturities) if maturities else set(MATURITY_MAP.keys())

        # ── Build contracts ───────────────────────────────────────────────
        contracts: List[OptionContract] = []

        for col_name, iv_pct in row.items():
            # Skip missing or non-positive implied vols
            if pd.isna(iv_pct) or iv_pct <= 0:
                continue

            try:
                mat_code, moneyness = _parse_column(str(col_name))
            except ValueError:
                continue

            if mat_code not in allowed_mats:
                continue
            if not (mon_min <= moneyness <= mon_max):
                continue

            texp   = MATURITY_MAP[mat_code]
            strike = (moneyness / 100.0) * spot
            iv     = iv_pct / 100.0                    # % → decimal

            call_price = _bs_call(spot, strike, texp, r, q, iv)
            if call_price <= 0:
                continue

            contracts.append(OptionContract(
                strike=strike,
                texp=texp,
                price=call_price,
                iv=iv,
            ))

        if not contracts:
            raise ValueError(
                f"No valid options extracted for date={ts.date()}, "
                f"spot={spot}, moneyness range=[{mon_min}, {mon_max}], "
                f"maturities={list(allowed_mats)}."
            )

        return contracts

    @staticmethod
    def surface_to_dataframe(
        surface: List[OptionContract],
        spot: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convert a list of OptionContracts to a tidy DataFrame for inspection.

        Parameters
        ----------
        surface : List[OptionContract]
        spot    : float | None   if provided, adds a 'moneyness' column

        Returns
        -------
        pd.DataFrame  with columns [strike, texp, price, iv_pct, weight]
        """
        rows = []
        for c in surface:
            row = {
                "strike":  c.strike,
                "texp":    c.texp,
                "price":   c.price,
                "iv_pct":  round(c.iv * 100, 4),
                "weight":  c.weight,
            }
            if spot is not None:
                row["moneyness_pct"] = round(c.strike / spot * 100, 2)
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["texp", "strike"]).reset_index(drop=True)

    @staticmethod
    def available_dates(mid_df: pd.DataFrame) -> pd.DatetimeIndex:
        """Return all trading dates available in mid_df."""
        return mid_df.index

    @staticmethod
    def latest_date(mid_df: pd.DataFrame) -> pd.Timestamp:
        """Return the most recent trading date in mid_df."""
        return mid_df.index[-1]


# ─────────────────────────────────────────────────────────────────────────────
# HESTON CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────

class HestonCalibrator:
    """
    Calibrates Heston parameters to a vol surface using the
    Levenberg-Marquardt algorithm with the analytical Jacobian from Cui et al.

    The calibration minimises:
        f(θ) = ½ Σᵢ wᵢ [C_model(θ; Kᵢ, Tᵢ) − C_market_i]²

    where wᵢ are the OptionContract weights (default 1.0, uniform).

    With a full SPY vol surface (165 options: 11 maturities × 15 strikes)
    the system is highly overdetermined (165 equations, 5 unknowns),
    giving a well-identified calibration consistent with the paper's results.

    Parameters
    ----------
    ftol, xtol, gtol : float   LM convergence tolerances (default 1e-10)
    max_nfev         : int     maximum LM function evaluations (default 2000)
    """

    # Feasible parameter bounds  [v0, kappa, vbar, eta, rho]
    BOUNDS_LO: np.ndarray = np.array([1e-6, 1e-4, 1e-6, 1e-4, -0.999])
    BOUNDS_HI: np.ndarray = np.array([2.0,  20.0, 2.0,  5.0,   0.999])

    def __init__(
        self,
        ftol:     float = 1e-10,
        xtol:     float = 1e-10,
        gtol:     float = 1e-10,
        max_nfev: int   = 2000,
    ):
        self.ftol     = ftol
        self.xtol     = xtol
        self.gtol     = gtol
        self.max_nfev = max_nfev

    def calibrate(
        self,
        surface: List[OptionContract],
        spot:    float,
        r:       float,
        q:       float,
        x0:      Optional[HestonParams] = None,
    ) -> CalibrationResult:
        """
        Run the Levenberg-Marquardt calibration.

        Parameters
        ----------
        surface : List[OptionContract]   from VolSurfaceBuilder.build_surface()
        spot    : float                  underlying spot price
        r       : float                  risk-free rate (decimal)
        q       : float                  dividend yield (decimal)
        x0      : HestonParams | None    initial guess; auto-estimated if None

        Returns
        -------
        CalibrationResult
        """
        if not surface:
            raise ValueError("surface is empty — nothing to calibrate to.")

        x0_params = x0 if x0 is not None else self._auto_initial_guess(surface)
        x0_arr    = np.clip(x0_params.to_array(), self.BOUNDS_LO, self.BOUNDS_HI)

        res = least_squares(
            self._residuals,
            x0_arr,
            jac     = self._jacobian,
            method  = "lm",        # Levenberg-Marquardt (matches paper's LEVMAR)
            ftol    = self.ftol,
            xtol    = self.xtol,
            gtol    = self.gtol,
            max_nfev= self.max_nfev,
            args    = (surface, spot, r, q),
        )

        # Clip final parameters to feasible region
        best   = np.clip(res.x, self.BOUNDS_LO, self.BOUNDS_HI)
        params = HestonParams.from_array(best)

        # Compute final residuals and MSE
        resid  = self._residuals(best, surface, spot, r, q)
        mse    = float(np.mean(resid ** 2))
        r_norm = float(np.linalg.norm(resid))

        return CalibrationResult(
            params        = params,
            residual_norm = r_norm,
            mse           = mse,
            n_options     = len(surface),
            n_iterations  = res.nfev,
            success       = res.success,
            message       = res.message,
        )

    # ── Internal: residuals and Jacobian ─────────────────────────────────

    @staticmethod
    def _residuals(
        x: np.ndarray,
        surface: List[OptionContract],
        spot: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """
        Residual vector r(θ) for LM.
        rᵢ(θ) = wᵢ · [C_model(θ; Kᵢ, Tᵢ) − C_market_i]
        """
        params = HestonParams.from_array(x)
        out    = np.empty(len(surface))
        for i, contract in enumerate(surface):
            call, _ = HestonPricer.price_and_grad(
                spot, contract.strike, contract.texp, r, q, params
            )
            out[i] = (call - contract.price) * contract.weight
        return out

    @staticmethod
    def _jacobian(
        x: np.ndarray,
        surface: List[OptionContract],
        spot: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """
        Analytical Jacobian J = ∂r/∂θ, shape (n_options, 5).
        Jᵢⱼ = wᵢ · ∂C(θ; Kᵢ, Tᵢ)/∂θⱼ
        """
        params = HestonParams.from_array(x)
        J      = np.empty((len(surface), 5))
        for i, contract in enumerate(surface):
            _, grad = HestonPricer.price_and_grad(
                spot, contract.strike, contract.texp, r, q, params
            )
            J[i] = grad * contract.weight
        return J

    @staticmethod
    def _auto_initial_guess(surface: List[OptionContract]) -> HestonParams:
        """
        Estimate an initial guess from the surface.

        v0 and vbar are set to the squared ATM implied vol of the shortest
        maturity available.  Other parameters use common equity defaults.
        """
        # Find the option closest to ATM in the shortest maturity bucket
        min_texp = min(c.texp for c in surface)
        atm_candidates = [c for c in surface if c.texp == min_texp]
        # Pick closest to ATM by choosing median moneyness (iv ~ 0.15-0.25 for SPY)
        atm = sorted(atm_candidates, key=lambda c: abs(c.iv - 0.20))[0]

        v_init = max(atm.iv ** 2, 1e-4)
        return HestonParams(
            v0    = v_init,
            kappa = 2.0,
            vbar  = v_init,
            eta   = 0.5,
            rho   = -0.5,
        )