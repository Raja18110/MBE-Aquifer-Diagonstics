"""
Analytical Aquifer Influx Models for Material Balance Equation (MBE) Diagnostics.

This module implements analytical water influx models in standard oilfield units:
- Cumulative Water Influx (We): reservoir barrels (rb)
- Aquifer Water in Place (Wi_aquifer): reservoir barrels (rb)
- Pressure (P, Pi): psi
- Time (t): days
- Geometry (ra, re, h): ft
- Dimensionless Radius (rD = ra / re): dimensionless
- Porosity (phi): fraction (0.0 to 1.0)
- Permeability (k): md
- Water Viscosity (mu_w): cp
- Compressibilities (cw, cf, ct): psi^-1
- Schilthuis Aquifer Constant (C): rb / (day * psi)
- Unsteady-State Aquifer Constant (B): rb / psi
- Encroachment Angle (theta): degrees (0 to 360)

Models implemented:
1. Pot Aquifer Model (small, confined aquifers)
2. Schilthuis Steady-State Model (trapezoidal integration over time)
3. Carter-Tracy Unsteady-State Model (non-iterative radial infinite-acting)
4. Van Everdingen-Hurst (VEH) Unsteady-State Model (Stehfest Laplace inversion + Duhamel superposition)
"""

from fractions import Fraction
from functools import lru_cache
import math
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import special

from engine.pvt import _format_return, _vectorize_input


# ============================================================================
# PHYSICAL CONSTANTS & CORRELATION COEFFICIENTS
# ============================================================================
# 1 bbl = 5.614583 cu ft (matches engine/pvt.py)
CUFT_PER_BBL: float = 5.614583

# Radial diffusivity dimensionless time constant for t in days:
# 0.0002637 * 24 hr/day = 0.0063288 -> standard oilfield constant 0.006328
DIFFUSIVITY_CONVERSION_CONSTANT: float = 0.006328

# Radial aquifer constant multiplier: 2 * pi / 5.614583 ~ 1.119083 rb/(cu ft * psi/psi)
AQUIFER_CONSTANT_FACTOR: float = 2.0 * np.pi / CUFT_PER_BBL

# Edwardson et al. (1962) radial infinite-acting pD rational approximation coefficients
_EDWARDSON_NUM_COEFFS: tuple[float, float, float] = (370.529, 137.582, 5.69549)
_EDWARDSON_DEN_COEFFS: tuple[float, float, float] = (328.834, 265.488, 45.2157)
_LOG_APPROX_CONSTANT: float = 0.80907
# Exact intersection where Edwardson rational pD equals 0.5 * (ln(tD) + 0.80907),
# guaranteeing strict continuity and monotonicity across the transition.
_EDWARDSON_LOG_CROSSOVER: float = 615.8715490849621


def _validate_positive(value: float, name: str) -> float:
    """Validate that a scalar parameter is finite and strictly positive."""
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")
    return val


def _validate_non_negative(value: float, name: str) -> float:
    """Validate that a scalar parameter is finite and non-negative."""
    val = float(value)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return val


def _validate_porosity(phi: float) -> float:
    """Validate aquifer porosity fraction in (0.0, 1.0]."""
    val = float(phi)
    if not np.isfinite(val) or val <= 0.0 or val > 1.0:
        raise ValueError(f"Porosity phi must be in (0.0, 1.0], got {phi}.")
    return val


def _validate_encroachment_angle(theta: float) -> float:
    """Validate aquifer encroachment angle theta in [0.0, 360.0] degrees."""
    val = float(theta)
    if not np.isfinite(val) or val < 0.0 or val > 360.0:
        raise ValueError(f"Encroachment angle theta must be in [0.0, 360.0] degrees, got {theta}.")
    return val


def _prepare_time_pressure_arrays(
    time_days: Union[list[float], np.ndarray, pd.Series],
    pressures: Union[list[float], np.ndarray, pd.Series],
    pi: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Validate and convert time (days) and pressure (psi) histories to 1D float64 arrays.

    Returns ``(time_arr, pressure_arr, pi_val)``.
    """
    t_arr = np.asarray(time_days, dtype=np.float64)
    p_arr = np.asarray(pressures, dtype=np.float64)

    if t_arr.ndim != 1 or p_arr.ndim != 1:
        raise ValueError("time_days and pressures must be 1D arrays.")
    if t_arr.size == 0:
        raise ValueError("time_days and pressures must not be empty.")
    if t_arr.size != p_arr.size:
        raise ValueError(
            f"time_days (length {t_arr.size}) and pressures (length {p_arr.size}) must have equal length."
        )
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(p_arr)):
        raise ValueError("time_days and pressures must contain only finite values.")
    if np.any(t_arr < 0.0):
        raise ValueError("time_days must be non-negative (>= 0 days).")
    if np.any(np.diff(t_arr) < 0.0):
        raise ValueError("time_days must be monotonically non-decreasing.")

    pi_val = float(p_arr[0]) if pi is None else float(pi)
    if not np.isfinite(pi_val):
        raise ValueError(f"Initial pressure Pi must be finite, got {pi}.")

    return t_arr, p_arr, pi_val


def _resolve_dimensionless_time(
    t_arr: np.ndarray,
    k: Optional[float],
    phi: Optional[float],
    mu_w: Optional[float],
    ct: Optional[float],
    re: Optional[float],
    t_d: Optional[Union[list[float], np.ndarray, pd.Series]],
) -> np.ndarray:
    """Validate precomputed tD or compute tD from time_days and diffusivity properties."""
    if t_d is not None:
        td_arr = np.asarray(t_d, dtype=np.float64)
        if td_arr.ndim != 1 or td_arr.size != t_arr.size:
            raise ValueError("t_d must be a 1D array with the same length as time_days and pressures.")
        if not np.all(np.isfinite(td_arr)) or np.any(td_arr < 0.0):
            raise ValueError("t_d must contain finite, non-negative values.")
        if np.any(np.diff(td_arr) < 0.0):
            raise ValueError("t_d must be monotonically non-decreasing.")
        return td_arr

    missing = [
        name
        for name, val in (("k", k), ("phi", phi), ("mu_w", mu_w), ("ct", ct), ("re", re))
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Provide 't_d' or all diffusivity parameters; missing: {missing}."
        )
    return np.asarray(
        dimensionless_time(t_arr, k, phi, mu_w, ct, re),  # type: ignore[arg-type]
        dtype=np.float64,
    )


def _edwardson_eval(td_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Edwardson's radial infinite-acting rational approximation for pD and p'D.

    Equation:
        pD(s) = (a1*s + a2*s^2 + a3*s^3) / (b0 + b1*s + b2*s^2 + s^3),  s = sqrt(tD)
    with p'D(tD) computed via the exact analytical quotient-rule derivative.
    """
    a1, a2, a3 = _EDWARDSON_NUM_COEFFS
    b0, b1, b2 = _EDWARDSON_DEN_COEFFS

    s = np.sqrt(td_pos)
    t_s = td_pos * s

    num = a1 * s + a2 * td_pos + a3 * t_s
    den = b0 + b1 * s + b2 * td_pos + t_s

    dnum_ds = a1 + 2.0 * a2 * s + 3.0 * a3 * td_pos
    dden_ds = b1 + 2.0 * b2 * s + 3.0 * td_pos

    pd_edw = num / den
    dpd_edw = ((dnum_ds * den - num * dden_ds) / (den * den)) / (2.0 * s)
    return pd_edw, dpd_edw


# ============================================================================
# 1. POT AQUIFER MODEL
# ============================================================================
def pot_aquifer_initial_water(
    ra: float,
    re: float,
    h: float,
    phi: float,
) -> float:
    """
    Compute initial water in place in a radial Pot Aquifer (Wi_aquifer) in rb.

    Computes the full 360-degree annular aquifer pore volume:
        Wi_aquifer = pi * (ra^2 - re^2) * h * phi / 5.614583

    Parameters
    ----------
    ra : float
        Outer radius of the aquifer in ft (must satisfy ra >= re).
    re : float
        Outer radius of the reservoir (inner aquifer boundary) in ft (must be > 0).
    h : float
        Net thickness of the aquifer in ft (must be > 0).
    phi : float
        Aquifer porosity as a fraction (0.0 < phi <= 1.0).

    Returns
    -------
    float
        Initial aquifer water volume (Wi_aquifer) in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If geometry or porosity parameters are physically invalid.
    """
    re_val = _validate_positive(re, "Reservoir radius re")
    ra_val = float(ra)
    if not np.isfinite(ra_val) or ra_val < re_val:
        raise ValueError(f"Aquifer radius ra ({ra}) must be >= reservoir radius re ({re}).")
    h_val = _validate_positive(h, "Aquifer thickness h")
    phi_val = _validate_porosity(phi)

    pore_vol_cuft = np.pi * (ra_val**2 - re_val**2) * h_val * phi_val
    return float(pore_vol_cuft / CUFT_PER_BBL)


def pot_aquifer_influx(
    p: Union[float, list[float], np.ndarray, pd.Series],
    pi: float,
    cw: float,
    cf: float,
    wi_aquifer: Optional[float] = None,
    theta: float = 360.0,
    ra: Optional[float] = None,
    re: Optional[float] = None,
    h: Optional[float] = None,
    phi: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate cumulative water influx (We) using the Pot Aquifer model in rb.

    Equation:
        We = (cw + cf) * Wi_aquifer * (Pi - P) * (theta / 360)

    Parameters
    ----------
    p : Union[float, list[float], np.ndarray, pd.Series]
        Current reservoir boundary pressure(s) in psi.
    pi : float
        Initial reservoir / aquifer pressure in psi.
    cw : float
        Aquifer water compressibility in psi^-1 (>= 0).
    cf : float
        Aquifer formation (pore rock) compressibility in psi^-1 (>= 0).
    wi_aquifer : Optional[float], optional
        Initial water in place in the full 360-degree aquifer in reservoir barrels (rb).
        If None, computed via ``pot_aquifer_initial_water(ra, re, h, phi)``.
    theta : float, default 360.0
        Water encroachment angle in degrees (0.0 to 360.0).
    ra : Optional[float], optional
        Aquifer outer radius in ft (used if ``wi_aquifer`` is None).
    re : Optional[float], optional
        Reservoir outer radius in ft (used if ``wi_aquifer`` is None).
    h : Optional[float], optional
        Aquifer thickness in ft (used if ``wi_aquifer`` is None).
    phi : Optional[float], optional
        Aquifer porosity fraction (used if ``wi_aquifer`` is None).

    Returns
    -------
    Union[float, np.ndarray]
        Cumulative water influx We in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If compressibilities, angle, or aquifer volume/geometry inputs are invalid.
    """
    cw_val = _validate_non_negative(cw, "Water compressibility cw")
    cf_val = _validate_non_negative(cf, "Formation compressibility cf")
    theta_val = _validate_encroachment_angle(theta)
    pi_val = float(pi)
    if not np.isfinite(pi_val):
        raise ValueError(f"Initial pressure Pi must be finite, got {pi}.")

    if wi_aquifer is None:
        if any(v is None for v in (ra, re, h, phi)):
            raise ValueError(
                "Either 'wi_aquifer' or all geometry parameters ('ra', 're', 'h', 'phi') must be provided."
            )
        wi_val = pot_aquifer_initial_water(ra, re, h, phi)  # type: ignore[arg-type]
    else:
        wi_val = _validate_non_negative(wi_aquifer, "Initial aquifer water volume Wi_aquifer")

    p_arr, is_scalar = _vectorize_input(p)
    we = (cw_val + cf_val) * wi_val * (pi_val - p_arr) * (theta_val / 360.0)
    return _format_return(we, is_scalar)


# ============================================================================
# 2. SCHILTHUIS STEADY-STATE MODEL
# ============================================================================
def schilthuis_aquifer_influx(
    time_days: Union[list[float], np.ndarray, pd.Series],
    pressures: Union[list[float], np.ndarray, pd.Series],
    c: float,
    pi: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate cumulative water influx (We) using the Schilthuis Steady-State model in rb.

    Equation:
        We(t) = C * integral_0^t (Pi - P(tau)) d(tau)

    Evaluated across discrete time steps using cumulative trapezoidal numerical
    integration over time (days).

    Parameters
    ----------
    time_days : Union[list[float], np.ndarray, pd.Series]
        Elapsed production/depletion times in days (non-negative, non-decreasing).
    pressures : Union[list[float], np.ndarray, pd.Series]
        Reservoir boundary pressures corresponding to ``time_days`` in psi.
    c : float
        Schilthuis water influx constant in rb/(day * psi) (>= 0).
    pi : Optional[float], optional
        Initial reservoir pressure in psi. Defaults to ``pressures[0]``.

    Returns
    -------
    np.ndarray
        Cumulative water influx We at each time step in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If time/pressure arrays are mismatched, non-monotonic, or C < 0.
    """
    c_val = _validate_non_negative(c, "Schilthuis constant C")
    t_arr, p_arr, pi_val = _prepare_time_pressure_arrays(time_days, pressures, pi)

    we = np.zeros_like(t_arr, dtype=np.float64)
    if t_arr.size > 1:
        delta_p = pi_val - p_arr
        dt = np.diff(t_arr)
        avg_dp = 0.5 * (delta_p[:-1] + delta_p[1:])
        we[1:] = c_val * np.cumsum(avg_dp * dt)

    return we


# ============================================================================
# 3. CARTER-TRACY UNSTEADY-STATE MODEL
# ============================================================================
def radial_aquifer_constant(
    phi: float,
    ct: float,
    re: float,
    h: float,
    theta: float = 360.0,
) -> float:
    """
    Compute radial aquifer influx constant B in rb/psi.

    Equation:
        B = (2 * pi / 5.614583) * phi * ct * re^2 * h * (theta / 360)
          ~ 1.119083 * phi * ct * re^2 * h * (theta / 360)

    Parameters
    ----------
    phi : float
        Aquifer porosity fraction (0.0 < phi <= 1.0).
    ct : float
        Total aquifer compressibility (cw + cf) in psi^-1 (>= 0).
    re : float
        Reservoir outer radius (aquifer inner radius) in ft (> 0).
    h : float
        Net aquifer thickness in ft (> 0).
    theta : float, default 360.0
        Water encroachment angle in degrees (0.0 to 360.0).

    Returns
    -------
    float
        Radial aquifer constant B in rb/psi.
    """
    phi_val = _validate_porosity(phi)
    ct_val = _validate_non_negative(ct, "Total compressibility ct")
    re_val = _validate_positive(re, "Reservoir radius re")
    h_val = _validate_positive(h, "Aquifer thickness h")
    theta_val = _validate_encroachment_angle(theta)

    return float(AQUIFER_CONSTANT_FACTOR * phi_val * ct_val * (re_val**2) * h_val * (theta_val / 360.0))


def dimensionless_time(
    time_days: Union[float, list[float], np.ndarray, pd.Series],
    k: float,
    phi: float,
    mu_w: float,
    ct: float,
    re: float,
) -> Union[float, np.ndarray]:
    """
    Compute radial dimensionless time (tD) in oilfield units.

    Equation:
        tD = (0.006328 * k * t) / (phi * mu_w * ct * re^2)

    Parameters
    ----------
    time_days : Union[float, list[float], np.ndarray, pd.Series]
        Elapsed time in days (>= 0).
    k : float
        Aquifer permeability in md (>= 0).
    phi : float
        Aquifer porosity as a fraction (0.0 < phi <= 1.0).
    mu_w : float
        Aquifer water viscosity in cp (> 0).
    ct : float
        Total aquifer compressibility (cw + cf) in psi^-1 (> 0).
    re : float
        Reservoir outer radius in ft (> 0).

    Returns
    -------
    Union[float, np.ndarray]
        Dimensionless time tD (dimensionless).

    Raises
    ------
    ValueError
        If any physical parameter is out of its valid domain or time is negative.
    """
    k_val = _validate_non_negative(k, "Aquifer permeability k")
    phi_val = _validate_porosity(phi)
    mu_w_val = _validate_positive(mu_w, "Water viscosity mu_w")
    ct_val = _validate_positive(ct, "Total compressibility ct")
    re_val = _validate_positive(re, "Reservoir radius re")

    t_arr, is_scalar = _vectorize_input(time_days)
    if np.any(~np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("time_days must be finite and non-negative (>= 0 days).")

    td = (DIFFUSIVITY_CONVERSION_CONSTANT * k_val * t_arr) / (
        phi_val * mu_w_val * ct_val * (re_val**2)
    )
    return _format_return(td, is_scalar)


def dimensionless_pressure(
    t_d: Union[float, list[float], np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Compute radial infinite-acting dimensionless pressure pD(tD).

    Uses Edwardson et al. (1962) rational polynomial approximation across small and
    intermediate dimensionless times (tD <= 615.8715) and the classic logarithmic
    line-source approximation pD = 0.5 * (ln(tD) + 0.80907) for large tD, ensuring
    continuity, strict monotonicity, and < 0.15% deviation from the exact
    van Everdingen-Hurst constant-terminal-rate solution across all tD > 0.

    At tD = 0, returns pD = 0.0.

    Parameters
    ----------
    t_d : Union[float, list[float], np.ndarray, pd.Series]
        Dimensionless time tD (>= 0).

    Returns
    -------
    Union[float, np.ndarray]
        Radial infinite-acting dimensionless pressure pD (dimensionless).

    Raises
    ------
    ValueError
        If any tD value is negative or non-finite.
    """
    td_arr, is_scalar = _vectorize_input(t_d)
    if np.any(~np.isfinite(td_arr)) or np.any(td_arr < 0.0):
        raise ValueError("Dimensionless time tD must be finite and non-negative (>= 0).")

    safe_td = np.where(td_arr > 0.0, td_arr, 1.0)
    pd_edw, _ = _edwardson_eval(safe_td)
    pd_log = 0.5 * (np.log(safe_td) + _LOG_APPROX_CONSTANT)

    pd_vals = np.where(td_arr <= _EDWARDSON_LOG_CROSSOVER, pd_edw, pd_log)
    pd_vals = np.where(td_arr == 0.0, 0.0, pd_vals)
    return _format_return(pd_vals, is_scalar)


def dimensionless_pressure_derivative(
    t_d: Union[float, list[float], np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Compute the time derivative of radial infinite-acting dimensionless pressure p'D(tD).

    Evaluates dpD/dtD consistently with ``dimensionless_pressure``:
    - Analytical quotient-rule derivative of Edwardson's rational approximation for
      0 < tD <= 615.8715.
    - Logarithmic derivative p'D = 1 / (2 * tD) for large tD.
    - Returns +inf at tD = 0.0.

    Parameters
    ----------
    t_d : Union[float, list[float], np.ndarray, pd.Series]
        Dimensionless time tD (>= 0).

    Returns
    -------
    Union[float, np.ndarray]
        Dimensionless pressure derivative p'D(tD) = dpD/dtD (dimensionless).

    Raises
    ------
    ValueError
        If any tD value is negative or non-finite.
    """
    td_arr, is_scalar = _vectorize_input(t_d)
    if np.any(~np.isfinite(td_arr)) or np.any(td_arr < 0.0):
        raise ValueError("Dimensionless time tD must be finite and non-negative (>= 0).")

    safe_td = np.where(td_arr > 0.0, td_arr, 1.0)
    _, dpd_edw = _edwardson_eval(safe_td)
    dpd_log = 0.5 / safe_td

    dpd_vals = np.where(td_arr <= _EDWARDSON_LOG_CROSSOVER, dpd_edw, dpd_log)
    dpd_vals = np.where(td_arr == 0.0, np.inf, dpd_vals)
    return _format_return(dpd_vals, is_scalar)


def carter_tracy_aquifer_influx(
    time_days: Union[list[float], np.ndarray, pd.Series],
    pressures: Union[list[float], np.ndarray, pd.Series],
    b: float,
    pi: Optional[float] = None,
    k: Optional[float] = None,
    phi: Optional[float] = None,
    mu_w: Optional[float] = None,
    ct: Optional[float] = None,
    re: Optional[float] = None,
    t_d: Optional[Union[list[float], np.ndarray, pd.Series]] = None,
) -> np.ndarray:
    """
    Calculate cumulative water influx (We) using the non-iterative Carter-Tracy model in rb.

    Step-by-step recurrence at timestep n:
        We_n = We_{n-1} + [ (B * (Pi - P_n) - We_{n-1} * p'D(tD_n))
                            / (pD(tD_n) - tD_{n-1} * p'D(tD_n)) ] * (tD_n - tD_{n-1})

    Either provide precomputed dimensionless times ``t_d`` directly, or provide
    all five aquifer diffusivity properties (``k``, ``phi``, ``mu_w``, ``ct``, ``re``)
    to compute ``t_d`` from ``time_days``.

    Parameters
    ----------
    time_days : Union[list[float], np.ndarray, pd.Series]
        Elapsed production/depletion times in days (non-negative, non-decreasing).
    pressures : Union[list[float], np.ndarray, pd.Series]
        Reservoir boundary pressures corresponding to ``time_days`` in psi.
    b : float
        Radial aquifer water influx constant B in rb/psi (>= 0).
    pi : Optional[float], optional
        Initial reservoir pressure in psi. Defaults to ``pressures[0]``.
    k : Optional[float], optional
        Aquifer permeability in md (required if ``t_d`` is None).
    phi : Optional[float], optional
        Aquifer porosity fraction (required if ``t_d`` is None).
    mu_w : Optional[float], optional
        Aquifer water viscosity in cp (required if ``t_d`` is None).
    ct : Optional[float], optional
        Total aquifer compressibility (cw + cf) in psi^-1 (required if ``t_d`` is None).
    re : Optional[float], optional
        Reservoir outer radius in ft (required if ``t_d`` is None).
    t_d : Optional[Union[list[float], np.ndarray, pd.Series]], optional
        Precomputed dimensionless time array matching ``time_days``.

    Returns
    -------
    np.ndarray
        Cumulative water influx We at each time step in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If inputs are invalid, mismatched, or diffusivity parameters are missing.
    """
    b_val = _validate_non_negative(b, "Aquifer constant B")
    t_arr, p_arr, pi_val = _prepare_time_pressure_arrays(time_days, pressures, pi)
    td_arr = _resolve_dimensionless_time(t_arr, k, phi, mu_w, ct, re, t_d)

    we = np.zeros_like(td_arr, dtype=np.float64)
    if b_val == 0.0:
        return we

    delta_p = pi_val - p_arr
    pd_vec = np.asarray(dimensionless_pressure(td_arr), dtype=np.float64)
    dpd_vec = np.asarray(dimensionless_pressure_derivative(td_arr), dtype=np.float64)

    if td_arr[0] > 0.0:
        we[0] = (b_val * delta_p[0] / pd_vec[0]) * td_arr[0]

    for n in range(1, td_arr.size):
        dt_d = td_arr[n] - td_arr[n - 1]
        if dt_d == 0.0:
            we[n] = we[n - 1]
            continue

        denom = pd_vec[n] - td_arr[n - 1] * dpd_vec[n]
        numer = b_val * delta_p[n] - we[n - 1] * dpd_vec[n]
        we[n] = we[n - 1] + (numer / denom) * dt_d

    return we


# ============================================================================
# 4. VAN EVERDINGEN-HURST (VEH) UNSTEADY-STATE MODEL
# ============================================================================
_VALID_VEH_GEOMETRIES: tuple[str, ...] = (
    "radial_infinite",
    "radial_finite",
    "linear_infinite",
)


@lru_cache(maxsize=16)
def _cached_stehfest_weights(n: int) -> tuple[float, ...]:
    """Compute and cache Stehfest inversion weights V_i via exact rational arithmetic."""
    half_n = n // 2
    weights: list[float] = []
    for i in range(1, n + 1):
        k_start = (i + 1) // 2
        k_end = min(i, half_n)
        acc = Fraction(0)
        for k in range(k_start, k_end + 1):
            num = (k ** half_n) * math.factorial(2 * k)
            den = (
                math.factorial(half_n - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            acc += Fraction(num, den)
        sign = -1 if (half_n + i) % 2 else 1
        weights.append(float(sign * acc))
    return tuple(weights)


def stehfest_weights(n: int = 12) -> np.ndarray:
    """
    Compute Stehfest numerical Laplace inversion weights V_i.

    Implements Stehfest (1970) Algorithm 368:
        V_i = (-1)^(N/2 + i) * sum_{k=floor((i+1)/2)}^{min(i, N/2)}
              k^(N/2) * (2k)! / ((N/2 - k)! * k! * (k - 1)! * (i - k)! * (2k - i)!)

    Parameters
    ----------
    n : int, default 12
        Number of Stehfest series terms (must be a positive even integer).

    Returns
    -------
    np.ndarray
        1D float64 array of Stehfest weights of length ``n`` (dimensionless).

    Raises
    ------
    ValueError
        If ``n`` is not a positive even integer.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0 or int(n) % 2 != 0:
        raise ValueError(f"Stehfest order n must be a positive even integer, got {n}.")
    return np.array(_cached_stehfest_weights(int(n)), dtype=np.float64)


def stehfest_invert(
    laplace_func: Callable[[Any], Any],
    t: Union[float, list[float], np.ndarray, pd.Series],
    n: int = 12,
) -> Union[float, np.ndarray]:
    """
    Numerically invert a Laplace-domain function F(s) using the Stehfest algorithm.

    Equation:
        f(t) ~ (ln(2) / t) * sum_{i=1}^{n} V_i * F(i * ln(2) / t)

    Parameters
    ----------
    laplace_func : Callable[[Any], Any]
        Laplace-domain function F(s) defined for positive real s > 0.
        Accepts NumPy arrays or scalar floats.
    t : Union[float, list[float], np.ndarray, pd.Series]
        Positive time(s) t > 0 at which to evaluate the inverted time-domain function
        (in consistent units with ``laplace_func``, e.g., days or dimensionless tD).
    n : int, default 12
        Number of Stehfest terms (positive even integer).

    Returns
    -------
    Union[float, np.ndarray]
        Inverted time-domain value(s) f(t).

    Raises
    ------
    TypeError
        If ``laplace_func`` is not callable.
    ValueError
        If ``n`` is invalid or any ``t`` value is non-finite or <= 0.
    """
    if not callable(laplace_func):
        raise TypeError("laplace_func must be callable.")
    weights = stehfest_weights(n)
    t_arr, is_scalar = _vectorize_input(t)
    if t_arr.size == 0:
        return _format_return(np.empty_like(t_arr, dtype=np.float64), is_scalar)
    if np.any(~np.isfinite(t_arr)) or np.any(t_arr <= 0.0):
        raise ValueError("Time t must be finite and strictly positive (> 0) for Stehfest inversion.")

    ln2_over_t = math.log(2.0) / t_arr
    i_idx = np.arange(1, weights.size + 1, dtype=np.float64)
    s_mat = ln2_over_t[..., None] * i_idx

    try:
        raw = laplace_func(s_mat)
        f_vals = np.asarray(raw, dtype=np.float64)
        if f_vals.ndim == 0:
            f_vals = np.broadcast_to(f_vals, s_mat.shape)
        elif f_vals.shape != s_mat.shape:
            raise TypeError("Laplace function output shape does not match input s shape.")
    except (TypeError, ValueError):
        flat_vals = [float(laplace_func(float(s_val))) for s_val in s_mat.ravel()]
        f_vals = np.asarray(flat_vals, dtype=np.float64).reshape(s_mat.shape)

    inverted = ln2_over_t * np.tensordot(f_vals, weights, axes=([-1], [0]))
    return _format_return(inverted, is_scalar)


def _validate_veh_geometry(geometry: str, rD: Optional[float]) -> tuple[str, Optional[float]]:
    """Validate VEH aquifer geometry keyword and finite outer radius ratio rD = ra / re."""
    if not isinstance(geometry, str):
        raise ValueError(f"geometry must be one of {_VALID_VEH_GEOMETRIES}, got {geometry!r}.")
    geom = geometry.strip().lower()
    if geom not in _VALID_VEH_GEOMETRIES:
        raise ValueError(
            f"Unsupported VEH geometry {geometry!r}. Must be one of {_VALID_VEH_GEOMETRIES}."
        )
    if geom == "radial_finite":
        if rD is None:
            raise ValueError("Dimensionless radius rD must be provided when geometry='radial_finite'.")
        rd_val = float(rD)
        if not np.isfinite(rd_val) or rd_val <= 1.0:
            raise ValueError(
                f"Dimensionless radius rD = ra / re must be finite and > 1.0, got {rD}."
            )
        return geom, rd_val
    return geom, None


def veh_dimensionless_influx(
    tD: Union[float, list[float], np.ndarray, pd.Series],
    geometry: str = "radial_infinite",
    rD: Optional[float] = None,
    n: int = 12,
) -> Union[float, np.ndarray]:
    """
    Compute Van Everdingen-Hurst dimensionless water influx WeD(tD).

    Supported aquifer geometries:
    - ``"radial_infinite"``: Laplace-domain solution
      ``WeD_bar(s) = K1(sqrt(s)) / (s^(3/2) * K0(sqrt(s)))``, inverted via Stehfest
      using ``scipy.special.k0e`` and ``scipy.special.k1e`` for numerical stability.
    - ``"radial_finite"``: No-flow outer boundary at dimensionless radius ``rD = ra / re > 1``,
      inverted via Stehfest using exponentially scaled Bessel functions
      (``i0e``, ``i1e``, ``k0e``, ``k1e``). Converges to ``(rD^2 - 1) / 2`` as ``tD -> inf``.
    - ``"linear_infinite"``: Exact analytical solution ``WeD(tD) = (2 / sqrt(pi)) * sqrt(tD)``.

    At ``tD = 0``, returns ``WeD = 0.0`` for all geometries.

    Parameters
    ----------
    tD : Union[float, list[float], np.ndarray, pd.Series]
        Dimensionless time tD (>= 0, dimensionless).
    geometry : str, default "radial_infinite"
        Aquifer geometry: ``"radial_infinite"``, ``"radial_finite"``, or ``"linear_infinite"``.
    rD : Optional[float], optional
        Dimensionless outer aquifer radius ``ra / re`` (> 1.0), required when
        ``geometry="radial_finite"``.
    n : int, default 12
        Stehfest Laplace inversion order (positive even integer).

    Returns
    -------
    Union[float, np.ndarray]
        Dimensionless cumulative water influx WeD (dimensionless).

    Raises
    ------
    ValueError
        If ``tD`` is negative/non-finite, ``geometry`` is unsupported, ``rD`` is invalid
        for ``"radial_finite"``, or ``n`` is invalid.
    """
    geom, rd_val = _validate_veh_geometry(geometry, rD)
    stehfest_weights(n)

    td_arr, is_scalar = _vectorize_input(tD)
    if np.any(~np.isfinite(td_arr)) or np.any(td_arr < 0.0):
        raise ValueError("Dimensionless time tD must be finite and non-negative (>= 0).")

    wed = np.zeros_like(td_arr, dtype=np.float64)
    pos_mask = td_arr > 0.0
    if not np.any(pos_mask):
        return _format_return(wed, is_scalar)

    td_pos = td_arr[pos_mask]
    if geom == "linear_infinite":
        wed[pos_mask] = (2.0 / math.sqrt(math.pi)) * np.sqrt(td_pos)
        return _format_return(wed, is_scalar)

    if geom == "radial_infinite":

        def _laplace_radial_infinite(s: np.ndarray) -> np.ndarray:
            u = np.sqrt(s)
            return special.k1e(u) / (s * (u * special.k0e(u)))

        wed[pos_mask] = np.asarray(
            stehfest_invert(_laplace_radial_infinite, td_pos, n=n),
            dtype=np.float64,
        )
        return _format_return(wed, is_scalar)

    assert rd_val is not None

    def _laplace_radial_finite(s: np.ndarray) -> np.ndarray:
        u = np.sqrt(s)
        alpha = np.exp(-2.0 * (rd_val - 1.0) * u)
        i1_ru = special.i1e(rd_val * u)
        k1_ru = special.k1e(rd_val * u)
        i0_u = special.i0e(u)
        i1_u = special.i1e(u)
        k0_u = special.k0e(u)
        k1_u = special.k1e(u)
        num = i1_ru * k1_u - alpha * k1_ru * i1_u
        den = s * (u * (k0_u * i1_ru + alpha * i0_u * k1_ru))
        return num / den

    wed[pos_mask] = np.asarray(
        stehfest_invert(_laplace_radial_finite, td_pos, n=n),
        dtype=np.float64,
    )
    return _format_return(wed, is_scalar)


def veh_aquifer_influx(
    time_days: Union[list[float], np.ndarray, pd.Series],
    pressures: Union[list[float], np.ndarray, pd.Series],
    b: float,
    pi: Optional[float] = None,
    k: Optional[float] = None,
    phi: Optional[float] = None,
    mu_w: Optional[float] = None,
    ct: Optional[float] = None,
    re: Optional[float] = None,
    t_d: Optional[Union[list[float], np.ndarray, pd.Series]] = None,
    geometry: str = "radial_infinite",
    rD: Optional[float] = None,
    ra: Optional[float] = None,
    n: int = 12,
) -> np.ndarray:
    """
    Calculate cumulative water influx (We) using the Van Everdingen-Hurst (VEH) model in rb.

    Evaluates full Duhamel superposition across arbitrary time (days) and pressure (psi)
    histories using standard midpoint pressure drops:
        dP_0 = (Pi - P_1) / 2               (when P_0 == Pi; Pi - (P_0 + P_1) / 2 generally)
        dP_j = (P_{j-1} - P_{j+1}) / 2      for j >= 1
        We(t_n) = B * sum_{j=0}^{n-1} dP_j * WeD(tD_n - tD_j)

    Either provide precomputed dimensionless times ``t_d`` directly, or provide
    all five aquifer diffusivity properties (``k``, ``phi``, ``mu_w``, ``ct``, ``re``)
    to compute ``t_d`` from ``time_days`` (for ``linear_infinite``, pass characteristic
    aquifer length as ``re`` if deriving ``t_d``).

    Parameters
    ----------
    time_days : Union[list[float], np.ndarray, pd.Series]
        Elapsed production/depletion times in days (non-negative, non-decreasing).
    pressures : Union[list[float], np.ndarray, pd.Series]
        Reservoir boundary pressures corresponding to ``time_days`` in psi.
    b : float
        Aquifer water influx constant B in rb/psi (>= 0).
    pi : Optional[float], optional
        Initial reservoir pressure in psi. Defaults to ``pressures[0]``.
    k : Optional[float], optional
        Aquifer permeability in md (required if ``t_d`` is None).
    phi : Optional[float], optional
        Aquifer porosity fraction (required if ``t_d`` is None).
    mu_w : Optional[float], optional
        Aquifer water viscosity in cp (required if ``t_d`` is None).
    ct : Optional[float], optional
        Total aquifer compressibility (cw + cf) in psi^-1 (required if ``t_d`` is None).
    re : Optional[float], optional
        Reservoir outer radius (or characteristic length for linear flow) in ft
        (required if ``t_d`` is None).
    t_d : Optional[Union[list[float], np.ndarray, pd.Series]], optional
        Precomputed dimensionless time array matching ``time_days`` (dimensionless).
    geometry : str, default "radial_infinite"
        Aquifer geometry: ``"radial_infinite"``, ``"radial_finite"``, or ``"linear_infinite"``.
    rD : Optional[float], optional
        Dimensionless outer radius ``ra / re`` (> 1.0) for ``geometry="radial_finite"``.
    ra : Optional[float], optional
        Aquifer outer radius in ft. Used to compute ``rD = ra / re`` if ``rD`` is None
        and ``re`` is provided for ``geometry="radial_finite"``.
    n : int, default 12
        Stehfest Laplace inversion order (positive even integer).

    Returns
    -------
    np.ndarray
        Cumulative water influx We at each time step in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If inputs are invalid, mismatched, geometry/rD are invalid, or diffusivity
        parameters are missing when ``t_d`` is None.
    """
    if rD is None and ra is not None and re is not None:
        ra_val = _validate_positive(ra, "Aquifer radius ra")
        re_val = _validate_positive(re, "Reservoir radius re")
        rD = ra_val / re_val

    geom, rd_val = _validate_veh_geometry(geometry, rD)
    stehfest_weights(n)
    b_val = _validate_non_negative(b, "Aquifer constant B")
    t_arr, p_arr, pi_val = _prepare_time_pressure_arrays(time_days, pressures, pi)
    td_arr = _resolve_dimensionless_time(t_arr, k, phi, mu_w, ct, re, t_d)

    m_steps = t_arr.size
    we = np.zeros(m_steps, dtype=np.float64)
    if b_val == 0.0 or m_steps <= 1:
        return we

    dp = np.empty(m_steps - 1, dtype=np.float64)
    if p_arr[0] == pi_val:
        dp[0] = 0.5 * (pi_val - p_arr[1])
    else:
        dp[0] = pi_val - 0.5 * (p_arr[0] + p_arr[1])
    if m_steps > 2:
        dp[1:] = 0.5 * (p_arr[:-2] - p_arr[2:])

    # Evaluate lower-triangular superposition in row chunks to bound memory for long histories.
    chunk_size = 512
    for start in range(1, m_steps, chunk_size):
        stop = min(m_steps, start + chunk_size)
        n_vec = np.arange(start, stop)
        j_vec = np.arange(stop - 1)
        mask = j_vec[None, :] < n_vec[:, None]
        dt_d = td_arr[n_vec[:, None]] - td_arr[j_vec[None, :]]

        wed_block = np.zeros(mask.shape, dtype=np.float64)
        wed_block[mask] = veh_dimensionless_influx(
            dt_d[mask], geometry=geom, rD=rd_val, n=n
        )
        we[start:stop] = b_val * (wed_block @ dp[: stop - 1])

    return we
