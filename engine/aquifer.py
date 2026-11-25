"""
Analytical Aquifer Influx Models for Material Balance Equation (MBE) Diagnostics.

This module implements the first three analytical water influx models in standard
oilfield units:
- Cumulative Water Influx (We): reservoir barrels (rb)
- Aquifer Water in Place (Wi_aquifer): reservoir barrels (rb)
- Pressure (P, Pi): psi
- Time (t): days
- Geometry (ra, re, h): ft
- Porosity (phi): fraction (0.0 to 1.0)
- Permeability (k): md
- Water Viscosity (mu_w): cp
- Compressibilities (cw, cf, ct): psi^-1
- Schilthuis Aquifer Constant (C): rb / (day * psi)
- Carter-Tracy Aquifer Constant (B): rb / psi
- Encroachment Angle (theta): degrees (0 to 360)

Models implemented:
1. Pot Aquifer Model (small, confined aquifers)
2. Schilthuis Steady-State Model (trapezoidal integration over time)
3. Carter-Tracy Unsteady-State Model (non-iterative radial infinite-acting)
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

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

    if t_d is not None:
        td_arr = np.asarray(t_d, dtype=np.float64)
        if td_arr.ndim != 1 or td_arr.size != t_arr.size:
            raise ValueError("t_d must be a 1D array with the same length as time_days and pressures.")
        if not np.all(np.isfinite(td_arr)) or np.any(td_arr < 0.0):
            raise ValueError("t_d must contain finite, non-negative values.")
        if np.any(np.diff(td_arr) < 0.0):
            raise ValueError("t_d must be monotonically non-decreasing.")
    else:
        missing = [
            name
            for name, val in (("k", k), ("phi", phi), ("mu_w", mu_w), ("ct", ct), ("re", re))
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Provide 't_d' or all diffusivity parameters; missing: {missing}."
            )
        td_arr = np.asarray(
            dimensionless_time(t_arr, k, phi, mu_w, ct, re),  # type: ignore[arg-type]
            dtype=np.float64,
        )

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
