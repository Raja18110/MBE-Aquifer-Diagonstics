"""
Black-Oil PVT Correlations and Expansion Engine.

This module provides clean, NumPy-vectorized functions for black-oil PVT properties
and Material Balance Equation (MBE) expansion terms in standard oilfield units:
- Pressure (P): psi
- Temperature (T): °F
- Solution Gas-Oil Ratio (Rs, Rsi): SCF/STB
- Oil Formation Volume Factor (Bo, Boi): rb/STB
- Gas Formation Volume Factor (Bg, Bgi): rb/SCF
- Two-Phase Formation Volume Factor (Bt): rb/STB
- Compressibilities (co, cw, cf): psi^-1
- Gas Cap Ratio (m): dimensionless (V_gi / V_oi)
- Connate Water Saturation (Swi): fraction (0.0 to 1.0)
- Expansion Terms (Eo, Eg, Ef,w): rb/STB
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


# ============================================================================
# PHYSICAL CONSTANTS & CONVERSION FACTORS
# ============================================================================
# Standard conditions: Psc = 14.7 psia, Tsc = 60 °F (519.67 °R)
# 1 bbl = 5.614583 cu ft
# Bg = (Psc / Tsc / 5.614583) * (Z * T_R / P) = 0.0050355 * (Z * T_R / P) rb/SCF
BG_CONVERSION_CONSTANT: float = 14.7 / (519.67 * 5.614583)  # ~0.0050355 rb*psi/(SCF*R)


def _vectorize_input(
    val: Union[float, int, list, np.ndarray, pd.Series],
) -> tuple[np.ndarray, bool]:
    """
    Convert scalar or array-like to 1D float64 NumPy array.

    Returns the NumPy array and a boolean indicating whether the input was a scalar.
    """
    if isinstance(val, (int, float, np.floating, np.integer)):
        return np.array([float(val)], dtype=np.float64), True
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        return np.array([float(arr.item())], dtype=np.float64), True
    return arr, False


def _format_return(res: np.ndarray, is_scalar: bool) -> Union[float, np.ndarray]:
    """Return a Python float if original input was scalar, else NumPy ndarray."""
    if is_scalar:
        return float(res[0])
    return res


def oil_specific_gravity(api: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate oil specific gravity (gamma_o, water = 1.0) from API gravity.

    Parameters
    ----------
    api : Union[float, np.ndarray]
        Oil gravity in °API.

    Returns
    -------
    Union[float, np.ndarray]
        Specific gravity of stock-tank oil (dimensionless).
    """
    arr, is_scalar = _vectorize_input(api)
    gamma_o = 141.5 / (131.5 + arr)
    return _format_return(gamma_o, is_scalar)


def standing_bubble_point_pressure(
    rs: Union[float, np.ndarray],
    temp_f: float,
    api: float,
    gas_gravity: float,
) -> Union[float, np.ndarray]:
    """
    Estimate bubble point pressure (Pb) using Standing's empirical correlation.

    Equation:
        Pb = 18.2 * [ (Rs / gamma_g)^0.83 * 10^(0.00091 * T - 0.0125 * API) - 1.4 ]

    Parameters
    ----------
    rs : Union[float, np.ndarray]
        Solution gas-oil ratio at the bubble point in SCF/STB.
    temp_f : float
        Reservoir temperature in °F.
    api : float
        Stock-tank oil gravity in °API.
    gas_gravity : float
        Separator gas specific gravity (air = 1.0).

    Returns
    -------
    Union[float, np.ndarray]
        Bubble point pressure in psi.
    """
    rs_arr, is_scalar = _vectorize_input(rs)

    # Exponent term: 0.00091 * T - 0.0125 * API
    a = 0.00091 * float(temp_f) - 0.0125 * float(api)

    # Zero or negative Rs yields 0 psi bubble point
    pos_mask = rs_arr > 0
    safe_rs = np.where(pos_mask, rs_arr, 0.0)

    term = np.power(safe_rs / float(gas_gravity), 0.83) * np.power(10.0, a) - 1.4
    pb = 18.2 * term
    pb = np.where(pos_mask, np.maximum(0.0, pb), 0.0)

    return _format_return(pb, is_scalar)


def standing_solution_gor(
    p: Union[float, np.ndarray],
    temp_f: float,
    api: float,
    gas_gravity: float,
    pb: Optional[float] = None,
    rsi: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate solution gas-oil ratio (Rs) using Standing's correlation.

    Below Pb, Rs decreases with pressure depletion as gas liberates from solution.
    At or above Pb, Rs is capped at initial / bubble-point GOR (Rsi).

    Equation (p < Pb):
        Rs = gamma_g * [ (P / 18.2 + 1.4) * 10^(0.0125 * API - 0.00091 * T) ]^(1 / 0.83)

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Reservoir pressure in psi.
    temp_f : float
        Reservoir temperature in °F.
    api : float
        Stock-tank oil gravity in °API.
    gas_gravity : float
        Separator gas specific gravity (air = 1.0).
    pb : Optional[float], optional
        Bubble point pressure in psi. If not provided, computed from rsi.
    rsi : Optional[float], optional
        Initial solution gas-oil ratio in SCF/STB. If not provided, computed from pb.

    Returns
    -------
    Union[float, np.ndarray]
        Solution gas-oil ratio in SCF/STB.

    Raises
    ------
    ValueError
        If neither pb nor rsi is provided.
    """
    p_arr, is_scalar = _vectorize_input(p)

    if pb is None and rsi is None:
        raise ValueError("Either 'pb' (bubble point pressure) or 'rsi' (initial GOR) must be provided.")

    if pb is None:
        pb = float(standing_bubble_point_pressure(rsi, temp_f, api, gas_gravity))
    elif rsi is None:
        # Compute rsi at pb
        x_pb = 0.0125 * float(api) - 0.00091 * float(temp_f)
        term_pb = (float(pb) / 18.2 + 1.4) * (10.0**x_pb)
        rsi = float(gas_gravity * (term_pb ** (1.0 / 0.83)))

    x = 0.0125 * float(api) - 0.00091 * float(temp_f)
    pow_factor = 1.0 / 0.83  # ~1.204819

    # Standing formula for saturated conditions
    p_safe = np.maximum(0.0, p_arr)
    bracket = (p_safe / 18.2 + 1.4) * (10.0**x)
    rs_calc = float(gas_gravity) * np.power(bracket, pow_factor)

    # Above or at Pb: capped at rsi
    # Below Pb: rs_calc capped at rsi
    # At or below 0 psi: 0.0
    rs = np.where(p_arr >= pb, float(rsi), np.minimum(float(rsi), rs_calc))
    rs = np.where(p_arr <= 0.0, 0.0, rs)

    return _format_return(rs, is_scalar)


def vasquez_beggs_oil_compressibility(
    p: Union[float, np.ndarray],
    rsb: float,
    temp_f: float,
    api: float,
    gas_gravity: float,
) -> Union[float, np.ndarray]:
    """
    Estimate isothermal oil compressibility (co) for undersaturated oil using Vasquez-Beggs (1980).

    Equation:
        co = (-1433 + 5 * Rsb + 17.2 * T - 1180 * gamma_g + 12.61 * API) / (10^5 * P)

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure in psi (must be > 0).
    rsb : float
        Bubble point solution GOR in SCF/STB.
    temp_f : float
        Reservoir temperature in °F.
    api : float
        Stock-tank oil gravity in °API.
    gas_gravity : float
        Gas specific gravity (air = 1.0).

    Returns
    -------
    Union[float, np.ndarray]
        Isothermal oil compressibility in psi^-1.
    """
    p_arr, is_scalar = _vectorize_input(p)
    numerator = (
        -1433.0
        + 5.0 * float(rsb)
        + 17.2 * float(temp_f)
        - 1180.0 * float(gas_gravity)
        + 12.61 * float(api)
    )
    # Safe division, ensure positive compressibility fallback
    co = np.where(p_arr > 0, numerator / (1e5 * np.maximum(p_arr, 1.0)), 1.2e-5)
    co = np.maximum(co, 1.0e-6)  # Physically realistic lower bound
    return _format_return(co, is_scalar)


def standing_oil_fvf(
    p: Union[float, np.ndarray],
    pb: float,
    rsi: float,
    temp_f: float,
    api: float,
    gas_gravity: float,
    co: Optional[float] = None,
    rs: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate oil formation volume factor (Bo) using Standing's correlation.

    At or below Pb (saturated oil):
        Bo = 0.9759 + 0.000120 * [ Rs * (gamma_g / gamma_o)^0.5 + 1.25 * T ]^1.2
    Above Pb (undersaturated oil):
        Bo = Bob * exp[ -co * (P - Pb) ]

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Reservoir pressure in psi.
    pb : float
        Bubble point pressure in psi.
    rsi : float
        Initial / bubble-point solution gas-oil ratio in SCF/STB.
    temp_f : float
        Reservoir temperature in °F.
    api : float
        Stock-tank oil gravity in °API.
    gas_gravity : float
        Gas specific gravity (air = 1.0).
    co : Optional[float], optional
        Undersaturated oil compressibility in psi^-1.
        If None, estimated at Pb via Vasquez-Beggs correlation.
    rs : Optional[Union[float, np.ndarray]], optional
        Precomputed solution GOR in SCF/STB. If None, computed via standing_solution_gor.

    Returns
    -------
    Union[float, np.ndarray]
        Oil formation volume factor in rb/STB.
    """
    p_arr, is_scalar = _vectorize_input(p)

    gamma_o = 141.5 / (131.5 + float(api))
    density_ratio_sqrt = np.sqrt(float(gas_gravity) / gamma_o)

    # Bob at bubble point pressure
    term_bob = float(rsi) * density_ratio_sqrt + 1.25 * float(temp_f)
    bob = 0.9759 + 0.000120 * np.power(term_bob, 1.2)

    # Compressibility for undersaturated regime
    if co is None:
        co_val = float(vasquez_beggs_oil_compressibility(pb, rsi, temp_f, api, gas_gravity))
    else:
        co_val = float(co)

    # Precomputed or calculated Rs
    if rs is None:
        rs_arr = standing_solution_gor(p_arr, temp_f, api, gas_gravity, pb=pb, rsi=rsi)
        if isinstance(rs_arr, float):
            rs_arr = np.array([rs_arr], dtype=np.float64)
    else:
        rs_arr, _ = _vectorize_input(rs)

    # Saturated Bo (p <= Pb)
    term_bo = rs_arr * density_ratio_sqrt + 1.25 * float(temp_f)
    bo_saturated = 0.9759 + 0.000120 * np.power(term_bo, 1.2)

    # Undersaturated Bo (p > Pb): exponential compressibility adjustment
    bo_undersaturated = bob * np.exp(-co_val * (p_arr - float(pb)))

    # Combine regimes
    bo = np.where(p_arr <= pb, bo_saturated, bo_undersaturated)

    return _format_return(bo, is_scalar)


def sutton_pseudocritical_properties(
    gas_gravity: Union[float, np.ndarray],
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate pseudo-critical pressure (Ppc) and temperature (Tpc) using Sutton (1985).

    Applicable for hydrocarbon natural gas systems with 0.57 < gamma_g < 1.68.

    Equations:
        Tpc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g^2  (°R)
        Ppc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g^2   (psia)

    Parameters
    ----------
    gas_gravity : Union[float, np.ndarray]
        Separator gas specific gravity (air = 1.0).

    Returns
    -------
    tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        (Ppc_psi, Tpc_R): Pseudo-critical pressure in psia and temperature in °R.
    """
    arr, is_scalar = _vectorize_input(gas_gravity)

    tpc_r = 169.2 + 349.5 * arr - 74.0 * np.power(arr, 2)
    ppc_psi = 756.8 - 131.0 * arr - 3.6 * np.power(arr, 2)

    return _format_return(ppc_psi, is_scalar), _format_return(tpc_r, is_scalar)


def gas_z_factor_papay(
    p: Union[float, np.ndarray],
    temp_f: float,
    gas_gravity: float,
) -> Union[float, np.ndarray]:
    """
    Calculate real gas compressibility factor (Z-factor) using Papay's explicit correlation (1968).

    Equation:
        Z = 1 - (3.52 * Ppr) / (10^(0.9813 * Tpr)) + (0.274 * Ppr^2) / (10^(0.8157 * Tpr))

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure in psi.
    temp_f : float
        Temperature in °F.
    gas_gravity : float
        Gas specific gravity (air = 1.0).

    Returns
    -------
    Union[float, np.ndarray]
        Z-factor (dimensionless).
    """
    p_arr, is_scalar = _vectorize_input(p)

    ppc, tpc = sutton_pseudocritical_properties(gas_gravity)
    ppc_val = float(ppc)
    tpc_val = float(tpc)

    t_r = float(temp_f) + 459.67
    tpr = t_r / tpc_val
    ppr = np.maximum(0.0, p_arr) / ppc_val

    term1 = (3.52 * ppr) / np.power(10.0, 0.9813 * tpr)
    term2 = (0.274 * np.power(ppr, 2)) / np.power(10.0, 0.8157 * tpr)
    z = 1.0 - term1 + term2

    # Physical bounds for natural gas Z-factor
    z = np.where(ppr == 0.0, 1.0, np.clip(z, 0.2, 3.5))

    return _format_return(z, is_scalar)


def gas_z_factor_hall_yarborough(
    p: Union[float, np.ndarray],
    temp_f: float,
    gas_gravity: float,
    max_iter: int = 25,
    tol: float = 1e-7,
) -> Union[float, np.ndarray]:
    """
    Calculate real gas compressibility factor (Z-factor) using Hall-Yarborough (1973) EOS.

    Solves for reduced density (y) via Newton-Raphson iteration:
        f(y) = (y + y^2 + y^3 - y^4) / (1 - y)^3 - A * Ppr - B * y^2 + C * y^D = 0
        Z = A * Ppr / y

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure in psi.
    temp_f : float
        Temperature in °F.
    gas_gravity : float
        Gas specific gravity (air = 1.0).
    max_iter : int, default 25
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-7
        Convergence tolerance for reduced density.

    Returns
    -------
    Union[float, np.ndarray]
        Z-factor (dimensionless).
    """
    p_arr, is_scalar = _vectorize_input(p)

    ppc, tpc = sutton_pseudocritical_properties(gas_gravity)
    ppc_val = float(ppc)
    tpc_val = float(tpc)

    t_r = float(temp_f) + 459.67
    tpr = t_r / tpc_val
    ppr = np.maximum(0.0, p_arr) / ppc_val

    t = 1.0 / tpr
    a = 0.06125 * t * np.exp(-1.2 * np.power(1.0 - t, 2))
    b = 14.76 * t - 9.76 * (t**2) + 4.58 * (t**3)
    c = 90.7 * t - 242.2 * (t**2) + 42.4 * (t**3)
    d = 2.18 + 2.82 * t

    # Initial guess for reduced density y
    y = np.clip(0.0125 * ppr * t * np.exp(-1.2 * np.power(1.0 - t, 2)), 1e-12, 0.85)

    zero_mask = ppr <= 1e-5

    for _ in range(max_iter):
        one_minus_y = 1.0 - y
        y_cubed = one_minus_y**3
        f = (y + y**2 + y**3 - y**4) / y_cubed - a * ppr - b * (y**2) + c * np.power(y, d)
        df = (
            (1.0 + 4.0 * y + 4.0 * (y**2) - 4.0 * (y**3) + y**4) / (one_minus_y**4)
            - 2.0 * b * y
            + c * d * np.power(np.maximum(y, 1e-15), d - 1.0)
        )
        dy = f / df
        y = np.clip(y - dy, 1e-12, 0.99)
        if float(np.max(np.abs(dy))) < tol:
            break

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(zero_mask, 1.0, (a * ppr) / y)
        z = np.clip(z, 0.2, 3.5)

    return _format_return(z, is_scalar)


def gas_z_factor(
    p: Union[float, np.ndarray],
    temp_f: float,
    gas_gravity: float,
    method: str = "papay",
) -> Union[float, np.ndarray]:
    """
    Calculate real gas compressibility factor (Z-factor).

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Pressure in psi.
    temp_f : float
        Temperature in °F.
    gas_gravity : float
        Gas specific gravity (air = 1.0).
    method : str, default 'papay'
        Correlation method: 'papay' (fast explicit) or 'hall_yarborough' (high-precision EOS).

    Returns
    -------
    Union[float, np.ndarray]
        Z-factor (dimensionless).
    """
    method_clean = method.lower().strip()
    if method_clean in ["papay", "p"]:
        return gas_z_factor_papay(p, temp_f, gas_gravity)
    elif method_clean in ["hall_yarborough", "hy", "hall-yarborough"]:
        return gas_z_factor_hall_yarborough(p, temp_f, gas_gravity)
    else:
        raise ValueError(f"Unknown Z-factor correlation method '{method}'. Choose 'papay' or 'hall_yarborough'.")


def gas_fvf(
    p: Union[float, np.ndarray],
    temp_f: float,
    z: Optional[Union[float, np.ndarray]] = None,
    gas_gravity: Optional[float] = None,
    z_method: str = "papay",
) -> Union[float, np.ndarray]:
    """
    Calculate gas formation volume factor (Bg) in reservoir barrels per standard cubic foot (rb/SCF).

    Equation:
        Bg = (Psc / Tsc / 5.614583) * (Z * T_R / P)
           = 0.0050355 * Z * (T + 459.67) / P  (rb/SCF)

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Reservoir pressure in psi.
    temp_f : float
        Reservoir temperature in °F.
    z : Optional[Union[float, np.ndarray]], optional
        Precomputed gas Z-factor. If None, gas_gravity must be provided to compute Z.
    gas_gravity : Optional[float], optional
        Separator gas specific gravity (air = 1.0), required if z is None.
    z_method : str, default 'papay'
        Z-factor correlation method ('papay' or 'hall_yarborough').

    Returns
    -------
    Union[float, np.ndarray]
        Gas formation volume factor in rb/SCF.

    Raises
    ------
    ValueError
        If both z and gas_gravity are None, or pressure contains values <= 0.
    """
    p_arr, is_scalar = _vectorize_input(p)

    if (p_arr <= 0).any():
        raise ValueError("Pressure must be strictly positive (P > 0 psi) to compute Bg.")

    t_r = float(temp_f) + 459.67

    if z is None:
        if gas_gravity is None:
            raise ValueError("Either 'z' or 'gas_gravity' must be provided to calculate Bg.")
        z_arr = gas_z_factor(p_arr, temp_f, gas_gravity, method=z_method)
        if isinstance(z_arr, float):
            z_arr = np.array([z_arr], dtype=np.float64)
    else:
        z_arr, _ = _vectorize_input(z)

    bg = BG_CONVERSION_CONSTANT * (z_arr * t_r) / p_arr
    return _format_return(bg, is_scalar)


def two_phase_fvf(
    bo: Union[float, np.ndarray],
    bg: Union[float, np.ndarray],
    rs: Union[float, np.ndarray],
    rsi: float,
) -> Union[float, np.ndarray]:
    """
    Calculate two-phase (total) formation volume factor (Bt) in rb/STB.

    Equation:
        Bt = Bo + (Rsi - Rs) * Bg

    Above bubble point (p >= Pb), Rs = Rsi, so Bt = Bo.
    Below bubble point (p < Pb), liberated free gas expands the two-phase system.

    Parameters
    ----------
    bo : Union[float, np.ndarray]
        Oil formation volume factor in rb/STB.
    bg : Union[float, np.ndarray]
        Gas formation volume factor in rb/SCF.
    rs : Union[float, np.ndarray]
        Solution gas-oil ratio in SCF/STB.
    rsi : float
        Initial solution gas-oil ratio in SCF/STB.

    Returns
    -------
    Union[float, np.ndarray]
        Two-phase formation volume factor in rb/STB.
    """
    bo_arr, is_scalar = _vectorize_input(bo)
    bg_arr, _ = _vectorize_input(bg)
    rs_arr, _ = _vectorize_input(rs)

    delta_rs = np.maximum(0.0, float(rsi) - rs_arr)
    bt = bo_arr + delta_rs * bg_arr
    return _format_return(bt, is_scalar)


def expansion_oil(
    bo: Union[float, np.ndarray],
    bg: Union[float, np.ndarray],
    rs: Union[float, np.ndarray],
    boi: float,
    rsi: float,
) -> Union[float, np.ndarray]:
    """
    Calculate oil and dissolved gas expansion term (Eo) in rb/STB.

    Equation:
        Eo = (Bo - Boi) + (Rsi - Rs) * Bg = Bt - Boi

    At initial conditions (P = Pi), Eo = 0.0.
    As pressure depletes, Eo >= 0.0 increases monotonically.

    Parameters
    ----------
    bo : Union[float, np.ndarray]
        Current oil formation volume factor in rb/STB.
    bg : Union[float, np.ndarray]
        Current gas formation volume factor in rb/SCF.
    rs : Union[float, np.ndarray]
        Current solution gas-oil ratio in SCF/STB.
    boi : float
        Initial oil formation volume factor at Pi in rb/STB.
    rsi : float
        Initial solution gas-oil ratio at Pi in SCF/STB.

    Returns
    -------
    Union[float, np.ndarray]
        Oil and dissolved gas expansion term (Eo) in rb/STB.
    """
    bt = two_phase_fvf(bo, bg, rs, rsi)
    bt_arr, is_scalar = _vectorize_input(bt)
    eo = bt_arr - float(boi)
    return _format_return(eo, is_scalar)


def expansion_gas(
    bg: Union[float, np.ndarray],
    bgi: float,
    boi: float,
) -> Union[float, np.ndarray]:
    """
    Calculate gas cap expansion term (Eg) in rb/STB of original oil in place.

    In standard Havlena-Odeh material balance formulation:
        Eg = Boi * (Bg / Bgi - 1.0)  (rb/STB)
    So that the gas cap expansion contribution is:
        m * N * Eg  (rb)

    At initial conditions (P = Pi, Bg = Bgi), Eg = 0.0.
    As pressure drops (Bg > Bgi), Eg increases monotonically.

    Parameters
    ----------
    bg : Union[float, np.ndarray]
        Current gas formation volume factor in rb/SCF.
    bgi : float
        Initial gas formation volume factor at Pi in rb/SCF.
    boi : float
        Initial oil formation volume factor at Pi in rb/STB.

    Returns
    -------
    Union[float, np.ndarray]
        Gas cap expansion term (Eg) in rb/STB.
    """
    bg_arr, is_scalar = _vectorize_input(bg)
    eg = float(boi) * (bg_arr / float(bgi) - 1.0)
    return _format_return(eg, is_scalar)


def expansion_rock_water(
    p: Union[float, np.ndarray],
    pi: float,
    boi: float,
    cw: float,
    cf: float,
    swi: float,
    m: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate formation pore and connate water expansion term (Ef,w) in rb/STB.

    Equation:
        Ef,w = (1 + m) * Boi * [ (cw * Swi + cf) / (1 - Swi) ] * (Pi - P)

    At initial pressure (P = Pi), Ef,w = 0.0.
    As reservoir pressure drops (Pi - P > 0), Ef,w increases monotonically.

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Current reservoir pressure in psi.
    pi : float
        Initial reservoir pressure in psi.
    boi : float
        Initial oil formation volume factor at Pi in rb/STB.
    cw : float
        Connate water compressibility in psi^-1.
    cf : float
        Formation / pore rock compressibility in psi^-1.
    swi : float
        Initial connate water saturation (fraction, 0.0 to 1.0).
    m : float, default 0.0
        Initial gas cap ratio (dimensionless, V_gi / V_oi).

    Returns
    -------
    Union[float, np.ndarray]
        Pore rock and connate water expansion term (Ef,w) in rb/STB.

    Raises
    ------
    ValueError
        If swi >= 1.0 or swi < 0.0.
    """
    if swi >= 1.0 or swi < 0.0:
        raise ValueError(f"Initial connate water saturation Swi must be in [0.0, 1.0), got {swi}.")

    p_arr, is_scalar = _vectorize_input(p)

    compaction_factor = (float(cw) * float(swi) + float(cf)) / (1.0 - float(swi))
    delta_p = float(pi) - p_arr
    efw = (1.0 + float(m)) * float(boi) * compaction_factor * delta_p

    return _format_return(efw, is_scalar)


def generate_pvt_table(
    pressures: Union[list[float], np.ndarray, pd.Series],
    temp_f: float,
    api: float,
    gas_gravity: float,
    pi: Optional[float] = None,
    pb: Optional[float] = None,
    rsi: Optional[float] = None,
    m: float = 0.0,
    cw: float = 3.0e-6,
    cf: float = 4.0e-6,
    swi: float = 0.20,
    co: Optional[float] = None,
    z_method: str = "papay",
) -> pd.DataFrame:
    """
    Generate a comprehensive Black-Oil PVT and MBE expansion table across pressures.

    Computes:
    - P: Pressure in psi
    - Rs: Solution gas-oil ratio in SCF/STB
    - Bo: Oil formation volume factor in rb/STB
    - Bg: Gas formation volume factor in rb/SCF
    - Bt: Two-phase formation volume factor in rb/STB
    - Eo: Oil + dissolved gas expansion in rb/STB
    - Eg: Gas cap expansion in rb/STB
    - Ef,w: Rock and connate water expansion in rb/STB

    Parameters
    ----------
    pressures : Union[list[float], np.ndarray, pd.Series]
        List or array of reservoir pressures in psi.
    temp_f : float
        Reservoir temperature in °F.
    api : float
        Stock-tank oil gravity in °API.
    gas_gravity : float
        Gas specific gravity (air = 1.0).
    pi : Optional[float], optional
        Initial reservoir pressure in psi. Defaults to max(pressures).
    pb : Optional[float], optional
        Bubble point pressure in psi. If None, derived from rsi (or set to pi).
    rsi : Optional[float], optional
        Initial solution GOR in SCF/STB. If None, derived from pb.
    m : float, default 0.0
        Gas cap ratio (dimensionless).
    cw : float, default 3.0e-6
        Connate water compressibility in psi^-1.
    cf : float, default 4.0e-6
        Formation rock compressibility in psi^-1.
    swi : float, default 0.20
        Initial connate water saturation (fraction).
    co : Optional[float], optional
        Undersaturated oil compressibility in psi^-1. If None, auto-calculated.
    z_method : str, default 'papay'
        Z-factor correlation method ('papay' or 'hall_yarborough').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['P', 'Rs', 'Bo', 'Bg', 'Bt', 'Eo', 'Eg', 'Ef,w']
    """
    p_arr, _ = _vectorize_input(pressures)

    if pi is None:
        pi = float(np.max(p_arr))

    # Resolve Pb and Rsi
    if pb is None and rsi is None:
        pb = float(pi)
        rsi = float(standing_solution_gor(pb, temp_f, api, gas_gravity, pb=pb, rsi=None))
    elif pb is None:
        pb = float(standing_bubble_point_pressure(rsi, temp_f, api, gas_gravity))
    elif rsi is None:
        rsi = float(standing_solution_gor(pb, temp_f, api, gas_gravity, pb=pb, rsi=None))

    # Initial properties at Pi
    boi = float(standing_oil_fvf(pi, pb, rsi, temp_f, api, gas_gravity, co=co))
    bgi = float(gas_fvf(pi, temp_f, gas_gravity=gas_gravity, z_method=z_method))

    # Pressure vector calculations
    rs_vec = standing_solution_gor(p_arr, temp_f, api, gas_gravity, pb=pb, rsi=rsi)
    bo_vec = standing_oil_fvf(p_arr, pb, rsi, temp_f, api, gas_gravity, co=co, rs=rs_vec)
    bg_vec = gas_fvf(p_arr, temp_f, gas_gravity=gas_gravity, z_method=z_method)
    bt_vec = two_phase_fvf(bo_vec, bg_vec, rs_vec, rsi)

    eo_vec = expansion_oil(bo_vec, bg_vec, rs_vec, boi, rsi)
    eg_vec = expansion_gas(bg_vec, bgi, boi)
    efw_vec = expansion_rock_water(p_arr, pi, boi, cw, cf, swi, m=m)

    df = pd.DataFrame(
        {
            "P": p_arr,
            "Rs": rs_vec,
            "Bo": bo_vec,
            "Bg": bg_vec,
            "Bt": bt_vec,
            "Eo": eo_vec,
            "Eg": eg_vec,
            "Ef,w": efw_vec,
        }
    )

    return df
