"""
Material Balance Equation (MBE) Engine, Diagnostics, Drive Indices & Optimizer.

This module implements core reservoir engineering material balance analysis in
standard oilfield units:
- Cumulative Oil / Water Production & Injection (Np, Wp, Wi): STB
- Cumulative Gas Production (Gp): SCF
- Producing & Solution Gas-Oil Ratios (Rp, Rs, Rsi): SCF/STB
- Reservoir Pressure (P, Pi): psi
- Time (t): days
- Formation Volume Factors (Bo, Boi, Bw): rb/STB
- Gas Formation Volume Factor (Bg, Bgi): rb/SCF
- Underground Fluid Withdrawal (F): reservoir barrels (rb)
- Cumulative Aquifer Water Influx (We): reservoir barrels (rb)
- Expansion Terms (Eo, Eg, Ef,w, Et): rb/STB
- Original Oil in Place (N): STB (and MMSTB = N / 1e6)
- Gas-Cap Ratio (m = V_gi / V_oi): dimensionless
- Compressibilities (cw, cf): psi^-1
- Aquifer Constants:
  - Pot Aquifer (pot_coeff): rb/psi
  - Schilthuis Steady-State (C): rb/(day * psi)
  - Carter-Tracy / Van Everdingen-Hurst (B): rb/psi
"""

from dataclasses import dataclass, fields
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from engine.aquifer import (
    carter_tracy_aquifer_influx,
    pot_aquifer_influx,
    schilthuis_aquifer_influx,
    veh_aquifer_influx,
)
from engine.pvt import (
    _format_return,
    _vectorize_input,
    expansion_gas,
    expansion_oil,
    expansion_rock_water,
)

ArrayLike = Union[float, int, list[float], np.ndarray, pd.Series]

_VALID_AQUIFER_MODELS: tuple[str, ...] = (
    "none",
    "pot",
    "schilthuis",
    "carter_tracy",
    "veh",
)

_AQUIFER_MODEL_ALIASES: dict[str, str] = {
    "none": "none",
    "volumetric": "none",
    "pot": "pot",
    "schilthuis": "schilthuis",
    "carter_tracy": "carter_tracy",
    "carter-tracy": "carter_tracy",
    "cartertracy": "carter_tracy",
    "veh": "veh",
    "van_everdingen_hurst": "veh",
    "van-everdingen-hurst": "veh",
}

_ALLOWED_AQUIFER_KWARGS: dict[str, set[str]] = {
    "none": set(),
    "pot": {"pi", "theta", "cw", "cf"},
    "schilthuis": {"pi"},
    "carter_tracy": {"pi", "k", "phi", "mu_w", "ct", "re", "t_d"},
    "veh": {"pi", "k", "phi", "mu_w", "ct", "re", "t_d", "geometry", "rD", "ra", "n"},
}


@dataclass
class MBEResult:
    """
    Structured history-matching result returned by ``fit_mbe_model``.

    Attributes
    ----------
    N : float
        Fitted Original Oil in Place (OOIP) in stock-tank barrels (STB).
    N_mmstb : float
        Fitted OOIP in million stock-tank barrels (MMSTB).
    m : float
        Gas-cap ratio (dimensionless, V_gi / V_oi).
    aquifer_model : str
        Canonical aquifer model identifier (``"none"``, ``"pot"``, ``"schilthuis"``,
        ``"carter_tracy"``, or ``"veh"``).
    aquifer_constant : float
        Fitted aquifer influx constant: ``0.0`` (none), ``pot_coeff`` in rb/psi (pot),
        ``C`` in rb/(day*psi) (schilthuis), or ``B`` in rb/psi (carter_tracy / veh).
    aquifer_constant_name : str
        Name of the fitted aquifer constant (``"none"``, ``"pot_coeff"``, ``"C"``, or ``"B"``).
    aquifer_constant_unit : str
        Field unit string for ``aquifer_constant``.
    F_observed : np.ndarray
        Observed underground fluid withdrawal F in rb.
    F_model : np.ndarray
        Modeled underground fluid withdrawal N * Et + We in rb.
    We : np.ndarray
        Modeled cumulative aquifer water influx in rb.
    Eo : np.ndarray
        Oil and dissolved gas expansion term in rb/STB.
    Eg : np.ndarray
        Gas-cap expansion term in rb/STB.
    Efw : np.ndarray
        Formation rock and connate water expansion term in rb/STB.
    Et : np.ndarray
        Total expansion term Eo + m * Eg + Efw in rb/STB.
    residuals : np.ndarray
        Fit residuals (F_observed - F_model) in rb.
    R2 : float
        Coefficient of determination (R^2) comparing F_observed and F_model.
    RMSE : float
        Root-mean-square error in reservoir barrels (rb).
    diagnostic_table : pd.DataFrame
        Havlena-Odeh and Campbell diagnostic table from ``compute_diagnostic_tables``.
    drive_indices : pd.DataFrame
        Dynamic drive indices (DDI, SDI, WDI, CDI) normalized across active timesteps.
    success : bool
        Optimizer convergence flag from ``scipy.optimize.least_squares``.
    message : str
        Optimizer termination message.
    nfev : int
        Number of objective function evaluations performed by ``least_squares``.
    """

    N: float
    N_mmstb: float
    m: float
    aquifer_model: str
    aquifer_constant: float
    aquifer_constant_name: str
    aquifer_constant_unit: str
    F_observed: np.ndarray
    F_model: np.ndarray
    We: np.ndarray
    Eo: np.ndarray
    Eg: np.ndarray
    Efw: np.ndarray
    Et: np.ndarray
    residuals: np.ndarray
    R2: float
    RMSE: float
    diagnostic_table: pd.DataFrame
    drive_indices: pd.DataFrame
    success: bool
    message: str
    nfev: int

    @property
    def N_stb(self) -> float:
        """Alias for ``N`` in stock-tank barrels (STB)."""
        return self.N

    @property
    def r2(self) -> float:
        """Lowercase alias for ``R2``."""
        return self.R2

    @property
    def rmse(self) -> float:
        """Lowercase alias for ``RMSE`` in rb."""
        return self.RMSE

    @property
    def f_model(self) -> np.ndarray:
        """Lowercase alias for ``F_model`` in rb."""
        return self.F_model

    @property
    def f_observed(self) -> np.ndarray:
        """Lowercase alias for ``F_observed`` in rb."""
        return self.F_observed

    @property
    def we(self) -> np.ndarray:
        """Lowercase alias for ``We`` in rb."""
        return self.We

    @property
    def B(self) -> Optional[float]:
        """Unsteady-state aquifer constant B (rb/psi) when applicable."""
        return self.aquifer_constant if self.aquifer_constant_name == "B" else None

    @property
    def C(self) -> Optional[float]:
        """Schilthuis aquifer constant C (rb/(day*psi)) when applicable."""
        return self.aquifer_constant if self.aquifer_constant_name == "C" else None

    @property
    def pot_coeff(self) -> Optional[float]:
        """Pot aquifer coefficient (rb/psi) when applicable."""
        return self.aquifer_constant if self.aquifer_constant_name == "pot_coeff" else None

    def to_dict(self) -> dict[str, Any]:
        """Return result fields as a dictionary."""
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["N_stb"] = self.N
        return result

    def keys(self) -> list[str]:
        """Return dictionary-style keys supported by ``MBEResult``."""
        return list(self.to_dict().keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str) or not hasattr(self, key):
            raise KeyError(f"Unknown MBEResult key: {key!r}")
        return getattr(self, key)


@dataclass
class _MBEInputs:
    """Resolved internal arrays extracted from an MBE input DataFrame."""

    pressure_psi: Optional[np.ndarray]
    time_days: Optional[np.ndarray]
    np_stb: Optional[np.ndarray]
    f_obs: np.ndarray
    eo: np.ndarray
    eg: np.ndarray
    efw_m0: np.ndarray
    efw_fixed: np.ndarray
    efw_varies_with_m: bool
    we_df: Optional[np.ndarray]


# ============================================================================
# VALIDATION & EXTRACTION HELPERS
# ============================================================================
def _broadcast_arrays(*values: ArrayLike) -> tuple[list[np.ndarray], bool]:
    """
    Convert scalar or 1D inputs to finite float64 NumPy arrays of matching length.

    Returns ``(arrays, all_scalar)``.
    """
    arrays: list[np.ndarray] = []
    all_scalar = True
    target_len: Optional[int] = None

    for val in values:
        arr, is_scalar = _vectorize_input(val)
        if arr.ndim != 1:
            raise ValueError(f"Inputs must be scalars or 1D sequences, got {arr.ndim}D array.")
        if arr.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Inputs must contain only finite numeric values.")
        if not is_scalar:
            all_scalar = False
            if target_len is None:
                target_len = arr.size
            elif arr.size != target_len:
                raise ValueError(
                    f"Mismatched array lengths: expected {target_len}, got {arr.size}."
                )
        arrays.append(arr)

    if target_len is not None:
        arrays = [
            np.full(target_len, a[0], dtype=np.float64) if a.size == 1 else a
            for a in arrays
        ]
    return arrays, all_scalar


def _validate_scalar_positive(value: float, name: str) -> float:
    """Validate that a scalar value is finite and strictly positive (> 0)."""
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be strictly positive (> 0), got {value}.")
    return val


def _validate_scalar_non_negative(value: float, name: str) -> float:
    """Validate that a scalar value is finite and non-negative (>= 0)."""
    val = float(value)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be non-negative (>= 0), got {value}.")
    return val


def _validate_non_negative_array(arr: np.ndarray, name: str) -> None:
    """Ensure every entry of a float64 array is >= 0."""
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (>= 0).")


def _validate_positive_array(arr: np.ndarray, name: str) -> None:
    """Ensure every entry of a float64 array is > 0."""
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive (> 0).")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first matching column name in ``df`` from ``candidates``, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _read_numeric_column(df: pd.DataFrame, col_name: str) -> np.ndarray:
    """Extract a 1D finite float64 NumPy array from a DataFrame column."""
    arr = df[col_name].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Column '{col_name}' contains NaN or non-finite values.")
    return arr


def _resolve_time_days(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Resolve elapsed time in days from ``time_days``, ``days``, ``t``, or ``date``."""
    time_col = _resolve_column(df, ("time_days", "days", "t"))
    if time_col is not None:
        t_arr = _read_numeric_column(df, time_col)
        if np.any(t_arr < 0.0):
            raise ValueError(f"Time column '{time_col}' must be non-negative (days >= 0).")
        return t_arr

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Column 'date' contains invalid or missing timestamps.")
        return (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0

    return None


def _normalize_aquifer_model(aquifer_model: Optional[str]) -> str:
    """Normalize and validate the aquifer model selector string."""
    if aquifer_model is None:
        return "none"
    if not isinstance(aquifer_model, str):
        raise ValueError(
            f"aquifer_model must be one of {_VALID_AQUIFER_MODELS}, got {aquifer_model!r}."
        )
    key = aquifer_model.strip().lower()
    if key not in _AQUIFER_MODEL_ALIASES:
        raise ValueError(
            f"Unknown aquifer_model {aquifer_model!r}. "
            f"Supported models: {_VALID_AQUIFER_MODELS}."
        )
    return _AQUIFER_MODEL_ALIASES[key]


def _extract_mbe_inputs(
    df: pd.DataFrame,
    *,
    cw: float = 0.0,
    cf: float = 0.0,
    swi: float = 0.0,
    require_eg: bool = True,
    pi: Optional[float] = None,
    boi: Optional[float] = None,
    bgi: Optional[float] = None,
    rsi: Optional[float] = None,
) -> _MBEInputs:
    """
    Validate ``df`` and resolve standardized MBE terms (F, Eo, Eg, Efw, P, time).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    cw_val = _validate_scalar_non_negative(cw, "Water compressibility cw")
    cf_val = _validate_scalar_non_negative(cf, "Formation compressibility cf")
    swi_val = float(swi)
    if not np.isfinite(swi_val) or swi_val < 0.0 or swi_val >= 1.0:
        raise ValueError(f"Connate water saturation swi must be in [0.0, 1.0), got {swi}.")

    n_rows = len(df)
    p_col = _resolve_column(df, ("pressure_psi", "P", "pressure"))
    pressure_psi = _read_numeric_column(df, p_col) if p_col is not None else None
    time_days = _resolve_time_days(df)

    np_col = _resolve_column(df, ("Np_STB", "Np"))
    np_stb = _read_numeric_column(df, np_col) if np_col is not None else None

    bo_col = _resolve_column(df, ("Bo_rb_STB", "Bo"))
    rs_col = _resolve_column(df, ("Rs_SCF_STB", "Rs"))
    bg_col = _resolve_column(df, ("Bg_rb_SCF", "Bg"))

    bo_arr = _read_numeric_column(df, bo_col) if bo_col is not None else None
    rs_arr = _read_numeric_column(df, rs_col) if rs_col is not None else None
    bg_arr = _read_numeric_column(df, bg_col) if bg_col is not None else None

    # Resolve observed underground withdrawal F (rb)
    f_col = _resolve_column(df, ("F", "F_rb"))
    if f_col is not None:
        f_obs = _read_numeric_column(df, f_col)
    else:
        if np_stb is None or bo_arr is None or rs_arr is None or bg_arr is None:
            raise ValueError(
                "DataFrame must contain either precomputed 'F' or production/PVT columns "
                "('Np_STB'/'Np', 'Bo_rb_STB'/'Bo', 'Rs_SCF_STB'/'Rs', 'Bg_rb_SCF'/'Bg')."
            )
        rp_col = _resolve_column(df, ("Rp_SCF_STB", "Rp"))
        if rp_col is not None:
            rp_arr = _read_numeric_column(df, rp_col)
        else:
            gp_col = _resolve_column(df, ("Gp_SCF", "Gp"))
            if gp_col is None:
                raise ValueError(
                    "DataFrame must contain 'Rp_SCF_STB'/'Rp' or 'Gp_SCF'/'Gp' to compute F."
                )
            gp_arr = _read_numeric_column(df, gp_col)
            _validate_non_negative_array(gp_arr, "Cumulative gas Gp")
            with np.errstate(divide="ignore", invalid="ignore"):
                rp_arr = np.where(np_stb > 0.0, gp_arr / np_stb, 0.0)

        wp_col = _resolve_column(df, ("Wp_STB", "Wp"))
        wi_col = _resolve_column(df, ("Wi_STB", "Wi"))
        bw_col = _resolve_column(df, ("Bw_rb_STB", "Bw"))
        wp_val: Union[float, np.ndarray] = _read_numeric_column(df, wp_col) if wp_col else 0.0
        wi_val: Union[float, np.ndarray] = _read_numeric_column(df, wi_col) if wi_col else 0.0
        bw_val: Union[float, np.ndarray] = _read_numeric_column(df, bw_col) if bw_col else 1.0

        f_obs = np.asarray(
            calculate_underground_withdrawal(
                Np=np_stb,
                Rp=rp_arr,
                Bo=bo_arr,
                Rs=rs_arr,
                Bg=bg_arr,
                Wp=wp_val,
                Wi=wi_val,
                Bw=bw_val,
            ),
            dtype=np.float64,
        )

    # Resolve oil + dissolved gas expansion Eo (rb/STB)
    if "Eo" in df.columns:
        eo = _read_numeric_column(df, "Eo")
        boi_val = float(boi) if boi is not None else (float(bo_arr[0]) if bo_arr is not None else None)
    else:
        if bo_arr is None or rs_arr is None or bg_arr is None:
            raise ValueError(
                "DataFrame must contain either 'Eo' or PVT columns "
                "('Bo_rb_STB'/'Bo', 'Rs_SCF_STB'/'Rs', 'Bg_rb_SCF'/'Bg')."
            )
        boi_val = float(bo_arr[0]) if boi is None else _validate_scalar_positive(boi, "Initial Boi")
        rsi_val = float(rs_arr[0]) if rsi is None else _validate_scalar_non_negative(rsi, "Initial Rsi")
        eo = np.asarray(
            expansion_oil(bo_arr, bg_arr, rs_arr, boi=boi_val, rsi=rsi_val),
            dtype=np.float64,
        )

    # Resolve gas cap expansion Eg (rb/STB)
    if "Eg" in df.columns:
        eg = _read_numeric_column(df, "Eg")
    elif bg_arr is not None and boi_val is not None:
        bgi_val = float(bg_arr[0]) if bgi is None else _validate_scalar_positive(bgi, "Initial Bgi")
        _validate_scalar_positive(bgi_val, "Initial Bgi")
        eg = np.asarray(expansion_gas(bg_arr, bgi=bgi_val, boi=boi_val), dtype=np.float64)
    elif not require_eg:
        eg = np.zeros(n_rows, dtype=np.float64)
    else:
        raise ValueError(
            "DataFrame must contain 'Eg' or ('Bg_rb_SCF'/'Bg' and 'Bo_rb_STB'/'Bo') "
            "to evaluate gas-cap expansion."
        )

    # Resolve formation rock & connate water expansion Ef,w (rb/STB)
    efw_col = _resolve_column(df, ("Efw", "Ef,w", "Efw_rb_STB"))
    if efw_col is not None:
        efw_fixed = _read_numeric_column(df, efw_col)
        efw_m0 = np.zeros(n_rows, dtype=np.float64)
        efw_varies_with_m = False
    elif cw_val > 0.0 or cf_val > 0.0:
        if pressure_psi is None or boi_val is None:
            raise ValueError(
                "Pressure column ('pressure_psi'/'P') and initial Boi are required to compute Efw "
                "when cw or cf > 0."
            )
        pi_val = float(pressure_psi[0]) if pi is None else float(pi)
        if not np.isfinite(pi_val):
            raise ValueError(f"Initial pressure Pi must be finite, got {pi}.")
        efw_m0 = np.asarray(
            expansion_rock_water(
                pressure_psi,
                pi=pi_val,
                boi=boi_val,
                cw=cw_val,
                cf=cf_val,
                swi=swi_val,
                m=0.0,
            ),
            dtype=np.float64,
        )
        efw_fixed = efw_m0
        efw_varies_with_m = True
    else:
        efw_fixed = np.zeros(n_rows, dtype=np.float64)
        efw_m0 = np.zeros(n_rows, dtype=np.float64)
        efw_varies_with_m = False

    we_col = _resolve_column(df, ("We", "We_rb", "We_bbl"))
    we_df = _read_numeric_column(df, we_col) if we_col is not None else None

    return _MBEInputs(
        pressure_psi=pressure_psi,
        time_days=time_days,
        np_stb=np_stb,
        f_obs=f_obs,
        eo=eo,
        eg=eg,
        efw_m0=efw_m0,
        efw_fixed=efw_fixed,
        efw_varies_with_m=efw_varies_with_m,
        we_df=we_df,
    )


def _compute_unit_aquifer_influx(
    aquifer_model: str,
    inputs: _MBEInputs,
    aquifer_kwargs: Optional[dict[str, Any]],
    pi_override: Optional[float],
) -> tuple[np.ndarray, str, str]:
    """
    Compute unit-constant aquifer influx We(coeff=1.0) once outside the optimizer loop.

    All supported aquifer models are strictly homogeneous linear in their aquifer
    influx constant (``pot_coeff``, ``C``, or ``B``), so ``We(coeff) = coeff * unit_we``.
    """
    kwargs = dict(aquifer_kwargs) if aquifer_kwargs is not None else {}
    unknown = set(kwargs.keys()) - _ALLOWED_AQUIFER_KWARGS[aquifer_model]
    if unknown:
        raise ValueError(
            f"Unsupported aquifer_kwargs {sorted(unknown)} for aquifer_model='{aquifer_model}'. "
            f"Allowed keys: {sorted(_ALLOWED_AQUIFER_KWARGS[aquifer_model])}."
        )

    n_rows = inputs.f_obs.size
    if aquifer_model == "none":
        return np.zeros(n_rows, dtype=np.float64), "none", ""

    if inputs.pressure_psi is None:
        raise ValueError(
            f"Pressure column ('pressure_psi' or 'P') is required for aquifer_model='{aquifer_model}'."
        )
    pressures = inputs.pressure_psi
    pi_val = float(kwargs.pop("pi", pi_override if pi_override is not None else pressures[0]))
    if not np.isfinite(pi_val):
        raise ValueError(f"Initial pressure Pi must be finite, got {pi_val}.")

    if aquifer_model == "pot":
        # Validate optional physical parameters if supplied; pot_coeff absorbs (cw+cf)*Wi*(theta/360)
        if "cw" in kwargs:
            _validate_scalar_non_negative(kwargs.pop("cw"), "Aquifer cw")
        if "cf" in kwargs:
            _validate_scalar_non_negative(kwargs.pop("cf"), "Aquifer cf")
        if "theta" in kwargs:
            theta_val = float(kwargs.pop("theta"))
            if not np.isfinite(theta_val) or theta_val <= 0.0 or theta_val > 360.0:
                raise ValueError(
                    f"Encroachment angle theta must be in (0.0, 360.0] degrees, got {theta_val}."
                )
        unit_we = np.asarray(
            pot_aquifer_influx(
                p=pressures,
                pi=pi_val,
                cw=1.0,
                cf=0.0,
                wi_aquifer=1.0,
                theta=360.0,
            ),
            dtype=np.float64,
        )
        return unit_we, "pot_coeff", "rb/psi"

    if inputs.time_days is None:
        raise ValueError(
            f"A time column ('time_days', 'days', or 'date') is required for "
            f"aquifer_model='{aquifer_model}'."
        )

    if aquifer_model == "schilthuis":
        unit_we = schilthuis_aquifer_influx(
            time_days=inputs.time_days,
            pressures=pressures,
            c=1.0,
            pi=pi_val,
        )
        return unit_we, "C", "rb/(day*psi)"

    if aquifer_model == "carter_tracy":
        unit_we = carter_tracy_aquifer_influx(
            time_days=inputs.time_days,
            pressures=pressures,
            b=1.0,
            pi=pi_val,
            **kwargs,
        )
        return unit_we, "B", "rb/psi"

    # aquifer_model == "veh"
    unit_we = veh_aquifer_influx(
        time_days=inputs.time_days,
        pressures=pressures,
        b=1.0,
        pi=pi_val,
        **kwargs,
    )
    return unit_we, "B", "rb/psi"


def _compute_initial_guess(
    f_obs: np.ndarray,
    eo: np.ndarray,
    eg: np.ndarray,
    efw_m0: np.ndarray,
    efw_fixed: np.ndarray,
    efw_varies_with_m: bool,
    unit_we: np.ndarray,
    fit_m: bool,
    has_aquifer: bool,
    m_init: float,
) -> tuple[float, float, float]:
    """Compute a well-scaled non-negative starting point ``(N0, m0, coeff0)`` via NNLS."""
    e_oil = eo + (efw_m0 if efw_varies_with_m else efw_fixed)
    e_gas = eg + (efw_m0 if efw_varies_with_m else 0.0)
    et_fixed = eo + m_init * eg + ((1.0 + m_init) * efw_m0 if efw_varies_with_m else efw_fixed)

    def _ols_n(basis: np.ndarray) -> float:
        denom = float(np.dot(basis, basis))
        if denom <= 1e-30:
            return 1.0
        return max(float(np.dot(basis, f_obs) / denom), 1e-3)

    if not fit_m and not has_aquifer:
        return _ols_n(et_fixed), m_init, 0.0

    if not fit_m and has_aquifer:
        sol, _ = nnls(np.column_stack([et_fixed, unit_we]), f_obs)
        n0 = float(sol[0]) if sol[0] > 1e-3 else _ols_n(et_fixed)
        c0 = max(float(sol[1]), 0.0)
        return n0, m_init, c0

    if fit_m and not has_aquifer:
        sol, _ = nnls(np.column_stack([e_oil, e_gas]), f_obs)
        if sol[0] > 1e-3:
            return float(sol[0]), max(float(sol[1] / sol[0]), 0.0), 0.0
        return _ols_n(e_oil), max(m_init, 0.0), 0.0

    sol, _ = nnls(np.column_stack([e_oil, e_gas, unit_we]), f_obs)
    if sol[0] > 1e-3:
        return float(sol[0]), max(float(sol[1] / sol[0]), 0.0), max(float(sol[2]), 0.0)
    return _ols_n(e_oil), max(m_init, 0.0), 0.0


def _compute_fit_metrics(f_obs: np.ndarray, f_model: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute residual vector (rb), R^2, and RMSE (rb)."""
    residuals = f_obs - f_model
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((f_obs - float(np.mean(f_obs))) ** 2))
    if ss_tot <= 0.0:
        r2 = 1.0 if ss_res <= 1e-24 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / f_obs.size))
    return residuals, float(r2), rmse


# ============================================================================
# 1. UNDERGROUND FLUID WITHDRAWAL (F)
# ============================================================================
def calculate_underground_withdrawal(
    Np: ArrayLike,
    Rp: ArrayLike,
    Bo: ArrayLike,
    Rs: ArrayLike,
    Bg: ArrayLike,
    Wp: ArrayLike = 0.0,
    Wi: ArrayLike = 0.0,
    Bw: ArrayLike = 1.0,
) -> Union[float, np.ndarray]:
    """
    Calculate total underground fluid withdrawal (F) in reservoir barrels (rb).

    Equation:
        F = Np * (Bo + (Rp - Rs) * Bg) + (Wp - Wi) * Bw

    Parameters
    ----------
    Np : Union[float, list[float], np.ndarray, pd.Series]
        Cumulative oil production in STB (>= 0).
    Rp : Union[float, list[float], np.ndarray, pd.Series]
        Cumulative producing gas-oil ratio (Gp / Np) in SCF/STB (>= 0).
    Bo : Union[float, list[float], np.ndarray, pd.Series]
        Oil formation volume factor in rb/STB (> 0).
    Rs : Union[float, list[float], np.ndarray, pd.Series]
        Solution gas-oil ratio in SCF/STB (>= 0).
    Bg : Union[float, list[float], np.ndarray, pd.Series]
        Gas formation volume factor in rb/SCF (>= 0).
    Wp : Union[float, list[float], np.ndarray, pd.Series], default 0.0
        Cumulative water production in STB (>= 0).
    Wi : Union[float, list[float], np.ndarray, pd.Series], default 0.0
        Cumulative water injection in STB (>= 0).
    Bw : Union[float, list[float], np.ndarray, pd.Series], default 1.0
        Water formation volume factor in rb/STB (> 0).

    Returns
    -------
    Union[float, np.ndarray]
        Underground fluid withdrawal F in reservoir barrels (rb).

    Raises
    ------
    ValueError
        If inputs are non-finite, shape-mismatched, or outside physical bounds.
    """
    (np_arr, rp_arr, bo_arr, rs_arr, bg_arr, wp_arr, wi_arr, bw_arr), is_scalar = _broadcast_arrays(
        Np, Rp, Bo, Rs, Bg, Wp, Wi, Bw
    )
    _validate_non_negative_array(np_arr, "Cumulative oil Np")
    _validate_non_negative_array(rp_arr, "Producing GOR Rp")
    _validate_positive_array(bo_arr, "Oil formation volume factor Bo")
    _validate_non_negative_array(rs_arr, "Solution GOR Rs")
    _validate_non_negative_array(bg_arr, "Gas formation volume factor Bg")
    _validate_non_negative_array(wp_arr, "Cumulative water production Wp")
    _validate_non_negative_array(wi_arr, "Cumulative water injection Wi")
    _validate_positive_array(bw_arr, "Water formation volume factor Bw")

    f_rb = np_arr * (bo_arr + (rp_arr - rs_arr) * bg_arr) + (wp_arr - wi_arr) * bw_arr
    return _format_return(f_rb, is_scalar)


# ============================================================================
# 2. TOTAL EXPANSION TERM (Et)
# ============================================================================
def calculate_total_expansion(
    Eo: ArrayLike,
    Eg: ArrayLike,
    Efw: ArrayLike = 0.0,
    m: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate total reservoir fluid and pore/water expansion (Et) in rb/STB.

    Equation:
        Et = Eo + m * Eg + Efw

    Parameters
    ----------
    Eo : Union[float, list[float], np.ndarray, pd.Series]
        Oil and dissolved gas expansion term in rb/STB.
    Eg : Union[float, list[float], np.ndarray, pd.Series]
        Gas-cap expansion term in rb/STB.
    Efw : Union[float, list[float], np.ndarray, pd.Series], default 0.0
        Formation rock and connate water expansion term in rb/STB.
    m : float, default 0.0
        Initial gas-cap ratio (dimensionless, V_gi / V_oi, >= 0).

    Returns
    -------
    Union[float, np.ndarray]
        Total expansion Et in rb/STB.

    Raises
    ------
    ValueError
        If ``m < 0`` or array inputs are invalid or shape-mismatched.
    """
    m_val = _validate_scalar_non_negative(m, "Gas-cap ratio m")
    (eo_arr, eg_arr, efw_arr), is_scalar = _broadcast_arrays(Eo, Eg, Efw)
    et = eo_arr + m_val * eg_arr + efw_arr
    return _format_return(et, is_scalar)


# ============================================================================
# 3. HAVLENA-ODEH & CAMPBELL DIAGNOSTICS
# ============================================================================
def compute_diagnostic_tables(
    df: pd.DataFrame,
    N: float,
    m: float = 0.0,
    We: Optional[ArrayLike] = None,
    cw: float = 0.0,
    cf: float = 0.0,
    swi: float = 0.0,
    pi: Optional[float] = None,
    boi: Optional[float] = None,
    bgi: Optional[float] = None,
    rsi: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute Havlena-Odeh and Campbell material balance diagnostic curves.

    Safely masks zero-expansion initial rows (``Eo == 0`` or ``Et == 0``) with ``NaN``.
    Computes:
    - Havlena-Odeh Plot 1 (gas cap vs oil): ``X1 = Eg / Eo``, ``Y1 = F / Eo``
    - Havlena-Odeh Plot 2 (water drive):    ``X2 = We / Et``, ``Y2 = F / Et``
    - Campbell Plot (aquifer detection):    ``X_campbell = Np``, ``Y_campbell = F / Et``

    Parameters
    ----------
    df : pd.DataFrame
        Reservoir production & PVT table (or precomputed F, Eo, Eg, Efw columns).
    N : float
        Original Oil in Place (OOIP) in STB (> 0).
    m : float, default 0.0
        Initial gas-cap ratio (dimensionless, >= 0).
    We : Optional[ArrayLike], optional
        Cumulative water influx in rb. If None, uses ``df['We']`` / ``df['We_rb']`` /
        ``df['We_bbl']`` when present, otherwise defaults to ``0.0``.
    cw : float, default 0.0
        Connate water compressibility in psi^-1 (>= 0), used if Efw is not in ``df``.
    cf : float, default 0.0
        Formation rock compressibility in psi^-1 (>= 0), used if Efw is not in ``df``.
    swi : float, default 0.0
        Initial connate water saturation (fraction in [0.0, 1.0)).
    pi, boi, bgi, rsi : Optional[float], optional
        Initial pressure (psi), Boi (rb/STB), Bgi (rb/SCF), and Rsi (SCF/STB).
        Defaults to row 0 of ``df``.

    Returns
    -------
    pd.DataFrame
        Diagnostic DataFrame containing ``F``, ``Eo``, ``Eg``, ``Efw``, ``Et``, ``We``,
        ``F_model``, ``X1``, ``Y1``, ``X2``, ``Y2``, ``X_campbell``, ``Y_campbell``
        (plus ``time_days``, ``pressure_psi``, and ``Np_STB`` when available).
    """
    n_val = _validate_scalar_positive(N, "Original Oil in Place N")
    m_val = _validate_scalar_non_negative(m, "Gas-cap ratio m")

    inputs = _extract_mbe_inputs(
        df,
        cw=cw,
        cf=cf,
        swi=swi,
        require_eg=True,
        pi=pi,
        boi=boi,
        bgi=bgi,
        rsi=rsi,
    )
    n_rows = inputs.f_obs.size
    efw = (1.0 + m_val) * inputs.efw_m0 if inputs.efw_varies_with_m else inputs.efw_fixed
    et = np.asarray(calculate_total_expansion(inputs.eo, inputs.eg, efw, m=m_val), dtype=np.float64)

    if We is not None:
        we_arr, is_scalar_we = _vectorize_input(We)
        if we_arr.ndim != 1 or not np.all(np.isfinite(we_arr)):
            raise ValueError("We must be a finite scalar or 1D array.")
        if is_scalar_we:
            we_vec = np.full(n_rows, we_arr[0], dtype=np.float64)
        elif we_arr.size == n_rows:
            we_vec = we_arr
        else:
            raise ValueError(f"We length ({we_arr.size}) must match DataFrame length ({n_rows}).")
    elif inputs.we_df is not None:
        we_vec = inputs.we_df
    else:
        we_vec = np.zeros(n_rows, dtype=np.float64)

    eo = inputs.eo
    f_obs = inputs.f_obs
    eg = inputs.eg

    with np.errstate(divide="ignore", invalid="ignore"):
        eo_mask = np.abs(eo) > 0.0
        et_mask = np.abs(et) > 0.0

        x1 = np.where(eo_mask, eg / eo, np.nan)
        y1 = np.where(eo_mask, f_obs / eo, np.nan)
        x2 = np.where(et_mask, we_vec / et, np.nan)
        y2 = np.where(et_mask, f_obs / et, np.nan)

    x_campbell = inputs.np_stb if inputs.np_stb is not None else np.full(n_rows, np.nan)
    y_campbell = y2.copy()

    data: dict[str, np.ndarray] = {}
    if inputs.time_days is not None:
        data["time_days"] = inputs.time_days
    if inputs.pressure_psi is not None:
        data["pressure_psi"] = inputs.pressure_psi
    if inputs.np_stb is not None:
        data["Np_STB"] = inputs.np_stb

    data.update(
        {
            "F": f_obs,
            "Eo": eo,
            "Eg": eg,
            "Efw": efw,
            "Et": et,
            "We": we_vec,
            "F_model": n_val * et + we_vec,
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "X_campbell": x_campbell,
            "Y_campbell": y_campbell,
        }
    )
    return pd.DataFrame(data)


# ============================================================================
# 4. DYNAMIC DRIVE INDICES (DDI, SDI, WDI, CDI)
# ============================================================================
def calculate_drive_indices(
    F: ArrayLike,
    N: float,
    Eo: ArrayLike,
    Eg: ArrayLike,
    Efw: ArrayLike,
    We: ArrayLike,
    m: float = 0.0,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Calculate dynamic reservoir drive indices across timesteps.

    Equations (``normalize=False``):
        DDI = (N * Eo) / F
        SDI = (N * m * Eg) / F
        WDI = We / F
        CDI = (N * Efw) / F

    When ``F`` matches the MBE balance ``N * (Eo + m * Eg + Efw) + We``, the sum
    ``DDI + SDI + WDI + CDI == 1.0`` holds across every active timestep (``F > 0``).
    When ``normalize=True``, terms are divided by the modeled balance ``F_model = N * Et + We``
    at active timesteps (``F > 0`` and ``F_model > 0``) so the indices sum strictly to 1.0
    even on noisy field production data. Initial zero-withdrawal points (``F == 0``) are
    safely masked as ``NaN``.

    Parameters
    ----------
    F : ArrayLike
        Underground fluid withdrawal in reservoir barrels (rb).
    N : float
        Original Oil in Place (OOIP) in STB (> 0).
    Eo : ArrayLike
        Oil and dissolved gas expansion in rb/STB.
    Eg : ArrayLike
        Gas-cap expansion in rb/STB.
    Efw : ArrayLike
        Formation rock and connate water expansion in rb/STB.
    We : ArrayLike
        Cumulative aquifer water influx in rb.
    m : float, default 0.0
        Initial gas-cap ratio (dimensionless, >= 0).
    normalize : bool, default False
        If True, divide by ``N * (Eo + m * Eg + Efw) + We`` at active timesteps.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['DDI', 'SDI', 'WDI', 'CDI']``.

    Raises
    ------
    ValueError
        If ``N <= 0``, ``m < 0``, or array inputs are invalid or shape-mismatched.
    """
    n_val = _validate_scalar_positive(N, "Original Oil in Place N")
    m_val = _validate_scalar_non_negative(m, "Gas-cap ratio m")
    (f_arr, eo_arr, eg_arr, efw_arr, we_arr), _ = _broadcast_arrays(F, Eo, Eg, Efw, We)

    ddi_num = n_val * eo_arr
    sdi_num = n_val * m_val * eg_arr
    wdi_num = we_arr
    cdi_num = n_val * efw_arr

    if normalize:
        denom = ddi_num + sdi_num + wdi_num + cdi_num
        active_mask = (np.abs(f_arr) > 0.0) & (np.abs(denom) > 0.0)
    else:
        denom = f_arr
        active_mask = np.abs(f_arr) > 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ddi = np.where(active_mask, ddi_num / denom, np.nan)
        sdi = np.where(active_mask, sdi_num / denom, np.nan)
        wdi = np.where(active_mask, wdi_num / denom, np.nan)
        cdi = np.where(active_mask, cdi_num / denom, np.nan)

    return pd.DataFrame({"DDI": ddi, "SDI": sdi, "WDI": wdi, "CDI": cdi})


# ============================================================================
# 5. HISTORY-MATCHING OPTIMIZER
# ============================================================================
def fit_mbe_model(
    df: pd.DataFrame,
    aquifer_model: str = "none",
    fit_m: bool = False,
    aquifer_kwargs: Optional[dict[str, Any]] = None,
    m: float = 0.0,
    cw: float = 0.0,
    cf: float = 0.0,
    swi: float = 0.0,
    pi: Optional[float] = None,
    boi: Optional[float] = None,
    bgi: Optional[float] = None,
    rsi: Optional[float] = None,
) -> MBEResult:
    """
    Fit a General Material Balance Equation (MBE) model via non-linear least squares.

    Minimizes residuals between observed underground withdrawal ``F`` and modeled
    ``N * (Eo + m * Eg + Ef,w) + We`` using ``scipy.optimize.least_squares`` with
    physically valid non-negative bounds (``N > 0``, ``m >= 0``, aquifer constant ``>= 0``).

    Parameters
    ----------
    df : pd.DataFrame
        Production history and PVT DataFrame (or precomputed F, Eo, Eg, Efw columns).
    aquifer_model : str, default "none"
        Water influx model: ``"none"``, ``"pot"``, ``"schilthuis"``,
        ``"carter_tracy"``, or ``"veh"``.
    fit_m : bool, default False
        If True, jointly optimizes gas-cap ratio ``m >= 0``. If False, fixes ``m``.
    aquifer_kwargs : Optional[dict[str, Any]], optional
        Model-specific parameters forwarded to the aquifer solver (e.g., ``t_d`` or
        ``k``, ``phi``, ``mu_w``, ``ct``, ``re``, ``geometry``, ``rD``, ``ra``, ``n``, ``pi``).
    m : float, default 0.0
        Fixed gas-cap ratio (when ``fit_m=False``) or fallback initial guess (when ``fit_m=True``).
    cw : float, default 0.0
        Connate water compressibility in psi^-1 (>= 0), used if Efw is not in ``df``.
    cf : float, default 0.0
        Formation rock compressibility in psi^-1 (>= 0), used if Efw is not in ``df``.
    swi : float, default 0.0
        Initial connate water saturation in [0.0, 1.0).
    pi, boi, bgi, rsi : Optional[float], optional
        Initial pressure (psi), Boi (rb/STB), Bgi (rb/SCF), and Rsi (SCF/STB).
        Defaults to row 0 of ``df``.

    Returns
    -------
    MBEResult
        Structured fit results containing fitted ``N`` (STB & MMSTB), ``m``,
        ``aquifer_constant``, ``F_model``, ``We``, ``R2``, ``RMSE``,
        ``diagnostic_table``, and ``drive_indices``.

    Raises
    ------
    ValueError
        If ``df`` is invalid, has insufficient rows, ``aquifer_model`` is unknown,
        required aquifer diffusivity/time inputs are missing, or inputs are degenerate.
    """
    model_name = _normalize_aquifer_model(aquifer_model)
    m_fixed = _validate_scalar_non_negative(m, "Gas-cap ratio m")

    inputs = _extract_mbe_inputs(
        df,
        cw=cw,
        cf=cf,
        swi=swi,
        require_eg=bool(fit_m or m_fixed > 0.0),
        pi=pi,
        boi=boi,
        bgi=bgi,
        rsi=rsi,
    )

    f_obs = inputs.f_obs
    eo = inputs.eo
    eg = inputs.eg
    efw_m0 = inputs.efw_m0
    efw_fixed = inputs.efw_fixed
    efw_varies = inputs.efw_varies_with_m

    unit_we, const_name, const_unit = _compute_unit_aquifer_influx(
        model_name,
        inputs,
        aquifer_kwargs,
        pi,
    )
    has_aquifer = model_name != "none"

    n_params = 1 + int(fit_m) + int(has_aquifer)
    if f_obs.size < max(2, n_params):
        raise ValueError(
            f"At least {max(2, n_params)} timesteps are required to fit {n_params} MBE parameter(s); "
            f"got {f_obs.size}."
        )
    if np.all(f_obs == 0.0):
        raise ValueError("Cannot fit MBE model when underground withdrawal F is identically zero.")

    et_check = eo + m_fixed * eg + ((1.0 + m_fixed) * efw_m0 if efw_varies else efw_fixed)
    e_oil_check = eo + (efw_m0 if efw_varies else efw_fixed)
    if (not fit_m and np.all(et_check == 0.0)) or (
        fit_m and np.all(e_oil_check == 0.0) and np.all(eg == 0.0)
    ):
        raise ValueError("Cannot identify OOIP (N) when reservoir expansion terms are identically zero.")

    n0, m0, c0 = _compute_initial_guess(
        f_obs=f_obs,
        eo=eo,
        eg=eg,
        efw_m0=efw_m0,
        efw_fixed=efw_fixed,
        efw_varies_with_m=efw_varies,
        unit_we=unit_we,
        fit_m=fit_m,
        has_aquifer=has_aquifer,
        m_init=m_fixed,
    )

    f_scale = max(float(np.max(np.abs(f_obs))), 1.0)
    n_lb = 1.0e-6
    n_scale = max(n0, 1.0)
    m_scale = max(m0, 1.0)
    we_max = max(float(np.max(np.abs(unit_we))), 1.0) if has_aquifer else 1.0
    c_scale = max(c0, f_scale / we_max, 1.0) if has_aquifer else 1.0

    # Assemble parameter vector, scaling, and bounds uniformly across active free parameters
    idx_m = 1 if fit_m else None
    idx_c = (2 if fit_m else 1) if has_aquifer else None

    scales_list = [n_scale]
    x0_list = [max(n0, n_lb)]
    lb_list = [n_lb]
    if fit_m:
        scales_list.append(m_scale)
        x0_list.append(max(m0, 0.0))
        lb_list.append(0.0)
    if has_aquifer:
        scales_list.append(c_scale)
        x0_list.append(max(c0, 0.0))
        lb_list.append(0.0)

    scales = np.asarray(scales_list, dtype=np.float64)
    x0 = np.asarray(x0_list, dtype=np.float64)
    lb = np.asarray(lb_list, dtype=np.float64)
    ub = np.full_like(lb, np.inf)

    def _residual_fn(theta: np.ndarray) -> np.ndarray:
        vals = theta * scales
        n_curr = float(vals[0])
        m_curr = float(vals[idx_m]) if idx_m is not None else m_fixed
        c_curr = float(vals[idx_c]) if idx_c is not None else 0.0
        efw_curr = (1.0 + m_curr) * efw_m0 if efw_varies else efw_fixed
        et_curr = eo + m_curr * eg + efw_curr
        return (n_curr * et_curr + c_curr * unit_we - f_obs) / f_scale

    opt = least_squares(
        _residual_fn,
        x0 / scales,
        bounds=(lb / scales, ub / scales),
        method="trf",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    fitted = opt.x * scales

    n_opt = float(fitted[0])
    m_opt = float(fitted[idx_m]) if idx_m is not None else m_fixed
    coeff_opt = float(fitted[idx_c]) if idx_c is not None else 0.0

    efw_opt = (1.0 + m_opt) * efw_m0 if efw_varies else efw_fixed
    et_opt = np.asarray(calculate_total_expansion(eo, eg, efw_opt, m=m_opt), dtype=np.float64)
    we_opt = coeff_opt * unit_we if has_aquifer else np.zeros_like(f_obs)
    f_model = n_opt * et_opt + we_opt

    residuals, r2, rmse = _compute_fit_metrics(f_obs, f_model)
    diag_df = compute_diagnostic_tables(
        df=df,
        N=n_opt,
        m=m_opt,
        We=we_opt,
        cw=cw,
        cf=cf,
        swi=swi,
        pi=pi,
        boi=boi,
        bgi=bgi,
        rsi=rsi,
    )
    drive_df = calculate_drive_indices(
        F=f_obs,
        N=n_opt,
        Eo=eo,
        Eg=eg,
        Efw=efw_opt,
        We=we_opt,
        m=m_opt,
        normalize=True,
    )

    return MBEResult(
        N=n_opt,
        N_mmstb=n_opt / 1.0e6,
        m=m_opt,
        aquifer_model=model_name,
        aquifer_constant=coeff_opt,
        aquifer_constant_name=const_name,
        aquifer_constant_unit=const_unit,
        F_observed=f_obs,
        F_model=f_model,
        We=we_opt,
        Eo=eo,
        Eg=eg,
        Efw=efw_opt,
        Et=et_opt,
        residuals=residuals,
        R2=r2,
        RMSE=rmse,
        diagnostic_table=diag_df,
        drive_indices=drive_df,
        success=bool(opt.success),
        message=str(opt.message),
        nfev=int(opt.nfev),
    )


__all__ = [
    "MBEResult",
    "calculate_underground_withdrawal",
    "calculate_total_expansion",
    "compute_diagnostic_tables",
    "calculate_drive_indices",
    "fit_mbe_model",
]
