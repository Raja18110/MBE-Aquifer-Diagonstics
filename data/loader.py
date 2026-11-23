"""
Data loader module for Material Balance Equation (MBE) & Reservoir Diagnostics.

This module provides data ingestion, unit conversion, and preprocessing functions for:
1. Equinor Volve production data (Excel/CSV), aggregating daily volumes to monthly sums
   and downhole pressures to monthly averages.
2. Standard unit conversions between metric (Sm3, bar) and oilfield units (STB, SCF, psi).
3. Cumulative production/injection metrics (Np, Gp, Wp, Wi) and producing GOR (Rp).
4. Benchmark reservoir datasets (volumetric depletion and water-drive benchmarks)
   and custom production CSV uploads.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

# ============================================================================
# STANDARD OILFIELD UNIT CONVERSION CONSTANTS
# ============================================================================
# Oil / Water volume conversion: 1 standard cubic meter (Sm3) to stock-tank barrels (STB)
SM3_TO_STB: float = 6.2898

# Gas volume conversion: 1 standard cubic meter (Sm3) to standard cubic feet (SCF)
SM3_GAS_TO_SCF: float = 35.3147

# Pressure conversion: 1 bar to pounds per square inch (psi)
BAR_TO_PSI: float = 14.5038


DEFAULT_VOLVE_PATH: str = "data/raw/Volve production data.xlsx"


def _get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def _resolve_file_path(
    file_path: Union[str, Path],
    fallback_subpaths: Optional[list[str]] = None,
    allow_fallbacks: bool = True,
) -> Path:
    """
    Resolve a file path dynamically, supporting both absolute paths and
    relative paths against the current working directory or repository root.

    Parameters
    ----------
    file_path : Union[str, Path]
        Input file path to resolve.
    fallback_subpaths : Optional[list[str]], optional
        List of candidate relative subpaths to check if file_path is not found
        and allow_fallbacks is True.
    allow_fallbacks : bool, default True
        Whether to try fallback paths if the primary path is not found.

    Returns
    -------
    Path
        Resolved existing Path object.

    Raises
    ------
    FileNotFoundError
        If file cannot be located at given or fallback locations.
    """
    path_obj = Path(file_path)
    if path_obj.is_file():
        return path_obj.resolve()

    # Try relative to project root
    root = _get_project_root()
    candidate = (root / file_path).resolve()
    if candidate.is_file():
        return candidate

    # Try fallback candidate subpaths only when allowed
    if allow_fallbacks and fallback_subpaths:
        for sub in fallback_subpaths:
            cand = (root / sub).resolve()
            if cand.is_file():
                return cand
            if Path(sub).is_file():
                return Path(sub).resolve()

    raise FileNotFoundError(f"Could not locate file at '{file_path}' or within project root '{root}'.")


def load_volve_production(
    file_path: Union[str, Path] = "data/raw/Volve production data.xlsx",
    sheet_name: str = "Daily Production Data",
    well_bore_code: Optional[str] = None,
    exclude_zero_pressure: bool = True,
    drop_pre_production: bool = False,
) -> pd.DataFrame:
    """
    Load Equinor Volve daily production Excel/CSV data, aggregate daily volumes
    to monthly sums, and calculate average downhole pressure.

    Converts metric volumes and pressures to standard oilfield units (STB, SCF, psi)
    and computes cumulative production (Np_STB, Gp_SCF, Wp_STB, Wi_STB) as well as
    producing gas-oil ratio Rp (SCF/STB).

    Parameters
    ----------
    file_path : Union[str, Path], default "data/raw/Volve production data.xlsx"
        Path to the Volve production dataset (.xlsx, .xls, or .csv).
    sheet_name : str, default "Daily Production Data"
        Excel sheet name containing daily production records.
    well_bore_code : Optional[str], default None
        Specific wellbore code to filter (e.g., 'NO 15/9-F-14 H').
        If None, aggregates across all wells in the field.
    exclude_zero_pressure : bool, default True
        If True, ignores downhole gauge readings <= 0 bar when computing
        the monthly average pressure (0 bar indicates gauge offline or shut-in).
    drop_pre_production : bool, default False
        If True, drops early months where cumulative oil production Np is 0.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated DataFrame containing:
        - 'year_month': Monthly period string ('YYYY-MM')
        - 'date': Timestamp at beginning of month (datetime64[ns])
        - 'days': Elapsed time in days from first record
        - 'oil_vol_Sm3': Monthly oil production in Sm3
        - 'gas_vol_Sm3': Monthly gas production in Sm3
        - 'water_vol_Sm3': Monthly water production in Sm3
        - 'water_inj_Sm3': Monthly water injection in Sm3
        - 'pressure_bar': Monthly average downhole pressure in bar
        - 'oil_vol_STB': Monthly oil production in STB
        - 'gas_vol_SCF': Monthly gas production in SCF
        - 'water_vol_STB': Monthly water production in STB
        - 'water_inj_STB': Monthly water injection in STB
        - 'pressure_psi': Monthly average downhole pressure in psi
        - 'Np_STB': Cumulative oil production in STB
        - 'Gp_SCF': Cumulative gas production in SCF
        - 'Wp_STB': Cumulative water production in STB
        - 'Wi_STB': Cumulative water injection in STB
        - 'Rp_SCF_STB': Cumulative producing gas-oil ratio in SCF/STB (Gp_SCF / Np_STB)
        - 'Rp': Alias for 'Rp_SCF_STB'
        - 'monthly_GOR_SCF_STB': Monthly producing gas-oil ratio in SCF/STB

    Raises
    ------
    FileNotFoundError
        If the file cannot be located.
    KeyError
        If required production columns are missing in the input file.
    """
    is_default_path = (
        str(file_path) == DEFAULT_VOLVE_PATH
        or str(file_path) == "data/raw/Volve production data.xlsx"
    )
    resolved_path = _resolve_file_path(
        file_path,
        fallback_subpaths=[
            "data/raw/Volve production data.xlsx",
            "data/Volve production data.xlsx",
        ],
        allow_fallbacks=is_default_path,
    )

    if resolved_path.suffix.lower() in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(resolved_path, sheet_name=sheet_name)
    else:
        df_raw = pd.read_csv(resolved_path)

    # Validate required columns
    required_cols = [
        "DATEPRD",
        "BORE_OIL_VOL",
        "BORE_GAS_VOL",
        "BORE_WAT_VOL",
        "BORE_WI_VOL",
        "AVG_DOWNHOLE_PRESSURE",
    ]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        raise KeyError(f"Input file is missing required columns: {missing_cols}")

    df = df_raw.copy()

    # Optional filter by wellbore
    if well_bore_code is not None:
        if "WELL_BORE_CODE" in df.columns:
            df = df[df["WELL_BORE_CODE"] == well_bore_code].copy()
            if df.empty:
                raise ValueError(f"No records found for WELL_BORE_CODE '{well_bore_code}'.")
        else:
            raise KeyError("Column 'WELL_BORE_CODE' not present in data for filtering.")

    # Parse date and build monthly grouping period
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    df["YEAR_MONTH"] = df["DATEPRD"].dt.to_period("M")

    # Clean volume columns (replace NaNs with 0.0)
    vol_cols = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "BORE_WI_VOL"]
    for c in vol_cols:
        df[c] = df[c].fillna(0.0)

    # Clean pressure column
    p_clean = df["AVG_DOWNHOLE_PRESSURE"].copy()
    if exclude_zero_pressure:
        p_clean = p_clean.where(p_clean > 0, np.nan)
    df["P_CLEAN"] = p_clean

    # Aggregate daily data to monthly timesteps
    monthly = (
        df.groupby("YEAR_MONTH")
        .agg(
            oil_vol_Sm3=("BORE_OIL_VOL", "sum"),
            gas_vol_Sm3=("BORE_GAS_VOL", "sum"),
            water_vol_Sm3=("BORE_WAT_VOL", "sum"),
            water_inj_Sm3=("BORE_WI_VOL", "sum"),
            pressure_bar=("P_CLEAN", "mean"),
        )
        .reset_index()
    )

    # Vectorized unit conversion from metric to standard oilfield units
    oil_sm3 = monthly["oil_vol_Sm3"].to_numpy(dtype=np.float64)
    gas_sm3 = monthly["gas_vol_Sm3"].to_numpy(dtype=np.float64)
    wat_sm3 = monthly["water_vol_Sm3"].to_numpy(dtype=np.float64)
    wi_sm3 = monthly["water_inj_Sm3"].to_numpy(dtype=np.float64)
    p_bar = monthly["pressure_bar"].to_numpy(dtype=np.float64)

    oil_stb = oil_sm3 * SM3_TO_STB
    gas_scf = gas_sm3 * SM3_GAS_TO_SCF
    wat_stb = wat_sm3 * SM3_TO_STB
    wi_stb = wi_sm3 * SM3_TO_STB
    p_psi = p_bar * BAR_TO_PSI

    monthly["oil_vol_STB"] = oil_stb
    monthly["gas_vol_SCF"] = gas_scf
    monthly["water_vol_STB"] = wat_stb
    monthly["water_inj_STB"] = wi_stb
    monthly["pressure_psi"] = p_psi

    # Cumulative production and injection volumes (vectorized cumsum)
    np_stb = np.cumsum(oil_stb)
    gp_scf = np.cumsum(gas_scf)
    wp_stb = np.cumsum(wat_stb)
    wi_stb_cum = np.cumsum(wi_stb)

    monthly["Np_STB"] = np_stb
    monthly["Gp_SCF"] = gp_scf
    monthly["Wp_STB"] = wp_stb
    monthly["Wi_STB"] = wi_stb_cum

    # Producing gas-oil ratio Rp (cumulative GOR: Gp / Np in SCF/STB)
    with np.errstate(divide="ignore", invalid="ignore"):
        rp_scf_stb = np.where(np_stb > 0, gp_scf / np_stb, 0.0)
        monthly_gor = np.where(oil_stb > 0, gas_scf / oil_stb, 0.0)

    monthly["Rp_SCF_STB"] = rp_scf_stb
    monthly["Rp"] = rp_scf_stb  # Alias for standard MBE notation
    monthly["monthly_GOR_SCF_STB"] = monthly_gor

    # Timestep dates and elapsed days
    date_series = monthly["YEAR_MONTH"].dt.to_timestamp()
    monthly["date"] = date_series
    monthly["year_month"] = monthly["YEAR_MONTH"].astype(str)

    first_date = date_series.iloc[0]
    monthly["days"] = (date_series - first_date).dt.days.to_numpy(dtype=np.int64)

    # Reorder columns logically
    ordered_cols = [
        "year_month",
        "date",
        "days",
        "oil_vol_Sm3",
        "gas_vol_Sm3",
        "water_vol_Sm3",
        "water_inj_Sm3",
        "pressure_bar",
        "oil_vol_STB",
        "gas_vol_SCF",
        "water_vol_STB",
        "water_inj_STB",
        "pressure_psi",
        "Np_STB",
        "Gp_SCF",
        "Wp_STB",
        "Wi_STB",
        "Rp_SCF_STB",
        "Rp",
        "monthly_GOR_SCF_STB",
    ]
    result = monthly[ordered_cols].copy()

    if drop_pre_production:
        result = result[result["Np_STB"] > 0].reset_index(drop=True)
        if not result.empty:
            result["days"] = (result["date"] - result["date"].iloc[0]).dt.days.to_numpy(dtype=np.int64)

    return result


def load_benchmark_data(
    benchmark_name_or_path: Union[str, Path] = "volumetric_oil",
) -> pd.DataFrame:
    """
    Load a standardized reservoir benchmark dataset (volumetric depletion or water-drive).

    Parameters
    ----------
    benchmark_name_or_path : Union[str, Path], default "volumetric_oil"
        Predefined benchmark keyword ('volumetric_oil', 'water_drive')
        or a file path to a benchmark CSV file.

    Returns
    -------
    pd.DataFrame
        Clean benchmark DataFrame with standardized oilfield units:
        - 'time_days': Elapsed production time (days)
        - 'date': Timestep date string or timestamp
        - 'pressure_psi': Average reservoir pressure (psi)
        - 'Np_STB': Cumulative oil production (STB)
        - 'Gp_SCF': Cumulative gas production (SCF)
        - 'Wp_STB': Cumulative water production (STB)
        - 'Wi_STB': Cumulative water injection (STB)
        - 'Rp_SCF_STB': Cumulative producing GOR (SCF/STB)
        - 'Rp': Alias for Rp_SCF_STB
        - 'Bo_rb_STB': Oil formation volume factor (rb/STB)
        - 'Rs_SCF_STB': Solution gas-oil ratio (SCF/STB)
        - 'Bg_rb_SCF': Gas formation volume factor (rb/SCF)
        - 'Bw_rb_STB': Water formation volume factor (rb/STB)
        - 'We_bbl': Cumulative water influx (bbl), present for water-drive benchmarks

    Raises
    ------
    FileNotFoundError
        If the benchmark CSV file cannot be located.
    ValueError
        If an unknown benchmark name is supplied and does not resolve to a file.
    """
    benchmarks_map = {
        "volumetric": "data/samples/volumetric_oil_benchmark.csv",
        "volumetric_oil": "data/samples/volumetric_oil_benchmark.csv",
        "volumetric_oil_benchmark": "data/samples/volumetric_oil_benchmark.csv",
        "water_drive": "data/samples/water_drive_benchmark.csv",
        "water_drive_benchmark": "data/samples/water_drive_benchmark.csv",
    }

    path_str = str(benchmark_name_or_path).lower().strip()
    is_named = path_str in benchmarks_map
    target_path: Union[str, Path] = benchmarks_map.get(path_str, benchmark_name_or_path)

    resolved_path = _resolve_file_path(
        target_path,
        fallback_subpaths=[
            "data/samples/volumetric_oil_benchmark.csv",
            "data/samples/water_drive_benchmark.csv",
        ]
        if is_named
        else None,
        allow_fallbacks=is_named,
    )

    df = pd.read_csv(resolved_path)

    # Ensure cumulative GOR Rp alias exists
    if "Rp_SCF_STB" in df.columns and "Rp" not in df.columns:
        df["Rp"] = df["Rp_SCF_STB"]
    elif "Rp" in df.columns and "Rp_SCF_STB" not in df.columns:
        df["Rp_SCF_STB"] = df["Rp"]
    elif "Np_STB" in df.columns and "Gp_SCF" in df.columns and "Rp_SCF_STB" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            rp = np.where(df["Np_STB"] > 0, df["Gp_SCF"] / df["Np_STB"], 0.0)
        df["Rp_SCF_STB"] = rp
        df["Rp"] = rp

    return df


def load_custom_production_csv(
    file_path: Union[str, Path],
    is_metric: bool = False,
    date_col: str = "date",
    pressure_col: str = "pressure",
    oil_col: str = "oil",
    gas_col: str = "gas",
    water_col: str = "water",
    wi_col: Optional[str] = None,
    is_cumulative: bool = False,
) -> pd.DataFrame:
    """
    Ingest user-uploaded custom reservoir production CSV files,
    standardize column nomenclature, convert metric inputs if necessary,
    and compute cumulative metrics.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to custom CSV file.
    is_metric : bool, default False
        Set to True if input volumes are Sm3 and pressure is bar.
        Set to False if input is already in oilfield units (STB, SCF, psi).
    date_col : str, default "date"
        Column name containing dates or timesteps.
    pressure_col : str, default "pressure"
        Column name containing reservoir pressure (bar or psi).
    oil_col : str, default "oil"
        Column name containing oil volume (Sm3 or STB).
    gas_col : str, default "gas"
        Column name containing gas volume (Sm3 or SCF).
    water_col : str, default "water"
        Column name containing water volume (Sm3 or STB).
    wi_col : Optional[str], default None
        Column name containing water injection volume (Sm3 or STB), if present.
    is_cumulative : bool, default False
        True if the input volumes are already cumulative (Np, Gp, Wp, Wi).

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with field units:
        - 'date': Parsed timestamp
        - 'pressure_psi': Pressure in psi
        - 'Np_STB': Cumulative oil production (STB)
        - 'Gp_SCF': Cumulative gas production (SCF)
        - 'Wp_STB': Cumulative water production (STB)
        - 'Wi_STB': Cumulative water injection (STB)
        - 'Rp_SCF_STB': Producing GOR (SCF/STB)
        - 'Rp': Alias for Rp_SCF_STB
    """
    resolved = _resolve_file_path(file_path)
    df_raw = pd.read_csv(resolved)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(df_raw[date_col])

    # Pressure
    p_raw = df_raw[pressure_col].to_numpy(dtype=np.float64)
    if is_metric:
        df["pressure_psi"] = p_raw * BAR_TO_PSI
    else:
        df["pressure_psi"] = p_raw

    # Volumes
    oil_raw = df_raw[oil_col].fillna(0.0).to_numpy(dtype=np.float64)
    gas_raw = df_raw[gas_col].fillna(0.0).to_numpy(dtype=np.float64)
    wat_raw = df_raw[water_col].fillna(0.0).to_numpy(dtype=np.float64)
    wi_raw = (
        df_raw[wi_col].fillna(0.0).to_numpy(dtype=np.float64)
        if wi_col and wi_col in df_raw.columns
        else np.zeros_like(oil_raw)
    )

    if is_metric:
        oil_stb = oil_raw * SM3_TO_STB
        gas_scf = gas_raw * SM3_GAS_TO_SCF
        wat_stb = wat_raw * SM3_TO_STB
        wi_stb = wi_raw * SM3_TO_STB
    else:
        oil_stb = oil_raw
        gas_scf = gas_raw
        wat_stb = wat_raw
        wi_stb = wi_raw

    if is_cumulative:
        df["Np_STB"] = oil_stb
        df["Gp_SCF"] = gas_scf
        df["Wp_STB"] = wat_stb
        df["Wi_STB"] = wi_stb
    else:
        df["Np_STB"] = np.cumsum(oil_stb)
        df["Gp_SCF"] = np.cumsum(gas_scf)
        df["Wp_STB"] = np.cumsum(wat_stb)
        df["Wi_STB"] = np.cumsum(wi_stb)

    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(df["Np_STB"] > 0, df["Gp_SCF"] / df["Np_STB"], 0.0)
    df["Rp_SCF_STB"] = rp
    df["Rp"] = rp

    return df
