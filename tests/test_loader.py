"""
Unit tests for data/loader.py (Day 2 Milestone).

Tests Volve Excel/CSV loading, unit conversions (Sm3->STB, Sm3 gas->SCF, bar->psi),
monthly resampling, cumulative volume calculations (Np, Gp, Wp, Wi), producing GOR (Rp),
and benchmark dataset loaders.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data.loader import (
    BAR_TO_PSI,
    SM3_GAS_TO_SCF,
    SM3_TO_STB,
    load_benchmark_data,
    load_custom_production_csv,
    load_volve_production,
)


def test_conversion_constants():
    """Verify that unit conversion constants match exact requirements."""
    assert SM3_TO_STB == pytest.approx(6.2898, rel=1e-5)
    assert SM3_GAS_TO_SCF == pytest.approx(35.3147, rel=1e-5)
    assert BAR_TO_PSI == pytest.approx(14.5038, rel=1e-5)


def test_load_volve_production_structure():
    """Verify Volve loader loads the actual dataset and returns expected columns and types."""
    df = load_volve_production()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 112  # Total months in Volve dataset from 2007-09 to 2016-12

    # Check key columns
    required_cols = [
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
    for col in required_cols:
        assert col in df.columns, f"Missing expected column '{col}'"

    # Verify unit conversions on the dataset
    np.testing.assert_allclose(df["oil_vol_STB"], df["oil_vol_Sm3"] * SM3_TO_STB)
    np.testing.assert_allclose(df["gas_vol_SCF"], df["gas_vol_Sm3"] * SM3_GAS_TO_SCF)
    np.testing.assert_allclose(df["water_vol_STB"], df["water_vol_Sm3"] * SM3_TO_STB)
    np.testing.assert_allclose(df["water_inj_STB"], df["water_inj_Sm3"] * SM3_TO_STB)

    # Valid pressure entries
    valid_p = df["pressure_bar"].dropna()
    valid_psi = df.loc[valid_p.index, "pressure_psi"]
    np.testing.assert_allclose(valid_psi, valid_p * BAR_TO_PSI)


def test_load_volve_cumulatives_and_gor():
    """Verify cumulative calculations and producing GOR logic."""
    df = load_volve_production()

    # Cumulatives must be monotonically non-decreasing
    assert (np.diff(df["Np_STB"]) >= -1e-6).all(), "Np_STB must be non-decreasing"
    assert (np.diff(df["Gp_SCF"]) >= -1e-6).all(), "Gp_SCF must be non-decreasing"
    assert (np.diff(df["Wp_STB"]) >= -1e-6).all(), "Wp_STB must be non-decreasing"
    assert (np.diff(df["Wi_STB"]) >= -1e-6).all(), "Wi_STB must be non-decreasing"

    # Final known field cumulative volumes for Volve (approx ~63 MMSTB oil)
    final_np_mmstb = df["Np_STB"].iloc[-1] / 1e6
    assert 60.0 < final_np_mmstb < 65.0, f"Expected ~63 MMSTB, got {final_np_mmstb}"

    # GOR Rp = Gp / Np where Np > 0, else 0.0
    for idx, row in df.iterrows():
        if row["Np_STB"] > 0:
            expected_rp = row["Gp_SCF"] / row["Np_STB"]
            assert row["Rp_SCF_STB"] == pytest.approx(expected_rp, rel=1e-5)
            assert row["Rp"] == pytest.approx(expected_rp, rel=1e-5)
        else:
            assert row["Rp_SCF_STB"] == 0.0
            assert row["Rp"] == 0.0


def test_load_volve_well_filter():
    """Verify wellbore filtering in Volve dataset."""
    well = "NO 15/9-F-14 H"
    df_well = load_volve_production(well_bore_code=well)
    assert not df_well.empty
    # Single well cumulative oil should be less than total field cumulative oil
    df_all = load_volve_production()
    assert df_well["Np_STB"].iloc[-1] < df_all["Np_STB"].iloc[-1]


def test_load_volumetric_benchmark():
    """Verify loading and properties of volumetric oil benchmark dataset."""
    df = load_benchmark_data("volumetric_oil")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8

    expected_cols = [
        "time_days",
        "date",
        "pressure_psi",
        "Np_STB",
        "Gp_SCF",
        "Wp_STB",
        "Wi_STB",
        "Rp_SCF_STB",
        "Rp",
        "Bo_rb_STB",
        "Rs_SCF_STB",
        "Bg_rb_SCF",
        "Bw_rb_STB",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Initial conditions at t=0
    assert df["pressure_psi"].iloc[0] == 4000.0
    assert df["Np_STB"].iloc[0] == 0.0
    assert df["Wp_STB"].sum() == 0.0  # Volumetric depletion drive: zero water production

    # Monotonic pressure depletion
    assert (np.diff(df["pressure_psi"]) < 0).all()


def test_load_water_drive_benchmark():
    """Verify loading and properties of water drive benchmark dataset."""
    df = load_benchmark_data("water_drive")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11

    assert "We_bbl" in df.columns
    assert "Wp_STB" in df.columns
    assert df["We_bbl"].iloc[-1] > 0
    assert df["Wp_STB"].iloc[-1] > 0

    # Monotonic pressure depletion and water influx
    assert (np.diff(df["pressure_psi"]) < 0).all()
    assert (np.diff(df["We_bbl"]) >= 0).all()


def test_custom_production_csv(tmp_path):
    """Verify custom production CSV loader with metric conversion."""
    custom_file = tmp_path / "custom_prod.csv"
    data = {
        "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "pressure": [300.0, 290.0, 280.0],
        "oil": [1000.0, 2000.0, 1500.0],
        "gas": [10000.0, 20000.0, 15000.0],
        "water": [50.0, 100.0, 80.0],
    }
    pd.DataFrame(data).to_csv(custom_file, index=False)

    df = load_custom_production_csv(custom_file, is_metric=True)
    assert len(df) == 3
    assert df["pressure_psi"].iloc[0] == pytest.approx(300.0 * BAR_TO_PSI)
    assert df["Np_STB"].iloc[0] == pytest.approx(1000.0 * SM3_TO_STB)
    assert df["Np_STB"].iloc[-1] == pytest.approx(4500.0 * SM3_TO_STB)
    assert df["Gp_SCF"].iloc[-1] == pytest.approx(45000.0 * SM3_GAS_TO_SCF)


def test_error_handling():
    """Verify appropriate exceptions for missing files or invalid wells."""
    with pytest.raises(FileNotFoundError):
        load_volve_production("non_existent_path.xlsx")

    with pytest.raises(FileNotFoundError):
        load_benchmark_data("non_existent_benchmark_name")

    with pytest.raises(ValueError):
        load_volve_production(well_bore_code="NON_EXISTENT_WELL_12345")
