"""
Unit tests for engine/mbe.py (Day 6 Milestone).

Tests Material Balance Equation (MBE) calculations, diagnostics, dynamic drive indices,
and non-linear history-matching optimization in oilfield units:
1. Underground fluid withdrawal F against hand-verified scalar and array cases.
2. Total expansion term Et = Eo + m*Eg + Efw.
3. Havlena-Odeh (Plot 1 & Plot 2) and Campbell diagnostic tables with zero-expansion masking.
4. Dynamic drive indices (DDI, SDI, WDI, CDI) summing to 1.0 across active timesteps.
5. History-matching optimizer across all aquifer models ('none', 'pot', 'schilthuis',
   'carter_tracy', 'veh') on exact synthetic and published benchmark datasets
   (volumetric_oil_benchmark.csv and water_drive_benchmark.csv).
6. Input validation and safe handling of zero-production/initial rows.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from data.loader import load_benchmark_data
from engine import (
    MBEResult,
    calculate_drive_indices,
    calculate_total_expansion,
    calculate_underground_withdrawal,
    carter_tracy_aquifer_influx,
    compute_diagnostic_tables,
    expansion_gas,
    expansion_oil,
    expansion_rock_water,
    fit_mbe_model,
    generate_pvt_table,
    pot_aquifer_influx,
    radial_aquifer_constant,
    schilthuis_aquifer_influx,
    veh_aquifer_influx,
)


# ============================================================================
# 1. UNDERGROUND WITHDRAWAL & TOTAL EXPANSION
# ============================================================================
def test_underground_withdrawal_hand_verified():
    """Verify F = Np * (Bo + (Rp - Rs) * Bg) + (Wp - Wi) * Bw on hand-calculated values."""
    # Single scalar timestep above and below bubble point
    np_stb = 2_600_000.0
    rp = 700.0
    bo = 1.340
    rs = 530.0
    bg = 0.00125
    wp = 150_000.0
    wi = 40_000.0
    bw = 1.02

    # F = 2.6e6 * (1.34 + (700 - 530) * 0.00125) + (150e3 - 40e3) * 1.02
    #   = 2.6e6 * (1.34 + 0.2125) + 110e3 * 1.02
    #   = 2.6e6 * 1.5525 + 112_200 = 4_036_500 + 112_200 = 4_148_700 rb
    f_scalar = calculate_underground_withdrawal(
        Np=np_stb, Rp=rp, Bo=bo, Rs=rs, Bg=bg, Wp=wp, Wi=wi, Bw=bw
    )
    assert isinstance(f_scalar, float)
    assert f_scalar == pytest.approx(4_148_700.0, rel=1e-12)

    # Default Wp=0, Wi=0, Bw=1.0
    f_no_water = calculate_underground_withdrawal(np_stb, rp, bo, rs, bg)
    assert f_no_water == pytest.approx(4_036_500.0, rel=1e-12)

    # Vectorized array including initial zero-production row (Rp=0 < Rs=600)
    np_arr = np.array([0.0, 580_000.0, 2_600_000.0])
    rp_arr = np.array([0.0, 600.0, 700.0])
    bo_arr = np.array([1.350, 1.365, 1.340])
    rs_arr = np.array([600.0, 600.0, 530.0])
    bg_arr = np.array([0.00085, 0.00095, 0.00125])

    f_arr = calculate_underground_withdrawal(np_arr, rp_arr, bo_arr, rs_arr, bg_arr)
    assert isinstance(f_arr, np.ndarray)
    np.testing.assert_allclose(f_arr, [0.0, 791_700.0, 4_036_500.0], rtol=1e-12)


def test_total_expansion_calculation():
    """Verify Et = Eo + m * Eg + Efw for scalar and vectorized inputs."""
    eo = np.array([0.0, 0.05, 0.15])
    eg = np.array([0.0, 0.40, 1.10])
    efw = np.array([0.0, 0.004, 0.010])
    m = 0.25

    et = calculate_total_expansion(eo, eg, efw, m=m)
    assert isinstance(et, np.ndarray)
    expected = np.array([0.0, 0.05 + 0.25 * 0.40 + 0.004, 0.15 + 0.25 * 1.10 + 0.010])
    np.testing.assert_allclose(et, expected, rtol=1e-12)

    et_scalar = calculate_total_expansion(0.10, 0.80, 0.005, m=0.5)
    assert isinstance(et_scalar, float)
    assert et_scalar == pytest.approx(0.10 + 0.5 * 0.80 + 0.005, rel=1e-12)


# ============================================================================
# 2. DRIVE INDICES & DIAGNOSTIC TABLES
# ============================================================================
def test_drive_indices_sum_to_unity_and_handle_zero_f():
    """
    Verify DDI + SDI + WDI + CDI == 1.0 across active timesteps for exact MBE data
    and safe NaN masking when F == 0 without divide-by-zero warnings.
    """
    n_true = 60_000_000.0
    m_true = 0.30
    eo = np.array([0.0, 0.02, 0.08, 0.20, 0.40])
    eg = np.array([0.0, 0.15, 0.45, 0.90, 1.60])
    efw = np.array([0.0, 0.003, 0.007, 0.012, 0.018])
    we = np.array([0.0, 500_000.0, 2_200_000.0, 5_500_000.0, 9_500_000.0])

    et = calculate_total_expansion(eo, eg, efw, m=m_true)
    f_exact = n_true * et + we

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        di = calculate_drive_indices(f_exact, n_true, eo, eg, efw, we, m=m_true)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)

    assert list(di.columns) == ["DDI", "SDI", "WDI", "CDI"]
    # Initial zero-withdrawal row must be safely masked as NaN
    assert di.iloc[0].isna().all()

    # Active timesteps sum identically to 1.0 and stay bounded in [0, 1]
    active = di.iloc[1:]
    assert not active.isna().any().any()
    np.testing.assert_allclose(active.sum(axis=1).to_numpy(), 1.0, rtol=1e-12, atol=1e-12)
    assert ((active >= 0.0) & (active <= 1.0)).all().all()

    # Hand check one active timestep (index 2)
    f2 = f_exact[2]
    assert di.loc[2, "DDI"] == pytest.approx((n_true * eo[2]) / f2, rel=1e-12)
    assert di.loc[2, "SDI"] == pytest.approx((n_true * m_true * eg[2]) / f2, rel=1e-12)
    assert di.loc[2, "WDI"] == pytest.approx(we[2] / f2, rel=1e-12)
    assert di.loc[2, "CDI"] == pytest.approx((n_true * efw[2]) / f2, rel=1e-12)

    # Normalized mode enforces sum == 1.0 even when observed F has measurement noise
    f_noisy = f_exact * np.array([1.0, 1.04, 0.97, 1.02, 0.99])
    di_norm = calculate_drive_indices(f_noisy, n_true, eo, eg, efw, we, m=m_true, normalize=True)
    assert di_norm.iloc[0].isna().all()
    np.testing.assert_allclose(di_norm.iloc[1:].sum(axis=1).to_numpy(), 1.0, rtol=1e-12)


def test_diagnostic_tables_havlena_odeh_and_campbell():
    """
    Verify Havlena-Odeh Plot 1 (Eg/Eo vs F/Eo), Plot 2 (We/Et vs F/Et),
    and Campbell (Np vs F/Et) relationships and safe masking of zero-expansion rows.
    """
    pressures = np.array([3600.0, 3600.0, 3200.0, 2800.0, 2400.0, 2000.0])
    pvt = generate_pvt_table(
        pressures=pressures,
        temp_f=195.0,
        api=34.0,
        gas_gravity=0.72,
        pi=3600.0,
        pb=3600.0,
        m=0.35,
        cw=0.0,
        cf=0.0,
        swi=0.20,
    )
    n_true = 75_000_000.0
    m_true = 0.35
    we_true = np.array([0.0, 0.0, 1.2e6, 3.1e6, 5.8e6, 9.2e6])
    et = calculate_total_expansion(pvt["Eo"], pvt["Eg"], pvt["Ef,w"], m=m_true)
    f_exact = n_true * et + we_true

    # Include 2 leading zero-expansion rows (P == Pi)
    np_stb = np.array([0.0, 0.0, 1.5e6, 3.8e6, 6.9e6, 10.4e6])
    df_in = pd.DataFrame(
        {
            "pressure_psi": pressures,
            "Np_STB": np_stb,
            "F": f_exact,
            "Bo_rb_STB": pvt["Bo"],
            "Rs_SCF_STB": pvt["Rs"],
            "Bg_rb_SCF": pvt["Bg"],
            "Eo": pvt["Eo"],
            "Eg": pvt["Eg"],
            "Ef,w": pvt["Ef,w"],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diag = compute_diagnostic_tables(df_in, N=n_true, m=m_true, We=we_true)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)

    # First two rows (Eo == 0 and Et == 0) must be masked as NaN without warnings
    for col in ("X1", "Y1", "X2", "Y2", "Y_campbell"):
        assert diag[col].iloc[:2].isna().all()

    # Havlena-Odeh Plot 2: Y2 = F / Et == N + We / Et = N + X2 (unit slope, intercept N)
    active = diag.iloc[2:]
    np.testing.assert_allclose(active["Y2"], n_true + active["X2"], rtol=1e-12)
    np.testing.assert_allclose(active["X_campbell"], np_stb[2:], rtol=1e-12)
    np.testing.assert_allclose(active["Y_campbell"], active["Y2"], rtol=1e-12)

    # When We == 0 (pure gas-cap + depletion), Plot 1 satisfies Y1 = N + (m * N) * X1
    f_no_aq = n_true * et
    df_no_aq = df_in.assign(F=f_no_aq)
    diag_gc = compute_diagnostic_tables(df_no_aq, N=n_true, m=m_true, We=0.0)
    act_gc = diag_gc.iloc[2:]
    np.testing.assert_allclose(
        act_gc["Y1"], n_true + (m_true * n_true) * act_gc["X1"], rtol=1e-12
    )
    # Campbell plot is flat at N for volumetric/gas-cap reservoir with We == 0
    np.testing.assert_allclose(act_gc["Y_campbell"], n_true, rtol=1e-12)


# ============================================================================
# 3. HISTORY-MATCHING OPTIMIZER: SYNTHETIC & BENCHMARK VALIDATION
# ============================================================================
def test_fit_mbe_volumetric_oil_benchmark():
    """
    Verify history-matching optimizer recovers OOIP (N) on volumetric_oil_benchmark.csv
    within <1% error and R^2 > 0.999, and matches exact synthetic volumetric data.
    """
    df_vol = load_benchmark_data("volumetric_oil")

    # 1. Fit without rock/water compressibility (matches analytical Havlena-Odeh OLS ~51.81 MMSTB)
    res_no_cf = fit_mbe_model(df_vol, aquifer_model="none", fit_m=False)
    assert isinstance(res_no_cf, MBEResult)
    assert res_no_cf.success is True
    assert res_no_cf.aquifer_model == "none"
    assert res_no_cf.aquifer_constant == 0.0
    assert res_no_cf.m == 0.0
    assert res_no_cf.R2 > 0.999
    assert res_no_cf.N_mmstb == pytest.approx(52.0, rel=0.01)  # within <1% of 52 MMSTB

    # Verify match against closed-form analytical OLS N = (Et . F) / (Et . Et)
    n_analytical = float(np.dot(res_no_cf.Et, res_no_cf.F_observed) / np.dot(res_no_cf.Et, res_no_cf.Et))
    assert res_no_cf.N == pytest.approx(n_analytical, rel=1e-7)
    assert res_no_cf.N_mmstb == pytest.approx(res_no_cf.N / 1e6, rel=1e-12)

    # Drive indices: pure depletion drive (DDI == 1.0) across all active timesteps
    active_di = res_no_cf.drive_indices.iloc[1:]
    np.testing.assert_allclose(active_di["DDI"], 1.0, rtol=1e-12)
    np.testing.assert_allclose(active_di["SDI"], 0.0, atol=1e-12)
    np.testing.assert_allclose(active_di["WDI"], 0.0, atol=1e-12)
    np.testing.assert_allclose(active_di["CDI"], 0.0, atol=1e-12)

    # 2. With standard rock/water compressibility (cw=3e-6, cf=4e-6, swi=0.20), recovers 50 MMSTB <0.1%
    res_with_efw = fit_mbe_model(
        df_vol, aquifer_model="none", fit_m=False, cw=3.0e-6, cf=4.0e-6, swi=0.20
    )
    assert res_with_efw.R2 > 0.999
    assert res_with_efw.N == pytest.approx(50_000_000.0, rel=0.01)
    assert (res_with_efw.drive_indices.iloc[1:]["CDI"] > 0.0).all()
    np.testing.assert_allclose(res_with_efw.drive_indices.iloc[1:].sum(axis=1), 1.0, rtol=1e-12)

    # 3. Exact synthetic volumetric benchmark generated from generate_pvt_table (N = 50 MMSTB)
    n_exact = 50_000_000.0
    pvt = generate_pvt_table(
        pressures=[4000.0, 3500.0, 3000.0, 2500.0, 2000.0, 1500.0],
        temp_f=200.0,
        api=35.0,
        gas_gravity=0.75,
        pi=4000.0,
        rsi=600.0,
    )
    et_exact = calculate_total_expansion(pvt["Eo"], pvt["Eg"], pvt["Ef,w"], m=0.0)
    f_synth = n_exact * et_exact
    df_synth = pd.DataFrame(
        {
            "pressure_psi": pvt["P"],
            "Np_STB": np.array([0.0, 6e5, 1.3e6, 2.7e6, 5.1e6, 8.6e6]),
            "F": f_synth,
            "Bo_rb_STB": pvt["Bo"],
            "Rs_SCF_STB": pvt["Rs"],
            "Bg_rb_SCF": pvt["Bg"],
            "Eo": pvt["Eo"],
            "Eg": pvt["Eg"],
            "Ef,w": pvt["Ef,w"],
        }
    )
    res_synth = fit_mbe_model(df_synth, aquifer_model="none")
    assert res_synth.N == pytest.approx(n_exact, rel=1e-7)
    assert res_synth.R2 == pytest.approx(1.0, abs=1e-12)
    assert res_synth.RMSE == pytest.approx(0.0, abs=1e-6)


def test_fit_mbe_water_drive_benchmark():
    """
    Verify history-matching optimizer accurately recovers OOIP (N) and water influx
    on water_drive_benchmark.csv with high R^2 and positive WDI.
    """
    df_wd = load_benchmark_data("water_drive")

    # 1. Schilthuis steady-state fit (no external diffusivity inputs needed)
    res_sch = fit_mbe_model(df_wd, aquifer_model="schilthuis")
    assert res_sch.success is True
    assert res_sch.aquifer_model == "schilthuis"
    assert res_sch.aquifer_constant_name == "C"
    assert res_sch.aquifer_constant_unit == "rb/(day*psi)"
    assert res_sch.N > 0.0
    assert res_sch.aquifer_constant > 0.0
    assert res_sch.R2 > 0.999
    assert (res_sch.We[1:] > 0.0).all()
    assert (res_sch.drive_indices.iloc[1:]["WDI"] > 0.0).all()
    np.testing.assert_allclose(res_sch.drive_indices.iloc[1:].sum(axis=1), 1.0, rtol=1e-12)

    # 2. Carter-Tracy unsteady-state fit
    ct_kwargs = {"k": 40.0, "phi": 0.20, "mu_w": 0.5, "ct": 7.0e-6, "re": 3000.0}
    res_ct = fit_mbe_model(df_wd, aquifer_model="carter_tracy", aquifer_kwargs=ct_kwargs)
    assert res_ct.success is True
    assert res_ct.aquifer_constant_name == "B"
    assert res_ct.aquifer_constant > 0.0
    assert res_ct.R2 > 0.999
    assert (res_ct.drive_indices.iloc[1:]["WDI"] > 0.0).all()

    # 3. Van Everdingen-Hurst finite-aquifer fit with rock/water compressibility
    # Matches benchmark ~100 MMSTB OOIP and ~70% water drive dominance
    t_days = df_wd["time_days"].to_numpy(dtype=np.float64)
    res_veh = fit_mbe_model(
        df_wd,
        aquifer_model="veh",
        aquifer_kwargs={"t_d": 0.001 * t_days, "geometry": "radial_finite", "rD": 1.5},
        cw=3.0e-6,
        cf=4.0e-6,
        swi=0.20,
    )
    assert res_veh.success is True
    assert res_veh.aquifer_constant_name == "B"
    assert res_veh.N == pytest.approx(100_000_000.0, rel=0.015)
    assert res_veh.R2 > 0.9995
    assert res_veh.drive_indices.iloc[-1]["WDI"] > 0.65
    np.testing.assert_allclose(res_veh.drive_indices.iloc[1:].sum(axis=1), 1.0, rtol=1e-12)

    # Verify dictionary-style access on MBEResult
    assert res_veh["N"] == res_veh.N
    assert res_veh["R2"] == res_veh.R2
    assert "drive_indices" in res_veh


@pytest.mark.parametrize("aquifer_model", ["pot", "schilthuis", "carter_tracy", "veh"])
def test_fit_mbe_exact_synthetic_recovery_all_models(aquifer_model):
    """
    Verify fit_mbe_model jointly recovers exact N, m (gas cap), and aquifer constant
    across all analytical aquifer models ('pot', 'schilthuis', 'carter_tracy', 'veh').
    """
    pressures = np.array([3600.0, 3400.0, 3150.0, 2850.0, 2500.0, 2100.0, 1750.0])
    time_days = np.array([0.0, 180.0, 365.0, 545.0, 730.0, 910.0, 1095.0])
    n_true = 80_000_000.0
    m_true = 0.25
    cw_val, cf_val, swi_val = 3.0e-6, 4.0e-6, 0.20

    pvt = generate_pvt_table(
        pressures=pressures,
        temp_f=195.0,
        api=34.0,
        gas_gravity=0.72,
        pi=3600.0,
        pb=3600.0,
        m=m_true,
        cw=cw_val,
        cf=cf_val,
        swi=swi_val,
    )
    aq_props = {"k": 50.0, "phi": 0.20, "mu_w": 0.5, "ct": 7.0e-6, "re": 3000.0}

    if aquifer_model == "pot":
        true_const = 25_000.0  # rb/psi
        we_true = pot_aquifer_influx(pressures, pi=3600.0, cw=1.0, cf=0.0, wi_aquifer=true_const)
        aq_kwargs = {}
    elif aquifer_model == "schilthuis":
        true_const = 18.5  # rb/(day*psi)
        we_true = schilthuis_aquifer_influx(time_days, pressures, c=true_const, pi=3600.0)
        aq_kwargs = {}
    elif aquifer_model == "carter_tracy":
        true_const = radial_aquifer_constant(phi=0.20, ct=7.0e-6, re=3000.0, h=80.0)
        we_true = carter_tracy_aquifer_influx(
            time_days, pressures, b=true_const, pi=3600.0, **aq_props
        )
        aq_kwargs = aq_props
    else:  # "veh"
        true_const = radial_aquifer_constant(phi=0.20, ct=7.0e-6, re=3000.0, h=80.0)
        we_true = veh_aquifer_influx(
            time_days, pressures, b=true_const, pi=3600.0, geometry="radial_infinite", **aq_props
        )
        aq_kwargs = {**aq_props, "geometry": "radial_infinite"}

    et_true = calculate_total_expansion(pvt["Eo"], pvt["Eg"], pvt["Ef,w"], m=m_true)
    f_exact = n_true * et_true + we_true

    # Construct production columns consistent with F_exact (Wp=0, Wi=0 -> Np = F / Bt)
    bt = pvt["Bo"] + (pvt["Rs"].iloc[0] - pvt["Rs"]) * pvt["Bg"]
    np_stb = f_exact / bt
    rp_scf_stb = np.where(np_stb > 0.0, pvt["Rs"].iloc[0], 0.0)

    df_case = pd.DataFrame(
        {
            "time_days": time_days,
            "pressure_psi": pressures,
            "Np_STB": np_stb,
            "Rp_SCF_STB": rp_scf_stb,
            "Bo_rb_STB": pvt["Bo"],
            "Rs_SCF_STB": pvt["Rs"],
            "Bg_rb_SCF": pvt["Bg"],
        }
    )

    res = fit_mbe_model(
        df_case,
        aquifer_model=aquifer_model,
        fit_m=True,
        aquifer_kwargs=aq_kwargs,
        cw=cw_val,
        cf=cf_val,
        swi=swi_val,
    )

    assert res.success is True
    assert res.N == pytest.approx(n_true, rel=1e-4)
    assert res.m == pytest.approx(m_true, rel=1e-3, abs=1e-4)
    assert res.aquifer_constant == pytest.approx(true_const, rel=1e-4)
    assert res.R2 == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(res.We, we_true, rtol=1e-4, atol=1.0)
    np.testing.assert_allclose(res.drive_indices.iloc[1:].sum(axis=1), 1.0, rtol=1e-12)


# ============================================================================
# 4. INPUT VALIDATION & ZERO-PRODUCTION INITIAL ROW SAFETY
# ============================================================================
def test_mbe_validation_and_zero_initial_rows():
    """Verify strict domain validation and safe handling of multiple zero-production initial rows."""
    # Multiple leading shut-in / initial rows (Np=0, P=Pi)
    df_vol = load_benchmark_data("volumetric_oil")
    df_multi_zero = pd.concat([df_vol.iloc[[0]], df_vol], ignore_index=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_mbe_model(df_multi_zero, aquifer_model="none")
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert res.diagnostic_table[["X1", "Y1", "X2", "Y2", "Y_campbell"]].iloc[:2].isna().all().all()
    assert res.drive_indices.iloc[:2].isna().all().all()
    np.testing.assert_allclose(res.drive_indices.iloc[2:].sum(axis=1), 1.0, rtol=1e-12)

    # Validation errors on calculate_underground_withdrawal
    with pytest.raises(ValueError):
        calculate_underground_withdrawal(-100.0, 500.0, 1.2, 500.0, 0.001)
    with pytest.raises(ValueError):
        calculate_underground_withdrawal(100.0, 500.0, 0.0, 500.0, 0.001)
    with pytest.raises(ValueError):
        calculate_underground_withdrawal(100.0, 500.0, 1.2, 500.0, 0.001, Bw=-1.0)
    with pytest.raises(ValueError):
        calculate_underground_withdrawal([100.0, 200.0], [500.0], 1.2, 500.0, 0.001)
    with pytest.raises(ValueError):
        calculate_underground_withdrawal(np.nan, 500.0, 1.2, 500.0, 0.001)

    # Validation errors on calculate_total_expansion
    with pytest.raises(ValueError):
        calculate_total_expansion(0.1, 0.5, 0.0, m=-0.1)

    # Validation errors on calculate_drive_indices & compute_diagnostic_tables
    with pytest.raises(ValueError):
        calculate_drive_indices(100.0, -1e6, 0.1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        compute_diagnostic_tables(df_vol, N=0.0)
    with pytest.raises(ValueError):
        compute_diagnostic_tables(df_vol, N=50e6, m=-0.2)
    with pytest.raises(ValueError):
        compute_diagnostic_tables(df_vol, N=50e6, We=[1.0, 2.0])

    # Validation errors on fit_mbe_model
    with pytest.raises(ValueError):
        fit_mbe_model(pd.DataFrame())
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol.iloc[[0]])
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol, aquifer_model="invalid_model")
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol, aquifer_model="carter_tracy")  # missing diffusivity/t_d
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol, aquifer_model="schilthuis", aquifer_kwargs={"bad_key": 1.0})
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol, cw=3.0e-6, cf=4.0e-6, swi=1.0)
    with pytest.raises(ValueError):
        fit_mbe_model(df_vol.assign(pressure_psi=np.nan))

    # All-zero production DataFrame must raise ValueError cleanly
    df_all_zero = pd.concat([df_vol.iloc[[0]], df_vol.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        fit_mbe_model(df_all_zero)
