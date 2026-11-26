"""
Unit tests for engine/aquifer.py (Day 4 & Day 5 Milestones).

Tests analytical aquifer influx models in oilfield units:
1. Pot Aquifer Model (Wi_aquifer geometry helper and We formula).
2. Schilthuis Steady-State Model (trapezoidal integration across time steps).
3. Carter-Tracy Unsteady-State Model (dimensionless time tD, radial pD & p'D,
   and step-by-step non-iterative We_n).
4. Van Everdingen-Hurst (VEH) Unsteady-State Model (Stehfest Laplace inversion,
   radial infinite/finite and linear infinite WeD, and Duhamel superposition).
"""

import math

import numpy as np
import pandas as pd
import pytest

from engine import (
    AQUIFER_CONSTANT_FACTOR,
    CUFT_PER_BBL,
    DIFFUSIVITY_CONVERSION_CONSTANT,
    carter_tracy_aquifer_influx,
    dimensionless_pressure,
    dimensionless_pressure_derivative,
    dimensionless_time,
    pot_aquifer_influx,
    pot_aquifer_initial_water,
    radial_aquifer_constant,
    schilthuis_aquifer_influx,
    stehfest_invert,
    stehfest_weights,
    veh_aquifer_influx,
    veh_dimensionless_influx,
)


# Standard reservoir & aquifer test properties
PI = 3500.0
RE = 3000.0        # ft
RA = 12000.0       # ft
H = 60.0           # ft
PHI = 0.20         # fraction
CW = 3.0e-6        # psi^-1
CF = 4.0e-6        # psi^-1
CT = CW + CF       # 7.0e-6 psi^-1
K = 50.0           # md
MU_W = 0.5         # cp


# ============================================================================
# 1. POT AQUIFER TESTS
# ============================================================================
def test_pot_aquifer_zero_drop_and_analytical_accuracy():
    """Verify Pot Aquifer Wi_aquifer, zero influx at P=Pi, and exact hand values."""
    wi = pot_aquifer_initial_water(ra=RA, re=RE, h=H, phi=PHI)
    expected_wi = np.pi * (RA**2 - RE**2) * H * PHI / CUFT_PER_BBL
    assert wi == pytest.approx(expected_wi, rel=1e-12)

    # Zero pressure drop -> We == 0
    assert pot_aquifer_influx(PI, PI, CW, CF, wi) == pytest.approx(0.0, abs=1e-12)

    # Hand-verifiable pressure drop of 200 psi with 180-degree encroachment
    p_curr = 3300.0
    theta = 180.0
    we = pot_aquifer_influx(p_curr, PI, CW, CF, wi, theta=theta)
    expected_we = CT * expected_wi * 200.0 * (180.0 / 360.0)
    assert isinstance(we, float)
    assert we == pytest.approx(expected_we, rel=1e-12)

    # Direct geometry call matches explicit wi_aquifer call
    we_geom = pot_aquifer_influx(
        p_curr, PI, CW, CF, theta=theta, ra=RA, re=RE, h=H, phi=PHI
    )
    assert we_geom == pytest.approx(expected_we, rel=1e-12)


def test_pot_aquifer_monotonicity():
    """Verify strictly monotonic We growth under declining pressure."""
    wi = pot_aquifer_initial_water(RA, RE, H, PHI)
    pressures = np.array([3500.0, 3450.0, 3400.0, 3300.0, 3150.0])
    we = pot_aquifer_influx(pressures, PI, CW, CF, wi)

    assert isinstance(we, np.ndarray)
    assert we[0] == pytest.approx(0.0, abs=1e-12)
    assert (np.diff(we) > 0.0).all()


# ============================================================================
# 2. SCHILTHUIS STEADY-STATE TESTS
# ============================================================================
def test_schilthuis_zero_drop_and_analytical_schedules():
    """Verify Schilthuis We=0 at P=Pi and exact trapezoidal integration on analytical schedules."""
    c_val = 25.0  # rb/(day*psi)
    t_days = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

    # Zero pressure drop across entire schedule
    p_const = np.full_like(t_days, PI)
    we_zero = schilthuis_aquifer_influx(t_days, p_const, c=c_val, pi=PI)
    np.testing.assert_allclose(we_zero, 0.0, atol=1e-12)

    # Linear pressure decline: P(t) = Pi - a*t => Pi - P(t) = a*t
    # Exact analytical integral: We(t) = C * 0.5 * a * t^2 (trapezoidal rule is exact for linear integrand)
    slope = 0.5  # psi/day
    p_linear = PI - slope * t_days
    we_linear = schilthuis_aquifer_influx(t_days, p_linear, c=c_val)
    expected_linear = c_val * 0.5 * slope * (t_days**2)
    np.testing.assert_allclose(we_linear, expected_linear, rtol=1e-12)
    assert we_linear[0] == 0.0
    assert (np.diff(we_linear) > 0.0).all()

    # Constant pressure drop below Pi starting from t0=0 with explicit Pi > P(0)
    # P(t) = 3400 psi for t in [0, 50, 150]; Pi = 3500 => We(t) = C * 100 * t
    t_step = [0.0, 50.0, 150.0]
    p_step = [3400.0, 3400.0, 3400.0]
    we_step = schilthuis_aquifer_influx(t_step, p_step, c=c_val, pi=PI)
    np.testing.assert_allclose(we_step, [0.0, c_val * 100.0 * 50.0, c_val * 100.0 * 150.0], rtol=1e-12)


# ============================================================================
# 3. CARTER-TRACY UNSTEADY-STATE TESTS
# ============================================================================
def test_dimensionless_time_and_radial_aquifer_constant():
    """Verify tD and radial aquifer constant B equations in oilfield units."""
    t = 180.0
    td = dimensionless_time(t, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE)
    expected_td = (DIFFUSIVITY_CONVERSION_CONSTANT * K * t) / (PHI * MU_W * CT * (RE**2))
    assert isinstance(td, float)
    assert td == pytest.approx(expected_td, rel=1e-12)
    assert dimensionless_time(0.0, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE) == 0.0

    # Array input
    t_arr = np.array([0.0, 180.0, 365.0])
    td_arr = dimensionless_time(t_arr, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE)
    assert isinstance(td_arr, np.ndarray)
    assert td_arr[0] == 0.0
    assert td_arr[1] == pytest.approx(expected_td, rel=1e-12)

    # Radial aquifer constant B (rb/psi)
    b_full = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=360.0)
    expected_b = AQUIFER_CONSTANT_FACTOR * PHI * CT * (RE**2) * H
    assert b_full == pytest.approx(expected_b, rel=1e-12)
    b_half = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=180.0)
    assert b_half == pytest.approx(0.5 * b_full, rel=1e-12)


def test_dimensionless_pressure_and_derivative_accuracy():
    """
    Verify radial pD(tD) and p'D(tD) across small, intermediate, and large tD:
    1. pD(0) = 0 and p'D(0) = inf.
    2. Strict monotonicity and concavity of pD across 1e-4 to 1e5.
    3. Match against published van Everdingen-Hurst radial values within 0.2%.
    4. Exact consistency between p'D(tD) and numerical derivative dpD/dtD.
    """
    assert dimensionless_pressure(0.0) == 0.0
    assert np.isinf(dimensionless_pressure_derivative(0.0))

    # Published exact radial infinite-acting constant-terminal-rate values
    veh_reference = {
        1e-4: 0.011234,
        1e-2: 0.108103,
        0.1: 0.314234,
        1.0: 0.802145,
        10.0: 1.650895,
        100.0: 2.722894,
        500.0: 3.515837,
        1000.0: 3.860591,
        1e4: 5.009985,
    }
    for td_val, expected_pd in veh_reference.items():
        pd_val = dimensionless_pressure(td_val)
        assert pd_val == pytest.approx(expected_pd, rel=2e-3)

    # Monotonicity across 9 orders of magnitude including crossover around 615.87
    td_grid = np.logspace(-4, 5, 200)
    pd_grid = dimensionless_pressure(td_grid)
    dpd_grid = dimensionless_pressure_derivative(td_grid)
    assert (np.diff(pd_grid) > 0.0).all()
    assert (dpd_grid > 0.0).all()
    assert (pd_grid - td_grid * dpd_grid > 0.0).all()

    # Check derivative accuracy via central finite difference away from crossover
    for td_test in [1e-3, 0.05, 1.0, 50.0, 300.0, 1000.0, 5000.0]:
        eps = td_test * 1e-6
        fd = (dimensionless_pressure(td_test + eps) - dimensionless_pressure(td_test - eps)) / (2.0 * eps)
        analytic = dimensionless_pressure_derivative(td_test)
        assert analytic == pytest.approx(fd, rel=1e-6)


def test_carter_tracy_zero_drop_monotonicity_and_linearity():
    """
    Verify Carter-Tracy physical consistency:
    1. We = 0 when reservoir pressure equals Pi.
    2. Strictly monotonic growth under declining pressure.
    3. Exact linear scaling with aquifer constant B.
    4. Agreement between diffusivity-derived and precomputed tD calls.
    """
    t_days = np.array([0.0, 90.0, 180.0, 365.0, 545.0, 730.0, 1095.0])
    b_val = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=360.0)

    # Zero pressure drop
    p_flat = np.full_like(t_days, PI)
    we_zero = carter_tracy_aquifer_influx(
        t_days, p_flat, b=b_val, pi=PI, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    np.testing.assert_allclose(we_zero, 0.0, atol=1e-12)

    # Declining pressure schedule
    p_declining = np.array([3500.0, 3470.0, 3435.0, 3380.0, 3320.0, 3250.0, 3120.0])
    we_b1 = carter_tracy_aquifer_influx(
        t_days, p_declining, b=b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    assert we_b1[0] == pytest.approx(0.0, abs=1e-12)
    assert (we_b1[1:] > 0.0).all()
    assert (np.diff(we_b1) > 0.0).all()

    # Exact linear scaling with B
    scale = 3.5
    we_scaled = carter_tracy_aquifer_influx(
        t_days, p_declining, b=scale * b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    np.testing.assert_allclose(we_scaled, scale * we_b1, rtol=1e-12)

    # Precomputed tD yields identical result
    td_pre = dimensionless_time(t_days, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE)
    we_td = carter_tracy_aquifer_influx(t_days, p_declining, b=b_val, t_d=td_pre)
    np.testing.assert_allclose(we_td, we_b1, rtol=1e-12)

    # Zero B yields zero influx
    we_zero_b = carter_tracy_aquifer_influx(t_days, p_declining, b=0.0, t_d=td_pre)
    np.testing.assert_allclose(we_zero_b, 0.0, atol=1e-12)


def test_carter_tracy_hand_verified_first_step():
    """Verify Carter-Tracy first-step formula against manual hand calculation."""
    t_days = [0.0, 100.0, 200.0]
    pressures = [3500.0, 3400.0, 3300.0]
    td = [0.0, 1.0, 2.0]
    b_val = 500.0

    we = carter_tracy_aquifer_influx(t_days, pressures, b=b_val, pi=3500.0, t_d=td)

    pd1 = dimensionless_pressure(1.0)
    expected_we1 = (b_val * 100.0 / pd1) * 1.0
    assert we[1] == pytest.approx(expected_we1, rel=1e-12)

    pd2 = dimensionless_pressure(2.0)
    dpd2 = dimensionless_pressure_derivative(2.0)
    expected_we2 = expected_we1 + ((b_val * 200.0 - expected_we1 * dpd2) / (pd2 - 1.0 * dpd2)) * (2.0 - 1.0)
    assert we[2] == pytest.approx(expected_we2, rel=1e-12)


def test_carter_tracy_handles_duplicate_time_and_series_inputs():
    """Verify Carter-Tracy handles duplicate timestamps (dt=0) and pandas Series cleanly."""
    t_series = pd.Series([0.0, 100.0, 100.0, 200.0])
    p_series = pd.Series([3500.0, 3450.0, 3450.0, 3380.0])
    we = carter_tracy_aquifer_influx(
        t_series, p_series, b=100.0, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    assert we[1] == pytest.approx(we[2], rel=1e-12)
    assert we[3] > we[2]


# ============================================================================
# 4. INPUT VALIDATION & ERROR HANDLING TESTS
# ============================================================================
def test_aquifer_validation_errors():
    """Verify invalid physical parameters and time histories raise ValueError."""
    # Pot aquifer invalid geometry / parameters
    with pytest.raises(ValueError):
        pot_aquifer_initial_water(ra=2000.0, re=3000.0, h=50.0, phi=0.2)
    with pytest.raises(ValueError):
        pot_aquifer_initial_water(ra=5000.0, re=3000.0, h=-10.0, phi=0.2)
    with pytest.raises(ValueError):
        pot_aquifer_initial_water(ra=5000.0, re=3000.0, h=50.0, phi=1.2)
    with pytest.raises(ValueError):
        pot_aquifer_influx(3400.0, PI, CW, CF, wi_aquifer=1e6, theta=400.0)
    with pytest.raises(ValueError):
        pot_aquifer_influx(3400.0, PI, CW, CF, wi_aquifer=None)

    # Schilthuis invalid inputs
    with pytest.raises(ValueError):
        schilthuis_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], c=-5.0)
    with pytest.raises(ValueError):
        schilthuis_aquifer_influx([0.0, 20.0, 10.0], [3500.0, 3400.0, 3300.0], c=10.0)
    with pytest.raises(ValueError):
        schilthuis_aquifer_influx([0.0, 10.0], [3500.0], c=10.0)

    # Carter-Tracy / dimensionless helpers invalid inputs
    with pytest.raises(ValueError):
        dimensionless_time(-1.0, K, PHI, MU_W, CT, RE)
    with pytest.raises(ValueError):
        dimensionless_time(10.0, K, PHI, mu_w=0.0, ct=CT, re=RE)
    with pytest.raises(ValueError):
        dimensionless_pressure(-0.5)
    with pytest.raises(ValueError):
        dimensionless_pressure_derivative(-0.5)
    with pytest.raises(ValueError):
        carter_tracy_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], b=100.0)
    with pytest.raises(ValueError):
        carter_tracy_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], b=-10.0, t_d=[0.0, 1.0])
    with pytest.raises(ValueError):
        carter_tracy_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], b=100.0, t_d=[0.0, 2.0, 3.0])

    # VEH Stehfest / dimensionless / superposition invalid inputs
    for bad_n in (0, -2, 11, 12.0, True):
        with pytest.raises(ValueError):
            stehfest_weights(bad_n)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        stehfest_invert("not_callable", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stehfest_invert(lambda s: 1.0 / s, 0.0)
    with pytest.raises(ValueError):
        stehfest_invert(lambda s: 1.0 / s, -1.0)
    with pytest.raises(ValueError):
        veh_dimensionless_influx(-0.1)
    with pytest.raises(ValueError):
        veh_dimensionless_influx(1.0, geometry="spherical")
    with pytest.raises(ValueError):
        veh_dimensionless_influx(1.0, geometry="radial_finite", rD=None)
    with pytest.raises(ValueError):
        veh_dimensionless_influx(1.0, geometry="radial_finite", rD=1.0)
    with pytest.raises(ValueError):
        veh_dimensionless_influx(1.0, geometry="radial_finite", rD=0.8)
    with pytest.raises(ValueError):
        veh_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], b=100.0)
    with pytest.raises(ValueError):
        veh_aquifer_influx([0.0, 10.0], [3500.0, 3400.0], b=-5.0, t_d=[0.0, 1.0])
    with pytest.raises(ValueError):
        veh_aquifer_influx(
            [0.0, 10.0],
            [3500.0, 3400.0],
            b=100.0,
            t_d=[0.0, 1.0],
            geometry="radial_finite",
            rD=None,
        )


# ============================================================================
# 5. VAN EVERDINGEN-HURST (VEH) UNSTEADY-STATE TESTS
# ============================================================================
def test_stehfest_weights_and_known_laplace_inversions():
    """Verify Stehfest weights and inversion accuracy on known analytical transforms."""
    weights = stehfest_weights(12)
    assert weights.shape == (12,)
    assert weights.dtype == np.float64
    assert np.sum(weights) == pytest.approx(0.0, abs=1e-6)

    # 1 / s -> 1.0 (scalar and array)
    assert isinstance(stehfest_invert(lambda s: 1.0 / s, 1.0), float)
    assert stehfest_invert(lambda s: 1.0 / s, 1.0) == pytest.approx(1.0, rel=1e-8)
    t_grid = np.array([0.1, 1.0, 5.0, 25.0])
    np.testing.assert_allclose(
        stehfest_invert(lambda s: 1.0 / s, t_grid),
        np.ones_like(t_grid),
        rtol=1e-8,
    )

    # 1 / (s + a) -> exp(-a * t)
    a = 0.5
    t_exp = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    inv_exp = stehfest_invert(lambda s: 1.0 / (s + a), t_exp)
    np.testing.assert_allclose(inv_exp[:3], np.exp(-a * t_exp[:3]), rtol=1e-5)
    np.testing.assert_allclose(inv_exp, np.exp(-a * t_exp), rtol=1e-3)

    # Scalar-only callable (using Python math) works identically to vectorized callable
    inv_scalar_fn = stehfest_invert(lambda s: 1.0 / (s + a), [0.5, 2.0])
    np.testing.assert_allclose(inv_scalar_fn, np.exp(-a * np.array([0.5, 2.0])), rtol=1e-4)

    # 1 / s^2 -> t and 1 / s^(3/2) -> 2 * sqrt(t / pi)
    np.testing.assert_allclose(
        stehfest_invert(lambda s: 1.0 / (s**2), t_grid),
        t_grid,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        stehfest_invert(lambda s: 1.0 / (s**1.5), t_grid),
        (2.0 / math.sqrt(math.pi)) * np.sqrt(t_grid),
        rtol=1e-5,
    )


def test_veh_dimensionless_influx_radial_infinite():
    """
    Verify radial_infinite WeD(tD) against exact Fourier-Bessel values and
    published Van Everdingen-Hurst / Dake reference tables without overflow.
    """
    assert veh_dimensionless_influx(0.0, geometry="radial_infinite") == 0.0

    # Exact VEH Fourier-Bessel analytical values for radial infinite aquifer
    exact_veh = {
        0.01: 0.117721,
        0.1: 0.404310,
        1.0: 1.568263,
        10.0: 7.401495,
        100.0: 43.025102,
        1000.0: 292.611277,
        1e4: 2198.347055,
    }
    td_vals = np.array(list(exact_veh.keys()))
    wed_vals = veh_dimensionless_influx(td_vals, geometry="radial_infinite")
    assert isinstance(wed_vals, np.ndarray)
    np.testing.assert_allclose(wed_vals, np.array(list(exact_veh.values())), rtol=1e-3)

    # Published textbook rounded VEH table points (tD = 1, 10, 100)
    assert veh_dimensionless_influx(1.0) == pytest.approx(1.569, rel=2e-3)
    assert veh_dimensionless_influx(10.0) == pytest.approx(7.411, rel=2e-3)
    assert veh_dimensionless_influx(100.0) == pytest.approx(43.01, rel=2e-3)

    # Extreme tD range (large s at small tD -> no Bessel overflow/NaN; strict monotonicity)
    extreme_td = np.logspace(-8, 6, 30)
    extreme_wed = veh_dimensionless_influx(extreme_td, geometry="radial_infinite")
    assert np.all(np.isfinite(extreme_wed))
    assert np.all(extreme_wed > 0.0)
    assert (np.diff(extreme_wed) > 0.0).all()


def test_veh_dimensionless_influx_radial_finite_and_linear():
    """
    Verify radial_finite WeD(tD) converges to (rD^2 - 1)/2 at large tD, matches
    infinite-acting at early time, and linear_infinite matches (2/sqrt(pi))*sqrt(tD).
    """
    assert veh_dimensionless_influx(0.0, geometry="radial_finite", rD=5.0) == 0.0
    assert veh_dimensionless_influx(0.0, geometry="linear_infinite") == 0.0

    # Linear infinite exact analytical solution
    td_lin = np.array([0.0, 0.04, 1.0, 9.0, 100.0])
    wed_lin = veh_dimensionless_influx(td_lin, geometry="linear_infinite")
    expected_lin = (2.0 / math.sqrt(math.pi)) * np.sqrt(td_lin)
    np.testing.assert_allclose(wed_lin, expected_lin, rtol=1e-12)

    # Radial finite matches exact analytical Fourier-Bessel series for rD = 2.0
    wed_rd2 = veh_dimensionless_influx([0.1, 1.0, 10.0], geometry="radial_finite", rD=2.0)
    np.testing.assert_allclose(wed_rd2, [0.404337, 1.295106, 1.500000], rtol=1e-4)

    # Steady-state capacity limit (rD^2 - 1)/2 at large tD across multiple rD ratios
    for rd in (1.5, 2.0, 5.0, 10.0):
        steady_limit = (rd**2 - 1.0) / 2.0
        wed_early = veh_dimensionless_influx(1e-3, geometry="radial_finite", rD=rd)
        wed_inf_early = veh_dimensionless_influx(1e-3, geometry="radial_infinite")
        assert wed_early == pytest.approx(wed_inf_early, rel=1e-4)

        wed_grid = veh_dimensionless_influx(
            np.logspace(-2, 5, 15), geometry="radial_finite", rD=rd
        )
        assert np.all(np.isfinite(wed_grid))
        # Bounded by steady-state limit (within ~1e-4 Stehfest plateau overshoot)
        assert np.all(wed_grid <= steady_limit * (1.0 + 1e-4))
        assert (np.diff(wed_grid) >= -1e-4 * steady_limit).all()
        assert wed_grid[-1] == pytest.approx(steady_limit, rel=1e-4)


def test_veh_superposition_physical_properties_and_hand_calculation():
    """
    Verify VEH Duhamel superposition We=0 at constant pressure, monotonicity under
    declining pressure, linear scaling with B, and exact midpoint formula.
    """
    t_days = np.array([0.0, 90.0, 180.0, 365.0, 545.0, 730.0])
    b_val = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=360.0)

    # Constant pressure -> We == 0
    p_flat = np.full_like(t_days, PI)
    we_zero = veh_aquifer_influx(
        t_days, p_flat, b=b_val, pi=PI, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    np.testing.assert_allclose(we_zero, 0.0, atol=1e-12)

    # Monotonic growth under declining pressure
    p_dec = np.array([3500.0, 3470.0, 3435.0, 3380.0, 3320.0, 3250.0])
    we_rad = veh_aquifer_influx(
        t_days, p_dec, b=b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    assert we_rad[0] == pytest.approx(0.0, abs=1e-12)
    assert (we_rad[1:] > 0.0).all()
    assert (np.diff(we_rad) > 0.0).all()

    # Exact linear scaling with B and zero B -> zero influx
    we_scaled = veh_aquifer_influx(
        t_days, p_dec, b=2.5 * b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    np.testing.assert_allclose(we_scaled, 2.5 * we_rad, rtol=1e-12)
    we_zero_b = veh_aquifer_influx(
        t_days, p_dec, b=0.0, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )
    np.testing.assert_allclose(we_zero_b, 0.0, atol=1e-12)

    # Precomputed t_d matches diffusivity-derived t_d; ra computes rD = ra / re
    td_pre = dimensionless_time(t_days, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE)
    we_pre = veh_aquifer_influx(t_days, p_dec, b=b_val, t_d=td_pre)
    np.testing.assert_allclose(we_pre, we_rad, rtol=1e-12)

    we_fin_ra = veh_aquifer_influx(
        t_days, p_dec, b=b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE,
        geometry="radial_finite", ra=RA,
    )
    we_fin_rd = veh_aquifer_influx(
        t_days, p_dec, b=b_val, t_d=td_pre, geometry="radial_finite", rD=RA / RE
    )
    np.testing.assert_allclose(we_fin_ra, we_fin_rd, rtol=1e-12)
    # Finite aquifer yields less influx than infinite aquifer at late time
    assert we_fin_rd[-1] <= we_rad[-1]

    # Exact midpoint superposition hand check on 4 points
    t_hand = [0.0, 100.0, 200.0, 300.0]
    p_hand = [3500.0, 3460.0, 3400.0, 3320.0]
    td_hand = [0.0, 1.0, 2.0, 3.0]
    b_hand = 200.0
    we_hand = veh_aquifer_influx(t_hand, p_hand, b=b_hand, pi=3500.0, t_d=td_hand)

    dp0 = (3500.0 - 3460.0) / 2.0
    dp1 = (3500.0 - 3400.0) / 2.0
    dp2 = (3460.0 - 3320.0) / 2.0
    expected_we3 = b_hand * (
        dp0 * veh_dimensionless_influx(3.0)
        + dp1 * veh_dimensionless_influx(2.0)
        + dp2 * veh_dimensionless_influx(1.0)
    )
    assert we_hand[3] == pytest.approx(expected_we3, rel=1e-9)


def test_veh_agrees_with_carter_tracy_and_pot_asymptote():
    """
    Verify VEH radial_infinite agrees closely with Carter-Tracy on a smooth pressure
    decline, and radial_finite step-drop converges to the Pot Aquifer capacity.
    """
    b_val = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=360.0)

    # Smooth monthly pressure decline over 5 years
    t_days = np.linspace(0.0, 1825.0, 61)
    p_smooth = PI - 500.0 * (1.0 - np.exp(-t_days / 900.0))

    we_veh = veh_aquifer_influx(
        t_days, p_smooth, b=b_val, pi=PI, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE,
        geometry="radial_infinite",
    )
    we_ct = carter_tracy_aquifer_influx(
        t_days, p_smooth, b=b_val, pi=PI, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE
    )

    # After early-time step-discretization transient, VEH and Carter-Tracy agree within 5%
    np.testing.assert_allclose(we_veh[20:], we_ct[20:], rtol=0.05)

    # Finite radial aquifer under held step drop converges to Pot Aquifer capacity
    ra_small = 2.0 * RE
    b_small = radial_aquifer_constant(phi=PHI, ct=CT, re=RE, h=H, theta=360.0)
    we_pot_limit = pot_aquifer_influx(
        3300.0, PI, CW, CF, ra=ra_small, re=RE, h=H, phi=PHI, theta=360.0
    )
    t_long = np.array([0.0, 1e4, 1e5, 1e6])
    p_step = np.array([3300.0, 3300.0, 3300.0, 3300.0])
    we_veh_fin = veh_aquifer_influx(
        t_long, p_step, b=b_small, pi=PI, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE,
        geometry="radial_finite", ra=ra_small,
    )
    assert we_veh_fin[-1] == pytest.approx(we_pot_limit, rel=1e-4)

    # Linear infinite influx scales properly and handles >512 chunked steps cleanly
    t_chunk = np.linspace(0.0, 5200.0, 520)
    p_chunk = PI - 0.1 * t_chunk
    we_lin = veh_aquifer_influx(
        pd.Series(t_chunk), pd.Series(p_chunk), b=b_val, k=K, phi=PHI, mu_w=MU_W, ct=CT, re=RE,
        geometry="linear_infinite",
    )
    assert we_lin.shape == (520,)
    assert we_lin[0] == 0.0
    assert (np.diff(we_lin) > 0.0).all()
