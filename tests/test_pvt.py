"""
Unit tests for engine/pvt.py (Day 3 Milestone).

Tests Black-Oil PVT correlations and Material Balance expansion terms:
1. Oil specific gravity conversion.
2. Standing's bubble point pressure (Pb) and solution GOR (Rs) capping.
3. Standing's oil formation volume factor (Bo) and compressibility adjustment.
4. Sutton pseudo-critical properties and Z-factor (Papay and Hall-Yarborough).
5. Gas formation volume factor (Bg).
6. Two-phase formation volume factor (Bt).
7. MBE expansion terms (Eo, Eg, Ef,w) and initial conditions (Eo=Eg=Ef,w=0 at Pi).
8. Physical monotonicity across pressure depletion.
9. generate_pvt_table helper function.
"""

import pytest
import numpy as np
import pandas as pd

from engine.pvt import (
    BG_CONVERSION_CONSTANT,
    expansion_gas,
    expansion_oil,
    expansion_rock_water,
    gas_fvf,
    gas_z_factor,
    gas_z_factor_hall_yarborough,
    gas_z_factor_papay,
    generate_pvt_table,
    oil_specific_gravity,
    standing_bubble_point_pressure,
    standing_oil_fvf,
    standing_solution_gor,
    sutton_pseudocritical_properties,
    two_phase_fvf,
    vasquez_beggs_oil_compressibility,
)


# Standard reservoir oil & gas test parameters (from Ahmed / McCain textbook benchmarks)
TEMP_F = 200.0          # Reservoir temperature (°F)
API = 35.0              # Oil gravity (°API)
GAS_GRAVITY = 0.75      # Separator gas gravity (air = 1.0)
RSI = 600.0             # Initial / bubble-point solution GOR (SCF/STB)
PI = 4000.0             # Initial reservoir pressure (psi)


def test_oil_specific_gravity():
    """Verify specific gravity calculation from API gravity for scalar and array."""
    # Water has API = 10 -> gamma_o = 1.0
    assert oil_specific_gravity(10.0) == pytest.approx(1.0, rel=1e-5)

    # 35 API oil
    expected_gamma = 141.5 / (131.5 + 35.0)  # ~0.84985
    assert oil_specific_gravity(35.0) == pytest.approx(expected_gamma, rel=1e-5)

    # Vectorized array input
    apis = np.array([10.0, 35.0, 50.0])
    gammas = oil_specific_gravity(apis)
    assert isinstance(gammas, np.ndarray)
    assert len(gammas) == 3
    assert gammas[0] == pytest.approx(1.0, rel=1e-5)
    assert gammas[1] == pytest.approx(expected_gamma, rel=1e-5)


def test_standing_bubble_point_and_gor():
    """Verify Standing's Pb and Rs calculations and inverse consistency."""
    # At RSI = 600 SCF/STB, T = 200 F, API = 35, gamma_g = 0.75
    pb = standing_bubble_point_pressure(RSI, TEMP_F, API, GAS_GRAVITY)
    assert isinstance(pb, float)
    # Known textbook value: ~2569.5 psi
    assert 2500.0 < pb < 2650.0
    assert pb == pytest.approx(2569.51, rel=1e-3)

    # Invert at Pb: Rs should recover RSI exactly
    rs_at_pb = standing_solution_gor(pb, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI)
    assert rs_at_pb == pytest.approx(RSI, rel=1e-4)

    # Undersaturated regime (P > Pb): Rs must be strictly capped at RSI
    p_above = np.array([2600.0, 3000.0, 3500.0, 4000.0])
    rs_above = standing_solution_gor(p_above, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI)
    assert (rs_above == RSI).all(), "Rs must be capped at RSI when P >= Pb"

    # Saturated regime (P < Pb): Rs must decrease monotonically with pressure
    p_below = np.array([2500.0, 2000.0, 1500.0, 1000.0, 500.0])
    rs_below = standing_solution_gor(p_below, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI)
    assert (np.diff(rs_below) < 0).all(), "Rs must strictly decrease as pressure drops below Pb"
    assert (rs_below < RSI).all()

    # Zero pressure gives 0 GOR
    assert standing_solution_gor(0.0, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI) == 0.0


def test_standing_oil_fvf_peak_at_bubble_point():
    """
    Verify Standing's Bo behavior:
    1. Bo peaks at the bubble point pressure Pb.
    2. Above Pb, Bo decreases as pressure rises (compression).
    3. Below Pb, Bo decreases as pressure drops (gas liberation).
    """
    pb = float(standing_bubble_point_pressure(RSI, TEMP_F, API, GAS_GRAVITY))
    bob = standing_oil_fvf(pb, pb, RSI, TEMP_F, API, GAS_GRAVITY)
    # Textbook value for Bo at Pb is ~1.3489 rb/STB
    assert 1.30 < bob < 1.40
    assert bob == pytest.approx(1.3489, rel=1e-3)

    # Above Pb (P > Pb)
    p_undersat = np.array([pb + 200.0, pb + 500.0, pb + 1000.0])
    bo_undersat = standing_oil_fvf(p_undersat, pb, RSI, TEMP_F, API, GAS_GRAVITY)
    # As pressure increases above Pb, Bo decreases monotonically due to compressibility
    assert (np.diff(bo_undersat) < 0).all()
    assert (bo_undersat < bob).all(), "Bo above Pb must be less than Bob"

    # Below Pb (P < Pb)
    p_sat = np.array([pb - 200.0, pb - 500.0, pb - 1000.0])
    bo_sat = standing_oil_fvf(p_sat, pb, RSI, TEMP_F, API, GAS_GRAVITY)
    # As pressure decreases below Pb, Bo decreases monotonically
    assert (np.diff(bo_sat) < 0).all()
    assert (bo_sat < bob).all(), "Bo below Pb must be less than Bob"


def test_sutton_pseudocritical_properties():
    """Verify Sutton pseudo-critical correlation against published values."""
    ppc, tpc = sutton_pseudocritical_properties(0.75)
    # Sutton: Tpc = 169.2 + 349.5(0.75) - 74(0.75^2) = 389.70 °R
    # Ppc = 756.8 - 131.0(0.75) - 3.6(0.75^2) = 656.52 psi
    assert tpc == pytest.approx(389.70, rel=1e-4)
    assert ppc == pytest.approx(656.52, rel=1e-4)

    # Vectorized call
    ppcs, tpcs = sutton_pseudocritical_properties(np.array([0.65, 0.75]))
    assert len(ppcs) == 2
    assert len(tpcs) == 2


def test_gas_z_factor_ideal_gas_limit_and_correlations():
    """Verify Z-factor correlations approach 1.0 at low pressure and match expected values."""
    # Low pressure limit -> Z approaches 1.0
    z_papay_low = gas_z_factor(0.1, TEMP_F, GAS_GRAVITY, method="papay")
    z_hy_low = gas_z_factor(0.1, TEMP_F, GAS_GRAVITY, method="hall_yarborough")
    assert z_papay_low == pytest.approx(1.0, abs=0.01)
    assert z_hy_low == pytest.approx(1.0, abs=0.01)

    # At reservoir conditions P = 2570 psi, T = 200 F, gamma_g = 0.75
    z_p = gas_z_factor(2570.0, TEMP_F, GAS_GRAVITY, method="papay")
    z_hy = gas_z_factor(2570.0, TEMP_F, GAS_GRAVITY, method="hall_yarborough")
    assert 0.80 < z_p < 0.95
    assert 0.80 < z_hy < 0.95
    # Two correlations typically agree within 2%
    assert abs(z_p - z_hy) / z_hy < 0.03


def test_gas_fvf_monotonic_expansion():
    """Verify gas formation volume factor (Bg) strictly increases as pressure drops."""
    pressures = np.array([4000.0, 3000.0, 2000.0, 1000.0, 500.0])
    bg = gas_fvf(pressures, TEMP_F, gas_gravity=GAS_GRAVITY)

    # Monotonicity: as pressure decreases, Bg must strictly increase
    assert (np.diff(bg) > 0).all(), "Bg must strictly increase as pressure depletes"
    # Expected order of magnitude: ~0.0008 to ~0.007 rb/SCF
    assert 0.0005 < bg[0] < 0.0015
    assert bg[-1] > bg[0]

    # Constant validation: ~0.005038 rb*psi/(SCF*R)
    assert BG_CONVERSION_CONSTANT == pytest.approx(0.005038, rel=1e-3)

    # Zero or negative pressure raises ValueError
    with pytest.raises(ValueError):
        gas_fvf(0.0, TEMP_F, gas_gravity=GAS_GRAVITY)


def test_two_phase_fvf():
    """Verify two-phase formation volume factor Bt = Bo + (Rsi - Rs) * Bg."""
    pb = float(standing_bubble_point_pressure(RSI, TEMP_F, API, GAS_GRAVITY))
    pressures = np.array([3500.0, pb, 2000.0, 1500.0])

    rs = standing_solution_gor(pressures, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI)
    bo = standing_oil_fvf(pressures, pb, RSI, TEMP_F, API, GAS_GRAVITY, rs=rs)
    bg = gas_fvf(pressures, TEMP_F, gas_gravity=GAS_GRAVITY)
    bt = two_phase_fvf(bo, bg, rs, RSI)

    # At and above Pb: Rs == RSI -> Bt must equal Bo
    assert bt[0] == pytest.approx(bo[0], rel=1e-6)
    assert bt[1] == pytest.approx(bo[1], rel=1e-6)

    # Below Pb: liberated gas causes Bt > Bo
    assert bt[2] > bo[2]
    assert bt[3] > bo[3]

    # Bt increases dramatically as pressure drops below Pb
    assert bt[3] > bt[2]


def test_mbe_expansion_terms():
    """
    Verify Material Balance Equation expansion terms:
    1. At Pi, Eo = 0, Eg = 0, Ef,w = 0.
    2. Below Pi, all expansion terms are strictly positive and monotonically increasing.
    """
    pb = float(standing_bubble_point_pressure(RSI, TEMP_F, API, GAS_GRAVITY))
    boi = float(standing_oil_fvf(PI, pb, RSI, TEMP_F, API, GAS_GRAVITY))
    bgi = float(gas_fvf(PI, TEMP_F, gas_gravity=GAS_GRAVITY))

    # At initial conditions Pi
    eo_i = expansion_oil(boi, bgi, RSI, boi, RSI)
    eg_i = expansion_gas(bgi, bgi, boi)
    efw_i = expansion_rock_water(PI, PI, boi, cw=3.0e-6, cf=4.0e-6, swi=0.20, m=0.0)

    assert eo_i == pytest.approx(0.0, abs=1e-10)
    assert eg_i == pytest.approx(0.0, abs=1e-10)
    assert efw_i == pytest.approx(0.0, abs=1e-10)

    # Depleting pressure series
    p_depl = np.array([3500.0, 3000.0, 2500.0, 2000.0, 1500.0])
    rs_depl = standing_solution_gor(p_depl, TEMP_F, API, GAS_GRAVITY, pb=pb, rsi=RSI)
    bo_depl = standing_oil_fvf(p_depl, pb, RSI, TEMP_F, API, GAS_GRAVITY, rs=rs_depl)
    bg_depl = gas_fvf(p_depl, TEMP_F, gas_gravity=GAS_GRAVITY)

    eo = expansion_oil(bo_depl, bg_depl, rs_depl, boi, RSI)
    eg = expansion_gas(bg_depl, bgi, boi)
    efw = expansion_rock_water(p_depl, PI, boi, cw=3.0e-6, cf=4.0e-6, swi=0.20, m=0.2)

    # All terms must be positive and strictly increasing as pressure depletes
    assert (eo > 0).all()
    assert (eg > 0).all()
    assert (efw > 0).all()

    assert (np.diff(eo) > 0).all(), "Eo must strictly increase with pressure drop"
    assert (np.diff(eg) > 0).all(), "Eg must strictly increase with pressure drop"
    assert (np.diff(efw) > 0).all(), "Ef,w must strictly increase with pressure drop"


def test_generate_pvt_table():
    """Verify generate_pvt_table returns valid DataFrame with all required columns and properties."""
    pressures = [4000.0, 3500.0, 3000.0, 2500.0, 2000.0, 1500.0]
    df = generate_pvt_table(
        pressures=pressures,
        temp_f=TEMP_F,
        api=API,
        gas_gravity=GAS_GRAVITY,
        pi=PI,
        rsi=RSI,
        m=0.1,
        cw=3.0e-6,
        cf=4.0e-6,
        swi=0.22,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(pressures)

    expected_cols = ["P", "Rs", "Bo", "Bg", "Bt", "Eo", "Eg", "Ef,w"]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column '{col}'"

    # Row at initial pressure Pi (4000 psi)
    row_pi = df[df["P"] == PI].iloc[0]
    assert row_pi["Eo"] == pytest.approx(0.0, abs=1e-8)
    assert row_pi["Eg"] == pytest.approx(0.0, abs=1e-8)
    assert row_pi["Ef,w"] == pytest.approx(0.0, abs=1e-8)
    assert row_pi["Rs"] == pytest.approx(RSI, rel=1e-5)

    # Monotonicity of expansion terms
    df_sorted = df.sort_values("P", ascending=False)
    assert (np.diff(df_sorted["Eo"]) >= -1e-8).all()
    assert (np.diff(df_sorted["Eg"]) >= -1e-8).all()
    assert (np.diff(df_sorted["Ef,w"]) >= -1e-8).all()


def test_pvt_error_handling():
    """Verify error handling for invalid physical parameters."""
    # Invalid Swi (>= 1.0 or < 0.0)
    with pytest.raises(ValueError):
        expansion_rock_water(3000.0, 4000.0, 1.3, 3e-6, 4e-6, swi=1.0)

    with pytest.raises(ValueError):
        expansion_rock_water(3000.0, 4000.0, 1.3, 3e-6, 4e-6, swi=-0.1)

    # Missing both pb and rsi in standing_solution_gor
    with pytest.raises(ValueError):
        standing_solution_gor(3000.0, TEMP_F, API, GAS_GRAVITY, pb=None, rsi=None)

    # Unknown Z-factor method
    with pytest.raises(ValueError):
        gas_z_factor(3000.0, TEMP_F, GAS_GRAVITY, method="invalid_method_name")
