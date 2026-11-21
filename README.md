# Material Balance Equation (MBE) & Analytical Aquifer Influx Modeling for Reservoir Drive Diagnostics

## Problem Statement
Accurate determination of original oil in place (OOIP / $N$), identification of dominant reservoir drive mechanisms, and quantitative modeling of water encroachment from contiguous aquifers are critical challenges in reservoir engineering. This analytical suite implements Havlena-Odeh material balance diagnostics, unsteady-state aquifer influx models (Pot, Schilthuis, Van Everdingen-Hurst, Carter-Tracy), dynamic drive index partitioning, and future reservoir pressure depletion forecasting.

## Governing MBE Equations

### General Material Balance Equation
$$F = N \left[ E_o + m E_g + E_{f,w} \right] + W_e$$

Where:
- $F$: Total underground withdrawal (rb)
- $N$: Initial oil in place (STB)
- $m$: Ratio of initial gas-cap hydrocarbon pore volume to initial oil hydrocarbon pore volume (dimensionless)
- $E_o$: Unit expansion of oil and dissolved gas (rb/STB)
- $E_g$: Unit expansion of gas-cap gas (rb/STB)
- $E_{f,w}$: Unit expansion of connate water and pore volume reduction (rb/STB)
- $W_e$: Cumulative aquifer water influx (rb)

### Dynamic Drive Indices
$$\text{DDI} + \text{SDI} + \text{WDI} + \text{CDI} = 1.0$$

Where:
- $\text{DDI}$: Depletion Drive Index (solution gas drive)
- $\text{SDI}$: Segregation Drive Index (gas-cap expansion)
- $\text{WDI}$: Water Drive Index (aquifer influx and water injection)
- $\text{CDI}$: Compaction Drive Index (formation rock and connate water expansion)

## Repository Structure
```
mbe-aquifer-diagnostics/
├── data/
│   ├── __init__.py
│   ├── samples/
│   │   └── __init__.py
│   └── loader.py              # Ingestion & unit conversion
├── engine/
│   ├── __init__.py
│   ├── pvt.py                 # Black-oil PVT correlations & expansion terms
│   ├── aquifer.py             # Analytical water influx solvers
│   ├── mbe.py                 # General MBE engine, Drive Indices, optimizer
│   └── forecast.py            # Reservoir pressure depletion forecaster
├── utils/
│   ├── __init__.py
│   ├── export.py              # Automated Excel report generator
│   └── plots.py               # Diagnostic plotting module
├── tests/
│   └── __init__.py            # Pytest validation test suite
├── app.py                     # Interactive Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```
