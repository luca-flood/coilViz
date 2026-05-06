# coilViz

Physics-informed simulation and analysis toolkit for **wireless power transfer (WPT)** under **coil curvature and misalignment** conditions.

This project generates large WPT datasets from a configurable simulation model, then runs a full analysis pipeline to produce figures, lookup tables (LUTs), and statistical reports.

---

## What this project does

- Simulates resonant WPT behavior at 6.78 MHz for curved receiver coils.
- Sweeps curvature, lateral offsets, and angular misalignment across a large pose grid.
- Compares multiple physics ablations (e.g., skin effect, proximity effect, parasitic capacitance, curvature geometry).
- Exports CSV datasets for downstream analysis and experiment planning.
- Produces publication-style plots, LUTs, and statistical summaries.

---

## Repository structure

```text
coil_viz/
├── data/
│   └── wpt_results.csv                  # Main generated dataset
├── simulation/
│   ├── wpt_simulation.py                # Core WPT physics engine + ablation configs
│   ├── generate_and_fill.py             # Bulk dataset generator
│   └── wpt_csv_filler_v2                # Template/filler workflow for custom CSVs
├── data_analysis/
│   ├── 01_efficiency_vs_curvature.py
│   ├── 02_efficiency_heatmap.py
│   ├── 03_misalignment_analysis.py
│   ├── 04_lut.py
│   ├── 05_statistical_tests.py
│   ├── 06_coupling_analysis.py
│   ├── 07_run_all.py                    # Runs scripts 01-06 in sequence
│   ├── 08_visualize_luts.py             # Interactive 3D LUT surfaces
│   └── *.png / *.csv / stats_report.txt # Generated analysis outputs
└── data_visualization/
    ├── README.md
    ├── requirements.txt
    └── visualization scripts
```

---

## Requirements

- Python 3.9+
- Packages:
  - `numpy>=1.20.0`
  - `pandas>=1.3.0`
  - `matplotlib>=3.4.0`
  - `scipy>=1.7.0`
  - `seaborn>=0.11.0`

## Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternative:

```bash
pip install -r data_visualization/requirements.txt
```

---

## Quick start

### 1) Generate simulation dataset

From repo root:

```bash
cd simulation
python3 generate_and_fill.py --outdir ../data
```

This creates:

- `data/wpt_results.csv`

Default sweep characteristics in current configuration:

- 18 curvature heights (`0..85 mm` in 5 mm steps)
- 10 angle values (`0..90°` in 10° steps)
- X-axis, Y-axis, and diagonal offset sets
- 9 ablation configurations
- ~49,500 rows total

### 2) Run full analysis pipeline

```bash
cd ../data_analysis
python3 07_run_all.py
```

This runs scripts 01 through 06 and generates figures/reports such as:

- `efficiency_vs_curvature.png`
- `ablation_comparison.png`
- `heatmap_h0mm.png`, `heatmap_h85mm.png`, `heatmap_comparison.png`
- `lateral_misalignment.png`, `angular_misalignment.png`, `diagonal_misalignment.png`
- `lut_full.csv`, `lut_aligned.csv`, `lut_pivot_xh.csv`, `lut_pivot_angleh.csv`, `lut_summary.csv`
- `regression_plot.png`, `effect_size_bar.png`, `correlation_matrix.png`
- `k_vs_curvature.png`, `kQ_regime_plot.png`, `frequency_splitting.png`
- `stats_report.txt`

### 3) Explore LUTs interactively (optional)

```bash
cd data_analysis
python3 08_visualize_luts.py
```

Opens interactive 3D surfaces for LUT-derived efficiency landscapes.

---

## Simulation model highlights

Core engine: `simulation/wpt_simulation.py`

### Dataclasses

- `CoilParams`
- `Pose2D`
- `AblationConfig`
- `WPTResult`

### System model

- `WPTSystem` computes coil inductances, coupling, Q factors, S-parameters, split frequencies, power, and efficiency.

### Predefined ablations

- `full_model`
- `no_skin_effect`
- `no_proximity_effect`
- `no_parasitic_cap`
- `no_medhurst_correction`
- `no_curvature_geometry`
- `no_temp_correction`
- Additional fixed-cap variants are documented in the module header.

---

## Dataset schema (main CSV)

`data/wpt_results.csv` includes:

### Experiment inputs

- `conic_curve_mm`, `x_cord`, `y_cord`, `angle`, `temp`, `Vin`, `alignment`, `group`

### Core outputs

- `Vout`, `Cin`, `Cout`, `S21_param`, `inductance`, `efficiency`

### Electrical/physics diagnostics

- Inductance terms, parasitic capacitance, SRF, resistances, `Q_tx`, `Q_rx`, `M_nH`, `k`, `kQ`

### Additional outputs

- Frequency splitting and coupling regime
- S-parameters and power

### Manual measurement placeholders

- `Vout_measured`, `Iout_measured`, `efficiency_measured`, `notes`

---

## Typical workflow

1. Update simulation parameters in `simulation/generate_and_fill.py` (heights, angles, offsets, source conditions).
2. Generate `data/wpt_results.csv`.
3. Run `data_analysis/07_run_all.py`.
4. Use generated LUT CSVs for control logic, interpolation, or experiment planning.
5. Review `stats_report.txt` and key figures for findings.

---

## Notes

- Analysis scripts expect to run from `data_analysis/` and read input from `../data/wpt_results.csv`.
- The simulation script writes where you point `--outdir`; using `../data` keeps analysis paths consistent.
- Generated plots and reports are saved in `data_analysis/`.

---

## License

No license file is currently defined. Add a `LICENSE` file (MIT, BSD-3-Clause, Apache-2.0, etc.) before public release.
