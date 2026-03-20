"""
generate_and_fill.py  — v2.0
─────────────────────────────────────────────────────────────────
Generates the full experiment CSV from methodology parameters,
runs the WPT simulation on every row, outputs a filled CSV.

Pose coverage:
  X-axis:   Y=0, X = 0→100mm in 5mm steps
  Y-axis:   X=0, Y = 5→100mm in 5mm steps (no duplicate origin)
  Diagonal: X=Y, both sweep 5→70mm in 5mm steps
            (45° diagonal — equal X and Y offset)

Angles: 0° to 90° in 10° steps (0,10,20,...,90)

Curvature: 0mm (control) + 5→45mm in 5mm steps (treatment)
"""

import csv
import sys
import os
import argparse
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpt_simulation import CoilParams, WPTSystem, Pose2D

# ═════════════════════════════════════════════════════════════
# EXPERIMENT PARAMETERS
# ═════════════════════════════════════════════════════════════

COIL = dict(
    N          = 12,
    R_outer    = 43e-3,
    R_inner    = 10e-3,
    pitch      = 1.2e-3,
    wire_diam  = 1.024e-3,
)

CURVATURE_HEIGHTS_MM = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# Lateral offsets for axis-aligned sweeps
OFFSETS_MM = list(range(0, 105, 5))       # 0→100mm

# Diagonal offsets — X=Y, swept to 70mm
# (beyond ~70mm the coils are >99mm apart diagonally
#  which exceeds meaningful coupling range)
DIAGONAL_OFFSETS_MM = list(range(5, 75, 5))  # 5→70mm

# Angles: 0→90° in 10° steps
ANGLES_DEG = list(range(0, 100, 10))          # [0,10,20,...,90]

D_AXIAL_MM = 50.0
V_IN       = 12.0
TEMP_C     = 75.0

# ═════════════════════════════════════════════════════════════
# CSV COLUMNS
# ═════════════════════════════════════════════════════════════

CORE_COLS = [
    'conic_curve_mm',
    'x_cord',
    'y_cord',
    'angle',
    'temp',
    'Vin',
    'Vout',
    'Cin',
    'Cout',
    'S21_param',
    'inductance',
    'efficiency',
]

EXTENDED_COLS = [
    'L_tx_uH',
    'L_tx_eff_uH',
    'L_rx_eff_uH',
    'C_parasitic_pF',
    'f_srf_MHz',
    'R_skin_ohm',
    'R_total_ohm',
    'proximity_factor',
    'Q_tx',
    'Q_rx',
    'M_nH',
    'k',
    'kQ',
    'k_critical',
    'S11_dB',
    'S22_dB',
    'S12_dB',
    'P_in_W',
    'P_out_W',
    'f_split_low_MHz',
    'f_split_high_MHz',
    'split_gap_kHz',
    'coupling_regime',
    'd_lateral_mm',
    'alignment',
    'group',
]

MEASUREMENT_COLS = [
    'Vout_measured',
    'Iout_measured',
    'efficiency_measured',
    'notes',
]

ALL_COLS = CORE_COLS + EXTENDED_COLS + MEASUREMENT_COLS


# ═════════════════════════════════════════════════════════════
# POSE GENERATOR
# ═════════════════════════════════════════════════════════════

def generate_poses():
    """
    Generate all (x_mm, y_mm, angle, alignment_label) combinations.

    Three alignment types:
      x-axis:   Y=0, X sweeps 0→100mm
      y-axis:   X=0, Y sweeps 5→100mm  (skip 0 to avoid duplicate)
      diagonal: X=Y, both sweep 5→70mm (45° direction)

    Each combined with angles 0→90° in 10° steps.
    """
    poses = []

    for angle in ANGLES_DEG:

        # ── X-axis: Y=0, X varies ────────────────────────────
        for x in OFFSETS_MM:
            poses.append((x, 0, angle, 'x-axis'))

        # ── Y-axis: X=0, Y varies (skip Y=0) ─────────────────
        for y in OFFSETS_MM:
            if y == 0:
                continue
            poses.append((0, y, angle, 'y-axis'))

        # ── Diagonal: X=Y ────────────────────────────────────
        # Represents coil moved at 45° from center
        # d_lateral = sqrt(x²+y²) = x*sqrt(2)
        # e.g. x=y=20mm → d_lateral ≈ 28.3mm
        for d in DIAGONAL_OFFSETS_MM:
            poses.append((d, d, angle, 'diagonal'))

    return poses


# ═════════════════════════════════════════════════════════════
# SYSTEM BUILDER
# ═════════════════════════════════════════════════════════════

def build_systems(d_axial_mm, v_in, temp_c):
    systems = {}
    print("  Building WPT systems...")
    for h_mm in CURVATURE_HEIGHTS_MM:
        tx = CoilParams(**COIL, curvature_h=0.0)
        rx = CoilParams(**COIL, curvature_h=h_mm * 1e-3)
        system = WPTSystem(
            tx=tx, rx=rx,
            freq    = 6.78e6,
            V_in    = v_in,
            I_in    = v_in / 50.0,
            d_axial = d_axial_mm * 1e-3,
            temp_c  = temp_c,
        )
        systems[h_mm] = system
        group = 'control' if h_mm == 0 else 'treatment'
        print(f"    h={h_mm:>3}mm [{group}]: "
              f"L_rx={system.L_rx*1e6:.3f}µH  "
              f"Q={system.Q_rx:.0f}  "
              f"R_total={system.R_rx_total:.3f}Ω")
    return systems


# ═════════════════════════════════════════════════════════════
# ROW BUILDER
# ═════════════════════════════════════════════════════════════

def build_row(h_mm, x_mm, y_mm, angle, alignment, system):
    pose   = Pose2D(x=x_mm * 1e-3, y=y_mm * 1e-3, theta=float(angle))
    result = system.evaluate(pose)
    group  = 'control' if h_mm == 0 else 'treatment'

    return {
        'conic_curve_mm'        : h_mm,
        'x_cord'                : x_mm,
        'y_cord'                : y_mm,
        'angle'                 : angle,
        'temp'                  : TEMP_C,
        'Vin'                   : round(result.V_in, 3),
        'Vout'                  : round(result.V_out, 6),
        'Cin'                   : round(result.C_tx_pF, 3),
        'Cout'                  : round(result.C_rx_pF, 3),
        'S21_param'             : round(result.S21_dB, 3),
        'inductance'            : round(result.L_rx_eff_uH, 5),
        'efficiency'            : round(result.efficiency_pct, 5),
        'L_tx_uH'               : round(result.L_tx_uH, 5),
        'L_tx_eff_uH'           : round(result.L_tx_eff_uH, 5),
        'L_rx_eff_uH'           : round(result.L_rx_eff_uH, 5),
        'C_parasitic_pF'        : round(result.C_parasitic_rx_pF, 4),
        'f_srf_MHz'             : round(result.f_srf_rx_MHz, 1),
        'R_skin_ohm'            : round(result.R_rx_skin, 5),
        'R_total_ohm'           : round(result.R_rx_total, 5),
        'proximity_factor'      : round(result.proximity_factor, 4),
        'Q_tx'                  : round(result.Q_tx, 2),
        'Q_rx'                  : round(result.Q_rx, 2),
        'M_nH'                  : round(result.M_nH, 5),
        'k'                     : round(result.k, 8),
        'kQ'                    : round(result.kQ, 5),
        'k_critical'            : round(result.k_critical, 8),
        'S11_dB'                : round(result.S11_dB, 3),
        'S22_dB'                : round(result.S22_dB, 3),
        'S12_dB'                : round(result.S12_dB, 3),
        'P_in_W'                : round(result.P_in_W, 5),
        'P_out_W'               : round(result.P_out_W, 8),
        'f_split_low_MHz'       : round(result.f_split_low_MHz, 4),
        'f_split_high_MHz'      : round(result.f_split_high_MHz, 4),
        'split_gap_kHz'         : round(result.split_gap_kHz, 2),
        'coupling_regime'       : result.coupling_regime,
        'd_lateral_mm'          : round(result.d_lateral_mm, 3),
        'alignment'             : alignment,
        'group'                 : group,
        'Vout_measured'         : '',
        'Iout_measured'         : '',
        'efficiency_measured'   : '',
        'notes'                 : '',
    }


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate and fill WPT experiment CSV'
    )
    parser.add_argument('--output',  default='wpt_experiment_results.csv')
    parser.add_argument('--daxial',  type=float, default=D_AXIAL_MM)
    parser.add_argument('--vin',     type=float, default=V_IN)
    parser.add_argument('--temp',    type=float, default=TEMP_C)
    args = parser.parse_args()

    poses      = generate_poses()
    total_rows = len(CURVATURE_HEIGHTS_MM) * len(poses)

    # Count pose types
    x_count   = sum(1 for p in poses if p[3] == 'x-axis')
    y_count   = sum(1 for p in poses if p[3] == 'y-axis')
    d_count   = sum(1 for p in poses if p[3] == 'diagonal')

    print("=" * 65)
    print("  WPT Experiment CSV Generator  v2.0")
    print("=" * 65)
    print(f"\n  Coil: 18AWG | OD=86mm | ID=20mm | N=12 | pitch=1.2mm")
    print(f"  Curvature heights: {CURVATURE_HEIGHTS_MM} mm")
    print(f"  Angles: {ANGLES_DEG}°")
    print(f"  d_axial: {args.daxial}mm | Vin: {args.vin}V | "
          f"Temp: {args.temp}°C")
    print(f"\n  Pose breakdown per curvature height:")
    print(f"    X-axis  (Y=0, X=0→100mm):    {x_count} poses")
    print(f"    Y-axis  (X=0, Y=5→100mm):    {y_count} poses")
    print(f"    Diagonal (X=Y, 5→70mm):      {d_count} poses")
    print(f"    Total poses per height:       {len(poses)}")
    print(f"\n  Total rows: {len(CURVATURE_HEIGHTS_MM)} heights × "
          f"{len(poses)} poses = {total_rows}\n")

    t0      = time.time()
    systems = build_systems(args.daxial, args.vin, args.temp)

    print(f"\n  Simulating {total_rows} poses...")
    rows    = []
    regimes = {'undercoupled': 0, 'critical': 0, 'overcoupled': 0}
    n_done  = 0
    step    = max(1, total_rows // 10)

    for h_mm in CURVATURE_HEIGHTS_MM:
        system = systems[h_mm]
        for x_mm, y_mm, angle, alignment in poses:
            row = build_row(h_mm, x_mm, y_mm, angle,
                            alignment, system)
            rows.append(row)
            regimes[row['coupling_regime']] = \
                regimes.get(row['coupling_regime'], 0) + 1
            n_done += 1

            if n_done % step == 0:
                pct = 100 * n_done / total_rows
                print(f"    {pct:>5.1f}%  [{n_done}/{total_rows}]  "
                      f"h={h_mm}mm x={x_mm} y={y_mm} "
                      f"θ={angle}° [{alignment}] → "
                      f"η={row['efficiency']:.3f}% "
                      f"S21={row['S21_param']:.1f}dB")

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    ctrl    = sum(1 for r in rows if r['group'] == 'control')
    treat   = sum(1 for r in rows if r['group'] == 'treatment')

    print(f"\n{'='*65}")
    print(f"  ✓ Complete in {elapsed:.1f}s")
    print(f"  ✓ {len(rows)} rows → {args.output}")
    print(f"  ✓ {len(ALL_COLS)} columns per row")
    print(f"\n  Coupling regime breakdown:")
    for regime, count in regimes.items():
        pct = 100 * count / len(rows)
        print(f"    {regime:<15}: {count:>5} rows  ({pct:.1f}%)")
    print(f"\n  Groups:")
    print(f"    control   : {ctrl}  rows  (h=0mm)")
    print(f"    treatment : {treat} rows  (h=5→45mm)")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
