"""
generate_and_fill.py  v3.2
────────────────────────────────────────────────────────────────
Generates WPT simulation results across all ablation configs.

Default: single unified CSV (best for pandas analysis)
  wpt_results.csv  —  49,500 rows, ablation_name column to filter

Optional: separate CSV per ablation (--separate flag)
  wpt_full_model.csv
  wpt_no_skin_effect.csv
  ... (9 files)

Usage:
    python3 generate_and_fill.py
    python3 generate_and_fill.py --outdir results/
    python3 generate_and_fill.py --separate --outdir results/
    python3 generate_and_fill.py --daxial 40 --vin 12

Expected runtime: ~3-5 minutes on a modern laptop.
Expected output:  49,500 rows × 50 columns.
"""

import csv, os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpt_simulation import (CoilParams, WPTSystem, Pose2D,
                             AblationConfig, ABLATIONS)

# ════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════
COIL = dict(N=12, R_outer=43e-3, R_inner=10e-3,
            pitch=1.2e-3, wire_diam=1.024e-3)
CURVATURE_HEIGHTS_MM = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
OFFSETS_MM           = list(range(0, 105, 5))
DIAGONAL_OFFSETS_MM  = list(range(5, 75, 5))
ANGLES_DEG           = list(range(0, 100, 10))
D_AXIAL_MM           = 50.0
V_IN, TEMP_C         = 12.0, 23.9

# ════════════════════════════════════════════════════════════
# CSV COLUMNS
# ════════════════════════════════════════════════════════════
ALL_COLS = [
    # Ablation metadata
    'ablation_name',
    'ablation_description',
    # Experimental inputs
    'conic_curve_mm','x_cord','y_cord','angle','temp','Vin',
    'alignment','group',
    # Core outputs (matching your Google Sheets headers)
    'Vout','Cin','Cout','S21_param','inductance','efficiency',
    # Coil properties
    'L_tx_raw_uH','L_rx_raw_uH','L_tx_eff_uH','L_rx_eff_uH',
    'f_resonant_tx_MHz','f_resonant_rx_MHz','rx_detuning_kHz',
    'C_parasitic_tx_pF','C_parasitic_rx_pF',
    'f_srf_tx_MHz','f_srf_rx_MHz',
    # Resistance model
    'R_tx_dc','R_tx_skin','R_tx_total',
    'R_rx_dc','R_rx_skin','R_rx_total',
    'proximity_factor','temp_factor',
    # Quality factors
    'Q_tx','Q_rx',
    # Coupling
    'M_nH','k','kQ','k_critical','d_lateral_mm',
    # Frequency splitting
    'f_split_low_MHz','f_split_high_MHz','split_gap_kHz',
    'coupling_regime',
    # S-parameters
    'S11_dB','S22_dB','S21_dB','S12_dB',
    # Power
    'P_in_W','P_out_W',
    # Manual measurement columns (blank — fill after experiment)
    'Vout_measured','Iout_measured','efficiency_measured','notes',
]

# ════════════════════════════════════════════════════════════
# POSES
# ════════════════════════════════════════════════════════════
def generate_poses():
    """
    Returns list of (x_mm, y_mm, angle_deg, alignment_label).
    Three alignment types × 10 angles:
      x-axis:   Y=0, X = 0→100mm
      y-axis:   X=0, Y = 5→100mm
      diagonal: X=Y, 5→70mm (45° direction)
    """
    poses = []
    for angle in ANGLES_DEG:
        for x in OFFSETS_MM:
            poses.append((x, 0, angle, 'x-axis'))
        for y in OFFSETS_MM:
            if y > 0:
                poses.append((0, y, angle, 'y-axis'))
        for d in DIAGONAL_OFFSETS_MM:
            poses.append((d, d, angle, 'diagonal'))
    return poses

# ════════════════════════════════════════════════════════════
# ROW BUILDER
# ════════════════════════════════════════════════════════════
def build_row(h_mm, x_mm, y_mm, angle, alignment, system):
    pose  = Pose2D(x=x_mm*1e-3, y=y_mm*1e-3, theta=float(angle))
    group = 'control' if h_mm == 0 else 'treatment'
    r     = system.evaluate(pose, alignment=alignment,
                             group=group, conic_curve_mm=h_mm)
    return {
        'ablation_name'         : r.ablation_name,
        'ablation_description'  : r.ablation_description,
        'conic_curve_mm'        : h_mm,
        'x_cord'                : x_mm,
        'y_cord'                : y_mm,
        'angle'                 : angle,
        'temp'                  : TEMP_C,
        'Vin'                   : round(r.V_in, 3),
        'alignment'             : alignment,
        'group'                 : group,
        'Vout'                  : round(r.V_out, 6),
        'Cin'                   : round(r.C_tx_pF, 3),
        'Cout'                  : round(r.C_rx_pF, 3),
        'S21_param'             : round(r.S21_dB, 3),
        'inductance'            : round(r.L_rx_eff_uH, 5),
        'efficiency'            : round(r.efficiency_pct, 5),
        'L_tx_raw_uH'           : round(r.L_tx_raw_uH, 5),
        'L_rx_raw_uH'           : round(r.L_rx_raw_uH, 5),
        'L_tx_eff_uH'           : round(r.L_tx_eff_uH, 5),
        'L_rx_eff_uH'           : round(r.L_rx_eff_uH, 5),
        'f_resonant_tx_MHz'     : round(r.f_resonant_tx_MHz, 4),
        'f_resonant_rx_MHz'     : round(r.f_resonant_rx_MHz, 4),
        'rx_detuning_kHz'       : round(r.rx_detuning_kHz, 2),
        'C_parasitic_tx_pF'     : round(r.C_parasitic_tx_pF, 5),
        'C_parasitic_rx_pF'     : round(r.C_parasitic_rx_pF, 5),
        'f_srf_tx_MHz'          : round(r.f_srf_tx_MHz, 2),
        'f_srf_rx_MHz'          : round(r.f_srf_rx_MHz, 2),
        'R_tx_dc'               : round(r.R_tx_dc, 6),
        'R_tx_skin'             : round(r.R_tx_skin, 6),
        'R_tx_total'            : round(r.R_tx_total, 6),
        'R_rx_dc'               : round(r.R_rx_dc, 6),
        'R_rx_skin'             : round(r.R_rx_skin, 6),
        'R_rx_total'            : round(r.R_rx_total, 6),
        'proximity_factor'      : round(r.proximity_factor, 4),
        'temp_factor'           : round(r.temp_factor, 5),
        'Q_tx'                  : round(r.Q_tx, 2),
        'Q_rx'                  : round(r.Q_rx, 2),
        'M_nH'                  : round(r.M_nH, 5),
        'k'                     : round(r.k, 8),
        'kQ'                    : round(r.kQ, 5),
        'k_critical'            : round(r.k_critical, 8),
        'd_lateral_mm'          : round(r.d_lateral_mm, 3),
        'f_split_low_MHz'       : round(r.f_split_low_MHz, 4),
        'f_split_high_MHz'      : round(r.f_split_high_MHz, 4),
        'split_gap_kHz'         : round(r.split_gap_kHz, 2),
        'coupling_regime'       : r.coupling_regime,
        'S11_dB'                : round(r.S11_dB, 3),
        'S22_dB'                : round(r.S22_dB, 3),
        'S21_dB'                : round(r.S21_dB, 3),
        'S12_dB'                : round(r.S12_dB, 3),
        'P_in_W'                : round(r.P_in_W, 5),
        'P_out_W'               : round(r.P_out_W, 8),
        'Vout_measured'         : '',
        'Iout_measured'         : '',
        'efficiency_measured'   : '',
        'notes'                 : '',
    }

# ════════════════════════════════════════════════════════════
# WRITE CSV
# ════════════════════════════════════════════════════════════
def write_csv(path, rows, mode='w'):
    write_header = (mode == 'w')
    with open(path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS,
                           extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerows(rows)

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Generate WPT simulation CSV(s)'
    )
    parser.add_argument('--outdir',   default='.',
                        help='Output directory (default: current)')
    parser.add_argument('--separate', action='store_true',
                        help='Write one CSV per ablation instead of unified')
    parser.add_argument('--both', action='store_true',
                        help='Write both unified CSV AND per-ablation CSVs')
    parser.add_argument('--daxial',   type=float, default=D_AXIAL_MM,
                        help='Axial separation mm (default: 50)')
    parser.add_argument('--vin',      type=float, default=V_IN,
                        help='Input voltage V (default: 12)')
    parser.add_argument('--temp',     type=float, default=TEMP_C,
                        help='Temperature °C (default: 75)')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    poses      = generate_poses()
    n_per_abl  = len(CURVATURE_HEIGHTS_MM) * len(poses)
    total_rows = n_per_abl * len(ABLATIONS)

    x_ct = sum(1 for p in poses if p[3]=='x-axis')
    y_ct = sum(1 for p in poses if p[3]=='y-axis')
    d_ct = sum(1 for p in poses if p[3]=='diagonal')

    # Output path(s)
    do_unified  = not args.separate or args.both
    do_separate = args.separate or args.both

    if args.both:
        mode_str = "unified CSV + per-ablation CSVs"
    elif args.separate:
        mode_str = "separate CSVs"
    else:
        mode_str = "unified CSV"

    if do_unified:
        unified_path = os.path.join(args.outdir, 'wpt_results.csv')
        write_csv(unified_path, [])  # create with header
    else:
        unified_path = None

    print("=" * 65)
    print(f"  WPT Generator v3.2  —  {mode_str}")
    print("=" * 65)
    print(f"\n  Coil: 18AWG | OD=86mm | ID=20mm | N=12 | pitch=1.2mm")
    print(f"  Heights (mm): {CURVATURE_HEIGHTS_MM}")
    print(f"  Angles (°):   {ANGLES_DEG}")
    print(f"  Poses/height: {len(poses)}"
          f"  (X-axis:{x_ct}  Y-axis:{y_ct}  Diagonal:{d_ct})")
    print(f"  d_axial={args.daxial}mm  Vin={args.vin}V  Temp={args.temp}°C")
    print(f"\n  Ablations:    {len(ABLATIONS)}")
    print(f"  Rows/ablation:{n_per_abl:,}")
    print(f"  Total rows:   {total_rows:,}")
    if unified_path:
        print(f"  Output:       {unified_path}")
    else:
        print(f"  Output dir:   {os.path.abspath(args.outdir)}")
    print(f"\n  Expected runtime: ~3-5 minutes\n")

    t_all = time.time()

    for idx, abl in enumerate(ABLATIONS):
        t0 = time.time()
        print(f"  [{idx+1}/{len(ABLATIONS)}] {abl.description()}")

        # Build one WPTSystem per curvature height
        systems = {}
        for h_mm in CURVATURE_HEIGHTS_MM:
            systems[h_mm] = WPTSystem(
                tx      = CoilParams(**COIL, curvature_h=0.0),
                rx      = CoilParams(**COIL, curvature_h=h_mm*1e-3),
                freq    = 6.78e6,
                V_in    = args.vin,
                I_in    = args.vin / 50.0,
                d_axial = args.daxial * 1e-3,
                temp_c  = args.temp,
                ablation= abl,
            )

        # Print per-height summary
        print(f"  {'h':>5}  {'R_total':>8}  {'Q_eff':>7}  "
              f"{'L_eff':>9}  {'f_res_rx':>10}  {'detune':>10}")
        for h_mm, sys in systems.items():
            print(f"  {h_mm:>4}mm  "
                  f"{sys.R_rx_total:>8.4f}  "
                  f"{sys.Q_rx_eff:>7.1f}  "
                  f"{sys.L_rx*1e6:>9.4f}µH  "
                  f"{sys.f_resonant_rx/1e6:>10.4f}MHz  "
                  f"{sys.rx_detuning/1e3:>+10.1f}kHz")

        # Simulate all rows
        rows   = []
        n_done = 0
        step   = max(1, n_per_abl // 4)

        for h_mm in CURVATURE_HEIGHTS_MM:
            for x_mm, y_mm, angle, alignment in poses:
                rows.append(build_row(
                    h_mm, x_mm, y_mm, angle, alignment,
                    systems[h_mm]
                ))
                n_done += 1
                if n_done % step == 0:
                    pct = 100 * n_done // n_per_abl
                    print(f"    {pct:>3}%  [{n_done}/{n_per_abl}]  "
                          f"h={h_mm}mm θ={angle}° "
                          f"→ η={rows[-1]['efficiency']:.3f}%  "
                          f"S21={rows[-1]['S21_param']:.1f}dB")

        # Write output
        if args.separate:
            fpath = os.path.join(args.outdir, f"wpt_{abl.name}.csv")
            write_csv(fpath, rows, mode='w')
            print(f"  ✓ {len(rows):,} rows → wpt_{abl.name}.csv  "
                  f"({time.time()-t0:.1f}s)\n")
        else:
            # Append to unified CSV (skip header after first ablation)
            mode = 'a' if idx > 0 else 'w'
            write_csv(unified_path, rows, mode=mode)
            print(f"  ✓ {len(rows):,} rows appended to wpt_results.csv  "
                  f"({time.time()-t0:.1f}s)\n")

    elapsed = time.time() - t_all

    # Final summary
    print("=" * 65)
    print(f"  DONE in {elapsed:.1f}s")
    if do_unified:
        print(f"  Unified: {unified_path}")
        print(f"           {total_rows:,} rows × {len(ALL_COLS)} columns")
    if do_separate:
        print(f"  Separate: {os.path.abspath(args.outdir)}/wpt_*.csv")
        print(f"            {len(ABLATIONS)} files × {n_per_abl:,} rows each")

    print(f"\n  Quick pandas usage:")
    fname = 'wpt_results.csv' if do_unified else 'wpt_full_model.csv'
    print(f"    import pandas as pd")
    print(f"    df = pd.read_csv('{fname}')")
    if not args.separate:
        print(f"    full = df[df['ablation_name'] == 'full_model']")
        print(f"    df.groupby('ablation_name')['efficiency'].mean()")
        print(f"    df.pivot_table(values='efficiency',")
        print(f"                   index='conic_curve_mm',")
        print(f"                   columns='ablation_name',")
        print(f"                   aggfunc='mean')")
    print("=" * 65)


if __name__ == '__main__':
    main()
