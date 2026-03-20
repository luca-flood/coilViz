"""
wpt_csv_filler.py
─────────────────────────────────────────────────────────────────
Reads a CSV with measurement headers + pose columns,
fills all simulation-derived columns using wpt_simulation.py,
and writes the completed CSV.

Expected CSV format (headers in row 0, data from row 1):
  Required input columns:
    X_mm       - lateral X offset (mm)
    Y_mm       - lateral Y offset (mm)
    theta_deg  - angular misalignment (degrees)
    curvature_h_mm - Rx coil curvature height (mm)

  All other columns will be filled by simulation.

Usage:
    python wpt_csv_filler.py input.csv output.csv [options]

    Options:
      --freq      Operating frequency Hz   (default: 6.78e6)
      --vin       Input voltage V          (default: 5.0)
      --iin       Input current A          (default: 0.1)
      --daxial    Axial separation mm      (default: 50.0)
      --N         Number of turns          (default: 12)
      --router    Outer radius mm          (default: 43.0)
      --rinner    Inner radius mm          (default: 10.0)
      --pitch     Pitch mm                 (default: 1.2)
      --wdiam     Wire diameter mm         (default: 1.024)
      --rload     Load resistance Ohm      (default: matched)
"""

import csv
import sys
import argparse
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wpt_simulation import CoilParams, WPTSystem, Pose2D, WPTResult

# ─────────────────────────────────────────────────────────────
# CSV COLUMN DEFINITIONS
# These are ALL columns that will appear in the output CSV.
# Columns marked INPUT must be provided; SIMULATED are filled.
# ─────────────────────────────────────────────────────────────

INPUT_COLS = [
    'X_mm',
    'Y_mm',
    'theta_deg',
    'curvature_h_mm',
]

SIMULATED_COLS = [
    # Pose / geometry
    'd_lateral_mm',
    'd_axial_mm',

    # Coil properties
    'L_tx_uH',
    'L_rx_uH',
    'C_tx_pF',
    'C_rx_pF',
    'R_tx_ohm',
    'R_rx_ohm',
    'Q_tx',
    'Q_rx',

    # Coupling
    'M_nH',
    'k',
    'kQ',

    # S-parameters
    'S11_dB',
    'S22_dB',
    'S21_dB',
    'S12_dB',

    # Electrical
    'V_in',
    'I_in',
    'P_in_W',
    'V_out',
    'I_out',
    'P_out_W',
    'efficiency_pct',
]

# Optional measurement columns (left blank for manual entry)
MEASUREMENT_COLS = [
    'V_out_measured',
    'I_out_measured',
    'efficiency_measured_pct',
    'notes',
]

ALL_COLS = INPUT_COLS + SIMULATED_COLS + MEASUREMENT_COLS


def result_to_row(r: WPTResult, input_row: dict) -> dict:
    """Map WPTResult fields to CSV row dict"""
    row = {}

    # Pass through input columns
    for c in INPUT_COLS:
        row[c] = input_row.get(c, '')

    # Simulated values
    row['d_lateral_mm']     = round(r.d_lateral * 1e3, 4)
    row['d_axial_mm']       = round(r.d_axial * 1e3, 4)
    row['L_tx_uH']          = round(r.L_tx * 1e6, 6)
    row['L_rx_uH']          = round(r.L_rx * 1e6, 6)
    row['C_tx_pF']          = round(r.C_tx * 1e12, 4)
    row['C_rx_pF']          = round(r.C_rx * 1e12, 4)
    row['R_tx_ohm']         = round(r.R_tx, 6)
    row['R_rx_ohm']         = round(r.R_rx, 6)
    row['Q_tx']             = round(r.Q_tx, 2)
    row['Q_rx']             = round(r.Q_rx, 2)
    row['M_nH']             = round(r.M * 1e9, 6)
    row['k']                = round(r.k, 8)
    row['kQ']               = round(r.kQ, 6)
    row['S11_dB']           = round(r.S11_dB, 4)
    row['S22_dB']           = round(r.S22_dB, 4)
    row['S21_dB']           = round(r.S21_dB, 4)
    row['S12_dB']           = round(r.S12_dB, 4)
    row['V_in']             = round(r.V_in, 4)
    row['I_in']             = round(r.I_in, 4)
    row['P_in_W']           = round(r.P_in, 6)
    row['V_out']            = round(r.V_out, 6)
    row['I_out']            = round(r.I_out, 6)
    row['P_out_W']          = round(r.P_out, 8)
    row['efficiency_pct']   = round(r.efficiency, 6)

    # Blank measurement columns
    for c in MEASUREMENT_COLS:
        row[c] = input_row.get(c, '')

    return row


def generate_template_csv(path: str):
    """
    Generate an empty template CSV with all column headers.
    User fills in INPUT_COLS; simulation fills the rest.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        # Write one example row
        example = {c: '' for c in ALL_COLS}
        example['X_mm']            = '0'
        example['Y_mm']            = '0'
        example['theta_deg']       = '0'
        example['curvature_h_mm']  = '0'
        writer.writerow(example)
    print(f"Template CSV written to: {path}")
    print(f"Fill in: {INPUT_COLS}")
    print(f"Simulation will fill: {SIMULATED_COLS}")


def fill_csv(input_path: str, output_path: str, args):
    """
    Read input CSV, simulate each row, write output CSV.
    """
    # Build Tx coil (always flat)
    tx = CoilParams(
        N          = args.N,
        R_outer    = args.router * 1e-3,
        R_inner    = args.rinner * 1e-3,
        pitch      = args.pitch * 1e-3,
        wire_diam  = args.wdiam * 1e-3,
        curvature_h = 0.0,
    )

    # Cache of WPTSystem per curvature height
    system_cache = {}

    def get_system(h_mm: float) -> WPTSystem:
        if h_mm not in system_cache:
            rx = CoilParams(
                N           = args.N,
                R_outer     = args.router * 1e-3,
                R_inner     = args.rinner * 1e-3,
                pitch       = args.pitch * 1e-3,
                wire_diam   = args.wdiam * 1e-3,
                curvature_h = h_mm * 1e-3,
            )
            system_cache[h_mm] = WPTSystem(
                tx      = tx,
                rx      = rx,
                freq    = args.freq,
                V_in    = args.vin,
                I_in    = args.iin,
                d_axial = args.daxial * 1e-3,
                R_load  = args.rload if args.rload > 0 else None,
            )
        return system_cache[h_mm]

    # Read input
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)

    if not input_rows:
        print("ERROR: Input CSV is empty.")
        return

    # Check required columns
    missing = [c for c in INPUT_COLS if c not in input_rows[0]]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Required: {INPUT_COLS}")
        return

    # Process each row
    output_rows = []
    for i, row in enumerate(input_rows):
        try:
            x_mm     = float(row.get('X_mm', 0) or 0)
            y_mm     = float(row.get('Y_mm', 0) or 0)
            theta    = float(row.get('theta_deg', 0) or 0)
            h_mm     = float(row.get('curvature_h_mm', 0) or 0)

            pose     = Pose2D(x=x_mm*1e-3, y=y_mm*1e-3, theta=theta)
            system   = get_system(h_mm)
            result   = system.evaluate(pose)
            out_row  = result_to_row(result, row)
            output_rows.append(out_row)

            print(f"  Row {i+1:>4}: x={x_mm:.1f}mm y={y_mm:.1f}mm "
                  f"θ={theta:.1f}° h={h_mm:.1f}mm → "
                  f"η={result.efficiency:.4f}% k={result.k:.6f}")

        except Exception as e:
            print(f"  Row {i+1}: ERROR — {e}")
            output_rows.append({c: 'ERROR' for c in ALL_COLS})

    # Write output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✓ Processed {len(output_rows)} rows")
    print(f"✓ Output written to: {output_path}")
    print(f"  Unique curvature heights simulated: "
          f"{sorted(system_cache.keys())} mm")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='WPT CSV filler — fill simulation metrics from pose data'
    )
    parser.add_argument('input',  nargs='?', help='Input CSV path')
    parser.add_argument('output', nargs='?', help='Output CSV path')
    parser.add_argument('--template', metavar='PATH',
                        help='Generate empty template CSV and exit')
    parser.add_argument('--freq',    type=float, default=6.78e6)
    parser.add_argument('--vin',     type=float, default=5.0)
    parser.add_argument('--iin',     type=float, default=0.1)
    parser.add_argument('--daxial',  type=float, default=50.0,
                        help='Axial separation in mm')
    parser.add_argument('--N',       type=int,   default=12)
    parser.add_argument('--router',  type=float, default=43.0,
                        help='Outer radius in mm')
    parser.add_argument('--rinner',  type=float, default=10.0,
                        help='Inner radius in mm')
    parser.add_argument('--pitch',   type=float, default=1.2,
                        help='Pitch in mm')
    parser.add_argument('--wdiam',   type=float, default=1.024,
                        help='Wire diameter in mm')
    parser.add_argument('--rload',   type=float, default=0,
                        help='Load resistance in Ohm (0=matched)')

    args = parser.parse_args()

    if args.template:
        generate_template_csv(args.template)
        return

    if not args.input or not args.output:
        parser.print_help()
        print("\nExample:")
        print("  python wpt_csv_filler.py --template template.csv")
        print("  python wpt_csv_filler.py data.csv results.csv "
              "--vin 5 --iin 0.1 --daxial 50")
        return

    print(f"\nWPT CSV Filler")
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    print(f"  Freq:     {args.freq/1e6:.3f} MHz")
    print(f"  V_in:     {args.vin} V")
    print(f"  I_in:     {args.iin} A")
    print(f"  d_axial:  {args.daxial} mm\n")

    fill_csv(args.input, args.output, args)


if __name__ == '__main__':
    main()
